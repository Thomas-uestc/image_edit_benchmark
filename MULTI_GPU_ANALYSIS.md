# 🖥️ 多GPU任务分配逻辑分析

## 📊 当前系统配置

### GPU硬件信息
```
系统GPU配置：6张 NVIDIA H100 80GB HBM3
├─ GPU 0: 80GB (可用 ~24GB)
├─ GPU 1: 80GB (可用 ~23GB)
├─ GPU 2: 80GB (可用 ~25GB)
├─ GPU 3: 80GB (可用 ~25GB)
├─ GPU 4: 80GB (可用 ~24GB)
└─ GPU 5: 80GB (可用 ~24GB)

总显存：480GB
当前可用：约144GB (其他任务占用约336GB)
```

---

## 🔍 当前任务分配逻辑分析

### 1. Qwen-Image-Edit（扩散模型）当前实现

#### 代码位置
`src/models/diffusion/implementations/qwen_image_edit.py`

#### 关键代码
```python
def _initialize(self):
    from diffusers import QwenImageEditPipeline
    
    # 配置参数
    self.device = self.config.get("device", "cuda")  # ← 关键配置
    
    # 加载pipeline
    self.pipeline = QwenImageEditPipeline.from_pretrained(self.model_name)
    
    # 设置数据类型
    self.pipeline.to(torch.bfloat16)
    
    # 移动到设备
    self.pipeline.to(self.device)  # ← 这里决定任务分配
```

#### 当前配置
```yaml
# config.yaml
diffusion_model:
  params:
    device: "cuda"  # ← 单GPU配置
```

### 2. 当前任务分配策略

#### 策略：**单GPU串行处理**

```
Pipeline加载：
  self.pipeline.to("cuda")
  └─→ 整个pipeline加载到单个GPU (默认cuda:0)

图像编辑：
  for pair in pairs:
      edited_image = pipeline(image, prompt)
      └─→ 所有推理都在同一个GPU上串行执行

GPU使用情况：
  GPU 0: ████████████ (使用中)
  GPU 1: ____________ (空闲)
  GPU 2: ____________ (空闲)
  GPU 3: ____________ (空闲)
  GPU 4: ____________ (空闲)
  GPU 5: ____________ (空闲)
```

#### 特点
- ✅ **实现简单**：代码逻辑清晰
- ✅ **稳定可靠**：不涉及复杂的并行逻辑
- ❌ **GPU利用率低**：只使用1/6的GPU资源
- ❌ **处理速度慢**：所有图像串行处理

---

## 🚀 多GPU并行策略

### 策略1: Pipeline模型并行（Model Parallelism）

#### 原理
将Pipeline的不同组件分布到不同的GPU上。

#### 实现方式

##### A. 使用 `device_map="balanced"`

```python
# 修改 _initialize() 方法
def _initialize(self):
    from diffusers import QwenImageEditPipeline
    
    # 使用device_map自动分配
    self.pipeline = QwenImageEditPipeline.from_pretrained(
        self.model_name,
        device_map="balanced",  # 自动平衡分配到多个GPU
        torch_dtype=torch.bfloat16
    )
```

**配置**：
```yaml
diffusion_model:
  params:
    device: "auto"  # 或 "balanced"
    use_device_map: true
```

**效果**：
```
Pipeline组件分布：
  Text Encoder    → GPU 0
  VAE Encoder     → GPU 1
  UNet (部分)     → GPU 2, 3, 4
  VAE Decoder     → GPU 5
```

**优点**：
- ✅ 单个样本推理速度可能更快（如果模型很大）
- ✅ 可以处理超大模型

**缺点**：
- ❌ GPU间通信开销
- ❌ 每次只能处理一个样本
- ❌ 不适合Qwen-Image-Edit这种相对小的模型

---

### 策略2: 数据并行（Data Parallelism）⭐ 推荐

#### 原理
在多个GPU上加载相同的模型副本，每个GPU处理不同的图像。

#### 实现方式

##### A. 多进程并行（最推荐）

```python
import torch.multiprocessing as mp
from queue import Queue

class MultiGPUQwenImageEditModel(BaseDiffusionModel):
    """
    多GPU并行的Qwen-Image-Edit模型
    """
    
    def _initialize(self):
        # 获取可用GPU数量
        self.num_gpus = torch.cuda.device_count()
        self.device_ids = self.config.get("device_ids", list(range(self.num_gpus)))
        
        print(f"[MultiGPUQwenImageEdit] 检测到 {self.num_gpus} 个GPU")
        print(f"[MultiGPUQwenImageEdit] 使用GPU: {self.device_ids}")
        
        # 为每个GPU创建一个pipeline实例
        self.pipelines = {}
        for gpu_id in self.device_ids:
            self.pipelines[gpu_id] = self._load_pipeline_on_gpu(gpu_id)
    
    def _load_pipeline_on_gpu(self, gpu_id: int):
        """在指定GPU上加载pipeline"""
        from diffusers import QwenImageEditPipeline
        
        device = f"cuda:{gpu_id}"
        print(f"[GPU {gpu_id}] 加载Pipeline...")
        
        pipeline = QwenImageEditPipeline.from_pretrained(self.model_name)
        pipeline.to(torch.bfloat16)
        pipeline.to(device)
        
        print(f"[GPU {gpu_id}] Pipeline加载完成")
        return pipeline
    
    def batch_edit(self, images: list, instructions: list, **kwargs) -> list:
        """
        多GPU并行批量编辑
        """
        n = len(images)
        num_workers = len(self.device_ids)
        
        # 将任务分配到多个GPU
        tasks_per_gpu = [[] for _ in range(num_workers)]
        for idx, (img, inst) in enumerate(zip(images, instructions)):
            gpu_idx = idx % num_workers
            tasks_per_gpu[gpu_idx].append((idx, img, inst))
        
        # 使用多进程并行处理
        with mp.Pool(processes=num_workers) as pool:
            results = pool.starmap(
                self._worker_edit,
                [(gpu_id, tasks, kwargs) for gpu_id, tasks in enumerate(tasks_per_gpu)]
            )
        
        # 合并结果并按原顺序排列
        all_results = {}
        for gpu_results in results:
            all_results.update(gpu_results)
        
        edited_images = [all_results[i] for i in range(n)]
        return edited_images
    
    def _worker_edit(self, gpu_id: int, tasks: list, kwargs: dict) -> dict:
        """
        单个GPU上的worker进程
        """
        results = {}
        pipeline = self.pipelines[self.device_ids[gpu_id]]
        
        for idx, img, inst in tasks:
            edited = pipeline(
                image=img,
                prompt=inst,
                **kwargs
            ).images[0]
            results[idx] = edited
        
        return results
```

**配置**：
```yaml
diffusion_model:
  params:
    device_ids: [0, 1, 2, 3, 4, 5]  # 使用哪些GPU
    parallel_mode: "data_parallel"
```

**效果**：
```
6个GPU并行处理：
  GPU 0: 处理 pair_0, pair_6, pair_12, ...  (每6个取1个)
  GPU 1: 处理 pair_1, pair_7, pair_13, ...
  GPU 2: 处理 pair_2, pair_8, pair_14, ...
  GPU 3: 处理 pair_3, pair_9, pair_15, ...
  GPU 4: 处理 pair_4, pair_10, pair_16, ...
  GPU 5: 处理 pair_5, pair_11, pair_17, ...

处理50个样本：
  串行: 50 × 5秒 = 250秒 ≈ 4.2分钟
  6GPU并行: (50/6) × 5秒 ≈ 42秒 ≈ 0.7分钟
  加速比: 6倍
```

**优点**：
- ✅ **线性加速**：6个GPU接近6倍速度
- ✅ **高GPU利用率**：所有GPU都工作
- ✅ **实现相对简单**：逻辑清晰

**缺点**：
- ⚠️ **内存占用**：每个GPU都要加载完整模型
- ⚠️ **初始化时间**：需要在每个GPU上加载模型

---

##### B. 使用Ray框架（更灵活）

```python
import ray
from ray import remote

@ray.remote(num_gpus=1)
class GPUWorker:
    """
    运行在单个GPU上的worker
    """
    def __init__(self, model_name: str, gpu_id: int):
        import torch
        from diffusers import QwenImageEditPipeline
        
        self.device = f"cuda:{gpu_id}"
        self.pipeline = QwenImageEditPipeline.from_pretrained(model_name)
        self.pipeline.to(torch.bfloat16)
        self.pipeline.to(self.device)
    
    def edit_image(self, image, instruction, **kwargs):
        output = self.pipeline(image=image, prompt=instruction, **kwargs)
        return output.images[0]

class RayMultiGPUQwenImageEditModel(BaseDiffusionModel):
    def _initialize(self):
        # 初始化Ray
        if not ray.is_initialized():
            ray.init()
        
        # 创建GPU workers
        self.num_gpus = 6
        self.workers = [
            GPUWorker.remote(self.model_name, gpu_id)
            for gpu_id in range(self.num_gpus)
        ]
    
    def batch_edit(self, images: list, instructions: list, **kwargs) -> list:
        # 分配任务
        futures = []
        for idx, (img, inst) in enumerate(zip(images, instructions)):
            worker = self.workers[idx % self.num_gpus]
            future = worker.edit_image.remote(img, inst, **kwargs)
            futures.append(future)
        
        # 等待所有结果
        edited_images = ray.get(futures)
        return edited_images
```

**优点**：
- ✅ **更灵活的调度**：Ray自动负载均衡
- ✅ **容错性好**：worker失败可以重试
- ✅ **易于监控**：Ray Dashboard

**缺点**：
- ⚠️ **额外依赖**：需要安装Ray
- ⚠️ **学习曲线**：Ray的概念需要理解

---

### 策略3: Pipeline级并行

#### 原理
在Pipeline的不同阶段并行处理不同的样本。

```
阶段流水线：
  时刻1: GPU0编辑pair_1, GPU1空闲,     GPU2空闲
  时刻2: GPU0编辑pair_2, GPU1评分pair_1, GPU2空闲
  时刻3: GPU0编辑pair_3, GPU1评分pair_2, GPU2后处理pair_1
  ...
```

**优点**：
- ✅ 最大化GPU利用率
- ✅ 隐藏部分延迟

**缺点**：
- ❌ 实现非常复杂
- ❌ 不适合当前场景（两阶段处理已经很好）

---

## 📊 不同策略对比

| 策略 | GPU利用率 | 加速比 | 实现复杂度 | 内存占用 | 推荐度 |
|-----|----------|--------|-----------|---------|--------|
| **单GPU串行** | 16.7% (1/6) | 1x | ⭐ 简单 | 低 | ⭐ |
| **模型并行** | 100% | 1.2-1.5x | ⭐⭐⭐ 复杂 | 低 | ⭐⭐ |
| **数据并行（多进程）** | 100% | 5-6x | ⭐⭐ 中等 | 高 | ⭐⭐⭐⭐⭐ |
| **数据并行（Ray）** | 100% | 5-6x | ⭐⭐⭐ 较复杂 | 高 | ⭐⭐⭐⭐ |
| **Pipeline并行** | 100% | 2-3x | ⭐⭐⭐⭐ 很复杂 | 中 | ⭐⭐ |

---

## 🎯 针对您的场景的最佳方案

### 推荐：数据并行（多进程）+ 两阶段处理

```
整体流程：
  for category in [物理, 环境, 社会, 因果, 指代]:
      
      # 阶段1: 多GPU并行编辑
      Diffusion Models on GPU 0-5
      ├─ GPU 0: 编辑 pair_0, pair_6, pair_12, ...
      ├─ GPU 1: 编辑 pair_1, pair_7, pair_13, ...
      ├─ GPU 2: 编辑 pair_2, pair_8, pair_14, ...
      ├─ GPU 3: 编辑 pair_3, pair_9, pair_15, ...
      ├─ GPU 4: 编辑 pair_4, pair_10, pair_16, ...
      └─ GPU 5: 编辑 pair_5, pair_11, pair_17, ...
      
      # 保存所有编辑结果到CPU
      
      # 模型切换
      Diffusion → CPU
      Reward → GPU 0 (单GPU即可，VLM通常不支持很好的数据并行)
      
      # 阶段2: 单GPU评分
      Reward Model on GPU 0
      for pair in pairs:
          score = reward_model(edited_image, ...)
```

### 性能预估

#### 单个类别（50张）

**编辑阶段**：
- 单GPU: 50 × 5秒 = 250秒 ≈ 4.2分钟
- 6GPU并行: (50/6) × 5秒 ≈ 42秒 + 开销 ≈ **1分钟**

**评分阶段**：
- 单GPU: 50 × 2秒 = 100秒 ≈ **1.7分钟**

**总计**：约2.7分钟（vs 原来的6分钟）

#### 全部5个类别（270张）

**编辑阶段总计**：
- 6GPU并行: 约 **5分钟**（vs 原来的21分钟）

**评分阶段总计**：
- 单GPU: 约 **9分钟**

**总时间**：约14分钟（vs 原来的30-40分钟）

**加速比**：约2-3倍

---

## 💾 内存考虑

### Qwen-Image-Edit模型大小估算

```
模型组件：
  - Text Encoder: ~1GB
  - UNet: ~15GB
  - VAE: ~2GB
  - 其他: ~2GB
  ----------------
  总计: ~20GB

6个GPU加载：
  每个GPU: ~20GB
  总占用: ~120GB
```

### 显存可用性检查

```
当前可用显存: ~144GB
需要显存: ~120GB
剩余: ~24GB ✅ 足够
```

**结论**：您的6张H100完全可以支持数据并行！

---

## 🔧 实现建议

### 短期（立即可用）

1. **修改配置支持指定GPU**
   ```yaml
   diffusion_model:
     params:
       device: "cuda:0"  # 可以指定具体GPU
   ```

2. **手动并行测试**
   - 在不同GPU上启动多个进程
   - 每个进程处理不同的样本子集
   - 最后合并结果

### 中期（推荐实现）

1. **实现MultiGPUQwenImageEditModel类**
   - 支持自动多GPU数据并行
   - 保持与现有接口兼容
   - 添加配置选项启用/禁用

2. **配置增强**
   ```yaml
   diffusion_model:
     params:
       parallel_mode: "data_parallel"  # single, data_parallel, model_parallel
       device_ids: [0, 1, 2, 3, 4, 5]
       num_workers: 6
   ```

### 长期（进一步优化）

1. **Reward模型也支持多GPU**（如果可能）
2. **使用Ray实现更灵活的调度**
3. **添加自动负载均衡**
4. **实现动态GPU分配**

---

## 📝 总结

### 当前状态
- ✅ 使用**单GPU串行处理**
- ✅ 实现简单稳定
- ❌ GPU利用率低（16.7%）
- ❌ 处理速度慢

### 优化后（数据并行）
- ✅ **6个GPU并行处理**
- ✅ GPU利用率高（100%）
- ✅ **加速比: 2-3倍**
- ✅ 实现复杂度可控

### 关键优势
1. **充分利用硬件**：6张H100全部投入使用
2. **显著加速**：编辑阶段从21分钟降至5分钟
3. **架构兼容**：与现有两阶段处理完美结合
4. **扩展性好**：可以轻松扩展到更多GPU

---

**文档创建时间**: 2025-10-23 21:30  
**系统配置**: 6× NVIDIA H100 80GB  
**状态**: 多GPU分析完成，等待实现


