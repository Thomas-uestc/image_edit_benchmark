# 🚀 两阶段处理优化 - 提升GPU利用效率

## 📋 优化概述

### 优化目标
减少GPU显存压力，避免频繁的模型搬移，提升评测效率。

### 问题背景
- **Diffusion模型**（Qwen-Image-Edit）和**Reward模型**（Qwen3-VL-30B）参数量都很大
- 现有GPU显存**不足以同时容纳两个模型**
- 原有逻辑在每个pair处理时都需要两个模型，导致**频繁的GPU显存切换**

### 优化方案
将原来的**单阶段逐pair处理**改为**两阶段批量处理**：
- **阶段1**：批量编辑所有图像（Diffusion Model在GPU）
- **阶段2**：批量评分所有图像（Reward Model在GPU）

---

## 📊 效率对比

### 原有逻辑（单阶段）

```
for category in [物理, 环境, 社会, 因果, 指代]:
    for pair in category.pairs:  # 50-70个pair
        1. 解码图像
        2. 编辑图像 (Diffusion on GPU)  ← GPU显存切换
        3. 获取prompt
        4. 评分 (Reward on GPU)          ← GPU显存切换
        5. 记录
```

**模型切换次数**：
- 每个pair需要切换2次（Diffusion → Reward, Reward → Diffusion）
- 单个类别（50个pair）：**100次切换**
- 全部5个类别（270个pair）：**540次切换**

### 优化后逻辑（两阶段）

```
for category in [物理, 环境, 社会, 因果, 指代]:
    
    # 阶段1: 批量编辑
    Diffusion on GPU
    for pair in category.pairs:  # 50-70个pair
        1. 解码图像
        2. 编辑图像
        3. 保存到CPU
    
    # 模型切换：Diffusion → CPU, Reward → GPU
    
    # 阶段2: 批量评分
    Reward on GPU
    for pair in category.pairs:
        1. 获取prompt
        2. 评分
        3. 记录
```

**模型切换次数**：
- 每个类别只需要切换1次（Diffusion → Reward）
- 单个类别：**1次切换**
- 全部5个类别：**5次切换** + 4次类别间恢复 = **9次切换**

### 效率提升

```
原有切换次数: 540次
优化后切换次数: 9次
减少比例: 98.3%
```

**预估时间节省**：
- 假设每次模型搬移需要30秒
- 原有：540 × 30秒 = 4.5小时
- 优化后：9 × 30秒 = 4.5分钟
- **节省时间：~4.5小时**

---

## 🔧 实现细节

### 1. 基类添加GPU资源管理方法

#### `BaseModel` 基类

```python
# src/models/base.py

class BaseModel(ABC):
    # ... 原有方法 ...
    
    def unload_from_gpu(self):
        """将模型从GPU卸载到CPU，释放GPU内存"""
        pass
    
    def load_to_gpu(self):
        """将模型从CPU加载到GPU"""
        pass
```

### 2. 具体模型实现GPU管理

#### `QwenImageEditModel` (Diffusion Model)

```python
# src/models/diffusion/implementations/qwen_image_edit.py

def unload_from_gpu(self):
    """将模型从GPU卸载到CPU，释放GPU内存"""
    if hasattr(self, 'pipeline') and self.pipeline is not None:
        print(f"[QwenImageEditModel] 将模型从GPU卸载到CPU...")
        self.pipeline.to('cpu')
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[QwenImageEditModel] 模型已卸载到CPU")

def load_to_gpu(self):
    """将模型从CPU加载到GPU"""
    if hasattr(self, 'pipeline') and self.pipeline is not None:
        print(f"[QwenImageEditModel] 将模型从CPU加载到GPU...")
        self.pipeline.to(self.device)
        print(f"[QwenImageEditModel] 模型已加载到GPU: {self.device}")
```

#### `Qwen3VLRewardModel` (Reward Model)

```python
# src/models/reward/implementations/qwen3_vl_reward.py

def unload_from_gpu(self):
    """将模型从GPU卸载到CPU，释放GPU内存"""
    if hasattr(self, 'model') and self.model is not None:
        print(f"[Qwen3VLRewardModel] 将模型从GPU卸载到CPU...")
        self.model.to('cpu')
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[Qwen3VLRewardModel] 模型已卸载到CPU")

def load_to_gpu(self):
    """将模型从CPU加载到GPU"""
    if hasattr(self, 'model') and self.model is not None:
        print(f"[Qwen3VLRewardModel] 将模型从CPU加载到GPU...")
        if self.device == "cuda":
            target_device = "cuda"
        elif self.device == "auto":
            target_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            target_device = self.device
        self.model.to(target_device)
        print(f"[Qwen3VLRewardModel] 模型已加载到GPU: {target_device}")
```

### 3. Pipeline两阶段处理

#### `_process_category()` 方法重写

```python
# src/pipeline.py

def _process_category(self, category_data) -> list:
    """
    处理单个类别的数据（两阶段处理优化）
    
    阶段1: 批量图像编辑（Diffusion Model在GPU）
    阶段2: 批量图像评分（Reward Model在GPU）
    """
    category_name = category_data.category_name
    
    # ===== 阶段1: 批量图像编辑 =====
    self.logger.info(f"[阶段1/2] 开始批量图像编辑 - {category_name}")
    
    for pair in tqdm(category_data.data_pairs, desc=f"[{category_name}] 编辑图像"):
        # 解码原始图像
        if pair.original_image is None:
            pair.original_image = decode_base64_image(pair.original_image_b64)
        
        # 使用扩散模型编辑图像
        edited_image = self.diffusion_model.edit_image(
            original_image=pair.original_image,
            edit_instruction=pair.edit_instruction
        )
        
        # 保存编辑后的图像（自动在CPU）
        pair.edited_image = edited_image
    
    # ===== 模型切换 =====
    self.logger.info(f"[模型切换] 卸载Diffusion模型，加载Reward模型")
    self.diffusion_model.unload_from_gpu()
    self.reward_model.load_to_gpu()
    
    # ===== 阶段2: 批量图像评分 =====
    self.logger.info(f"[阶段2/2] 开始批量图像评分 - {category_name}")
    
    scores = []
    for pair in tqdm(category_data.data_pairs, desc=f"[{category_name}] 评分图像"):
        # 获取该类别的prompt
        prompts = self.prompt_manager.get_full_prompt(
            category=category_name,
            original_description=pair.original_description,
            edit_instruction=pair.edit_instruction
        )
        
        # 使用reward模型评分
        score = self.reward_model.score(
            edited_image=pair.edited_image,
            original_description=pair.original_description,
            edit_instruction=pair.edit_instruction,
            system_prompt=prompts["system_prompt"],
            user_prompt=prompts["user_prompt"],
            original_image=pair.original_image
        )
        
        pair.score = score
        scores.append(score)
    
    self.logger.info(f"[完成] {category_name} - 平均分: {sum(scores)/len(scores):.3f}")
    
    return scores
```

#### `run()` 方法更新

```python
def run(self) -> Dict[str, Any]:
    """运行完整的评测流程（两阶段处理优化）"""
    
    # 1. 加载benchmark数据
    benchmark_data = self._load_benchmark_data()
    
    # 2. 初始化模型状态：Diffusion在GPU，Reward在CPU
    self.diffusion_model.load_to_gpu()
    self.reward_model.unload_from_gpu()
    
    # 3. 按类别处理数据
    category_scores = {}
    
    for idx, category_name in enumerate(benchmark_data.category_names, 1):
        # 处理当前类别（两阶段）
        scores = self._process_category(category_data)
        category_scores[category_name] = scores
        
        # 在处理下一个类别前，恢复模型状态
        if idx < len(benchmark_data.category_names):
            self.reward_model.unload_from_gpu()
            self.diffusion_model.load_to_gpu()
    
    # 4. 计算统计指标
    # 5. 生成报告
    # 6. 保存报告
    
    return report
```

---

## 🎯 优化效果

### 模型搬移流程

#### 全部5个类别的处理流程

```
初始化:
  Diffusion → GPU
  Reward → CPU

类别1（物理）:
  阶段1: [Diffusion on GPU] 编辑50张图像
  模型切换: Diffusion → CPU, Reward → GPU
  阶段2: [Reward on GPU] 评分50张图像
  恢复: Reward → CPU, Diffusion → GPU

类别2（环境）:
  阶段1: [Diffusion on GPU] 编辑50张图像
  模型切换: Diffusion → CPU, Reward → GPU
  阶段2: [Reward on GPU] 评分50张图像
  恢复: Reward → CPU, Diffusion → GPU

类别3（社会）:
  阶段1: [Diffusion on GPU] 编辑70张图像
  模型切换: Diffusion → CPU, Reward → GPU
  阶段2: [Reward on GPU] 评分70张图像
  恢复: Reward → CPU, Diffusion → GPU

类别4（因果）:
  阶段1: [Diffusion on GPU] 编辑50张图像
  模型切换: Diffusion → CPU, Reward → GPU
  阶段2: [Reward on GPU] 评分50张图像
  恢复: Reward → CPU, Diffusion → GPU

类别5（指代）:
  阶段1: [Diffusion on GPU] 编辑50张图像
  模型切换: Diffusion → CPU, Reward → GPU
  阶段2: [Reward on GPU] 评分50张图像
  (最后一个类别，无需恢复)
```

**总切换次数**：
- 5个类别内的模型切换：5次
- 4次类别间的模型恢复：4次
- **合计：9次**

### 日志输出示例

```
================================================================================
Starting benchmark evaluation (Two-Stage Processing)
================================================================================

============================================================
[初始化] 设置模型状态
============================================================
[QwenImageEditModel] 模型已加载到GPU: cuda
[Qwen3VLRewardModel] 模型已卸载到CPU

################################################################################
# 处理类别 [1/5]: 物理
################################################################################

============================================================
[阶段1/2] 开始批量图像编辑 - 物理
============================================================
[物理] 编辑图像: 100%|██████████| 50/50 [04:10<00:00, 5.01s/it]

============================================================
[模型切换] 卸载Diffusion模型，加载Reward模型
============================================================
[QwenImageEditModel] 将模型从GPU卸载到CPU...
[QwenImageEditModel] 模型已卸载到CPU
[Qwen3VLRewardModel] 将模型从CPU加载到GPU...
[Qwen3VLRewardModel] 模型已加载到GPU: cuda

============================================================
[阶段2/2] 开始批量图像评分 - 物理
============================================================
[物理] 评分图像: 100%|██████████| 50/50 [01:40<00:00, 2.01s/it]

============================================================
[完成] 物理 - 共处理 50 个样本
平均分: 7.234
============================================================

============================================================
[准备下一类别] 恢复模型状态：Diffusion → GPU, Reward → CPU
============================================================
[Qwen3VLRewardModel] 将模型从GPU卸载到CPU...
[QwenImageEditModel] 将模型从CPU加载到GPU...

... (继续处理其他类别)
```

---

## 📈 内存管理

### GPU显存使用模式

#### 阶段1：图像编辑
```
GPU显存分配:
  ┌─────────────────────────────────┐
  │  Diffusion Model (~40GB)        │  ← 在GPU
  ├─────────────────────────────────┤
  │  Working Memory (~10GB)         │  ← 推理临时内存
  └─────────────────────────────────┘
  Total: ~50GB

Reward Model: 在CPU
Edited Images: 保存到CPU内存（PIL.Image对象）
```

#### 阶段2：图像评分
```
GPU显存分配:
  ┌─────────────────────────────────┐
  │  Reward Model (~30GB)           │  ← 在GPU
  ├─────────────────────────────────┤
  │  Working Memory (~15GB)         │  ← 推理临时内存
  └─────────────────────────────────┘
  Total: ~45GB

Diffusion Model: 在CPU
Edited Images: 从CPU读取
```

### 模型切换时的显存清理

```python
def unload_from_gpu(self):
    self.pipeline.to('cpu')  # 将模型移到CPU
    torch.cuda.empty_cache()  # 清理GPU缓存
```

**效果**：
- 模型参数移到CPU内存
- GPU显存立即释放
- 为下一个模型腾出空间

---

## ⚡ 性能优化建议

### 1. 使用更快的图像编码

当前编辑后的图像保存为PIL.Image对象在CPU。如果CPU内存紧张，可以考虑：

```python
# 选项A: 保存为base64编码（节省内存）
pair.edited_image_b64 = encode_base64_image(edited_image)

# 选项B: 直接保存到磁盘（节省内存，但增加IO）
save_image(edited_image, f"temp/{pair.pair_id}.png")
```

### 2. 批量推理优化

如果模型支持真正的batch inference，可以进一步优化：

```python
# 批量编辑（如果pipeline支持）
edited_images = diffusion_model.batch_edit(
    images=[p.original_image for p in pairs],
    instructions=[p.edit_instruction for p in pairs]
)

# 批量评分（如果模型支持）
scores = reward_model.batch_score(
    edited_images=edited_images,
    original_descriptions=[...],
    edit_instructions=[...],
    system_prompts=[...],
    user_prompts=[...]
)
```

### 3. 多GPU并行

如果有多张GPU，可以：
- GPU0: 专门用于Diffusion Model
- GPU1: 专门用于Reward Model
- 两个阶段可以并行处理不同类别

---

## 🔄 断点续传的影响

**注意**：当前实现**暂时移除了checkpoint功能**以简化逻辑。

### 未来如何恢复checkpoint

需要保存两个状态：
1. **编辑完成状态**：记录哪些pair已完成图像编辑
2. **评分完成状态**：记录哪些pair已完成评分

```python
checkpoint_data = {
    "物理": {
        "edited": ["pair_001", "pair_002", ...],
        "scored": ["pair_001", "pair_002", ...]
    }
}
```

恢复逻辑：
```python
# 阶段1: 跳过已编辑的pair
if pair.pair_id in checkpoint["edited"]:
    continue

# 阶段2: 跳过已评分的pair
if pair.pair_id in checkpoint["scored"]:
    continue
```

---

## 📝 总结

### 优化成果

✅ **显存效率提升**：避免两个大模型同时占用GPU  
✅ **模型切换减少98.3%**：从540次降至9次  
✅ **预估节省时间~4.5小时**：减少模型搬移时间  
✅ **代码结构清晰**：两阶段处理逻辑一目了然  
✅ **易于扩展**：基于抽象接口，方便添加其他模型  

### 代码修改文件

1. **`src/models/base.py`**
   - 添加 `unload_from_gpu()` 和 `load_to_gpu()` 方法

2. **`src/models/diffusion/implementations/qwen_image_edit.py`**
   - 实现 `unload_from_gpu()` 和 `load_to_gpu()` 方法

3. **`src/models/reward/implementations/qwen3_vl_reward.py`**
   - 实现 `unload_from_gpu()` 和 `load_to_gpu()` 方法

4. **`src/pipeline.py`**
   - 重写 `_process_category()` 为两阶段处理
   - 更新 `run()` 方法，添加模型状态管理

### 使用建议

1. **GPU显存要求**：
   - 至少需要50GB显存运行Diffusion Model
   - 至少需要45GB显存运行Reward Model

2. **CPU内存要求**：
   - 需要足够内存存储一个类别的所有编辑后图像
   - 每张图像约10-20MB，50张约500MB-1GB

3. **运行时间**：
   - 单个类别（50张）：约6-8分钟（编辑+评分）
   - 全部5个类别（270张）：约30-40分钟
   - 模型搬移时间：约5分钟

---

**文档创建时间**: 2025-10-23 21:10  
**优化版本**: v2.0  
**状态**: 两阶段处理优化完成


