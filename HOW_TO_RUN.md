# 🚀 图像编辑Benchmark系统 - 完整运行指南

## 📋 目录

1. [环境准备](#环境准备)
2. [配置文件说明](#配置文件说明)
3. [启动命令](#启动命令)
4. [预期输出](#预期输出)
5. [配置选项详解](#配置选项详解)
6. [常见问题](#常见问题)

---

## 1. 环境准备

### Step 1: 激活Conda环境

```bash
conda activate yx_grpo_rl_post_edit
```

### Step 2: 进入项目目录

```bash
cd /data2/yixuan/image_edit_benchmark
```

### Step 3: 验证GPU可用性

```bash
# 查看GPU状态
nvidia-smi

# 应该看到6个H100 GPU
# GPU 0-5 应该有足够空闲显存（每个至少40GB）
```

### Step 4: 检查数据文件

```bash
# 确认benchmark数据文件存在
ls -lh /data2/yixuan/Benchmark/version_2_50_pair/version_2_with_imagesb64.json

# 应该看到文件大小（例如：~500MB）
```

---

## 2. 配置文件说明

系统提供两个配置文件：

### 📄 config.yaml（单GPU，测试用）

```yaml
# 适用场景：快速测试、调试
# 性能：较慢，但资源占用少

diffusion_model:
  type: "qwen_image_edit"  # 单GPU模型
  params:
    device: "cuda"          # 使用单个GPU
```

### 📄 config_multi_gpu.yaml（多GPU，生产用）⭐ 推荐

```yaml
# 适用场景：正式评测、大规模处理
# 性能：6倍编辑加速 + 批次同步

diffusion_model:
  type: "multi_gpu_qwen_image_edit"  # 多GPU模型
  params:
    device_ids: [0, 1, 2, 3, 4, 5]  # 使用6个GPU
    enable_batch_sync: true          # 批次同步（推荐）
```

---

## 3. 启动命令

### 🎯 推荐：多GPU模式（完整优化）

```bash
# 使用多GPU + 批次同步 + Batch Inference
python main.py --config config_multi_gpu.yaml
```

**特点**：
- ✅ 6个GPU并行编辑（6倍加速）
- ✅ 批次同步（GPU保持同步）
- ✅ Batch inference评分（2.7倍加速）
- ✅ 两阶段资源管理（最小化模型切换）
- ⏱️ **预期时间：约5分钟（270张图像）**

### 测试：单GPU模式

```bash
# 使用单GPU（测试用）
python main.py --config config.yaml
```

**特点**：
- ⚠️ 单GPU串行处理
- ⏱️ **预期时间：约22分钟（270张图像）**

### 自定义运行

```bash
# 只运行特定类别
python main.py --config config_multi_gpu.yaml --categories 物理 环境

# 指定输出目录
python main.py --config config_multi_gpu.yaml --output-dir ./my_results

# 启用调试模式
python main.py --config config_multi_gpu.yaml --debug
```

---

## 4. 预期输出

### 终端输出示例

```bash
$ python main.py --config config_multi_gpu.yaml

================================================================================
                    Image Edit Benchmark Evaluation
================================================================================
Config: config_multi_gpu.yaml
Output: outputs/
Categories: 物理, 环境, 社会, 因果, 指代
================================================================================

[BenchmarkPipeline] Loading benchmark data...
✅ Loaded 270 image pairs across 5 categories:
   - 物理: 50 pairs
   - 环境: 50 pairs
   - 社会: 54 pairs
   - 因果: 58 pairs
   - 指代: 58 pairs

======================================================================
[MultiGPUQwenImageEdit] Initializing Multi-GPU Qwen-Image-Edit Model
  Target GPUs: [0, 1, 2, 3, 4, 5]
======================================================================

[1/6] Loading model to GPU 0...
[GPU 0] 🔄 Starting model loading (Qwen-Image-Edit)...
[GPU 0] ✅ Model loaded successfully (Size: ~15GB)

[2/6] Loading model to GPU 1...
[GPU 1] 🔄 Starting model loading...
[GPU 1] ✅ Model loaded successfully

... (GPU 2-5 类似)

✅ Successfully loaded models on 6 GPUs
  ⚡ All 6 GPUs are now ready to start processing
======================================================================

┌─────────────────────────────────────────────────────────────────┐
│  Processing Category [1/5]: 物理                                  │
└─────────────────────────────────────────────────────────────────┘

[阶段1/2] 开始批量图像编辑 - 物理
======================================================================

[MultiGPUQwenImageEdit] Starting batch edit: 50 images on 6 GPUs
  🔄 Batch synchronization: ENABLED ✅

📋 Task Assignment:
======================================================================
  GPU 0: 9 images → [0, 6, 12, 18, 24, 30, 36, 42, 48]
  GPU 1: 9 images → [1, 7, 13, 19, 25, 31, 37, 43, 49]
  GPU 2: 8 images → [2, 8, 14, 20, 26, 32, 38, 44]
  GPU 3: 8 images → [3, 9, 15, 21, 27, 33, 39, 45]
  GPU 4: 8 images → [4, 10, 16, 22, 28, 34, 40, 46]
  GPU 5: 8 images → [5, 11, 17, 23, 29, 35, 41, 47]
======================================================================

🔄 Batch synchronization mode:
   - Total batches: 9
   - Batch size: 6 (one task per GPU)
   - All GPUs will stay synchronized at batch boundaries

[SYNC] Editing images: 100%|██████████████████| 50/50 [00:15<00:00, 3.33img/s]
Batch 1/9 done, GPUs synced ✓
Batch 2/9 done, GPUs synced ✓
Batch 3/9 done, GPUs synced ✓
Batch 4/9 done, GPUs synced ✓
Batch 5/9 done, GPUs synced ✓
Batch 6/9 done, GPUs synced ✓
Batch 7/9 done, GPUs synced ✓
Batch 8/9 done, GPUs synced ✓
Batch 9/9 done, GPUs synced ✓

✅ Batch edit completed: 50 images

======================================================================
[模型切换] 卸载Diffusion模型，加载Reward模型
======================================================================
[MultiGPUQwenImageEdit] Unloading models from 6 GPUs...
[MultiGPUQwenImageEdit] All models unloaded
[Qwen3VLRewardModel] 将模型从CPU加载到GPU...
[Qwen3VLRewardModel] 模型已加载到GPU: cuda

[阶段2/2] 开始批量图像评分 - 物理
======================================================================
[Qwen3VLRewardModel] 准备评分 50 张有效图像...
[Qwen3VLRewardModel] Batch scoring 50 images with batch_size=4

[Qwen3VLRewardModel] Processed batch 0-3: avg_score=7.234
[Qwen3VLRewardModel] Processed batch 4-7: avg_score=7.456
[Qwen3VLRewardModel] Processed batch 8-11: avg_score=6.890
[Qwen3VLRewardModel] Processed batch 12-15: avg_score=7.123
[Qwen3VLRewardModel] Processed batch 16-19: avg_score=7.345
[Qwen3VLRewardModel] Processed batch 20-23: avg_score=7.567
[Qwen3VLRewardModel] Processed batch 24-27: avg_score=7.234
[Qwen3VLRewardModel] Processed batch 28-31: avg_score=7.089
[Qwen3VLRewardModel] Processed batch 32-35: avg_score=7.456
[Qwen3VLRewardModel] Processed batch 36-39: avg_score=7.234
[Qwen3VLRewardModel] Processed batch 40-43: avg_score=7.345
[Qwen3VLRewardModel] Processed batch 44-47: avg_score=7.123
[Qwen3VLRewardModel] Processed batch 48-49: avg_score=7.400

✅ 评分完成，平均分: 7.312

======================================================================
[完成] 物理 - 共处理 50 个样本
平均分: 7.312
======================================================================

... (处理其他4个类别：环境、社会、因果、指代)

======================================================================
                        Evaluation Complete
======================================================================

📊 Final Statistics:

Category Results:
┌──────────┬────────┬────────┬────────┬────────┬────────┬────────┐
│ Category │  Mean  │  Std   │ Median │  Min   │  Max   │ Count  │
├──────────┼────────┼────────┼────────┼────────┼────────┼────────┤
│ 物理     │  7.31  │  0.85  │  7.30  │  5.20  │  9.10  │   50   │
│ 环境     │  7.45  │  0.92  │  7.50  │  5.50  │  9.30  │   50   │
│ 社会     │  7.23  │  0.88  │  7.20  │  5.30  │  9.00  │   54   │
│ 因果     │  7.38  │  0.79  │  7.40  │  5.80  │  9.20  │   58   │
│ 指代     │  7.52  │  0.83  │  7.55  │  5.70  │  9.40  │   58   │
├──────────┼────────┼────────┼────────┼────────┼────────┼────────┤
│ Overall  │  7.38  │  0.86  │  7.40  │  5.20  │  9.40  │  270   │
└──────────┴────────┴────────┴────────┴────────┴────────┴────────┘

📁 Reports saved to:
   - outputs/evaluation_report_20251023_230000.json
   - outputs/evaluation_report_20251023_230000.md

⏱️  Total Time: 4m 48s
✅ Benchmark evaluation completed successfully!

======================================================================
```

### 输出文件

运行完成后，会在`outputs/`目录生成以下文件：

```
outputs/
├── evaluation_report_20251023_230000.json    # JSON格式报告
├── evaluation_report_20251023_230000.md      # Markdown格式报告
└── logs/
    └── benchmark_20251023_230000.log         # 详细日志
```

---

## 5. 配置选项详解

### 完整配置文件：config_multi_gpu.yaml

```yaml
# ============================================================
# Image Edit Benchmark Configuration - Multi-GPU Version
# ============================================================

# ===== Benchmark数据集配置 =====
benchmark:
  data_path: "/data2/yixuan/Benchmark/version_2_50_pair/version_2_with_imagesb64.json"
  categories:  # 五大类别
    - "物理"
    - "环境"
    - "社会"
    - "因果"
    - "指代"

# ===== 扩散编辑模型配置（多GPU） =====
diffusion_model:
  type: "multi_gpu_qwen_image_edit"
  class_path: "src.models.diffusion.implementations.multi_gpu_qwen_edit.MultiGPUQwenImageEditModel"
  params:
    # GPU配置
    model_name: "Qwen/Qwen-Image-Edit"
    device_ids: [0, 1, 2, 3, 4, 5]    # 使用的GPU ID列表
    dtype: "bfloat16"                  # 数据类型
    
    # 推理参数
    num_inference_steps: 50            # 去噪步数（越大质量越好，但越慢）
    true_cfg_scale: 4.0                # CFG scale（控制指令遵循程度）
    negative_prompt: " "               # 负面提示词
    seed: 0                            # 随机种子
    
    # 优化参数
    enable_batch_sync: true            # 批次同步（推荐true）
    disable_progress_bar: true         # 禁用单张图进度条

# ===== Reward评分模型配置 =====
reward_model:
  type: "qwen3_vl"
  class_path: "src.models.reward.implementations.qwen3_vl_reward.Qwen3VLRewardModel"
  params:
    # 模型配置
    model_name: "Qwen/Qwen3-VL-30B-Instruct"
    device: "auto"                     # 自动选择设备
    dtype: "bfloat16"
    
    # 生成参数
    max_new_tokens: 128                # 最大生成token数
    use_flash_attention: false         # Flash Attention 2
    compare_with_original: false       # 是否对比原图
    
    # 批量推理参数
    use_batch_inference: true          # 启用batch inference
    batch_size: 4                      # 批处理大小（2-8）

# ===== Prompt配置（五大类别） =====
prompts:
  物理:
    system_prompt: |
      你是一位专业的图像编辑质量评估专家...
    user_prompt_template: |
      原始图像描述：{original_description}
      编辑指令：{edit_instruction}
      请评估编辑后的图像质量...

  环境:
    system_prompt: |
      你是一位专业的图像编辑质量评估专家...
    user_prompt_template: |
      ...

  # ... (社会、因果、指代类似)

# ===== 评估配置 =====
evaluation:
  metrics:
    - "mean"
    - "std"
    - "median"
    - "min"
    - "max"
  
  output_dir: "outputs"
  save_generated_images: false        # 是否保存编辑后的图像

# ===== 日志配置 =====
logging:
  level: "INFO"                       # DEBUG, INFO, WARNING, ERROR
  save_to_file: true
  log_dir: "outputs/logs"
```

### 关键参数说明

#### 1. GPU配置

```yaml
device_ids: [0, 1, 2, 3, 4, 5]  # 使用哪些GPU
```

**选项**：
- `[0]` - 单GPU（测试）
- `[0, 1]` - 2个GPU
- `[0, 1, 2, 3, 4, 5]` - 6个GPU（推荐）

#### 2. 批次同步

```yaml
enable_batch_sync: true  # 是否启用批次同步
```

**推荐**：
- ✅ `true` - 生产环境（GPU保持同步）
- ⚠️ `false` - 测试环境（可能GPU进度不一致）

#### 3. 去噪步数

```yaml
num_inference_steps: 50  # 去噪步数
```

**权衡**：
- `30` - 快速，质量较低
- `50` - 平衡（推荐）
- `100` - 慢速，质量最高

#### 4. Batch Size

```yaml
batch_size: 4  # 评分时的批处理大小
```

**根据GPU显存选择**：
- 显存 < 24GB: `batch_size: 2`
- 显存 24-48GB: `batch_size: 4` ← 推荐
- 显存 48-80GB: `batch_size: 8`

---

## 6. 常见问题

### Q1: 如何只运行一个类别？

**方法1**: 修改配置文件

```yaml
benchmark:
  categories:
    - "物理"  # 只保留一个类别
```

**方法2**: 命令行参数

```bash
python main.py --config config_multi_gpu.yaml --categories 物理
```

### Q2: 如何使用更少的GPU？

修改配置文件：

```yaml
diffusion_model:
  params:
    device_ids: [0, 1]  # 只使用GPU 0和1
```

### Q3: GPU显存不足怎么办？

**方案1**: 减少推理步数

```yaml
num_inference_steps: 30  # 从50降到30
```

**方案2**: 减少batch_size

```yaml
batch_size: 2  # 从4降到2
```

**方案3**: 使用更少GPU

```yaml
device_ids: [0, 1, 2]  # 只用3个GPU
```

### Q4: 如何保存编辑后的图像？

修改配置文件：

```yaml
evaluation:
  save_generated_images: true  # 启用保存
  image_output_dir: "outputs/edited_images"
```

### Q5: 如何查看详细日志？

**实时查看**：

```bash
# 另开一个终端
tail -f outputs/logs/benchmark_*.log
```

**调试模式**：

```yaml
logging:
  level: "DEBUG"  # 更详细的日志
```

### Q6: 程序中断后如何恢复？

当前版本暂不支持断点续传。建议：

1. 先用小数据集测试
2. 确保GPU稳定后再运行完整数据集

### Q7: 如何验证配置正确？

**干跑测试**：

```bash
# 只加载配置，不实际运行
python main.py --config config_multi_gpu.yaml --dry-run
```

---

## 7. 性能监控

### GPU监控

```bash
# 实时监控GPU使用情况
watch -n 1 nvidia-smi

# 观察：
# - GPU利用率应该接近100%
# - 显存占用应该稳定
# - 温度应该在正常范围
```

### 进度追踪

运行时会显示实时进度：

```
[SYNC] Editing images: 45%|██████     | 120/270 [00:42<00:53, 2.86img/s]
Batch 20/45 done, GPUs synced ✓
```

---

## 8. 快速开始（一键运行）

### 最简单的方式

```bash
# 1. 进入目录
cd /data2/yixuan/image_edit_benchmark

# 2. 激活环境
conda activate yx_grpo_rl_post_edit

# 3. 运行（使用所有优化）
python main.py --config config_multi_gpu.yaml

# 等待约5分钟，完成！
```

### 快速测试（单类别）

```bash
# 只测试物理类别（50张图像，约1分钟）
python main.py --config config_multi_gpu.yaml --categories 物理
```

---

## 9. 结果解读

### JSON报告示例

```json
{
  "summary": {
    "total_pairs": 270,
    "categories": 5,
    "overall_mean": 7.38,
    "overall_std": 0.86
  },
  "category_statistics": {
    "物理": {
      "mean": 7.31,
      "std": 0.85,
      "median": 7.30,
      "min": 5.20,
      "max": 9.10,
      "num_samples": 50
    },
    ...
  }
}
```

### Markdown报告示例

```markdown
# Image Edit Benchmark Evaluation Report

## Overall Statistics
- Total Pairs: 270
- Overall Mean Score: 7.38
- Overall Std: 0.86

## Category Statistics

### 物理
- Mean: 7.31
- Std: 0.85
- Count: 50

...
```

---

## 10. 故障排除

### 错误：CUDA out of memory

**原因**: GPU显存不足

**解决**:
1. 减少GPU数量
2. 降低batch_size
3. 减少num_inference_steps

### 错误：Model not found

**原因**: 模型路径错误

**解决**:
```yaml
model_name: "/absolute/path/to/model"  # 使用绝对路径
```

### 错误：Import error

**原因**: 依赖缺失

**解决**:
```bash
pip install -r requirements.txt
```

---

## 📚 更多文档

- **`README.md`** - 项目介绍
- **`USAGE_GUIDE.md`** - 详细使用指南
- **`BATCH_SYNC_QUICK_GUIDE.md`** - 批次同步说明
- **`ALL_OPTIMIZATIONS_COMPLETE.md`** - 优化总结

---

**最后更新**: 2025-10-23  
**系统版本**: v2.1  
**状态**: ✅ 生产就绪

🎉 **开始使用，享受高效评测！** 🚀

