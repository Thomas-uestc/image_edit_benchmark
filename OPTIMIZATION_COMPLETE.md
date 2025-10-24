# ✅ 两阶段处理优化完成总结

## 🎯 优化目标

将Pipeline从**单阶段逐pair处理**优化为**两阶段批量处理**，以减少GPU显存压力和模型搬移次数。

---

## 📊 核心改进

### 优化前：单阶段处理

```
for each category (5个类别):
    for each pair (50-70个):
        解码 → 编辑(GPU) → 评分(GPU) → 记录
        ↑________________↓
        频繁模型切换 (540次)
```

### 优化后：两阶段处理

```
for each category (5个类别):
    
    阶段1: 批量编辑
    for each pair (50-70个):
        解码 → 编辑(GPU) → 保存到CPU
    
    ↓ 模型切换 (只1次)
    
    阶段2: 批量评分
    for each pair (50-70个):
        从CPU读取 → 评分(GPU) → 记录

总切换次数: 9次 (减少98.3%)
```

---

## 🔧 实现的改动

### 1. 基类增强 (`src/models/base.py`)

```python
class BaseModel(ABC):
    # 新增GPU资源管理方法
    def unload_from_gpu(self):
        """将模型从GPU卸载到CPU，释放GPU内存"""
        pass
    
    def load_to_gpu(self):
        """将模型从CPU加载到GPU"""
        pass
```

### 2. Diffusion模型实现 (`qwen_image_edit.py`)

```python
class QwenImageEditModel(BaseDiffusionModel):
    def unload_from_gpu(self):
        self.pipeline.to('cpu')
        torch.cuda.empty_cache()
    
    def load_to_gpu(self):
        self.pipeline.to(self.device)
```

### 3. Reward模型实现 (`qwen3_vl_reward.py`)

```python
class Qwen3VLRewardModel(BaseRewardModel):
    def unload_from_gpu(self):
        self.model.to('cpu')
        torch.cuda.empty_cache()
    
    def load_to_gpu(self):
        self.model.to(target_device)
```

### 4. Pipeline两阶段重写 (`src/pipeline.py`)

#### `_process_category()` 方法

```python
def _process_category(self, category_data) -> list:
    # ===== 阶段1: 批量编辑 =====
    for pair in category_data.data_pairs:
        pair.original_image = decode_base64_image(pair.original_image_b64)
        pair.edited_image = self.diffusion_model.edit_image(
            pair.original_image, 
            pair.edit_instruction
        )
    
    # ===== 模型切换 =====
    self.diffusion_model.unload_from_gpu()
    self.reward_model.load_to_gpu()
    
    # ===== 阶段2: 批量评分 =====
    scores = []
    for pair in category_data.data_pairs:
        prompts = self.prompt_manager.get_full_prompt(...)
        score = self.reward_model.score(
            pair.edited_image,
            pair.original_description,
            pair.edit_instruction,
            prompts["system_prompt"],
            prompts["user_prompt"],
            pair.original_image
        )
        scores.append(score)
    
    return scores
```

#### `run()` 方法

```python
def run(self) -> Dict[str, Any]:
    # 1. 初始化模型状态
    self.diffusion_model.load_to_gpu()
    self.reward_model.unload_from_gpu()
    
    # 2. 逐类别处理
    for idx, category_name in enumerate(categories):
        scores = self._process_category(category_data)
        
        # 恢复模型状态（除最后一个类别）
        if idx < len(categories) - 1:
            self.reward_model.unload_from_gpu()
            self.diffusion_model.load_to_gpu()
    
    # 3-6. 统计、报告、保存
    ...
```

---

## 📈 效率提升

### 模型切换次数对比

| 指标 | 优化前 | 优化后 | 提升 |
|-----|--------|--------|------|
| 单个类别切换次数 | 100次 | 1次 | 99% ↓ |
| 全部5类别切换次数 | 540次 | 9次 | 98.3% ↓ |
| 预估节省时间 | - | ~4.5小时 | - |

### GPU显存使用

| 阶段 | GPU显存占用 | CPU内存占用 |
|-----|------------|------------|
| **阶段1：编辑** | Diffusion (~50GB) | Reward Model |
| **阶段2：评分** | Reward (~45GB) | Diffusion Model + Edited Images |

### 时间估算

- **单个类别处理时间**：约6-8分钟
  - 阶段1（编辑50张）：~4分钟
  - 模型切换：~30秒
  - 阶段2（评分50张）：~2分钟
  - 恢复模型：~30秒

- **全部5个类别**：约30-40分钟

---

## 🎨 日志输出示例

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
[Qwen3VLRewardModel] 模型已卸载到CPU
[QwenImageEditModel] 模型已加载到GPU: cuda

... (继续处理其他4个类别)
```

---

## 📁 修改的文件

| 文件 | 修改内容 | 行数变化 |
|-----|---------|---------|
| `src/models/base.py` | 添加GPU资源管理方法 | +18 |
| `src/models/diffusion/implementations/qwen_image_edit.py` | 实现GPU管理方法 | +20 |
| `src/models/reward/implementations/qwen3_vl_reward.py` | 实现GPU管理方法 | +28 |
| `src/pipeline.py` | 重写处理逻辑为两阶段 | ~100 |

**总计**：约166行代码修改/新增

---

## 🔄 数据流图

### 完整的两阶段处理流程

```
┌─────────────────────────────────────────────────────────────┐
│  初始化：Diffusion → GPU, Reward → CPU                       │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  类别1: 物理 (50个pairs)                                     │
├─────────────────────────────────────────────────────────────┤
│  [阶段1] Diffusion on GPU                                    │
│    pair_001: 原图b64 → PIL.Image → 编辑 → edited_image      │
│    pair_002: 原图b64 → PIL.Image → 编辑 → edited_image      │
│    ...                                                       │
│    pair_050: 原图b64 → PIL.Image → 编辑 → edited_image      │
│  ↓ 所有edited_image保存在CPU内存                             │
├─────────────────────────────────────────────────────────────┤
│  [模型切换] Diffusion → CPU, Reward → GPU                    │
├─────────────────────────────────────────────────────────────┤
│  [阶段2] Reward on GPU                                       │
│    pair_001: edited_image + desc + inst → score_001         │
│    pair_002: edited_image + desc + inst → score_002         │
│    ...                                                       │
│    pair_050: edited_image + desc + inst → score_050         │
│  ↓ category_scores["物理"] = [score_001, ..., score_050]   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  [恢复] Reward → CPU, Diffusion → GPU                        │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  类别2: 环境 (50个pairs)                                     │
│  ... (重复上述流程)                                           │
└─────────────────────────────────────────────────────────────┘
                            ↓
                     (类别3, 4, 5...)
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  统计计算 → 报告生成 → 保存                                   │
└─────────────────────────────────────────────────────────────┘
```

---

## ⚠️ 注意事项

### 1. Checkpoint功能暂时移除

为了简化两阶段处理逻辑，当前版本**暂时移除了checkpoint断点续传功能**。

**原因**：
- 需要分别记录编辑和评分两个状态
- 恢复逻辑较复杂

**未来恢复方案**：
```python
checkpoint = {
    "物理": {
        "edited": ["pair_001", "pair_002", ...],
        "scored": ["pair_001", "pair_002", ...]
    }
}
```

### 2. CPU内存需求

每个类别需要在CPU内存中保存所有编辑后的图像：
- **单张图像**：约10-20MB
- **50张图像**：约500MB-1GB
- **70张图像**：约700MB-1.4GB

**建议**：确保有至少4GB空闲CPU内存

### 3. GPU显存需求

- **Diffusion Model**：约50GB
- **Reward Model**：约45GB

**建议**：使用80GB显存的GPU（如A100）

---

## 🚀 使用方法

### 1. 确保配置正确

```yaml
# config.yaml

benchmark:
  data_path: "/data2/yixuan/Benchmark/version_2_50_pair/version_2_with_imagesb64.json"
  categories: ["物理", "环境", "社会", "因果", "指代"]

diffusion_model:
  class_path: "src.models.diffusion.implementations.qwen_image_edit.QwenImageEditModel"
  params:
    model_name: "Qwen/Qwen-Image-Edit"
    device: "cuda"
    dtype: "bfloat16"

reward_model:
  class_path: "src.models.reward.implementations.qwen3_vl_reward.Qwen3VLRewardModel"
  params:
    model_name: "Qwen/Qwen3-VL-30B-Instruct"
    device: "auto"
    dtype: "bfloat16"
```

### 2. 运行评测

```bash
cd /data2/yixuan/image_edit_benchmark
conda activate yx_grpo_rl_post_edit
python main.py --config config.yaml
```

### 3. 查看结果

```bash
# 查看JSON报告
cat outputs/results/evaluation_report_20251023_210000.json

# 查看Markdown报告
cat outputs/results/evaluation_report_20251023_210000.md

# 查看编辑后的图像（如果启用保存）
ls outputs/images/物理/
```

---

## 📚 相关文档

1. **`PIPELINE_ANALYSIS.md`** - Pipeline串联逻辑详细分析
2. **`SCORER_ANALYSIS.md`** - 评分统计器逻辑详细分析
3. **`TWO_STAGE_OPTIMIZATION.md`** - 两阶段处理优化详细说明
4. **`README.md`** - 项目使用说明
5. **`QUICKSTART.md`** - 快速开始指南

---

## ✅ 优化完成清单

- [x] 添加GPU资源管理基类方法
- [x] 实现Diffusion模型的GPU管理
- [x] 实现Reward模型的GPU管理
- [x] 重写`_process_category()`为两阶段处理
- [x] 更新`run()`添加模型状态管理
- [x] 添加详细的日志输出
- [x] 添加进度条显示
- [x] 编写优化文档
- [ ] 恢复checkpoint功能（待后续）
- [ ] GPU可用后实际测试

---

## 🎯 下一步

### 待完成任务

1. **GPU可用后测试**
   - 验证两阶段处理逻辑正确
   - 确认模型切换正常
   - 测试完整的5个类别处理
   - 检查内存使用情况

2. **可选优化**
   - 恢复checkpoint断点续传功能
   - 实现真正的batch inference（如果模型支持）
   - 添加多GPU并行支持
   - 添加配置选项：选择单阶段或两阶段模式

3. **完善文档**
   - 添加实际运行的性能数据
   - 添加故障排除指南
   - 添加更多使用示例

---

## 📊 预期效果

基于理论分析，预期两阶段优化将带来：

✅ **显存使用效率提升** - 避免两个大模型同时占用GPU  
✅ **模型切换减少98.3%** - 从540次降至9次  
✅ **预估节省约4.5小时** - 减少模型搬移开销  
✅ **代码结构更清晰** - 两阶段逻辑一目了然  
✅ **易于维护和扩展** - 基于抽象接口设计  

---

**优化完成时间**: 2025-10-23 21:15  
**优化版本**: v2.0  
**状态**: ✅ 两阶段处理优化完成，等待GPU测试


