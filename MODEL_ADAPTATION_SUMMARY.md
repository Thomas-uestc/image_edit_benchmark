# 模型适配总结

## ✅ 已完成的模型适配

### 1. Qwen-Image-Edit 扩散编辑模型 ✅

**文件位置**: `src/models/diffusion/implementations/qwen_image_edit.py`

#### 模型信息
- **模型**: Qwen/Qwen-Image-Edit
- **类型**: 图像编辑扩散模型
- **数据类型**: bfloat16（推荐）
- **特点**: 基于Instruct-Pix2Pix架构，针对中文优化

#### 实现细节
```python
class QwenImageEditModel(BaseDiffusionModel):
    def _initialize(self):
        # 加载 QwenImageEditPipeline
        # 设置 bfloat16 精度
        # 移动到 GPU
        
    def edit_image(self, original_image, edit_instruction):
        # 执行图像编辑
        # 返回编辑后的图像
```

#### 配置参数
```yaml
diffusion_model:
  class_path: "src.models.diffusion.implementations.qwen_image_edit.QwenImageEditModel"
  params:
    model_name: "Qwen/Qwen-Image-Edit"
    device: "cuda"
    dtype: "bfloat16"
    num_inference_steps: 50
    true_cfg_scale: 4.0
    negative_prompt: " "
    seed: 0
```

#### 核心功能
- ✅ 单张图像编辑
- ✅ 批量图像编辑
- ✅ 随机种子控制
- ✅ GPU内存自动管理
- ✅ 进度条显示（可配置）

---

### 2. Qwen3-VL Reward评分模型 ✅

**文件位置**: `src/models/reward/implementations/qwen3_vl_reward.py`

#### 模型信息
- **模型**: Qwen/Qwen3-VL-30B-Instruct（或其他规格）
- **类型**: Vision-Language 多模态模型
- **数据类型**: bfloat16
- **特点**: 支持图像+文本理解，可进行图像质量评分

#### 实现细节
```python
class Qwen3VLRewardModel(BaseRewardModel):
    def _initialize(self):
        # 加载 AutoModelForImageTextToText
        # 加载 AutoProcessor
        # 支持 Flash Attention 2（可选）
        
    def score(self, edited_image, original_description, 
              edit_instruction, system_prompt, user_prompt):
        # 构建 messages（包含图像和文本）
        # 生成评分文本
        # 提取数字分数（0-10）
        # 返回分数
```

#### 配置参数
```yaml
reward_model:
  class_path: "src.models.reward.implementations.qwen3_vl_reward.Qwen3VLRewardModel"
  params:
    model_name: "Qwen/Qwen3-VL-30B-Instruct"
    device: "auto"  # 自动分配GPU
    dtype: "bfloat16"
    max_new_tokens: 128
    use_flash_attention: false
    compare_with_original: false  # 是否对比原图
```

#### 核心功能
- ✅ 单图像评分
- ✅ 批量图像评分
- ✅ 支持原图对比（可选）
- ✅ 多种分数格式自动解析
- ✅ Flash Attention 2支持
- ✅ 灵活的prompt系统

#### 分数提取
支持多种输出格式：
- `"Score: 8.5"` → 8.5
- `"8.5/10"` → 8.5
- `"Rating: 8.5"` → 8.5
- `"8.5"` → 8.5

如果无法解析，返回默认分数 5.0

---

## 📊 类别特定Prompt

每个类别都有专门的评分prompt：

### 物理类别
- **关注点**: 光照、阴影、反射、物理规律
- **评分标准**: 物理一致性

### 环境类别
- **关注点**: 季节、天气、光照、氛围
- **评分标准**: 环境一致性

### 社会类别
- **关注点**: 文化、社会适应性
- **评分标准**: 文化/社会一致性

### 因果类别
- **关注点**: 因果逻辑关系
- **评分标准**: 因果逻辑正确性

### 指代类别
- **关注点**: 目标对象识别准确性
- **评分标准**: 指代准确性

所有prompt都经过优化，引导模型输出标准化格式的分数。

---

## 🔧 使用指南

### 基本使用

#### 1. 配置模型
编辑 `config.yaml`，确保以下配置正确：
```yaml
benchmark:
  data_path: "/data2/yixuan/Benchmark/version_2_50_pair/version_2_with_imagesb64.json"
  categories: ["物理", "环境", "社会", "因果", "指代"]

diffusion_model:
  class_path: "src.models.diffusion.implementations.qwen_image_edit.QwenImageEditModel"
  params:
    model_name: "Qwen/Qwen-Image-Edit"

reward_model:
  class_path: "src.models.reward.implementations.qwen3_vl_reward.Qwen3VLRewardModel"
  params:
    model_name: "Qwen/Qwen3-VL-30B-Instruct"
```

#### 2. 运行评测
```bash
# 激活环境
source /data2/yixuan/miniconda3/etc/profile.d/conda.sh
conda activate yx_grpo_rl_post_edit

# 进入项目目录
cd /data2/yixuan/image_edit_benchmark

# 运行完整评测
python main.py --config config.yaml
```

#### 3. 断点续传
如果评测中断：
```bash
python main.py --config config.yaml --resume
```

---

## 🎯 当前状态

### 完成度：95%

- ✅ 数据加载模块（100%）
- ✅ Qwen-Image-Edit扩散模型（100%）
- ✅ Qwen3-VL Reward模型（100%）
- ✅ 评估统计模块（100%）
- ✅ Pipeline框架（100%）
- ✅ 配置系统（100%）
- ✅ 文档（100%）
- ⚠️ 模型测试（待GPU可用后测试）

### 待测试
- [ ] Qwen-Image-Edit模型加载和编辑功能
- [ ] Qwen3-VL模型加载和评分功能
- [ ] 完整Pipeline运行（270条数据）

---

## 🚀 下一步

### 当GPU可用时

1. **测试扩散模型**
   ```bash
   python tools/test_qwen_model.py
   ```

2. **测试完整Pipeline（小规模）**
   修改配置只测试少量数据：
   ```python
   # 在配置中或代码中限制数据量
   # 例如只测试前5条数据
   ```

3. **运行完整评测**
   ```bash
   python main.py --config config.yaml
   ```

---

## 🔍 高级配置

### 使用本地模型路径
如果模型已下载到本地：
```yaml
diffusion_model:
  params:
    model_name: "/path/to/local/Qwen-Image-Edit"

reward_model:
  params:
    model_name: "/path/to/local/Qwen3-VL-30B-Instruct"
```

### 指定GPU设备
```yaml
diffusion_model:
  params:
    device: "cuda:1"  # 使用第二块GPU

reward_model:
  params:
    device: "cuda:2"  # 使用第三块GPU
```

### 启用Flash Attention（Reward模型）
```yaml
reward_model:
  params:
    use_flash_attention: true  # 多图场景下推荐
```

### 启用原图对比评分
```yaml
reward_model:
  params:
    compare_with_original: true  # 评分时同时看原图和编辑图
```

---

## 📝 模型替换指南

### 替换扩散模型

1. 在 `src/models/diffusion/implementations/` 创建新文件
2. 继承 `BaseDiffusionModel`
3. 实现 `_initialize()` 和 `edit_image()` 方法
4. 在配置中指定新模型类路径

### 替换Reward模型

1. 在 `src/models/reward/implementations/` 创建新文件
2. 继承 `BaseRewardModel`
3. 实现 `_initialize()` 和 `score()` 方法
4. 在配置中指定新模型类路径

---

## 📊 预期输出

### 评测完成后

#### 1. JSON报告
```
outputs/results/evaluation_report_YYYYMMDD_HHMMSS.json
```

包含：
- 各类别详细统计（mean, std, median, min, max）
- 整体统计
- 元数据

#### 2. Markdown报告
```
outputs/results/evaluation_report_YYYYMMDD_HHMMSS.md
```

人类可读格式，包含所有统计数据

#### 3. 生成的图像
```
outputs/images/
├── 物理/
│   ├── 00000_物理_medium.png
│   └── ...
├── 环境/
├── 社会/
├── 因果/
└── 指代/
```

#### 4. 日志文件
```
outputs/logs/evaluation.log
```

详细的执行日志

---

## 🐛 故障排查

### GPU内存不足
1. 减少batch size
2. 使用更小的推理步数
3. 禁用保存生成图像
4. 使用CPU（不推荐）

### 模型加载失败
1. 检查模型路径是否正确
2. 检查网络连接（如果从HF下载）
3. 检查GPU内存是否足够
4. 查看日志文件了解详细错误

### 分数解析失败
1. 检查prompt格式
2. 查看模型输出（日志中会记录）
3. 调整 `_extract_score_from_response` 方法
4. 增加max_new_tokens

---

更新时间: 2025-10-23 20:05


