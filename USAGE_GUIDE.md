# Image Edit Benchmark Pipeline - 使用指南

## 📚 目录
1. [快速开始](#快速开始)
2. [配置文件详解](#配置文件详解)
3. [实现自定义模型](#实现自定义模型)
4. [运行评测](#运行评测)
5. [查看结果](#查看结果)
6. [常见问题](#常见问题)

---

## 快速开始

### 1. 安装依赖

```bash
cd /data2/yixuan/image_edit_benchmark
pip install -r requirements.txt
```

### 2. 准备配置文件

```bash
# 复制配置模板
cp config_template.yaml config.yaml

# 编辑配置文件
vim config.yaml
```

### 3. 配置你的数据和模型

编辑 `config.yaml`，设置以下关键参数：

- `benchmark.data_path`: 你的benchmark JSON文件路径
- `benchmark.categories`: 五个类别的名称
- `diffusion_model.class_path`: 你的扩散编辑模型类路径
- `reward_model.class_path`: 你的reward评分模型类路径
- `prompts`: 每个类别的评分prompt

### 4. 运行评测

```bash
python main.py --config config.yaml
```

---

## 配置文件详解

### Benchmark数据配置

```yaml
benchmark:
  data_path: "path/to/benchmark.json"  # 必填
  categories:  # 必填：五个类别名称
    - "category_1"
    - "category_2"
    - "category_3"
    - "category_4"
    - "category_5"
```

### 扩散模型配置

```yaml
diffusion_model:
  # 模型类的完整路径（模块.类名）
  class_path: "src.models.diffusion.implementations.example_model.ExampleDiffusionModel"
  
  params:  # 传递给模型的参数
    model_name: "timbrooks/instruct-pix2pix"
    device: "cuda"
    batch_size: 1
    num_inference_steps: 50
    guidance_scale: 7.5
```

### Reward模型配置

```yaml
reward_model:
  class_path: "src.models.reward.implementations.example_reward.ExampleRewardModel"
  
  params:
    model_name: "your-vlm-model"
    device: "cuda"
    temperature: 0.7
```

### Prompt配置

```yaml
prompts:
  category_1:  # 每个类别都需要配置
    system_prompt: "You are an expert image quality evaluator."
    user_prompt_template: |
      Original description: {original_description}
      Edit instruction: {edit_instruction}
      Please rate the edited image quality on a scale of 0-10.
```

**注意**：`user_prompt_template` 中可以使用以下变量：
- `{original_description}`: 原图描述
- `{edit_instruction}`: 编辑指令

---

## 实现自定义模型

### 步骤1: 创建自定义扩散模型

在 `src/models/diffusion/implementations/` 目录下创建新文件，例如 `my_model.py`：

```python
from PIL import Image
from ..base_diffusion import BaseDiffusionModel

class MyDiffusionModel(BaseDiffusionModel):
    def _initialize(self):
        """初始化你的模型"""
        self.model_name = self.config.get("model_name")
        # 加载模型...
        
    def edit_image(self, original_image: Image.Image, 
                   edit_instruction: str, **kwargs) -> Image.Image:
        """实现图像编辑逻辑"""
        # 你的编辑逻辑...
        return edited_image
```

### 步骤2: 创建自定义Reward模型

在 `src/models/reward/implementations/` 目录下创建新文件，例如 `my_reward.py`：

```python
from PIL import Image
from typing import Optional
from ..base_reward import BaseRewardModel

class MyRewardModel(BaseRewardModel):
    def _initialize(self):
        """初始化你的reward模型"""
        # 加载模型...
        
    def score(self, edited_image: Image.Image,
              original_description: str,
              edit_instruction: str,
              system_prompt: str,
              user_prompt: str,
              original_image: Optional[Image.Image] = None,
              **kwargs) -> float:
        """实现评分逻辑"""
        # 你的评分逻辑...
        return score  # 返回0-10的分数
```

### 步骤3: 在配置文件中使用

```yaml
diffusion_model:
  class_path: "src.models.diffusion.implementations.my_model.MyDiffusionModel"
  params:
    model_name: "your-model-path"

reward_model:
  class_path: "src.models.reward.implementations.my_reward.MyRewardModel"
  params:
    model_name: "your-reward-model-path"
```

---

## 运行评测

### 基本运行

```bash
python main.py --config config.yaml
```

### 从断点继续

如果评测中断，可以从断点继续：

```bash
python main.py --config config.yaml --resume
```

### 查看帮助

```bash
python main.py --help
```

---

## 查看结果

评测完成后，结果会保存在 `outputs/results/` 目录：

### JSON报告

```bash
cat outputs/results/evaluation_report_YYYYMMDD_HHMMSS.json
```

包含完整的统计数据，适合程序处理。

### Markdown报告

```bash
cat outputs/results/evaluation_report_YYYYMMDD_HHMMSS.md
```

人类可读的格式，包含：
- 总体摘要
- 各类别详细统计
- 最好/最差类别

### 生成的图像

如果在配置中启用了 `save_generated_images: true`，编辑后的图像会保存在：

```
outputs/images/
├── category_1/
│   ├── pair_001.png
│   └── pair_002.png
├── category_2/
│   └── ...
```

---

## 常见问题

### Q1: 我的benchmark JSON格式不同怎么办？

**A**: 有两种方法：

1. **调整数据格式**：修改JSON使其符合默认格式
2. **自定义loader**：修改 `src/data/benchmark_loader.py` 中的 `_extract_category_data` 方法

默认格式：
```json
{
  "category_name": [
    {
      "id": "xxx",
      "original_image_b64": "...",
      "edit_instruction": "...",
      "original_description": "..."
    }
  ]
}
```

### Q2: 如何使用不同的评分标准？

**A**: 在配置文件的 `prompts` 部分为每个类别自定义prompt：

```yaml
prompts:
  object_addition:
    system_prompt: "Focus on whether the object was added correctly."
    user_prompt_template: |
      Instruction: {edit_instruction}
      Rate how well the object was added (0-10).
```

### Q3: 评测速度太慢怎么办？

**A**: 可以：

1. **减少inference steps**：在模型配置中降低 `num_inference_steps`
2. **启用批处理**：实现模型的 `batch_edit` 和 `batch_score` 方法
3. **使用更快的模型**：选择更轻量的扩散模型

### Q4: 内存不足怎么办？

**A**: 

1. **不保存生成图像**：设置 `save_generated_images: false`
2. **使用更小的batch size**：减小 `batch_size`
3. **使用float16**：在模型中使用 `torch.float16`

### Q5: 如何测试单个样本？

**A**: 使用Pipeline的 `run_single_pair` 方法：

```python
from src.pipeline import BenchmarkPipeline
import yaml

with open('config.yaml') as f:
    config = yaml.safe_load(f)

pipeline = BenchmarkPipeline(config)

result = pipeline.run_single_pair(
    original_image_b64="your_base64_string",
    edit_instruction="make it blue",
    original_description="a red car",
    category="category_1"
)

print(f"Score: {result['score']}")
```

---

## 高级功能

### 自定义评分指标

在配置文件中指定需要计算的统计指标：

```yaml
evaluation:
  metrics:
    - "mean"      # 平均值
    - "std"       # 标准差
    - "median"    # 中位数
    - "min"       # 最小值
    - "max"       # 最大值
```

### 自定义日志级别

```yaml
logging:
  level: "DEBUG"  # DEBUG, INFO, WARNING, ERROR
  console_output: true
  file_output: true
  log_file: "outputs/logs/evaluation.log"
```

---

## 示例代码

查看 `examples/` 目录获取更多示例：

- `run_evaluation.py`: 完整评测示例
- `custom_model_example.py`: 自定义模型示例

---

## 获取帮助

如果遇到问题：

1. 查看日志文件：`outputs/logs/evaluation.log`
2. 检查配置文件格式是否正确
3. 确认模型类路径是否正确
4. 查看示例代码

---

祝评测顺利！🎉


