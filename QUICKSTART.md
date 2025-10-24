# 🚀 快速启动指南

## 1️⃣ 安装依赖（1分钟）

```bash
cd /data2/yixuan/image_edit_benchmark
pip install -r requirements.txt
```

## 2️⃣ 准备配置文件（2分钟）

```bash
# 复制配置模板
cp config_template.yaml config.yaml

# 编辑配置文件
vim config.yaml
```

### 必须修改的配置项：

```yaml
benchmark:
  data_path: "path/to/your/benchmark.json"  # ← 修改为你的数据路径
  categories:  # ← 修改为你的5个类别名称
    - "category_1"
    - "category_2"
    - "category_3"
    - "category_4"
    - "category_5"

diffusion_model:
  class_path: "src.models.diffusion.implementations.YOUR_MODEL.YourModelClass"  # ← 修改为你的模型
  params:
    model_name: "your-model-name"
    device: "cuda"

reward_model:
  class_path: "src.models.reward.implementations.YOUR_REWARD.YourRewardClass"  # ← 修改为你的模型
  params:
    model_name: "your-reward-model"
    device: "cuda"

prompts:  # ← 为每个类别配置prompt
  category_1:
    system_prompt: "You are an expert..."
    user_prompt_template: "Original: {original_description}..."
```

## 3️⃣ 实现你的模型（10-30分钟）

### 方法A：使用示例模型测试流程

直接运行即可（会使用占位符模型）：
```bash
python main.py --config config.yaml
```

### 方法B：实现真实模型

#### 创建扩散编辑模型
```bash
# 在 src/models/diffusion/implementations/ 创建你的模型
vim src/models/diffusion/implementations/my_model.py
```

```python
from PIL import Image
from ..base_diffusion import BaseDiffusionModel

class MyDiffusionModel(BaseDiffusionModel):
    def _initialize(self):
        # 加载你的模型
        from diffusers import StableDiffusionInstructPix2PixPipeline
        self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            self.config.get("model_name")
        ).to(self.config.get("device"))
    
    def edit_image(self, original_image: Image.Image, 
                   edit_instruction: str, **kwargs) -> Image.Image:
        # 实现编辑逻辑
        return self.pipe(
            prompt=edit_instruction,
            image=original_image,
            num_inference_steps=50
        ).images[0]
```

#### 创建Reward评分模型
```bash
vim src/models/reward/implementations/my_reward.py
```

```python
from PIL import Image
from typing import Optional
from ..base_reward import BaseRewardModel

class MyRewardModel(BaseRewardModel):
    def _initialize(self):
        # 加载你的评分模型
        pass
    
    def score(self, edited_image: Image.Image,
              original_description: str,
              edit_instruction: str,
              system_prompt: str,
              user_prompt: str,
              original_image: Optional[Image.Image] = None,
              **kwargs) -> float:
        # 实现评分逻辑，返回0-10的分数
        score = 8.5  # 示例
        return score
```

## 4️⃣ 运行评测（根据数据量）

```bash
# 运行完整评测
python main.py --config config.yaml

# 如果中断了，从断点继续
python main.py --config config.yaml --resume
```

## 5️⃣ 查看结果（1分钟）

```bash
# 查看Markdown报告（人类可读）
cat outputs/results/evaluation_report_*.md

# 查看JSON报告（程序处理）
cat outputs/results/evaluation_report_*.json

# 查看生成的图像（如果启用）
ls outputs/images/
```

---

## 📋 检查清单

在运行前，确保：

- [ ] 已安装所有依赖包
- [ ] 配置文件中的数据路径正确
- [ ] 配置文件中的类别名称与数据匹配
- [ ] 已实现或配置了扩散编辑模型
- [ ] 已实现或配置了Reward评分模型
- [ ] 为每个类别配置了评分prompt
- [ ] GPU/CPU设置正确

---

## 🆘 遇到问题？

### 问题1: 找不到模块
```
ModuleNotFoundError: No module named 'xxx'
```
**解决**: 检查requirements.txt是否安装完整
```bash
pip install -r requirements.txt
```

### 问题2: 配置文件错误
```
ValueError: benchmark.data_path not specified in config
```
**解决**: 检查config.yaml格式是否正确，是否包含所有必填字段

### 问题3: 内存不足
```
CUDA out of memory
```
**解决**: 在配置中设置：
```yaml
evaluation:
  save_generated_images: false
diffusion_model:
  params:
    batch_size: 1
```

### 问题4: JSON格式不匹配
如果你的benchmark JSON格式不同，需要修改数据加载器。
查看 `src/data/benchmark_loader.py` 中的 `_extract_category_data` 方法。

---

## 📚 更多文档

- `README.md`: 项目概述
- `USAGE_GUIDE.md`: 详细使用指南
- `PROJECT_STRUCTURE.md`: 项目结构说明
- `examples/`: 示例代码

---

## 🎯 下一步

1. 查看 `examples/custom_model_example.py` 了解如何实现自定义模型
2. 查看 `USAGE_GUIDE.md` 了解高级功能
3. 运行测试确保系统正常：`python -m pytest tests/`

祝评测顺利！🎉


