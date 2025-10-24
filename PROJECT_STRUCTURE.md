# 项目结构说明

## 📂 完整目录结构

```
image_edit_benchmark/
│
├── README.md                          # 项目说明文档
├── USAGE_GUIDE.md                     # 详细使用指南
├── PROJECT_STRUCTURE.md               # 本文件 - 项目结构说明
├── requirements.txt                   # Python依赖包
├── config_template.yaml               # 配置文件模板
├── config.yaml                        # 实际配置文件（需自行创建）
├── main.py                            # 主入口脚本
├── .gitignore                         # Git忽略文件配置
│
├── src/                               # 源代码目录
│   ├── __init__.py
│   ├── pipeline.py                    # 主Pipeline类 - 核心评测流程
│   │
│   ├── data/                          # 数据加载模块
│   │   ├── __init__.py
│   │   ├── benchmark_loader.py        # Benchmark数据加载器
│   │   └── data_types.py              # 数据类型定义（DataPair, CategoryData等）
│   │
│   ├── models/                        # 模型模块
│   │   ├── __init__.py
│   │   ├── base.py                    # 所有模型的基类
│   │   │
│   │   ├── diffusion/                 # 扩散编辑模型
│   │   │   ├── __init__.py
│   │   │   ├── base_diffusion.py      # 扩散模型抽象基类
│   │   │   └── implementations/       # 具体实现
│   │   │       ├── __init__.py
│   │   │       ├── example_model.py   # 示例实现（占位符）
│   │   │       └── [your_model.py]    # 在这里添加你的模型
│   │   │
│   │   └── reward/                    # Reward评分模型
│   │       ├── __init__.py
│   │       ├── base_reward.py         # Reward模型抽象基类
│   │       └── implementations/       # 具体实现
│   │           ├── __init__.py
│   │           ├── example_reward.py  # 示例实现（占位符）
│   │           └── [your_reward.py]   # 在这里添加你的模型
│   │
│   ├── evaluation/                    # 评估模块
│   │   ├── __init__.py
│   │   ├── scorer.py                  # 评分统计器（计算mean, std等）
│   │   └── reporter.py                # 报告生成器（JSON和Markdown）
│   │
│   └── utils/                         # 工具模块
│       ├── __init__.py
│       ├── image_utils.py             # 图像处理工具（base64编解码等）
│       ├── logger.py                  # 日志工具
│       └── prompt_manager.py          # Prompt管理器
│
├── examples/                          # 示例代码
│   ├── run_evaluation.py              # 运行评测示例
│   └── custom_model_example.py        # 自定义模型示例
│
├── tests/                             # 测试代码
│   ├── __init__.py
│   ├── test_data_loader.py            # 数据加载器测试
│   ├── test_models.py                 # 模型测试
│   └── test_pipeline.py               # Pipeline测试
│
└── outputs/                           # 输出目录（运行时生成）
    ├── results/                       # 评测结果（JSON和Markdown报告）
    ├── logs/                          # 日志文件
    └── images/                        # 生成的图像（可选）
        ├── category_1/
        ├── category_2/
        └── ...
```

---

## 🔑 核心模块说明

### 1. Pipeline (`src/pipeline.py`)
**作用**：整合所有模块，控制评测流程

**主要功能**：
- 加载benchmark数据
- 调用扩散模型编辑图像
- 调用reward模型评分
- 计算统计指标
- 生成报告
- 支持断点续传

**关键方法**：
- `run()`: 运行完整评测
- `run_single_pair()`: 测试单个样本

### 2. 数据加载 (`src/data/`)
**作用**：读取和组织benchmark数据

**核心类**：
- `BenchmarkLoader`: 从JSON加载数据
- `DataPair`: 单个数据对的数据结构
- `CategoryData`: 单个类别的数据集合
- `BenchmarkData`: 完整的benchmark数据

**支持的数据格式**：
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

### 3. 扩散编辑模型 (`src/models/diffusion/`)
**作用**：提供图像编辑功能

**抽象基类**: `BaseDiffusionModel`

**必须实现的方法**：
- `_initialize()`: 初始化模型
- `edit_image(original_image, edit_instruction)`: 编辑图像

**可选优化**：
- `batch_edit()`: 批量编辑以提高效率

### 4. Reward评分模型 (`src/models/reward/`)
**作用**：对编辑后的图像进行评分

**抽象基类**: `BaseRewardModel`

**必须实现的方法**：
- `_initialize()`: 初始化模型
- `score(edited_image, original_description, edit_instruction, system_prompt, user_prompt)`: 评分

**可选优化**：
- `batch_score()`: 批量评分以提高效率

### 5. 评估统计 (`src/evaluation/`)
**作用**：计算统计指标和生成报告

**核心类**：
- `Scorer`: 计算mean、std、median等统计指标
- `Reporter`: 生成JSON和Markdown格式的报告

### 6. 工具模块 (`src/utils/`)
**作用**：提供通用工具函数

**主要工具**：
- `image_utils`: Base64编解码、图像保存等
- `logger`: 日志配置
- `prompt_manager`: 管理不同类别的prompt

---

## 🔄 评测流程

```
1. 初始化Pipeline
   ├── 加载配置文件
   ├── 初始化扩散模型
   ├── 初始化Reward模型
   └── 初始化Prompt管理器

2. 加载Benchmark数据
   ├── 读取JSON文件
   └── 按类别组织数据

3. 按类别处理（对每个类别执行）
   ├── 遍历该类别的所有数据对
   │   ├── 解码原始图像（base64 → PIL.Image）
   │   ├── 调用扩散模型编辑图像
   │   ├── （可选）保存编辑后的图像
   │   ├── 获取该类别的prompt
   │   ├── 调用Reward模型评分
   │   └── 保存分数和断点
   └── 收集该类别的所有分数

4. 计算统计指标
   ├── 各类别统计（mean, std, median等）
   └── 整体统计

5. 生成报告
   ├── 创建报告数据结构
   ├── 保存JSON报告
   └── 保存Markdown报告

6. 完成
```

---

## 🎨 设计模式

### 1. 抽象基类模式
所有模型都继承自抽象基类，确保接口一致性：
- `BaseDiffusionModel` → 扩散模型实现
- `BaseRewardModel` → Reward模型实现

### 2. 策略模式
通过配置文件动态加载模型类：
```yaml
diffusion_model:
  class_path: "module.path.YourModel"
```

### 3. 模板方法模式
Pipeline定义评测流程框架，具体步骤由模型实现。

---

## 📝 扩展指南

### 添加新的扩散模型

1. 在 `src/models/diffusion/implementations/` 创建文件
2. 继承 `BaseDiffusionModel`
3. 实现必要方法
4. 在配置中指定类路径

示例：
```python
from ..base_diffusion import BaseDiffusionModel

class MyModel(BaseDiffusionModel):
    def _initialize(self):
        # 加载模型
        pass
    
    def edit_image(self, original_image, edit_instruction):
        # 编辑逻辑
        return edited_image
```

### 添加新的Reward模型

类似扩散模型，在 `src/models/reward/implementations/` 创建实现。

### 自定义数据格式

修改 `src/data/benchmark_loader.py` 的 `_extract_category_data` 方法。

### 添加新的评分指标

在 `src/evaluation/scorer.py` 的 `compute_category_statistics` 方法中添加。

---

## 🧪 测试

运行单元测试：
```bash
python -m pytest tests/
```

或单独运行：
```bash
python tests/test_data_loader.py
python tests/test_models.py
python tests/test_pipeline.py
```

---

## 📊 输出说明

### JSON报告
```json
{
  "timestamp": "2025-10-23T12:00:00",
  "category_statistics": {
    "category_1": {
      "mean": 7.5,
      "std": 1.2,
      "median": 7.8,
      "min": 5.0,
      "max": 9.5,
      "num_samples": 180
    }
  },
  "overall_statistics": {...},
  "summary": {...},
  "metadata": {...}
}
```

### Markdown报告
包含：
- 总体摘要
- 各类别详细统计
- 最好/最差类别
- 完整统计数据

---

## 🔧 配置说明

### 必填配置
- `benchmark.data_path`: 数据文件路径
- `benchmark.categories`: 类别列表
- `diffusion_model.class_path`: 扩散模型类路径
- `reward_model.class_path`: Reward模型类路径
- `prompts`: 各类别的prompt配置

### 可选配置
- `evaluation.save_generated_images`: 是否保存生成图像
- `evaluation.resume_from_checkpoint`: 是否断点续传
- `logging`: 日志配置

---

## 💡 使用建议

1. **开发阶段**：使用示例模型测试流程
2. **实现模型**：创建自己的模型实现
3. **小规模测试**：先用少量数据测试
4. **完整评测**：在完整数据集上运行
5. **分析结果**：查看生成的报告

---

更多详细信息请参考：
- `README.md`: 项目概述
- `USAGE_GUIDE.md`: 详细使用指南
- `examples/`: 示例代码


