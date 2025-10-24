# Image Edit Benchmark Pipeline

图像编辑模型评测系统 - 用于在benchmark数据集上测试扩散编辑模型的性能

## 📁 项目结构

```
image_edit_benchmark/
├── README.md                     # 项目说明文档
├── requirements.txt              # 依赖包列表
├── config.yaml                   # 配置文件
├── config_template.yaml          # 配置文件模板
├── main.py                       # 主入口脚本
│
├── src/                          # 源代码目录
│   ├── __init__.py
│   ├── pipeline.py               # 主Pipeline类
│   │
│   ├── data/                     # 数据加载模块
│   │   ├── __init__.py
│   │   ├── benchmark_loader.py   # Benchmark数据加载器
│   │   └── data_types.py         # 数据类型定义
│   │
│   ├── models/                   # 模型模块
│   │   ├── __init__.py
│   │   ├── base.py               # 模型基类定义
│   │   │
│   │   ├── diffusion/            # 扩散编辑模型
│   │   │   ├── __init__.py
│   │   │   ├── base_diffusion.py # 扩散模型抽象基类
│   │   │   └── implementations/  # 具体实现
│   │   │       ├── __init__.py
│   │   │       ├── example_model.py
│   │   │       └── custom_model.py
│   │   │
│   │   └── reward/               # Reward评分模型
│   │       ├── __init__.py
│   │       ├── base_reward.py    # Reward模型抽象基类
│   │       └── implementations/  # 具体实现
│   │           ├── __init__.py
│   │           ├── example_reward.py
│   │           └── custom_reward.py
│   │
│   ├── evaluation/               # 评估模块
│   │   ├── __init__.py
│   │   ├── scorer.py             # 评分统计器
│   │   └── reporter.py           # 报告生成器
│   │
│   └── utils/                    # 工具模块
│       ├── __init__.py
│       ├── image_utils.py        # 图像处理工具
│       ├── logger.py             # 日志工具
│       └── prompt_manager.py     # Prompt管理器
│
├── configs/                      # 配置文件目录
│   ├── prompts/                  # Prompt配置
│   │   ├── category_prompts.yaml # 各类别的评分prompt
│   │   └── default_prompts.yaml  # 默认prompt
│   └── models/                   # 模型配置
│       ├── diffusion_config.yaml # 扩散模型配置
│       └── reward_config.yaml    # Reward模型配置
│
├── examples/                     # 示例代码
│   ├── run_evaluation.py         # 运行评测示例
│   └── custom_model_example.py   # 自定义模型示例
│
├── outputs/                      # 输出目录
│   ├── results/                  # 评测结果
│   ├── logs/                     # 日志文件
│   └── images/                   # 生成的图像（可选）
│
└── tests/                        # 测试代码
    ├── __init__.py
    ├── test_data_loader.py
    ├── test_models.py
    └── test_pipeline.py
```

## 🎯 核心模块说明

### 1. 数据加载模块 (`src/data/`)
- 读取benchmark JSON文件
- 按类别组织数据（原图b64、编辑指令、原图描述）
- 提供数据迭代器

### 2. 扩散编辑模型模块 (`src/models/diffusion/`)
- 抽象基类定义统一接口
- 支持多种扩散模型实现
- 便于替换和扩展

### 3. Reward评分模型模块 (`src/models/reward/`)
- 抽象基类定义统一接口
- 按类别使用不同prompt
- 支持多种评分模型

### 4. 评估模块 (`src/evaluation/`)
- 按类别计算平均分
- 生成详细评测报告
- 支持多种统计指标

### 5. Pipeline (`src/pipeline.py`)
- 整合所有模块
- 控制评测流程
- 支持断点续传

## 🚀 快速开始

### 安装依赖
```bash
pip install -r requirements.txt
```

### 配置模型
```bash
cp config_template.yaml config.yaml
# 编辑 config.yaml，配置你的模型和参数
```

### 运行评测
```bash
python main.py --config config.yaml
```

## 📝 使用说明

详细使用说明请参考各模块的文档。

## 🔧 扩展指南

### 添加新的扩散编辑模型
1. 在 `src/models/diffusion/implementations/` 创建新的实现文件
2. 继承 `BaseDiffusionModel` 类
3. 实现 `edit_image()` 方法
4. 在配置文件中指定模型类

### 添加新的Reward模型
1. 在 `src/models/reward/implementations/` 创建新的实现文件
2. 继承 `BaseRewardModel` 类
3. 实现 `score()` 方法
4. 在配置文件中指定模型类

## 📊 评测流程

1. 加载benchmark数据集（JSON）
2. 按类别提取：原图b64、编辑指令、原图描述
3. 调用扩散编辑模型生成编辑后的图像
4. 调用Reward模型对每个pair进行评分
5. 按类别计算平均分
6. 生成完整评测报告

## 📄 License

MIT License


