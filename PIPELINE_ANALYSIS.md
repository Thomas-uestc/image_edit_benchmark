# 🚀 Pipeline串联逻辑详细分析

## 📋 目录
1. [整体架构](#整体架构)
2. [初始化流程](#初始化流程)
3. [主运行流程](#主运行流程)
4. [数据流分析](#数据流分析)
5. [各模块职责](#各模块职责)
6. [断点续传机制](#断点续传机制)
7. [配置驱动设计](#配置驱动设计)
8. [关键代码解析](#关键代码解析)

---

## 🏗️ 整体架构

### Pipeline的核心定位

`BenchmarkPipeline` 是整个系统的**协调者（Orchestrator）**，负责：
- 串联所有模块
- 控制数据流
- 管理评测流程
- 处理异常和断点

### 架构图

```
┌─────────────────────────────────────────────────────────────┐
│                    BenchmarkPipeline                        │
│                      (协调者/Orchestrator)                   │
└─────────────────────────────────────────────────────────────┘
       │
       ├─→ 初始化阶段 (__init__)
       │    ├─→ Logger (日志系统)
       │    ├─→ BenchmarkLoader (数据加载器)
       │    ├─→ BaseDiffusionModel (扩散编辑模型)
       │    ├─→ BaseRewardModel (评分模型)
       │    ├─→ PromptManager (Prompt管理器)
       │    ├─→ Scorer (统计计算器)
       │    └─→ Reporter (报告生成器)
       │
       └─→ 运行阶段 (run())
            ├─→ 数据加载
            ├─→ 逐类别处理
            │    ├─→ 图像编辑 (Diffusion)
            │    └─→ 图像评分 (Reward)
            ├─→ 统计计算 (Scorer)
            └─→ 报告生成 (Reporter)
```

---

## 🎬 初始化流程

### 初始化顺序（`__init__`方法）

```python
def __init__(self, config: Dict[str, Any]):
```

#### 步骤1: 接收配置
```python
self.config = config  # 从config.yaml加载的配置字典
```

#### 步骤2: 设置日志系统
```python
log_config = config.get("logging", {})
self.logger = setup_logger(
    name="benchmark_pipeline",
    level=log_config.get("level", "INFO"),
    log_file=log_config.get("log_file") if log_config.get("file_output") else None,
    console_output=log_config.get("console_output", True)
)
```
**作用**：创建统一的日志记录器，所有模块都可以使用

#### 步骤3: 创建输出目录
```python
self._setup_output_dirs()
```
**创建的目录**：
- `outputs/` - 主输出目录
- `outputs/results/` - 报告输出目录
- `outputs/images/` - 编辑后图像保存目录
- `outputs/logs/` - 日志文件目录

#### 步骤4: 初始化数据加载器
```python
self.data_loader = BenchmarkLoader(logger=self.logger)
```
**职责**：加载和解析JSON benchmark数据

#### 步骤5: 加载扩散编辑模型
```python
self.diffusion_model = self._load_diffusion_model()
```
**动态加载机制**：
```python
# 从配置读取类路径
class_path = "src.models.diffusion.implementations.qwen_image_edit.QwenImageEditModel"

# 分离模块路径和类名
module_path = "src.models.diffusion.implementations.qwen_image_edit"
class_name = "QwenImageEditModel"

# 动态导入
module = importlib.import_module(module_path)
model_class = getattr(module, class_name)

# 实例化模型
model = model_class(config.get("diffusion_model").get("params"))
```
**优势**：通过修改config.yaml即可替换不同的扩散模型实现

#### 步骤6: 加载Reward评分模型
```python
self.reward_model = self._load_reward_model()
```
**同样使用动态加载**，例如：
```python
class_path = "src.models.reward.implementations.qwen3_vl_reward.Qwen3VLRewardModel"
```

#### 步骤7: 初始化Prompt管理器
```python
self.prompt_manager = PromptManager(config.get("prompts", {}))
```
**配置示例**：
```yaml
prompts:
  物理:
    system_prompt: "You are an image editing reward model evaluator..."
    user_prompt_template: "Original scene: {original_description}\nEdit: {edit_instruction}"
  环境:
    system_prompt: "..."
    user_prompt_template: "..."
  # ... 其他类别
```

#### 步骤8: 初始化统计计算器
```python
self.scorer = Scorer(
    metrics=config.get("evaluation", {}).get("metrics", ["mean", "std", "median"]),
    logger=self.logger
)
```
**可配置指标**：mean, std, median, min, max

#### 步骤9: 初始化报告生成器
```python
self.reporter = Reporter(
    output_dir=self.config["evaluation"]["results_dir"],
    logger=self.logger
)
```

#### 步骤10: 设置断点续传
```python
self.checkpoint_path = Path(config.get("evaluation", {}).get("checkpoint_path", "outputs/checkpoint.json"))
self.resume_from_checkpoint = config.get("evaluation", {}).get("resume_from_checkpoint", False)
self.checkpoint_data = self._load_checkpoint() if self.resume_from_checkpoint else {}
```

### 初始化完成后的状态

```python
pipeline = BenchmarkPipeline(config)
# 此时pipeline包含以下已初始化的组件：
# - self.logger          ✓ 日志系统
# - self.data_loader     ✓ 数据加载器
# - self.diffusion_model ✓ 扩散编辑模型（已加载到GPU）
# - self.reward_model    ✓ 评分模型（已加载到GPU）
# - self.prompt_manager  ✓ Prompt管理器
# - self.scorer          ✓ 统计计算器
# - self.reporter        ✓ 报告生成器
```

---

## 🔄 主运行流程

### `run()` 方法 - 完整评测流程

```python
def run(self) -> Dict[str, Any]:
```

### 流程图

```
┌─────────────────┐
│  Pipeline.run() │
└────────┬────────┘
         │
         ├─→ [步骤1] 加载benchmark数据
         │   ↓
         │   benchmark_data = self._load_benchmark_data()
         │   ↓
         │   返回 BenchmarkData 对象:
         │   {
         │     categories: {
         │       "物理": CategoryData(50 pairs),
         │       "环境": CategoryData(50 pairs),
         │       "社会": CategoryData(70 pairs),
         │       "因果": CategoryData(50 pairs),
         │       "指代": CategoryData(50 pairs)
         │     },
         │     total_pairs: 270
         │   }
         │
         ├─→ [步骤2] 按类别处理数据 (for循环)
         │   ↓
         │   for category_name in ["物理", "环境", "社会", "因果", "指代"]:
         │       ↓
         │       category_data = benchmark_data.get_category(category_name)
         │       ↓
         │       scores = self._process_category(category_data)
         │       ↓
         │       category_scores[category_name] = scores
         │   ↓
         │   结果: category_scores = {
         │     "物理": [7.2, 8.1, 6.8, ..., 7.5],  # 50个分数
         │     "环境": [7.5, 8.2, 7.0, ..., 8.1],  # 50个分数
         │     "社会": [6.8, 7.5, 8.1, ..., 7.2],  # 70个分数
         │     "因果": [7.1, 6.9, 7.6, ..., 7.8],  # 50个分数
         │     "指代": [8.0, 7.3, 8.2, ..., 7.9]   # 50个分数
         │   }
         │
         ├─→ [步骤3] 计算统计指标
         │   ↓
         │   category_statistics = self.scorer.compute_all_statistics(category_scores)
         │   overall_statistics = self.scorer.compute_overall_statistics(category_scores)
         │   ↓
         │   结果: {
         │     "物理": {"mean": 7.23, "std": 1.12, ...},
         │     "环境": {"mean": 7.46, "std": 0.98, ...},
         │     ...
         │   }
         │
         ├─→ [步骤4] 生成报告
         │   ↓
         │   metadata = {...}  # 收集元数据
         │   report = self.reporter.generate_report(
         │       category_statistics, overall_statistics, metadata
         │   )
         │
         └─→ [步骤5] 保存报告
             ↓
             json_path = self.reporter.save_report(report)
             md_path = self.reporter.save_markdown_report(report)
             ↓
             返回 report
```

---

## 📦 数据流分析

### 完整数据流

```
┌──────────────────────────────────────────────────────────────┐
│ 1. 数据加载阶段                                                │
└──────────────────────────────────────────────────────────────┘

JSON文件 (/data2/yixuan/Benchmark/version_2_with_imagesb64.json)
    ↓ BenchmarkLoader.load()
BenchmarkData {
    categories: {
        "物理": CategoryData {
            category_name: "物理",
            data_pairs: [
                DataPair {
                    pair_id: "physical_001",
                    category: "物理",
                    original_image_b64: "iVBORw0KG...",
                    edit_instruction: "将红色汽车改为蓝色",
                    original_description: "一辆红色的汽车停在街道上",
                    original_image: None,        # 延迟解码
                    edited_image: None,
                    score: None
                },
                DataPair {...}, # 共50个
                ...
            ]
        },
        "环境": CategoryData {...},
        "社会": CategoryData {...},
        "因果": CategoryData {...},
        "指代": CategoryData {...}
    },
    total_pairs: 270
}

┌──────────────────────────────────────────────────────────────┐
│ 2. 逐类别处理阶段 (以"物理"类别为例)                            │
└──────────────────────────────────────────────────────────────┘

CategoryData("物理") → _process_category()
    ↓
for pair in category_data.data_pairs:  # 50次循环
    ↓
    ┌─────────────────────────────────────┐
    │ 2.1 解码原始图像                      │
    └─────────────────────────────────────┘
    pair.original_image_b64 (str)
        ↓ decode_base64_image()
    pair.original_image = PIL.Image.Image (RGB)
    
    ┌─────────────────────────────────────┐
    │ 2.2 编辑图像                         │
    └─────────────────────────────────────┘
    输入:
      - pair.original_image: PIL.Image
      - pair.edit_instruction: "将红色汽车改为蓝色"
    
    ↓ diffusion_model.edit_image()
    
    处理流程:
      1. 图像预处理（转RGB、resize等）
      2. 编码到潜空间
      3. 应用编辑指令
      4. 去噪过程（50步）
      5. 解码到图像空间
    
    输出:
      pair.edited_image = PIL.Image.Image (编辑后的蓝色汽车)
    
    ┌─────────────────────────────────────┐
    │ 2.3 获取评分Prompt                   │
    └─────────────────────────────────────┘
    输入:
      - category: "物理"
      - original_description: "一辆红色的汽车停在街道上"
      - edit_instruction: "将红色汽车改为蓝色"
    
    ↓ prompt_manager.get_full_prompt()
    
    输出:
      prompts = {
          "system_prompt": "You are an image editing reward model evaluator...",
          "user_prompt": "Original scene: 一辆红色的汽车停在街道上\n
                          Edit instruction: 将红色汽车改为蓝色\n
                          Based on the above information, evaluate..."
      }
    
    ┌─────────────────────────────────────┐
    │ 2.4 评分                             │
    └─────────────────────────────────────┘
    输入:
      - edited_image: PIL.Image (编辑后的图像)
      - original_description: str
      - edit_instruction: str
      - system_prompt: str
      - user_prompt: str
      - original_image: PIL.Image (可选)
    
    ↓ reward_model.score()
    
    处理流程:
      1. 构建messages (system + user + image)
      2. 处理器编码 (processor.apply_chat_template)
      3. 模型生成 (model.generate)
      4. 解析输出 (_parse_score)
           - 匹配 "Score: X.XXX"
           - 或提取第一个数字
           - 默认值: 5.0
    
    输出:
      score = 7.234  # 浮点数
    
    ┌─────────────────────────────────────┐
    │ 2.5 保存结果                         │
    └─────────────────────────────────────┘
    pair.score = 7.234
    scores.append(7.234)
    
    (可选) 保存编辑后的图像:
      outputs/images/物理/physical_001.png

┌──────────────────────────────────────────────────────────────┐
│ 3. 统计计算阶段                                               │
└──────────────────────────────────────────────────────────────┘

category_scores = {
    "物理": [7.234, 8.123, 6.789, ..., 7.456],  # 50个
    "环境": [7.456, 8.234, 7.012, ..., 8.123],  # 50个
    "社会": [6.789, 7.456, 8.123, ..., 7.234],  # 70个
    "因果": [7.123, 6.890, 7.567, ..., 7.890],  # 50个
    "指代": [8.012, 7.345, 8.234, ..., 7.678]   # 50个
}
    ↓
scorer.compute_all_statistics(category_scores)
    ↓
category_statistics = {
    "物理": {
        "mean": 7.234,
        "std": 1.123,
        "median": 7.456,
        "min": 4.567,
        "max": 9.123,
        "num_samples": 50
    },
    "环境": {...},
    ...
}
    ↓
scorer.compute_overall_statistics(category_scores)
    ↓
overall_statistics = {
    "mean": 7.423,
    "std": 1.156,
    "median": 7.512,
    "min": 3.456,
    "max": 9.876,
    "num_samples": 270
}

┌──────────────────────────────────────────────────────────────┐
│ 4. 报告生成阶段                                               │
└──────────────────────────────────────────────────────────────┘

输入:
  - category_statistics
  - overall_statistics
  - metadata (配置信息、模型信息等)
    ↓
reporter.generate_report()
    ↓
report = {
    "timestamp": "2025-10-23T20:35:00",
    "category_statistics": {...},
    "overall_statistics": {...},
    "summary": {
        "num_categories": 5,
        "total_samples": 270,
        "overall_mean": 7.423,
        "category_means": {...},
        "best_category": {"name": "指代", "score": 8.012},
        "worst_category": {"name": "社会", "score": 6.789}
    },
    "metadata": {...}
}
    ↓
reporter.save_report(report)
    ↓
outputs/results/evaluation_report_20251023_203500.json
    ↓
reporter.save_markdown_report(report)
    ↓
outputs/results/evaluation_report_20251023_203500.md
```

---

## 🔍 各模块职责

### 1. BenchmarkPipeline（协调者）
**职责**：
- ✅ 管理整体流程
- ✅ 协调各模块协作
- ✅ 控制数据流动
- ✅ 处理异常和断点
- ✅ 日志记录和进度显示

**不负责**：
- ❌ 具体的模型推理
- ❌ 具体的统计计算
- ❌ 具体的数据解析

### 2. BenchmarkLoader（数据加载器）
**职责**：
- ✅ 读取JSON文件
- ✅ 解析数据结构
- ✅ 按类别组织数据
- ✅ 创建DataPair对象
- ✅ 验证数据完整性

**输入**：JSON文件路径
**输出**：BenchmarkData对象

### 3. BaseDiffusionModel（扩散编辑模型）
**职责**：
- ✅ 加载扩散模型
- ✅ 图像编辑推理
- ✅ 管理GPU内存

**接口方法**：
```python
def edit_image(self, original_image: PIL.Image, edit_instruction: str) -> PIL.Image
def batch_edit(self, images: List[PIL.Image], instructions: List[str]) -> List[PIL.Image]
```

**输入**：PIL.Image + 编辑指令
**输出**：编辑后的PIL.Image

### 4. BaseRewardModel（评分模型）
**职责**：
- ✅ 加载VLM模型
- ✅ 图像评分推理
- ✅ 解析分数输出

**接口方法**：
```python
def score(self, edited_image: PIL.Image, 
          original_description: str,
          edit_instruction: str,
          system_prompt: str,
          user_prompt: str,
          original_image: PIL.Image = None) -> float
```

**输入**：图像 + 描述 + 指令 + Prompt
**输出**：浮点分数 (0.0 - 10.0)

### 5. PromptManager（Prompt管理器）
**职责**：
- ✅ 管理多类别Prompt
- ✅ 填充Prompt模板
- ✅ 验证Prompt配置

**接口方法**：
```python
def get_full_prompt(self, category: str, 
                   original_description: str,
                   edit_instruction: str) -> Dict[str, str]
```

**输入**：类别名 + 描述 + 指令
**输出**：{"system_prompt": str, "user_prompt": str}

### 6. Scorer（统计计算器）
**职责**：
- ✅ 计算描述性统计
- ✅ 分类别统计
- ✅ 整体统计

**接口方法**：
```python
def compute_all_statistics(self, category_scores: Dict[str, List[float]]) -> Dict
def compute_overall_statistics(self, category_scores: Dict[str, List[float]]) -> Dict
```

**输入**：category_scores字典
**输出**：统计结果字典

### 7. Reporter（报告生成器）
**职责**：
- ✅ 生成评测报告
- ✅ 保存JSON报告
- ✅ 生成Markdown报告
- ✅ 生成摘要信息

**接口方法**：
```python
def generate_report(self, category_statistics: Dict, 
                   overall_statistics: Dict,
                   metadata: Dict) -> Dict
def save_report(self, report: Dict) -> str
def save_markdown_report(self, report: Dict) -> str
```

**输入**：统计结果 + 元数据
**输出**：报告文件路径

---

## 💾 断点续传机制

### 设计目的
- 避免长时间运行中断导致重新计算
- 节省GPU资源和时间
- 支持分批处理

### 实现机制

#### 1. Checkpoint数据结构
```json
{
  "物理": [
    {"pair_id": "physical_001", "score": 7.234},
    {"pair_id": "physical_002", "score": 8.123},
    ...
  ],
  "环境": [
    {"pair_id": "environment_001", "score": 7.456},
    ...
  ]
}
```

#### 2. 保存时机
```python
# 在_process_category()方法中，每处理完一个pair就更新checkpoint
for pair in category_data.data_pairs:
    # ... 编辑和评分 ...
    
    # 保存到checkpoint
    if category_name not in self.checkpoint_data:
        self.checkpoint_data[category_name] = []
    self.checkpoint_data[category_name].append({
        "pair_id": pair.pair_id,
        "score": score
    })
    
    # 可以每N个pair保存一次，或每个pair都保存
    self._save_checkpoint(self.checkpoint_data)
```

#### 3. 恢复机制
```python
# 初始化时加载checkpoint
self.checkpoint_data = self._load_checkpoint() if self.resume_from_checkpoint else {}

# 处理时跳过已处理的pair
processed_ids = set(self.checkpoint_data.get(category_name, []))

for pair in pbar:
    if pair.pair_id in processed_ids:
        scores.append(pair.score)  # 使用已有分数
        continue  # 跳过处理
    
    # 执行编辑和评分...
```

#### 4. 配置
```yaml
evaluation:
  resume_from_checkpoint: true  # 是否启用断点续传
  checkpoint_path: "outputs/checkpoint.json"  # 断点文件路径
```

### 使用场景
1. **长时间运行**：270个样本，可能需要几小时
2. **中断恢复**：程序崩溃或手动停止后可以继续
3. **分批处理**：可以每次只处理一个类别

---

## ⚙️ 配置驱动设计

### 设计理念
- **单一配置源**：所有配置都在`config.yaml`
- **模块化配置**：每个模块独立配置区域
- **动态加载**：通过配置指定模型类路径
- **灵活替换**：修改配置即可替换模型

### 配置结构

```yaml
# ===== 数据配置 =====
benchmark:
  data_path: "/path/to/benchmark.json"
  categories: ["物理", "环境", "社会", "因果", "指代"]

# ===== 扩散模型配置 =====
diffusion_model:
  class_path: "src.models.diffusion.implementations.qwen_image_edit.QwenImageEditModel"
  params:
    model_name: "Qwen/Qwen-Image-Edit"
    device: "cuda"
    dtype: "bfloat16"
    num_inference_steps: 50
    # ... 其他参数

# ===== Reward模型配置 =====
reward_model:
  class_path: "src.models.reward.implementations.qwen3_vl_reward.Qwen3VLRewardModel"
  params:
    model_name: "Qwen/Qwen3-VL-30B-Instruct"
    device: "auto"
    dtype: "bfloat16"
    # ... 其他参数

# ===== Prompt配置 =====
prompts:
  物理:
    system_prompt: |
      You are an image editing reward model evaluator...
    user_prompt_template: |
      Original scene: {original_description}
      Edit instruction: {edit_instruction}
      ...
  环境:
    system_prompt: |
      ...
    user_prompt_template: |
      ...
  # ... 其他类别

# ===== 评估配置 =====
evaluation:
  output_dir: "outputs"
  results_dir: "outputs/results"
  images_dir: "outputs/images"
  logs_dir: "outputs/logs"
  save_generated_images: false
  resume_from_checkpoint: false
  checkpoint_path: "outputs/checkpoint.json"
  metrics: ["mean", "std", "median", "min", "max"]

# ===== 日志配置 =====
logging:
  level: "INFO"
  console_output: true
  file_output: false
  log_file: "outputs/logs/benchmark.log"
```

### 配置的使用

#### 1. Pipeline初始化
```python
# 从配置加载所有组件
pipeline = BenchmarkPipeline(config)
# 所有模块都基于config初始化
```

#### 2. 动态模型加载
```python
# 扩散模型
class_path = config["diffusion_model"]["class_path"]
params = config["diffusion_model"]["params"]
model = load_model_by_path(class_path, params)

# 更换模型只需修改config.yaml：
# class_path: "src.models.diffusion.implementations.stable_diffusion.StableDiffusionEditModel"
```

#### 3. Prompt管理
```python
# 从配置加载所有类别的prompt
prompt_manager = PromptManager(config["prompts"])

# 使用时自动选择对应类别的prompt
prompts = prompt_manager.get_full_prompt(
    category="物理",
    original_description="...",
    edit_instruction="..."
)
```

### 优势
- ✅ **易于维护**：所有配置集中管理
- ✅ **灵活替换**：不修改代码即可替换模型
- ✅ **参数调优**：方便调整各种超参数
- ✅ **多配置管理**：可以有多个config文件用于不同实验

---

## 🔑 关键代码解析

### 1. 单个Pair的处理流程

```python
# 位置: _process_category() 方法中

for pair in tqdm(category_data.data_pairs):
    
    # ===== 步骤1: 解码原始图像 =====
    if pair.original_image is None:
        pair.original_image = decode_base64_image(pair.original_image_b64)
    # 输入: base64字符串
    # 输出: PIL.Image对象
    
    # ===== 步骤2: 编辑图像 =====
    edited_image = self.diffusion_model.edit_image(
        original_image=pair.original_image,
        edit_instruction=pair.edit_instruction
    )
    pair.edited_image = edited_image
    # 输入: PIL.Image + 编辑指令
    # 输出: 编辑后的PIL.Image
    # 耗时: ~3-10秒/张 (取决于推理步数和GPU)
    
    # ===== 步骤3: 保存图像（可选） =====
    if self.config.get("evaluation", {}).get("save_generated_images", False):
        self._save_edited_image(pair, category_name)
    # 保存到: outputs/images/{category}/{pair_id}.png
    
    # ===== 步骤4: 获取Prompt =====
    prompts = self.prompt_manager.get_full_prompt(
        category=category_name,
        original_description=pair.original_description,
        edit_instruction=pair.edit_instruction
    )
    # 返回: {"system_prompt": "...", "user_prompt": "..."}
    # 根据类别自动选择对应的prompt模板
    
    # ===== 步骤5: 评分 =====
    score = self.reward_model.score(
        edited_image=edited_image,
        original_description=pair.original_description,
        edit_instruction=pair.edit_instruction,
        system_prompt=prompts["system_prompt"],
        user_prompt=prompts["user_prompt"],
        original_image=pair.original_image
    )
    # 输入: 编辑后的图像 + 描述 + 指令 + prompt
    # 输出: 浮点分数 (0.0 - 10.0)
    # 耗时: ~1-5秒/张 (取决于模型大小)
    
    # ===== 步骤6: 记录结果 =====
    pair.score = score
    scores.append(score)
    
    # ===== 步骤7: 更新断点 =====
    self.checkpoint_data[category_name].append({
        "pair_id": pair.pair_id,
        "score": score
    })
```

### 2. 动态模型加载

```python
def _load_diffusion_model(self) -> BaseDiffusionModel:
    model_config = self.config.get("diffusion_model", {})
    class_path = model_config.get("class_path")
    # 例如: "src.models.diffusion.implementations.qwen_image_edit.QwenImageEditModel"
    
    if not class_path:
        raise ValueError("diffusion_model.class_path not specified")
    
    # 分离模块路径和类名
    module_path, class_name = class_path.rsplit(".", 1)
    # module_path = "src.models.diffusion.implementations.qwen_image_edit"
    # class_name = "QwenImageEditModel"
    
    # 动态导入模块
    module = importlib.import_module(module_path)
    # 相当于: from src.models.diffusion.implementations import qwen_image_edit
    
    # 获取类
    model_class = getattr(module, class_name)
    # 相当于: model_class = qwen_image_edit.QwenImageEditModel
    
    # 实例化模型
    model = model_class(model_config.get("params", {}))
    # 相当于: model = QwenImageEditModel(params)
    
    return model

# 优势: 
# 1. 不需要在代码中硬编码import
# 2. 通过修改config.yaml即可替换不同实现
# 3. 支持插件式扩展
```

### 3. Prompt获取和填充

```python
# 步骤1: 初始化PromptManager
self.prompt_manager = PromptManager(config.get("prompts", {}))
# 加载所有类别的prompt配置

# 步骤2: 获取特定类别的prompt
prompts = self.prompt_manager.get_full_prompt(
    category="物理",
    original_description="一辆红色的汽车停在街道上",
    edit_instruction="将红色汽车改为蓝色"
)

# 内部处理:
# 1. 获取"物理"类别的system_prompt (固定)
# 2. 获取"物理"类别的user_prompt_template
# 3. 使用.format()填充模板:
#    template = "Original scene: {original_description}\nEdit: {edit_instruction}"
#    filled = template.format(
#        original_description="一辆红色的汽车停在街道上",
#        edit_instruction="将红色汽车改为蓝色"
#    )

# 返回:
# {
#     "system_prompt": "You are an image editing reward model evaluator for Physical Consistency...",
#     "user_prompt": "Original scene: 一辆红色的汽车停在街道上\nEdit: 将红色汽车改为蓝色\n..."
# }
```

### 4. 异常处理

```python
for pair in pbar:
    try:
        # 编辑图像
        edited_image = self.diffusion_model.edit_image(...)
        
        # 评分
        score = self.reward_model.score(...)
        
        scores.append(score)
        
    except Exception as e:
        # 记录错误
        self.logger.error(f"Error processing pair {pair.pair_id}: {e}")
        
        # 使用默认分数
        scores.append(0.0)
        # 或者: scores.append(5.0)  # 中性分数
        
        # 继续处理下一个pair
        continue

# 优势:
# 1. 单个样本失败不影响整体评测
# 2. 记录详细错误信息便于调试
# 3. 可以选择合适的默认分数策略
```

---

## 📊 性能考虑

### 时间估算

假设：
- 扩散模型编辑: 5秒/张
- Reward模型评分: 2秒/张
- 总时间 = 7秒/张

**单个类别（50张）**：
```
50张 × 7秒 = 350秒 ≈ 6分钟
```

**全部5个类别（270张）**：
```
270张 × 7秒 = 1890秒 ≈ 31.5分钟
```

### 优化建议

1. **批处理**
```python
# 当前: 逐张处理
for pair in pairs:
    edited = diffusion.edit_image(pair.image, pair.instruction)

# 优化: 批量处理
edited_images = diffusion.batch_edit(
    [p.image for p in pairs],
    [p.instruction for p in pairs]
)
```

2. **延迟解码**
```python
# 当前实现已采用: 只在需要时解码图像
decode_images=False  # 加载数据时不解码
pair.original_image = decode_base64_image(pair.original_image_b64)  # 使用时才解码
```

3. **GPU内存管理**
```python
# 在模型推理后清理缓存
with torch.inference_mode():
    output = model.generate(...)

torch.cuda.empty_cache()  # 释放未使用的缓存
```

4. **并行处理**
```python
# 如果有多张GPU，可以并行处理不同类别
# 或使用 torch.multiprocessing
```

---

## 🎯 总结

### Pipeline的设计特点

1. **模块化设计**
   - 各模块职责清晰
   - 低耦合高内聚
   - 易于替换和扩展

2. **配置驱动**
   - 单一配置源
   - 动态加载模型
   - 灵活参数调整

3. **健壮性**
   - 完整的异常处理
   - 断点续传支持
   - 详细的日志记录

4. **可扩展性**
   - 抽象基类设计
   - 插件式模型加载
   - 易于添加新功能

### 数据流总览

```
JSON → BenchmarkData → CategoryData → DataPair
                                         ↓
                              [原图b64 + 编辑指令]
                                         ↓
                              DiffusionModel.edit_image()
                                         ↓
                              [编辑后的图像]
                                         ↓
                              RewardModel.score()
                                         ↓
                              [分数: 7.234]
                                         ↓
                              category_scores["物理"] = [...]
                                         ↓
                              Scorer.compute_statistics()
                                         ↓
                              Reporter.generate_report()
                                         ↓
                              JSON + Markdown报告
```

### 关键优势

✅ **完整性**：覆盖从数据加载到报告生成的完整流程  
✅ **灵活性**：通过配置即可替换模型和调整参数  
✅ **可靠性**：支持断点续传，避免重复计算  
✅ **可维护性**：代码结构清晰，易于理解和修改  
✅ **可扩展性**：基于抽象接口，方便添加新功能  

---

**文档创建时间**: 2025-10-23 20:50  
**代码版本**: v1.0  
**状态**: Pipeline分析完成


