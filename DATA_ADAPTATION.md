# 数据适配说明

## 📊 实际数据集信息

### 数据文件
- **路径**: `/data2/yixuan/Benchmark/version_2_50_pair/version_2_with_imagesb64.json`
- **格式**: JSON列表
- **总数据量**: 270条

### 数据分布
| 类别 | 数量 |
|------|------|
| 物理 | 50 |
| 环境 | 50 |
| 社会 | 70 |
| 因果 | 50 |
| 指代 | 50 |
| **总计** | **270** |

---

## 🔧 已完成的适配

### 1. JSON数据结构适配

**原始数据格式**:
```json
[
  {
    "subset": "物理",
    "original_description": "阳台窗边，一只猫正趴在阳光照射的木地板上...",
    "edit_instruction": "将阳光方向改为从右上方照射...",
    "original_description_en": "A cat lies on a sunlit wooden floor...",
    "edit_instruction_en": "Change the sunlight to come from the upper right...",
    "src_img_b64": "iVBORw0KGgoAAAANSU...",
    "rationale_short": "...",
    "rationale_short_en": "...",
    "tags": ["indoor", "lighting", "shadow", "animal", "geometry"],
    "difficulty": "medium",
    "original_image_path": "generated_images_2/00000_物理_medium.png",
    "seed": 42
  },
  ...
]
```

**关键字段映射**:
- `subset` → 类别标识（物理、环境、社会、因果、指代）
- `src_img_b64` → 原始图像的base64编码
- `original_description_en` → 原始图像英文描述
- `edit_instruction_en` → 编辑指令英文版
- `original_image_path` → 用作pair_id

### 2. 数据加载器修改

**文件**: `src/data/benchmark_loader.py`

**主要修改**:
1. 支持JSON列表格式（而非字典格式）
2. 按`subset`字段筛选类别
3. 使用`src_img_b64`字段作为图像数据源
4. 使用英文字段（`original_description_en`, `edit_instruction_en`）
5. 从`original_image_path`提取pair_id

**代码片段**:
```python
# 从列表中筛选指定subset的数据
for idx, item in enumerate(data_list):
    item_subset = item.get("subset", "")
    if item_subset != category:
        continue  # 跳过不匹配的类别
    
    # 提取字段
    pair_id = item.get("original_image_path", f"{category}_{idx}")
    original_image_b64 = item.get("src_img_b64", "")
    edit_instruction = item.get("edit_instruction_en", "")
    original_description = item.get("original_description_en", "")
```

### 3. 配置文件模板更新

**文件**: `config_template.yaml`

**更新内容**:
```yaml
benchmark:
  data_path: "/data2/yixuan/Benchmark/version_2_50_pair/version_2_with_imagesb64.json"
  categories:
    - "物理"
    - "环境"
    - "社会"
    - "因果"
    - "指代"
```

---

## ✅ 测试验证

### 测试脚本
`test_data_loading.py`

### 测试结果
```
✓ 数据加载成功：270条数据
✓ 类别分类正确：5个类别
✓ 字段提取正确：
  - ID: 00000_物理_medium
  - 描述: A cat lies on a sunlit wooden floor...
  - 指令: Change the sunlight to come from...
  - Base64长度: 2061488
✓ 图像解码成功：尺寸 (1472, 1104), 模式 RGB
```

---

## 📝 额外元数据

每个数据对的`metadata`字段包含完整的原始数据，包括：
- `subset`: 类别
- `original_description`: 中文描述
- `edit_instruction`: 中文指令
- `rationale_short`: 短理由（中文）
- `rationale_short_en`: 短理由（英文）
- `tags`: 标签列表
- `difficulty`: 难度（easy/medium/hard）
- `seed`: 随机种子

这些信息可用于：
- 更详细的分析
- 按难度分类
- 按标签过滤
- 多语言支持

**访问示例**:
```python
pair = benchmark_data.get_category("物理").data_pairs[0]
difficulty = pair.metadata.get("difficulty")
tags = pair.metadata.get("tags")
rationale = pair.metadata.get("rationale_short_en")
```

---

## 🔄 后续扩展

### 如果有更多版本的数据

例如 `version_3_200_pair` 等，只需：

1. 在配置文件中修改路径：
```yaml
benchmark:
  data_path: "/data2/yixuan/Benchmark/version_3_200_pair/version_3.json"
```

2. 确认数据格式一致（如果不一致，需要相应调整loader）

### 如果需要使用中文字段

修改 `benchmark_loader.py`:
```python
# 使用中文字段
edit_instruction = item.get("edit_instruction", "")  # 而非 edit_instruction_en
original_description = item.get("original_description", "")  # 而非 original_description_en
```

### 如果需要按难度或标签筛选

可以在 `BenchmarkLoader` 中添加过滤方法：
```python
def load_with_filter(self, data_path, categories, 
                    difficulty=None, tags=None):
    # 添加过滤逻辑
    pass
```

---

## 📊 数据质量检查

已验证的数据质量指标：
- ✅ 所有270条数据都有完整的必填字段
- ✅ 所有图像base64编码有效且可解码
- ✅ 所有类别分布符合预期
- ✅ 图像尺寸合理（约1472x1104）
- ✅ 描述和指令都有英文版本

---

## 🎯 下一步

数据加载模块已完成适配并验证通过。可以继续：

1. **实现扩散编辑模型** (`src/models/diffusion/implementations/`)
2. **实现Reward评分模型** (`src/models/reward/implementations/`)
3. **配置各类别的评分prompt** (`config.yaml` 中的 `prompts` 部分)
4. **运行完整的评测pipeline**

---

## 📞 问题排查

如果遇到数据加载问题：

1. **检查文件路径**: 确认JSON文件存在
2. **检查JSON格式**: 运行 `inspect_benchmark_structure.py`
3. **检查类别名称**: 确保配置中的类别名与数据中的`subset`字段一致
4. **查看日志**: 检查详细的错误信息

---

更新日期: 2025-10-23


