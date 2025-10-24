# 工具脚本

本目录包含用于开发和测试的工具脚本。

## 📝 可用工具

### 1. inspect_benchmark_structure.py
**功能**: 检查benchmark JSON文件的结构

**用法**:
```bash
python tools/inspect_benchmark_structure.py
```

**输出**:
- JSON数据类型和结构
- 字段信息
- Subset分布统计
- 示例数据预览

**适用场景**:
- 检查新的数据文件格式
- 验证数据完整性
- 了解数据分布

---

### 2. test_data_loading.py
**功能**: 测试数据加载模块是否正常工作

**用法**:
```bash
python tools/test_data_loading.py
```

**测试内容**:
- 加载benchmark数据
- 验证类别分类
- 检查字段提取
- 测试图像解码

**适用场景**:
- 验证数据加载器修改后是否正常
- 检查新数据文件是否兼容
- 调试数据加载问题

---

## 🔧 添加新工具

如果需要添加新的工具脚本，请：

1. 在此目录创建脚本
2. 在顶部添加docstring说明用途
3. 更新本README文件

---

## 📚 相关文档

- `DATA_ADAPTATION.md` - 数据适配说明
- `USAGE_GUIDE.md` - 使用指南
- `PROJECT_STRUCTURE.md` - 项目结构


