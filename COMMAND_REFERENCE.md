# 📋 命令参考速查表

## 🚀 基础命令

### 1️⃣ 完整运行（推荐）

```bash
# 激活环境
conda activate yx_grpo_rl_post_edit

# 进入目录
cd /data2/yixuan/image_edit_benchmark

# 运行（使用所有优化：6GPU并行 + 批次同步 + Batch Inference）
python main.py --config config_multi_gpu.yaml
```

**预期时间**: 约5分钟（270张图像）

---

### 2️⃣ 快速测试

```bash
# 只运行物理类别（50张图像）
python main.py --config config_multi_gpu.yaml --categories 物理
```

**预期时间**: 约1分钟

---

### 3️⃣ 使用脚本（最简单）

```bash
# 一键启动（交互式）
bash QUICK_START.sh
```

---

## ⚙️ 配置文件

### 多GPU配置（推荐）

```bash
python main.py --config config_multi_gpu.yaml
```

**特性**：
- ✅ 6个GPU并行编辑（6倍加速）
- ✅ 批次同步（GPU保持同步）
- ✅ Batch inference评分（2.7倍加速）

### 单GPU配置（测试）

```bash
python main.py --config config.yaml
```

**特性**：
- ⚠️ 单GPU串行处理
- ⏱️ 较慢（约22分钟）

---

## 🎯 常用选项

### 指定类别

```bash
# 单个类别
python main.py --config config_multi_gpu.yaml --categories 物理

# 多个类别
python main.py --config config_multi_gpu.yaml --categories 物理 环境 社会
```

### 指定输出目录

```bash
python main.py --config config_multi_gpu.yaml --output-dir ./my_results
```

### 调试模式

```bash
python main.py --config config_multi_gpu.yaml --debug
```

### 干跑测试（验证配置）

```bash
python main.py --config config_multi_gpu.yaml --dry-run
```

---

## 📊 监控命令

### 实时监控GPU

```bash
# 终端1: 运行程序
python main.py --config config_multi_gpu.yaml

# 终端2: 监控GPU
watch -n 1 nvidia-smi
```

### 查看日志

```bash
# 实时查看日志
tail -f outputs/logs/benchmark_*.log

# 查看最新日志
ls -lt outputs/logs/ | head -n 5
```

### 查看输出文件

```bash
# 查看生成的报告
ls -lh outputs/evaluation_report_*.{json,md}

# 查看JSON报告
cat outputs/evaluation_report_*.json | jq .

# 查看Markdown报告
cat outputs/evaluation_report_*.md
```

---

## 🔧 配置修改

### 临时修改GPU数量

编辑 `config_multi_gpu.yaml`:

```yaml
diffusion_model:
  params:
    device_ids: [0, 1, 2, 3]  # 改为使用4个GPU
```

### 临时禁用批次同步

```yaml
diffusion_model:
  params:
    enable_batch_sync: false  # 禁用批次同步
```

### 临时修改Batch Size

```yaml
reward_model:
  params:
    batch_size: 2  # 改为2（节省显存）
```

---

## 🛠️ 故障排除命令

### 检查环境

```bash
# 检查Conda环境
conda env list

# 检查Python版本
python --version

# 检查PyTorch
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

### 检查GPU

```bash
# 简洁显示
nvidia-smi --query-gpu=index,name,memory.free,memory.total --format=csv

# 详细显示
nvidia-smi
```

### 检查数据文件

```bash
# 检查文件存在
ls -lh /data2/yixuan/Benchmark/version_2_50_pair/version_2_with_imagesb64.json

# 检查文件内容
python -c "import json; data=json.load(open('/data2/yixuan/Benchmark/version_2_50_pair/version_2_with_imagesb64.json')); print(f'Total pairs: {len(data)}')"
```

### 清理缓存

```bash
# 清理Python缓存
find . -type d -name __pycache__ -exec rm -rf {} +

# 清理GPU缓存（Python内）
python -c "import torch; torch.cuda.empty_cache(); print('GPU cache cleared')"
```

---

## 📈 性能测试

### 测试不同GPU数量

```bash
# 测试单GPU
python main.py --config config.yaml --categories 物理

# 测试2GPU
# (修改config_multi_gpu.yaml: device_ids: [0, 1])
python main.py --config config_multi_gpu.yaml --categories 物理

# 测试6GPU
# (修改config_multi_gpu.yaml: device_ids: [0,1,2,3,4,5])
python main.py --config config_multi_gpu.yaml --categories 物理
```

### 测试不同Batch Size

```bash
# 修改 config_multi_gpu.yaml:
# batch_size: 2
python main.py --config config_multi_gpu.yaml --categories 物理

# batch_size: 4
python main.py --config config_multi_gpu.yaml --categories 物理

# batch_size: 8
python main.py --config config_multi_gpu.yaml --categories 物理
```

---

## 📚 查看文档

```bash
# 查看使用指南
cat HOW_TO_RUN.md

# 查看批次同步说明
cat BATCH_SYNC_QUICK_GUIDE.md

# 查看所有优化总结
cat ALL_OPTIMIZATIONS_COMPLETE.md

# 查看项目README
cat README.md
```

---

## 🎯 完整工作流

```bash
# 1. 激活环境
conda activate yx_grpo_rl_post_edit

# 2. 进入目录
cd /data2/yixuan/image_edit_benchmark

# 3. 检查GPU
nvidia-smi

# 4. 快速测试（1分钟）
python main.py --config config_multi_gpu.yaml --categories 物理

# 5. 查看结果
cat outputs/evaluation_report_*.md

# 6. 如果测试通过，运行完整评测（5分钟）
python main.py --config config_multi_gpu.yaml

# 7. 查看最终结果
ls -lh outputs/
cat outputs/evaluation_report_*.json
```

---

## 🔑 关键配置参数

| 参数 | 位置 | 推荐值 | 说明 |
|-----|------|--------|------|
| `device_ids` | diffusion_model.params | `[0,1,2,3,4,5]` | 使用的GPU |
| `enable_batch_sync` | diffusion_model.params | `true` | 批次同步 |
| `use_batch_inference` | reward_model.params | `true` | Batch推理 |
| `batch_size` | reward_model.params | `4` | 批处理大小 |
| `num_inference_steps` | diffusion_model.params | `50` | 去噪步数 |

---

## 💡 最佳实践

1. **首次运行**：先用单类别测试
   ```bash
   python main.py --config config_multi_gpu.yaml --categories 物理
   ```

2. **监控GPU**：运行时在另一个终端监控
   ```bash
   watch -n 1 nvidia-smi
   ```

3. **查看日志**：如有问题，查看详细日志
   ```bash
   tail -f outputs/logs/benchmark_*.log
   ```

4. **保存结果**：运行完后备份结果
   ```bash
   cp -r outputs/ outputs_backup_$(date +%Y%m%d_%H%M%S)/
   ```

---

**最后更新**: 2025-10-23  
**版本**: v2.1  
**状态**: ✅ 生产就绪


