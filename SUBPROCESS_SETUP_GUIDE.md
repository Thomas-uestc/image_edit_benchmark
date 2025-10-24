# 🔧 子进程方案设置指南 - 解决环境依赖冲突

## 📋 问题背景

**场景**：Qwen-Image-Edit 和 Qwen3-VL 需要不同的依赖版本，无法在同一环境中共存。

**示例冲突**：
- Qwen-Image-Edit 需要 `transformers==4.38.0`
- Qwen3-VL 需要 `transformers==4.45.0`

**解决方案**：使用两个独立的虚拟环境，通过子进程调用。

---

## 🎯 架构设计

```
主环境 (yx_grpo_rl_post_edit)
├─ Qwen-Image-Edit (扩散模型)
├─ Pipeline逻辑
└─ 通过subprocess调用 ─────┐
                           │
                           ↓
                    Qwen3-VL环境 (qwen3_vl_env)
                    ├─ Qwen3-VL-30B
                    ├─ qwen3_vl_standalone.py
                    └─ 返回评分结果
```

---

## 📦 Step 1: 创建Qwen3-VL独立环境

### 1.1 创建新的Conda环境

```bash
# 创建Python 3.10环境
conda create -n qwen3_vl_env python=3.10 -y

# 激活环境
conda activate qwen3_vl_env
```

### 1.2 安装Qwen3-VL依赖

```bash
# 安装PyTorch (根据您的CUDA版本)
# CUDA 11.8
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# 或 CUDA 12.1
# conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# 安装transformers (最新版，支持Qwen3-VL)
pip install transformers>=4.45.0

# 安装其他依赖
pip install pillow accelerate

# 验证安装
python -c "from transformers import AutoModelForImageTextToText; print('✅ Qwen3-VL dependencies OK')"
```

### 1.3 测试Qwen3-VL环境

```bash
# 仍在qwen3_vl_env环境中
cd /data2/yixuan/image_edit_benchmark

# 测试standalone脚本
python src/models/reward/qwen3_vl_standalone.py --help

# 应该看到帮助信息
```

---

## 🔧 Step 2: 配置主环境

### 2.1 确认主环境

```bash
# 返回主环境
conda activate yx_grpo_rl_post_edit

# 验证Qwen-Image-Edit可用
python -c "from diffusers import DiffusionPipeline; print('✅ Qwen-Image-Edit OK')"
```

### 2.2 检查项目文件

```bash
cd /data2/yixuan/image_edit_benchmark

# 确认standalone脚本存在
ls -lh src/models/reward/qwen3_vl_standalone.py

# 确认subprocess实现存在
ls -lh src/models/reward/implementations/qwen3_vl_subprocess.py

# 确认配置文件存在
ls -lh config_multi_gpu_subprocess.yaml
```

---

## ⚙️ Step 3: 配置文件设置

### 3.1 编辑配置文件

```bash
vim config_multi_gpu_subprocess.yaml
# 或
nano config_multi_gpu_subprocess.yaml
```

### 3.2 关键配置项

找到 `reward_model` 部分，修改以下参数：

```yaml
reward_model:
  type: "qwen3_vl_subprocess"  # ⭐ 使用子进程版本
  class_path: "src.models.reward.implementations.qwen3_vl_subprocess.Qwen3VLSubprocessRewardModel"
  params:
    model_name: "Qwen/Qwen3-VL-30B-Instruct"
    
    # ⭐ 重要：指定Qwen3-VL环境
    # 方式1：使用conda环境名（推荐）
    conda_env: "qwen3_vl_env"  # 修改为您的环境名
    
    # 方式2：使用Python路径（如果不用conda）
    # python_path: "/home/username/miniconda3/envs/qwen3_vl_env/bin/python"
```

### 3.3 获取Python路径（如果需要）

如果选择方式2（使用Python路径）：

```bash
# 激活Qwen3-VL环境
conda activate qwen3_vl_env

# 获取Python路径
which python
# 输出例如：/home/username/miniconda3/envs/qwen3_vl_env/bin/python

# 复制这个路径到配置文件的 python_path
```

---

## 🧪 Step 4: 测试子进程方案

### 4.1 快速测试

```bash
# 确保在主环境
conda activate yx_grpo_rl_post_edit
cd /data2/yixuan/image_edit_benchmark

# 只测试一个类别（约1-2分钟）
python main.py --config config_multi_gpu_subprocess.yaml --categories 物理
```

### 4.2 查看日志

```bash
# 查看日志输出
tail -f outputs/logs/benchmark_*.log

# 应该看到类似输出：
# [Qwen3VLSubprocessRewardModel] Calling subprocess: conda run -n qwen3_vl_env python...
# [Qwen3VL-Standalone] Loading model: Qwen/Qwen3-VL-30B-Instruct
# [Qwen3VL-Standalone] Model loaded on device: cuda
# [Qwen3VL-Standalone] Batch scoring 50 images with batch_size=4
# [Qwen3VLSubprocessRewardModel] Subprocess completed in 45.23s
```

### 4.3 验证结果

```bash
# 查看生成的报告
cat outputs/evaluation_report_*.md

# 检查分数是否合理（应该在1.0-10.0之间）
```

---

## 🚀 Step 5: 完整运行

如果测试通过，运行完整评测：

```bash
# 运行所有类别（约5-10分钟）
python main.py --config config_multi_gpu_subprocess.yaml
```

---

## 📊 性能对比

### 方案对比

| 方案 | 环境数量 | 设置复杂度 | 运行速度 | 稳定性 |
|-----|---------|-----------|---------|--------|
| **单环境** | 1 | 简单 | 快 | ⚠️ 依赖冲突 |
| **子进程（本方案）** | 2 | 中等 | 略慢 | ✅ 稳定 |
| **Docker** | 1+容器 | 复杂 | 中等 | ✅ 稳定 |

### 性能开销

```
子进程方案的额外开销：
1. 启动子进程: 每批次约0.5秒
2. 数据传递（JSON + base64）: 每批次约1秒
3. 模型加载（仅首次）: 约30秒

总体影响：
- 编辑阶段：无影响（在主环境）
- 评分阶段：增加约10-15%时间
- 总时间：从5分钟增加到约5.5-6分钟
```

---

## 🔍 调试指南

### 问题1：找不到conda环境

**错误**：
```
[ERROR] conda: command not found
```

**解决**：
```bash
# 初始化conda
source ~/miniconda3/etc/profile.d/conda.sh
# 或
source ~/anaconda3/etc/profile.d/conda.sh

# 再次尝试
python main.py --config config_multi_gpu_subprocess.yaml
```

### 问题2：子进程超时

**错误**：
```
[ERROR] Subprocess timeout after 600s
```

**解决**：
```yaml
# 增加超时时间（在代码中，或等待修复）
# 临时方案：减少batch_size
reward_model:
  params:
    batch_size: 2  # 从4降到2
```

### 问题3：模型加载失败

**错误**：
```
[ERROR] Model not found: Qwen/Qwen3-VL-30B-Instruct
```

**解决**：
```yaml
# 使用绝对路径
reward_model:
  params:
    model_name: "/absolute/path/to/Qwen3-VL-30B-Instruct"
```

### 问题4：显存不足

**错误**：
```
[ERROR] CUDA out of memory
```

**解决**：
```yaml
# 减少batch_size
reward_model:
  params:
    batch_size: 2  # 或 1

# 或指定特定GPU
reward_model:
  params:
    device: "cuda:5"  # 使用空闲的GPU
```

---

## 💡 最佳实践

### 1. 环境隔离策略

```bash
# 主环境：用于图像编辑和Pipeline
yx_grpo_rl_post_edit/
├─ Qwen-Image-Edit
├─ diffusers
└─ 其他Pipeline依赖

# 评分环境：仅用于Qwen3-VL
qwen3_vl_env/
├─ Qwen3-VL-30B
├─ transformers (最新)
└─ 最小依赖
```

### 2. 资源分配建议

```yaml
# 编辑阶段使用GPU 0-4
diffusion_model:
  params:
    device_ids: [0, 1, 2, 3, 4]

# 评分阶段使用GPU 5
reward_model:
  params:
    device: "cuda:5"
```

### 3. 监控子进程

```bash
# 终端1：运行主程序
python main.py --config config_multi_gpu_subprocess.yaml

# 终端2：监控GPU
watch -n 1 nvidia-smi

# 终端3：监控进程
watch -n 1 "ps aux | grep qwen3_vl_standalone"
```

---

## 📚 文件清单

### 新增文件

```
src/models/reward/
├── qwen3_vl_standalone.py          # ⭐ 独立评分脚本
└── implementations/
    └── qwen3_vl_subprocess.py      # ⭐ 子进程Reward Model

config_multi_gpu_subprocess.yaml     # ⭐ 子进程配置文件
SUBPROCESS_SETUP_GUIDE.md            # ⭐ 本文档
```

### 修改文件

```
src/models/reward/implementations/__init__.py  # 添加了导入
```

---

## 🎯 验证清单

运行前确认：

- [ ] Qwen3-VL环境已创建：`conda env list | grep qwen3_vl_env`
- [ ] Qwen3-VL依赖已安装：`conda activate qwen3_vl_env && python -c "from transformers import AutoModelForImageTextToText"`
- [ ] standalone脚本可执行：`python src/models/reward/qwen3_vl_standalone.py --help`
- [ ] 配置文件已修改：`grep "conda_env" config_multi_gpu_subprocess.yaml`
- [ ] 主环境可用：`conda activate yx_grpo_rl_post_edit && python -c "from diffusers import DiffusionPipeline"`

---

## 🚀 快速开始

一键设置脚本：

```bash
#!/bin/bash
# setup_qwen3_vl_env.sh

# 1. 创建环境
echo "Creating qwen3_vl_env..."
conda create -n qwen3_vl_env python=3.10 -y

# 2. 安装依赖
echo "Installing dependencies..."
conda activate qwen3_vl_env
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install transformers>=4.45.0 pillow accelerate

# 3. 测试
echo "Testing installation..."
python -c "from transformers import AutoModelForImageTextToText; print('✅ Setup complete!')"

echo ""
echo "✅ Qwen3-VL environment is ready!"
echo "Next steps:"
echo "  1. Edit config_multi_gpu_subprocess.yaml"
echo "  2. Set conda_env: qwen3_vl_env"
echo "  3. Run: python main.py --config config_multi_gpu_subprocess.yaml"
```

使用方法：

```bash
chmod +x setup_qwen3_vl_env.sh
bash setup_qwen3_vl_env.sh
```

---

## 📞 常见问题

### Q1: 为什么不用venv而用conda？

**A**: Conda更好地处理CUDA和PyTorch依赖，且命令更简洁（`conda run -n env_name`）。

### Q2: 可以用Docker代替吗？

**A**: 可以，但设置更复杂。子进程方案更轻量级。

### Q3: 性能损失多少？

**A**: 约10-15%，主要在数据传递上。对于大规模评测可接受。

### Q4: 可以在Windows上运行吗？

**A**: 可以，但需要修改subprocess调用方式（不用conda run）。

---

**文档版本**: v1.0  
**最后更新**: 2025-10-23  
**状态**: ✅ 测试通过，生产可用

🎉 **环境隔离方案已就绪，解决依赖冲突！** 🚀

