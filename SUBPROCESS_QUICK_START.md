# ⚡ 子进程方案快速开始

## 🎯 问题

Qwen-Image-Edit 和 Qwen3-VL 依赖冲突，无法在同一环境运行。

## 💡 解决方案

使用两个独立环境，通过子进程通信。

---

## 🚀 三步开始

### Step 1: 运行设置脚本

```bash
cd /data2/yixuan/image_edit_benchmark
bash setup_qwen3_vl_env.sh
```

这将自动：
- ✅ 创建 `qwen3_vl_env` 环境
- ✅ 安装 Qwen3-VL 依赖
- ✅ 测试环境

### Step 2: 修改配置文件

```bash
# 编辑配置
vim config_multi_gpu_subprocess.yaml

# 确认以下配置
reward_model:
  type: "qwen3_vl_subprocess"
  params:
    conda_env: "qwen3_vl_env"  # ← 确保此行正确
```

### Step 3: 运行测试

```bash
# 回到主环境
conda activate yx_grpo_rl_post_edit

# 快速测试（1-2分钟）
python main.py --config config_multi_gpu_subprocess.yaml --categories 物理

# 完整运行（5-6分钟）
python main.py --config config_multi_gpu_subprocess.yaml
```

---

## 📊 架构图

```
主进程 (yx_grpo_rl_post_edit)          子进程 (qwen3_vl_env)
┌──────────────────────────┐          ┌─────────────────────┐
│ Pipeline                  │          │                     │
│ ├─ Qwen-Image-Edit       │          │  Qwen3-VL          │
│ ├─ 图像编辑（6GPU并行）    │          │  ├─ 加载模型        │
│ └─ 编辑完成               │          │  ├─ 批量评分        │
│                          │          │  └─ 返回scores      │
│ 调用Reward模型            │  JSON    │                     │
│ ├─ 准备数据（base64）     ├─────────>│  输入: tasks[]      │
│ ├─ 调用subprocess        │          │  输出: scores[]     │
│ └─ 接收结果              │<─────────┤                     │
│                          │          │                     │
└──────────────────────────┘          └─────────────────────┘
```

---

## 🔍 验证清单

运行前检查：

```bash
# 1. 检查Qwen3-VL环境
conda env list | grep qwen3_vl_env
# 应该看到: qwen3_vl_env

# 2. 测试Qwen3-VL环境
conda activate qwen3_vl_env
python -c "from transformers import AutoModelForImageTextToText; print('✅ OK')"
# 应该输出: ✅ OK

# 3. 检查配置文件
grep "conda_env" config_multi_gpu_subprocess.yaml
# 应该看到: conda_env: "qwen3_vl_env"

# 4. 回到主环境
conda activate yx_grpo_rl_post_edit
```

---

## 📈 性能对比

| 指标 | 同环境方案 | 子进程方案 |
|-----|----------|----------|
| **环境冲突** | ❌ 有冲突 | ✅ 无冲突 |
| **设置复杂度** | 简单 | 中等 |
| **运行时间** | 5分钟 | 5.5-6分钟 |
| **额外开销** | 0% | ~10-15% |
| **稳定性** | ⚠️ 看运气 | ✅ 稳定 |

---

## 🔧 配置对比

### 原始配置（同环境）

```yaml
reward_model:
  type: "qwen3_vl"
  class_path: "src.models.reward.implementations.qwen3_vl_reward.Qwen3VLRewardModel"
  # 直接在当前环境运行
```

### 子进程配置

```yaml
reward_model:
  type: "qwen3_vl_subprocess"  # ← 改为subprocess
  class_path: "src.models.reward.implementations.qwen3_vl_subprocess.Qwen3VLSubprocessRewardModel"
  params:
    conda_env: "qwen3_vl_env"  # ← 指定独立环境
```

---

## 💡 工作原理

1. **主进程**（在 `yx_grpo_rl_post_edit`）：
   - 加载Qwen-Image-Edit模型
   - 执行图像编辑
   - 准备评分任务（图像→base64，prompts）

2. **数据传递**（JSON文件）：
   ```json
   {
     "tasks": [
       {
         "image_b64": "iVBORw0KGgo...",
         "system_prompt": "你是...",
         "user_prompt": "评估..."
       }
     ]
   }
   ```

3. **子进程**（在 `qwen3_vl_env`）：
   - 读取JSON文件
   - 加载Qwen3-VL模型
   - 批量评分
   - 返回结果JSON

4. **主进程接收**：
   ```json
   {
     "scores": [7.5, 8.2, 7.8, ...],
     "status": "success"
   }
   ```

---

## 🎓 故障排除

### 问题1: 找不到conda环境

```bash
# 确保conda初始化
source ~/miniconda3/etc/profile.d/conda.sh

# 重新运行
python main.py --config config_multi_gpu_subprocess.yaml
```

### 问题2: 子进程失败

```bash
# 查看详细日志
tail -f outputs/logs/benchmark_*.log

# 手动测试standalone脚本
conda run -n qwen3_vl_env python src/models/reward/qwen3_vl_standalone.py --help
```

### 问题3: 显存不足

```yaml
# 减小batch_size
reward_model:
  params:
    batch_size: 2  # 从4降到2
```

---

## 📚 相关文档

- **`SUBPROCESS_SETUP_GUIDE.md`** - 详细设置指南
- **`setup_qwen3_vl_env.sh`** - 自动化设置脚本
- **`config_multi_gpu_subprocess.yaml`** - 配置文件

---

## ✅ 完成清单

- [ ] 运行 `bash setup_qwen3_vl_env.sh`
- [ ] 修改 `config_multi_gpu_subprocess.yaml`
- [ ] 测试运行单类别
- [ ] 验证结果正常
- [ ] 运行完整评测

---

**文档版本**: v1.0  
**最后更新**: 2025-10-23  
**状态**: ✅ 生产可用

🎉 **3步解决环境冲突，开始使用！** 🚀


