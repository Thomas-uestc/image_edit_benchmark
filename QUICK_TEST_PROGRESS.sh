#!/bin/bash
# 快速测试进度显示优化

echo "========================================"
echo "测试进度显示优化"
echo "========================================"
echo ""

# 激活环境
echo "激活conda环境: yx_grpo_rl_post_edit"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate yx_grpo_rl_post_edit

# 进入项目目录
cd /data2/yixuan/image_edit_benchmark

echo ""
echo "========================================  "
echo "运行Pipeline (使用多GPU + 子进程方案)"
echo "========================================  "
echo ""
echo "你将看到："
echo "  1. 编辑阶段：各GPU的去噪进度条（实时显示每个step）"
echo "  2. 评分阶段：每个样本的详细分数和模型响应"
echo "  3. 评分阶段：批次统计和最终总结"
echo ""
echo "开始运行..."
echo ""

python main.py --config config_multi_gpu_subprocess.yaml

echo ""
echo "========================================"
echo "测试完成！"
echo "========================================"


