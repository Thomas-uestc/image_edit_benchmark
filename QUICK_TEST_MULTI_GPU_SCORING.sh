#!/bin/bash
# 快速测试多GPU评分

echo "========================================"
echo "测试多GPU评分解决方案"
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
echo "运行Full Multi-GPU Pipeline"
echo "========================================  "
echo ""
echo "配置："
echo "  - 编辑：6个GPU并行编辑"
echo "  - 评分：6个GPU并行评分 ⭐ NEW"
echo ""
echo "你将看到："
echo "  1. 编辑阶段：6个GPU的去噪进度条"
echo "  2. 评分阶段：6个GPU同时输出评分信息"
echo "  3. 监控建议：在另一个终端运行 'watch -n 1 nvidia-smi'"
echo ""
echo "开始运行..."
echo ""

python main.py --config config_full_multi_gpu.yaml

echo ""
echo "========================================"
echo "测试完成！"
echo "========================================"
echo ""
echo "如果要监控GPU使用率，请在另一个终端运行："
echo "  watch -n 1 nvidia-smi"

