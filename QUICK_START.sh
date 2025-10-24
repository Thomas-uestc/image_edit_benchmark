#!/bin/bash
# 图像编辑Benchmark系统 - 快速启动脚本
# 版本: v2.1
# 使用方法: bash QUICK_START.sh

echo "=========================================="
echo "  图像编辑Benchmark系统 v2.1"
echo "=========================================="
echo ""

# 1. 激活环境
echo "📦 激活Conda环境..."
source ~/miniconda3/etc/profile.d/conda.sh  # 根据实际路径调整
conda activate yx_grpo_rl_post_edit

# 2. 进入项目目录
echo "📂 进入项目目录..."
cd /data2/yixuan/image_edit_benchmark

# 3. 检查GPU
echo "🔍 检查GPU状态..."
nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader

echo ""
echo "✅ 环境准备完成！"
echo ""

# 4. 询问运行模式
echo "请选择运行模式："
echo "  1) 完整运行（270张图像，所有类别，约5分钟）"
echo "  2) 快速测试（50张图像，物理类别，约1分钟）"
echo "  3) 自定义"
echo ""
read -p "请输入选项 [1-3]: " choice

case $choice in
    1)
        echo ""
        echo "🚀 开始完整运行..."
        python main.py --config config_multi_gpu.yaml
        ;;
    2)
        echo ""
        echo "🚀 开始快速测试（仅物理类别）..."
        python main.py --config config_multi_gpu.yaml --categories 物理
        ;;
    3)
        echo ""
        echo "请手动运行："
        echo "  python main.py --config config_multi_gpu.yaml [选项]"
        echo ""
        echo "可用选项："
        echo "  --categories 物理 环境    # 指定类别"
        echo "  --output-dir ./results   # 指定输出目录"
        echo "  --debug                  # 调试模式"
        ;;
    *)
        echo "❌ 无效选项"
        exit 1
        ;;
esac

echo ""
echo "✅ 完成！"
echo "📁 结果保存在: outputs/"
echo ""

