#!/bin/bash
#================================================================
# Qwen3-VL 独立环境设置脚本
# 用于解决Qwen-Image-Edit和Qwen3-VL的依赖冲突问题
#================================================================

set -e  # 遇到错误立即退出

echo "================================================================"
echo "  Qwen3-VL 独立环境设置"
echo "================================================================"
echo ""

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 配置参数
ENV_NAME="qwen3_vl_env"
PYTHON_VERSION="3.10"
CUDA_VERSION="11.8"  # 根据您的CUDA版本修改

echo -e "${YELLOW}配置信息:${NC}"
echo "  环境名称: $ENV_NAME"
echo "  Python版本: $PYTHON_VERSION"
echo "  CUDA版本: $CUDA_VERSION"
echo ""

# 询问是否继续
read -p "是否继续? (y/n): " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "已取消"
    exit 1
fi

#----------------------------------------------------------------
# Step 1: 检查conda
#----------------------------------------------------------------
echo -e "${YELLOW}[1/6] 检查conda...${NC}"

if ! command -v conda &> /dev/null
then
    echo -e "${RED}❌ 错误: 找不到conda命令${NC}"
    echo "请确保已安装Anaconda或Miniconda"
    exit 1
fi

echo -e "${GREEN}✅ Conda已安装${NC}"
echo ""

#----------------------------------------------------------------
# Step 2: 检查环境是否已存在
#----------------------------------------------------------------
echo -e "${YELLOW}[2/6] 检查环境...${NC}"

if conda env list | grep -q "^$ENV_NAME "; then
    echo -e "${YELLOW}⚠️  环境 '$ENV_NAME' 已存在${NC}"
    read -p "是否删除并重新创建? (y/n): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]
    then
        echo "删除旧环境..."
        conda env remove -n $ENV_NAME -y
    else
        echo "跳过环境创建"
        SKIP_CREATE=true
    fi
fi
echo ""

#----------------------------------------------------------------
# Step 3: 创建conda环境
#----------------------------------------------------------------
if [ "$SKIP_CREATE" != "true" ]; then
    echo -e "${YELLOW}[3/6] 创建conda环境...${NC}"
    conda create -n $ENV_NAME python=$PYTHON_VERSION -y
    echo -e "${GREEN}✅ 环境创建成功${NC}"
else
    echo -e "${YELLOW}[3/6] 跳过环境创建${NC}"
fi
echo ""

#----------------------------------------------------------------
# Step 4: 安装PyTorch
#----------------------------------------------------------------
echo -e "${YELLOW}[4/6] 安装PyTorch...${NC}"

# 根据CUDA版本选择安装命令
if [ "$CUDA_VERSION" == "11.8" ]; then
    PYTORCH_CMD="conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y"
elif [ "$CUDA_VERSION" == "12.1" ]; then
    PYTORCH_CMD="conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y"
else
    echo -e "${RED}❌ 不支持的CUDA版本: $CUDA_VERSION${NC}"
    exit 1
fi

echo "执行: $PYTORCH_CMD"
conda run -n $ENV_NAME bash -c "$PYTORCH_CMD"

echo -e "${GREEN}✅ PyTorch安装成功${NC}"
echo ""

#----------------------------------------------------------------
# Step 5: 安装transformers和其他依赖
#----------------------------------------------------------------
echo -e "${YELLOW}[5/6] 安装transformers和其他依赖...${NC}"

conda run -n $ENV_NAME pip install \
    "transformers>=4.45.0" \
    pillow \
    accelerate \
    sentencepiece \
    protobuf

echo -e "${GREEN}✅ 依赖安装成功${NC}"
echo ""

#----------------------------------------------------------------
# Step 6: 测试安装
#----------------------------------------------------------------
echo -e "${YELLOW}[6/6] 测试安装...${NC}"

# 测试导入
TEST_SCRIPT='
import sys
try:
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from PIL import Image
    import torch
    print("✅ All imports successful")
    print(f"✅ PyTorch: {torch.__version__}")
    print(f"✅ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✅ CUDA device: {torch.cuda.get_device_name(0)}")
    sys.exit(0)
except Exception as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)
'

if conda run -n $ENV_NAME python -c "$TEST_SCRIPT"; then
    echo -e "${GREEN}✅ 环境测试通过${NC}"
else
    echo -e "${RED}❌ 环境测试失败${NC}"
    exit 1
fi
echo ""

#----------------------------------------------------------------
# 完成
#----------------------------------------------------------------
echo "================================================================"
echo -e "${GREEN}✅ Qwen3-VL环境设置完成！${NC}"
echo "================================================================"
echo ""
echo "环境名称: $ENV_NAME"
echo ""
echo "下一步操作："
echo "  1. 编辑配置文件:"
echo "     vim config_multi_gpu_subprocess.yaml"
echo ""
echo "  2. 修改以下配置:"
echo "     reward_model:"
echo "       params:"
echo "         conda_env: \"$ENV_NAME\"  # 设置环境名"
echo ""
echo "  3. 测试运行:"
echo "     conda activate yx_grpo_rl_post_edit"
echo "     python main.py --config config_multi_gpu_subprocess.yaml --categories 物理"
echo ""
echo "  4. 完整运行:"
echo "     python main.py --config config_multi_gpu_subprocess.yaml"
echo ""
echo "================================================================"


