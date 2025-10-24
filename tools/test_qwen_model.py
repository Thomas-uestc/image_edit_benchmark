"""
测试Qwen-Image-Edit模型
验证模型是否能正确加载和编辑图像
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from PIL import Image
from src.models.diffusion.implementations.qwen_image_edit import QwenImageEditModel
from src.utils import setup_logger, decode_base64_image

def test_qwen_model():
    """测试Qwen模型"""
    
    # 设置日志
    logger = setup_logger(name="test_qwen", level="INFO", console_output=True)
    
    logger.info("="*60)
    logger.info("测试 Qwen-Image-Edit 模型")
    logger.info("="*60)
    
    try:
        # 配置
        config = {
            "model_name": "Qwen/Qwen-Image-Edit",
            "device": "cuda",
            "dtype": "bfloat16",
            "num_inference_steps": 50,
            "true_cfg_scale": 4.0,
            "negative_prompt": " ",
            "seed": 42,
            "disable_progress_bar": False  # 显示进度条
        }
        
        logger.info("\n步骤1: 初始化模型")
        logger.info("-"*60)
        model = QwenImageEditModel(config)
        logger.info("✓ 模型初始化成功")
        
        logger.info("\n步骤2: 加载测试图像")
        logger.info("-"*60)
        
        # 从benchmark中加载第一张图像进行测试
        from src.data import BenchmarkLoader
        loader = BenchmarkLoader(logger=logger)
        
        data_path = "/data2/yixuan/Benchmark/version_2_50_pair/version_2_with_imagesb64.json"
        categories = ["物理"]
        
        benchmark_data = loader.load(
            data_path=data_path,
            categories=categories,
            decode_images=False
        )
        
        # 获取第一个数据对
        first_pair = benchmark_data.get_category("物理").data_pairs[0]
        original_image = decode_base64_image(first_pair.original_image_b64)
        edit_instruction = first_pair.edit_instruction
        
        logger.info(f"✓ 测试图像加载成功")
        logger.info(f"  图像尺寸: {original_image.size}")
        logger.info(f"  编辑指令: {edit_instruction[:80]}...")
        
        logger.info("\n步骤3: 执行图像编辑")
        logger.info("-"*60)
        logger.info("正在编辑图像...")
        
        edited_image = model.edit_image(
            original_image=original_image,
            edit_instruction=edit_instruction
        )
        
        logger.info("✓ 图像编辑成功")
        logger.info(f"  编辑后图像尺寸: {edited_image.size}")
        
        logger.info("\n步骤4: 保存结果")
        logger.info("-"*60)
        
        # 保存编辑后的图像
        output_dir = project_root / "outputs" / "test"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / "qwen_test_output.png"
        edited_image.save(output_path)
        
        logger.info(f"✓ 编辑后图像已保存到: {output_path}")
        
        # 也保存原图用于对比
        original_path = output_dir / "qwen_test_input.png"
        original_image.save(original_path)
        logger.info(f"✓ 原始图像已保存到: {original_path}")
        
        logger.info("\n" + "="*60)
        logger.info("✓ Qwen-Image-Edit模型测试通过！")
        logger.info("="*60)
        logger.info("\n提示: 你可以查看生成的图像来验证编辑效果")
        logger.info(f"  原图: {original_path}")
        logger.info(f"  编辑后: {output_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_qwen_model()
    sys.exit(0 if success else 1)


