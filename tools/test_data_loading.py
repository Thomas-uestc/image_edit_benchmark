"""
测试数据加载模块
验证是否能正确加载benchmark数据
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data import BenchmarkLoader
from src.utils import setup_logger

def test_data_loading():
    """测试数据加载功能"""
    
    # 设置日志
    logger = setup_logger(name="test_loader", level="INFO", console_output=True)
    
    # 创建数据加载器
    loader = BenchmarkLoader(logger=logger)
    
    # 数据路径和类别
    data_path = "/data2/yixuan/Benchmark/version_2_50_pair/version_2_with_imagesb64.json"
    categories = ["物理", "环境", "社会", "因果", "指代"]
    
    logger.info("="*60)
    logger.info("开始测试数据加载")
    logger.info("="*60)
    
    try:
        # 加载数据（不解码图像以节省时间）
        benchmark_data = loader.load(
            data_path=data_path,
            categories=categories,
            decode_images=False
        )
        
        logger.info("\n" + "="*60)
        logger.info("数据加载成功！")
        logger.info("="*60)
        
        # 显示统计信息
        logger.info(f"\n总数据对数量: {benchmark_data.total_pairs}")
        logger.info(f"类别数量: {len(benchmark_data.category_names)}")
        logger.info(f"类别列表: {benchmark_data.category_names}")
        
        # 显示每个类别的详细信息
        logger.info("\n" + "="*60)
        logger.info("各类别详细信息:")
        logger.info("="*60)
        
        for category_name in benchmark_data.category_names:
            category_data = benchmark_data.get_category(category_name)
            logger.info(f"\n类别: {category_name}")
            logger.info(f"  数据对数量: {len(category_data)}")
            
            if len(category_data) > 0:
                # 显示第一个数据对的信息
                first_pair = category_data.data_pairs[0]
                logger.info(f"  第一个数据对:")
                logger.info(f"    ID: {first_pair.pair_id}")
                logger.info(f"    原图描述: {first_pair.original_description[:80]}...")
                logger.info(f"    编辑指令: {first_pair.edit_instruction[:80]}...")
                logger.info(f"    Base64长度: {len(first_pair.original_image_b64)}")
                
                # 显示metadata中的其他信息
                if first_pair.metadata:
                    logger.info(f"    难度: {first_pair.metadata.get('difficulty', 'N/A')}")
                    logger.info(f"    标签: {first_pair.metadata.get('tags', [])}")
        
        # 测试解码一张图像
        logger.info("\n" + "="*60)
        logger.info("测试图像解码:")
        logger.info("="*60)
        
        from src.utils import decode_base64_image
        
        first_category = benchmark_data.category_names[0]
        first_pair = benchmark_data.get_category(first_category).data_pairs[0]
        
        logger.info(f"正在解码 {first_category} 类别的第一张图像...")
        
        try:
            image = decode_base64_image(first_pair.original_image_b64)
            logger.info(f"✓ 图像解码成功!")
            logger.info(f"  图像尺寸: {image.size}")
            logger.info(f"  图像模式: {image.mode}")
        except Exception as e:
            logger.error(f"✗ 图像解码失败: {e}")
        
        logger.info("\n" + "="*60)
        logger.info("测试完成！数据加载模块工作正常。")
        logger.info("="*60)
        
        return True
        
    except Exception as e:
        logger.error(f"\n✗ 数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_data_loading()
    sys.exit(0 if success else 1)

