"""
Main entry point for benchmark evaluation
主入口脚本
"""

import argparse
import yaml
from pathlib import Path
import sys

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.pipeline import BenchmarkPipeline


def load_config(config_path: str) -> dict:
    """
    加载YAML配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置字典
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="Image Edit Benchmark Evaluation Pipeline"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file (default: config.yaml)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint"
    )
    
    args = parser.parse_args()
    
    # 加载配置
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}")
        print("Please copy config_template.yaml to config.yaml and edit it.")
        sys.exit(1)
    
    config = load_config(args.config)
    
    # 如果指定了resume，覆盖配置
    if args.resume:
        config.setdefault("evaluation", {})["resume_from_checkpoint"] = True
    
    # 创建并运行pipeline
    try:
        pipeline = BenchmarkPipeline(config)
        report = pipeline.run()
        
        print("\n" + "="*60)
        print("Evaluation Summary:")
        print("="*60)
        
        summary = report.get("summary", {})
        print(f"Total Samples: {summary.get('total_samples', 0)}")
        print(f"Number of Categories: {summary.get('num_categories', 0)}")
        print(f"Overall Mean Score: {summary.get('overall_mean', 0):.3f}")
        
        print("\nCategory Mean Scores:")
        for cat, score in summary.get("category_means", {}).items():
            print(f"  - {cat}: {score:.3f}")
        
        print("="*60)
        print("\nEvaluation completed successfully!")
        
    except Exception as e:
        print(f"\nError during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()


