"""
Example: Run benchmark evaluation
运行评测示例
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yaml
from src.pipeline import BenchmarkPipeline


def main():
    """运行评测示例"""
    
    # 配置示例
    config = {
        "benchmark": {
            "data_path": "path/to/your/benchmark.json",
            "categories": ["category_1", "category_2", "category_3", "category_4", "category_5"]
        },
        "diffusion_model": {
            "class_path": "src.models.diffusion.implementations.example_model.ExampleDiffusionModel",
            "params": {
                "model_name": "your-model-name",
                "device": "cuda"
            }
        },
        "reward_model": {
            "class_path": "src.models.reward.implementations.example_reward.ExampleRewardModel",
            "params": {
                "model_name": "your-reward-model",
                "device": "cuda"
            }
        },
        "prompts": {
            "category_1": {
                "system_prompt": "You are an expert image quality evaluator.",
                "user_prompt_template": "Original: {original_description}\nInstruction: {edit_instruction}\nRate the edited image (0-10):"
            },
            # ... 其他类别的prompt
        },
        "evaluation": {
            "save_generated_images": True,
            "output_dir": "outputs",
            "results_dir": "outputs/results",
            "images_dir": "outputs/images",
            "logs_dir": "outputs/logs",
            "metrics": ["mean", "std", "median", "min", "max"]
        },
        "logging": {
            "level": "INFO",
            "console_output": True,
            "file_output": True,
            "log_file": "outputs/logs/evaluation.log"
        }
    }
    
    # 创建并运行pipeline
    pipeline = BenchmarkPipeline(config)
    report = pipeline.run()
    
    print("\nEvaluation completed!")
    print(f"Overall mean score: {report['overall_statistics']['mean']:.3f}")


if __name__ == "__main__":
    main()


