"""
Unit tests for pipeline
Pipeline测试
"""

import unittest
import tempfile
import json
from pathlib import Path
import sys

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.pipeline import BenchmarkPipeline


class TestBenchmarkPipeline(unittest.TestCase):
    """测试BenchmarkPipeline"""
    
    def setUp(self):
        """设置测试环境"""
        # 创建临时目录
        self.temp_dir = tempfile.mkdtemp()
        
        # 创建测试数据文件
        test_data = {
            "test_category": [
                {
                    "id": "test_001",
                    "original_image_b64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
                    "edit_instruction": "Test instruction",
                    "original_description": "Test description"
                }
            ]
        }
        
        self.test_data_file = Path(self.temp_dir) / "test_data.json"
        with open(self.test_data_file, 'w') as f:
            json.dump(test_data, f)
        
        # 创建测试配置
        self.config = {
            "benchmark": {
                "data_path": str(self.test_data_file),
                "categories": ["test_category"]
            },
            "diffusion_model": {
                "class_path": "src.models.diffusion.implementations.example_model.ExampleDiffusionModel",
                "params": {"device": "cpu"}
            },
            "reward_model": {
                "class_path": "src.models.reward.implementations.example_reward.ExampleRewardModel",
                "params": {"device": "cpu"}
            },
            "prompts": {
                "test_category": {
                    "system_prompt": "Test system prompt",
                    "user_prompt_template": "Test: {original_description} | {edit_instruction}"
                }
            },
            "evaluation": {
                "save_generated_images": False,
                "output_dir": str(Path(self.temp_dir) / "outputs"),
                "results_dir": str(Path(self.temp_dir) / "outputs/results"),
                "images_dir": str(Path(self.temp_dir) / "outputs/images"),
                "logs_dir": str(Path(self.temp_dir) / "outputs/logs"),
                "metrics": ["mean", "std"]
            },
            "logging": {
                "level": "WARNING",
                "console_output": False,
                "file_output": False
            }
        }
    
    def tearDown(self):
        """清理测试文件"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_pipeline_initialization(self):
        """测试Pipeline初始化"""
        pipeline = BenchmarkPipeline(self.config)
        
        self.assertIsNotNone(pipeline.diffusion_model)
        self.assertIsNotNone(pipeline.reward_model)
        self.assertIsNotNone(pipeline.prompt_manager)
    
    def test_pipeline_run(self):
        """测试Pipeline运行（简单测试）"""
        # 注意：这个测试会实际运行整个pipeline，但使用示例模型
        pipeline = BenchmarkPipeline(self.config)
        
        try:
            report = pipeline.run()
            
            # 验证报告结构
            self.assertIn("category_statistics", report)
            self.assertIn("overall_statistics", report)
            self.assertIn("summary", report)
            
        except Exception as e:
            self.fail(f"Pipeline run failed: {e}")


if __name__ == "__main__":
    unittest.main()


