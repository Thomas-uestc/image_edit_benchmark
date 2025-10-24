"""
Unit tests for models
模型测试
"""

import unittest
from PIL import Image
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.diffusion.implementations.example_model import ExampleDiffusionModel
from src.models.reward.implementations.example_reward import ExampleRewardModel


class TestDiffusionModel(unittest.TestCase):
    """测试扩散编辑模型"""
    
    def setUp(self):
        """初始化测试模型"""
        config = {
            "model_name": "test-model",
            "device": "cpu",
            "batch_size": 1
        }
        self.model = ExampleDiffusionModel(config)
    
    def test_edit_image(self):
        """测试图像编辑功能"""
        # 创建测试图像
        test_image = Image.new("RGB", (256, 256), color="white")
        instruction = "Make it blue"
        
        # 执行编辑
        edited_image = self.model.edit_image(test_image, instruction)
        
        # 验证输出
        self.assertIsInstance(edited_image, Image.Image)
        self.assertEqual(edited_image.size, test_image.size)
    
    def test_batch_edit(self):
        """测试批量编辑"""
        images = [Image.new("RGB", (256, 256), color="white") for _ in range(3)]
        instructions = ["Make it red", "Make it blue", "Make it green"]
        
        edited_images = self.model.batch_edit(images, instructions)
        
        self.assertEqual(len(edited_images), 3)
        for img in edited_images:
            self.assertIsInstance(img, Image.Image)


class TestRewardModel(unittest.TestCase):
    """测试Reward评分模型"""
    
    def setUp(self):
        """初始化测试模型"""
        config = {
            "model_name": "test-reward",
            "device": "cpu",
            "temperature": 0.7
        }
        self.model = ExampleRewardModel(config)
    
    def test_score(self):
        """测试评分功能"""
        test_image = Image.new("RGB", (256, 256), color="blue")
        
        score = self.model.score(
            edited_image=test_image,
            original_description="A white image",
            edit_instruction="Make it blue",
            system_prompt="You are an evaluator",
            user_prompt="Rate the quality (0-10)"
        )
        
        # 验证分数在合理范围内
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 10.0)


if __name__ == "__main__":
    unittest.main()


