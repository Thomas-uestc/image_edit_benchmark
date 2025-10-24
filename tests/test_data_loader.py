"""
Unit tests for data loader
数据加载器测试
"""

import unittest
import json
import tempfile
from pathlib import Path
import sys

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data import BenchmarkLoader, DataPair


class TestBenchmarkLoader(unittest.TestCase):
    """测试BenchmarkLoader"""
    
    def setUp(self):
        """设置测试数据"""
        self.loader = BenchmarkLoader()
        
        # 创建临时测试数据
        self.test_data = {
            "category_1": [
                {
                    "id": "test_001",
                    "original_image_b64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
                    "edit_instruction": "Make it red",
                    "original_description": "A blue dot"
                }
            ],
            "category_2": [
                {
                    "id": "test_002",
                    "original_image_b64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
                    "edit_instruction": "Add a tree",
                    "original_description": "Empty landscape"
                }
            ]
        }
        
        # 创建临时文件
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(self.test_data, self.temp_file)
        self.temp_file.close()
    
    def tearDown(self):
        """清理测试文件"""
        Path(self.temp_file.name).unlink(missing_ok=True)
    
    def test_load_benchmark(self):
        """测试加载benchmark数据"""
        categories = ["category_1", "category_2"]
        
        benchmark_data = self.loader.load(
            data_path=self.temp_file.name,
            categories=categories,
            decode_images=False
        )
        
        self.assertEqual(benchmark_data.total_pairs, 2)
        self.assertEqual(len(benchmark_data.category_names), 2)
        self.assertIn("category_1", benchmark_data.categories)
        self.assertIn("category_2", benchmark_data.categories)
    
    def test_extract_category_data(self):
        """测试提取类别数据"""
        pairs = self.loader._extract_category_data(
            self.test_data,
            "category_1",
            decode_images=False
        )
        
        self.assertEqual(len(pairs), 1)
        self.assertIsInstance(pairs[0], DataPair)
        self.assertEqual(pairs[0].pair_id, "test_001")
        self.assertEqual(pairs[0].edit_instruction, "Make it red")


if __name__ == "__main__":
    unittest.main()


