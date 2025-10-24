"""
Example reward model implementation
示例Reward模型实现（占位符）

这是一个示例实现，用于展示如何继承BaseRewardModel。
实际使用时，请根据你的模型API创建新的实现。
"""

import random
import time
from typing import Any, Dict, Optional
from PIL import Image

from ..base_reward import BaseRewardModel


class ExampleRewardModel(BaseRewardModel):
    """
    示例Reward评分模型
    
    这个类展示了如何实现一个Reward评分模型。
    在实际使用中，你应该：
    1. 在src/models/reward/implementations/目录下创建新的实现文件
    2. 继承BaseRewardModel类
    3. 实现_initialize()和score()方法
    4. 在配置文件中指定你的模型类路径
    """
    
    def _initialize(self):
        """
        初始化模型
        
        在这里加载模型权重、设置设备等
        """
        self.model_name = self.config.get("model_name", "example-reward-model")
        self.device = self.config.get("device", "cuda")
        self.temperature = self.config.get("temperature", 0.7)
        
        print(f"[ExampleRewardModel] Initializing {self.model_name} on {self.device}")
        
        # TODO: 在这里加载你的实际模型
        # 例如:
        # from transformers import AutoModel, AutoTokenizer
        # self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        # self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        print(f"[ExampleRewardModel] Model loaded successfully")
    
    def score(self,
              edited_image: Image.Image,
              original_description: str,
              edit_instruction: str,
              system_prompt: str,
              user_prompt: str,
              original_image: Optional[Image.Image] = None,
              **kwargs) -> float:
        """
        对编辑后的图像进行评分
        
        Args:
            edited_image: 编辑后的PIL图像
            original_description: 原始图像描述
            edit_instruction: 编辑指令
            system_prompt: 系统prompt
            user_prompt: 用户prompt
            original_image: 原始图像（可选）
            **kwargs: 其他参数
            
        Returns:
            评分（0-10的浮点数）
        """
        print(f"[ExampleRewardModel] Scoring edited image")
        print(f"  - System prompt: {system_prompt[:50]}...")
        print(f"  - User prompt: {user_prompt[:50]}...")
        
        # TODO: 实现实际的评分逻辑
        # 例如（Vision-Language Model）:
        # inputs = self.processor(
        #     text=[system_prompt, user_prompt],
        #     images=edited_image,
        #     return_tensors="pt"
        # ).to(self.device)
        # 
        # outputs = self.model.generate(**inputs, max_length=100)
        # response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # 
        # # 从响应中提取分数
        # score = self._extract_score_from_response(response)
        
        # 占位符实现：返回随机分数
        # 实际使用时请删除这部分并实现真实的评分逻辑
        print("[ExampleRewardModel] Warning: Using placeholder implementation, returning random score")
        time.sleep(0.1)  # 模拟处理时间
        score = random.uniform(5.0, 9.0)  # 随机分数 5-9
        
        return score
    
    def _extract_score_from_response(self, response: str) -> float:
        """
        从模型响应中提取分数
        
        这是一个辅助方法，根据你的模型输出格式来解析分数
        """
        # TODO: 根据实际的模型输出格式来实现
        # 例如，如果模型输出类似 "Score: 8.5"，则可以用正则表达式提取
        import re
        match = re.search(r'(\d+\.?\d*)', response)
        if match:
            score = float(match.group(1))
            return min(max(score, 0.0), 10.0)  # 限制在0-10范围内
        return 5.0  # 默认分数
    
    def batch_score(self,
                   edited_images: list,
                   original_descriptions: list,
                   edit_instructions: list,
                   system_prompts: list,
                   user_prompts: list,
                   original_images: Optional[list] = None,
                   **kwargs) -> list:
        """
        批量评分（可选：优化的批处理实现）
        
        如果你的模型支持批处理，可以在这里实现更高效的批处理逻辑
        """
        # 默认使用父类的逐个处理实现
        return super().batch_score(
            edited_images, 
            original_descriptions,
            edit_instructions,
            system_prompts,
            user_prompts,
            original_images,
            **kwargs
        )


