"""
Prompt management utility
Prompt管理工具
"""

from typing import Dict, Any


class PromptManager:
    """
    管理不同类别的评分prompt
    """
    
    def __init__(self, prompts_config: Dict[str, Dict[str, str]]):
        """
        初始化Prompt管理器
        
        Args:
            prompts_config: prompt配置字典，格式为:
                {
                    "category_name": {
                        "system_prompt": "...",
                        "user_prompt_template": "..."
                    }
                }
        """
        self.prompts_config = prompts_config
        self._validate_config()
    
    def _validate_config(self):
        """验证配置格式"""
        for category, config in self.prompts_config.items():
            if "system_prompt" not in config:
                raise ValueError(f"Missing 'system_prompt' for category: {category}")
            if "user_prompt_template" not in config:
                raise ValueError(f"Missing 'user_prompt_template' for category: {category}")
    
    def get_system_prompt(self, category: str) -> str:
        """
        获取指定类别的系统prompt
        
        Args:
            category: 类别名称
            
        Returns:
            系统prompt字符串
        """
        if category not in self.prompts_config:
            raise ValueError(f"Category '{category}' not found in prompts config")
        
        return self.prompts_config[category]["system_prompt"]
    
    def get_user_prompt(self, 
                       category: str,
                       original_description: str,
                       edit_instruction: str,
                       **kwargs) -> str:
        """
        获取指定类别的用户prompt（填充模板）
        
        Args:
            category: 类别名称
            original_description: 原始图像描述
            edit_instruction: 编辑指令
            **kwargs: 其他需要填充到模板的变量
            
        Returns:
            填充好的用户prompt字符串
        """
        if category not in self.prompts_config:
            raise ValueError(f"Category '{category}' not found in prompts config")
        
        template = self.prompts_config[category]["user_prompt_template"]
        
        # 准备替换变量
        format_vars = {
            "original_description": original_description,
            "edit_instruction": edit_instruction,
            **kwargs
        }
        
        try:
            return template.format(**format_vars)
        except KeyError as e:
            raise ValueError(f"Missing variable in template for category '{category}': {e}")
    
    def get_full_prompt(self,
                       category: str,
                       original_description: str,
                       edit_instruction: str,
                       **kwargs) -> Dict[str, str]:
        """
        获取完整的prompt（系统+用户）
        
        Args:
            category: 类别名称
            original_description: 原始图像描述
            edit_instruction: 编辑指令
            **kwargs: 其他需要填充到模板的变量
            
        Returns:
            包含system_prompt和user_prompt的字典
        """
        return {
            "system_prompt": self.get_system_prompt(category),
            "user_prompt": self.get_user_prompt(
                category, 
                original_description, 
                edit_instruction, 
                **kwargs
            )
        }
    
    def list_categories(self) -> list:
        """返回所有可用的类别"""
        return list(self.prompts_config.keys())


