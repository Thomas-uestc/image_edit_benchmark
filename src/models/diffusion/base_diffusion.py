"""
Base diffusion model class
扩散编辑模型抽象基类
"""

from abc import abstractmethod
from typing import Any, Dict
from PIL import Image

from ..base import BaseModel


class BaseDiffusionModel(BaseModel):
    """
    扩散编辑模型抽象基类
    
    所有具体的扩散模型实现都应该继承这个类，并实现edit_image方法
    """
    
    @abstractmethod
    def edit_image(self, 
                   original_image: Image.Image,
                   edit_instruction: str,
                   **kwargs) -> Image.Image:
        """
        执行图像编辑
        
        Args:
            original_image: 原始PIL图像
            edit_instruction: 编辑指令文本
            **kwargs: 其他参数
            
        Returns:
            编辑后的PIL图像
        """
        pass
    
    def __call__(self, 
                 original_image: Image.Image,
                 edit_instruction: str,
                 **kwargs) -> Image.Image:
        """
        调用接口，直接调用edit_image
        
        Args:
            original_image: 原始PIL图像
            edit_instruction: 编辑指令文本
            **kwargs: 其他参数
            
        Returns:
            编辑后的PIL图像
        """
        return self.edit_image(original_image, edit_instruction, **kwargs)
    
    def batch_edit(self,
                   images: list,
                   instructions: list,
                   **kwargs) -> list:
        """
        批量编辑图像（默认实现，可被子类覆盖以优化性能）
        
        Args:
            images: 原始图像列表
            instructions: 编辑指令列表
            **kwargs: 其他参数
            
        Returns:
            编辑后的图像列表
        """
        if len(images) != len(instructions):
            raise ValueError("Number of images must match number of instructions")
        
        edited_images = []
        for img, inst in zip(images, instructions):
            edited_img = self.edit_image(img, inst, **kwargs)
            edited_images.append(edited_img)
        
        return edited_images


