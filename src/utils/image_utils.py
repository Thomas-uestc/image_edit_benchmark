"""
Image utility functions
图像处理工具函数
"""

import base64
import io
from typing import Union
from PIL import Image
import numpy as np


def decode_base64_image(base64_string: str) -> Image.Image:
    """
    将base64编码的字符串解码为PIL图像
    
    Args:
        base64_string: base64编码的图像字符串
        
    Returns:
        PIL.Image对象
    """
    try:
        # 移除可能的data URL前缀
        if ',' in base64_string:
            base64_string = base64_string.split(',', 1)[1]
        
        # 解码base64
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data))
        
        # 转换为RGB模式（如果需要）
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        return image
    except Exception as e:
        raise ValueError(f"Failed to decode base64 image: {str(e)}")


def encode_image_to_base64(image: Union[Image.Image, np.ndarray], 
                           format: str = "PNG") -> str:
    """
    将PIL图像或numpy数组编码为base64字符串
    
    Args:
        image: PIL图像或numpy数组
        format: 图像格式（PNG, JPEG等）
        
    Returns:
        base64编码的字符串
    """
    try:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        buffered = io.BytesIO()
        image.save(buffered, format=format)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return img_str
    except Exception as e:
        raise ValueError(f"Failed to encode image to base64: {str(e)}")


def save_image(image: Union[Image.Image, np.ndarray], 
               save_path: str,
               format: str = None) -> None:
    """
    保存图像到文件
    
    Args:
        image: PIL图像或numpy数组
        save_path: 保存路径
        format: 图像格式（如果为None，从文件扩展名推断）
    """
    try:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        if format:
            image.save(save_path, format=format)
        else:
            image.save(save_path)
            
    except Exception as e:
        raise ValueError(f"Failed to save image to {save_path}: {str(e)}")


def resize_image(image: Image.Image, 
                max_size: int = 1024, 
                keep_aspect_ratio: bool = True) -> Image.Image:
    """
    调整图像大小
    
    Args:
        image: PIL图像
        max_size: 最大尺寸
        keep_aspect_ratio: 是否保持宽高比
        
    Returns:
        调整大小后的PIL图像
    """
    if keep_aspect_ratio:
        image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    else:
        image = image.resize((max_size, max_size), Image.Resampling.LANCZOS)
    
    return image


