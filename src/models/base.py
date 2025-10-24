"""
Base model class
模型基类
"""

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseModel(ABC):
    """
    所有模型的抽象基类
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化模型
        
        Args:
            config: 模型配置字典
        """
        self.config = config
        self._initialize()
    
    @abstractmethod
    def _initialize(self):
        """初始化模型（由子类实现）"""
        pass
    
    @abstractmethod
    def __call__(self, *args, **kwargs):
        """模型调用接口（由子类实现）"""
        pass
    
    def unload_from_gpu(self):
        """
        将模型从GPU卸载到CPU
        
        子类可以重写此方法以实现特定的卸载逻辑
        默认实现：不做任何操作
        """
        pass
    
    def load_to_gpu(self):
        """
        将模型从CPU加载到GPU
        
        子类可以重写此方法以实现特定的加载逻辑
        默认实现：不做任何操作
        """
        pass

