"""
Qwen-Image-Edit diffusion model implementation
Qwen图像编辑模型实现

基于官方 Qwen-Image-Edit 模型
"""

import torch
from PIL import Image
from typing import Any, Dict

from ..base_diffusion import BaseDiffusionModel


class QwenImageEditModel(BaseDiffusionModel):
    """
    Qwen-Image-Edit 扩散编辑模型实现
    
    官方仓库: https://huggingface.co/Qwen/Qwen-Image-Edit
    """
    
    def _initialize(self):
        """
        初始化Qwen-Image-Edit模型
        """
        from diffusers import QwenImageEditPipeline
        
        # 从配置获取参数
        self.model_name = self.config.get("model_name", "Qwen/Qwen-Image-Edit")
        self.device = self.config.get("device", "cuda")
        self.dtype = self.config.get("dtype", "bfloat16")  # Qwen推荐使用bfloat16
        self.num_inference_steps = self.config.get("num_inference_steps", 50)
        self.true_cfg_scale = self.config.get("true_cfg_scale", 4.0)
        self.negative_prompt = self.config.get("negative_prompt", " ")
        self.seed = self.config.get("seed", 0)
        
        print(f"[QwenImageEditModel] 正在加载模型: {self.model_name}")
        print(f"[QwenImageEditModel] 设备: {self.device}, 数据类型: {self.dtype}")
        
        # 加载pipeline
        self.pipeline = QwenImageEditPipeline.from_pretrained(self.model_name)
        print("[QwenImageEditModel] Pipeline加载完成")
        
        # 设置数据类型
        if self.dtype == "bfloat16":
            self.pipeline.to(torch.bfloat16)
        elif self.dtype == "float16":
            self.pipeline.to(torch.float16)
        elif self.dtype == "float32":
            self.pipeline.to(torch.float32)
        
        # 移动到设备
        self.pipeline.to(self.device)
        
        # 禁用进度条（可选）
        if self.config.get("disable_progress_bar", True):
            self.pipeline.set_progress_bar_config(disable=None)
        
        print(f"[QwenImageEditModel] 模型初始化完成")
    
    def edit_image(self, 
                   original_image: Image.Image,
                   edit_instruction: str,
                   **kwargs) -> Image.Image:
        """
        使用Qwen-Image-Edit编辑图像
        
        Args:
            original_image: 原始PIL图像
            edit_instruction: 编辑指令文本
            **kwargs: 其他参数，可以覆盖默认配置
                - num_inference_steps: 推理步数
                - true_cfg_scale: CFG scale
                - negative_prompt: 负面提示词
                - seed: 随机种子
            
        Returns:
            编辑后的PIL图像
        """
        # 确保图像是RGB格式
        if original_image.mode != 'RGB':
            original_image = original_image.convert('RGB')
        
        # 获取参数（优先使用kwargs，否则使用配置）
        num_steps = kwargs.get("num_inference_steps", self.num_inference_steps)
        cfg_scale = kwargs.get("true_cfg_scale", self.true_cfg_scale)
        neg_prompt = kwargs.get("negative_prompt", self.negative_prompt)
        seed = kwargs.get("seed", self.seed)
        
        # 准备输入
        inputs = {
            "image": original_image,
            "prompt": edit_instruction,
            "generator": torch.manual_seed(seed),
            "true_cfg_scale": cfg_scale,
            "negative_prompt": neg_prompt,
            "num_inference_steps": num_steps,
        }
        
        # 执行编辑
        with torch.inference_mode():
            output = self.pipeline(**inputs)
            edited_image = output.images[0]
        
        return edited_image
    
    def batch_edit(self,
                   images: list,
                   instructions: list,
                   **kwargs) -> list:
        """
        批量编辑图像
        
        注意：Qwen-Image-Edit默认不支持批处理，这里逐个处理
        如果需要更高效的批处理，可以考虑修改pipeline
        
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
        
        # 逐个处理（保持不同的seed以增加多样性）
        base_seed = kwargs.get("seed", self.seed)
        
        for idx, (img, inst) in enumerate(zip(images, instructions)):
            # 为每张图像使用不同的seed
            current_kwargs = kwargs.copy()
            current_kwargs["seed"] = base_seed + idx
            
            edited_img = self.edit_image(img, inst, **current_kwargs)
            edited_images.append(edited_img)
        
        return edited_images
    
    def unload_from_gpu(self):
        """
        将模型从GPU卸载到CPU，释放GPU内存
        """
        if hasattr(self, 'pipeline') and self.pipeline is not None:
            print(f"[QwenImageEditModel] 将模型从GPU卸载到CPU...")
            self.pipeline.to('cpu')
            # 清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"[QwenImageEditModel] 模型已卸载到CPU")
    
    def load_to_gpu(self):
        """
        将模型从CPU加载到GPU
        """
        if hasattr(self, 'pipeline') and self.pipeline is not None:
            print(f"[QwenImageEditModel] 将模型从CPU加载到GPU...")
            self.pipeline.to(self.device)
            print(f"[QwenImageEditModel] 模型已加载到GPU: {self.device}")
    
    def __del__(self):
        """
        清理资源
        """
        if hasattr(self, 'pipeline'):
            # 清理GPU内存
            del self.pipeline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

