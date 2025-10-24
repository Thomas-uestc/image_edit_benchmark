"""
Qwen3-VL Reward Model - Subprocess Implementation
通过子进程调用独立环境中的Qwen3-VL模型
"""

import json
import base64
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from io import BytesIO
from PIL import Image

from ..base_reward import BaseRewardModel
from ....utils import setup_logger


class Qwen3VLSubprocessRewardModel(BaseRewardModel):
    """
    Qwen3-VL奖励模型（子进程实现）
    
    通过子进程调用独立虚拟环境中的Qwen3-VL模型
    适用于Qwen3-VL与主环境依赖冲突的情况
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.logger = setup_logger(self.__class__.__name__)
        
        # 配置参数
        self.model_name = config.get("model_name", "Qwen/Qwen3-VL-30B-Instruct")
        self.device = config.get("device", "auto")
        self.dtype = config.get("dtype", "bfloat16")
        self.max_new_tokens = config.get("max_new_tokens", 128)
        self.batch_size = config.get("batch_size", 4)
        self.use_batch_inference = config.get("use_batch_inference", True)
        
        # 子进程相关配置
        self.python_path = config.get("python_path", None)  # 新环境的python路径
        self.conda_env = config.get("conda_env", None)      # 或者conda环境名
        self.script_path = config.get("script_path", None)  # standalone脚本路径
        
        # 调用父类初始化（会调用_initialize）
        super().__init__(config)
    
    def _initialize(self):
        """初始化模型（实现BaseModel的抽象方法）"""
        # 自动检测脚本路径
        if self.script_path is None:
            current_dir = Path(__file__).parent.parent
            self.script_path = current_dir / "qwen3_vl_standalone.py"
        else:
            self.script_path = Path(self.script_path)
        
        if not self.script_path.exists():
            raise FileNotFoundError(f"Standalone script not found: {self.script_path}")
        
        self.logger.info(f"Initialized Qwen3-VL Subprocess Reward Model")
        self.logger.info(f"  Model: {self.model_name}")
        self.logger.info(f"  Script: {self.script_path}")
        if self.conda_env:
            self.logger.info(f"  Conda Env: {self.conda_env}")
        elif self.python_path:
            self.logger.info(f"  Python: {self.python_path}")
    
    def _get_python_command(self) -> List[str]:
        """获取Python命令"""
        if self.conda_env:
            # 使用conda环境
            return ["conda", "run", "-n", self.conda_env, "python"]
        elif self.python_path:
            # 使用指定的python路径
            return [self.python_path]
        else:
            # 使用当前环境（不推荐，因为可能有依赖冲突）
            self.logger.warning("No separate environment specified, using current Python")
            return ["python"]
    
    def _encode_image_to_base64(self, image: Image.Image) -> str:
        """将PIL图像编码为base64"""
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return img_str
    
    def _call_subprocess(self, input_data: Dict, timeout: int = 600) -> Dict:
        """
        调用子进程执行评分
        
        Args:
            input_data: 输入数据（包含tasks列表）
            timeout: 超时时间（秒）
            
        Returns:
            输出数据（包含scores列表）
        """
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f_in:
            input_file = f_in.name
            json.dump(input_data, f_in, ensure_ascii=False)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f_out:
            output_file = f_out.name
        
        try:
            # 构建命令
            python_cmd = self._get_python_command()
            cmd = python_cmd + [
                str(self.script_path),
                '--input', input_file,
                '--output', output_file,
                '--model-name', self.model_name,
                '--device', self.device,
                '--dtype', self.dtype,
                '--batch-size', str(self.batch_size),
                '--max-new-tokens', str(self.max_new_tokens),
            ]
            
            if self.use_batch_inference:
                cmd.append('--use-batch-inference')
            
            self.logger.info(f"Calling subprocess: {' '.join(cmd[:5])}...")
            
            # 执行子进程（使用Popen实时捕获输出）
            start_time = time.time()
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # 实时打印stderr（包含评分进度）
            stderr_output = []
            while True:
                stderr_line = process.stderr.readline()
                if stderr_line:
                    # 打印评分进度
                    print(stderr_line.rstrip())
                    stderr_output.append(stderr_line)
                elif process.poll() is not None:
                    break
            
            # 获取剩余输出
            remaining_stderr = process.stderr.read()
            if remaining_stderr:
                print(remaining_stderr.rstrip())
                stderr_output.append(remaining_stderr)
            
            # 等待进程完成
            return_code = process.wait(timeout=timeout)
            elapsed = time.time() - start_time
            
            # 检查返回码
            if return_code != 0:
                stderr_text = ''.join(stderr_output)
                self.logger.error(f"Subprocess failed with return code {return_code}")
                self.logger.error(f"stderr: {stderr_text}")
                raise RuntimeError(f"Subprocess failed: {stderr_text}")
            
            # 读取输出
            with open(output_file, 'r') as f:
                output_data = json.load(f)
            
            if output_data.get('status') != 'success':
                raise RuntimeError(f"Subprocess error: {output_data.get('error', 'Unknown')}")
            
            self.logger.info(f"Subprocess completed in {elapsed:.2f}s")
            return output_data
        
        finally:
            # 清理临时文件
            Path(input_file).unlink(missing_ok=True)
            Path(output_file).unlink(missing_ok=True)
    
    def score(self,
             edited_image: Image.Image,
             original_description: str,
             edit_instruction: str,
             system_prompt: str,
             user_prompt: str,
             original_image: Optional[Image.Image] = None,
             **kwargs) -> float:
        """
        评分单张图像
        
        Args:
            edited_image: 编辑后的图像
            original_description: 原始图像描述
            edit_instruction: 编辑指令
            system_prompt: 系统提示
            user_prompt: 用户提示
            original_image: 原始图像（可选，当前未使用）
            
        Returns:
            评分
        """
        # 编码图像
        image_b64 = self._encode_image_to_base64(edited_image)
        
        # 构建输入数据
        input_data = {
            'tasks': [{
                'image_b64': image_b64,
                'system_prompt': system_prompt,
                'user_prompt': user_prompt
            }]
        }
        
        # 调用子进程
        output_data = self._call_subprocess(input_data)
        
        # 返回分数
        scores = output_data.get('scores', [5.0])
        return scores[0] if scores else 5.0
    
    def batch_score(self,
                   edited_images: list,
                   original_descriptions: list,
                   edit_instructions: list,
                   system_prompts: list,
                   user_prompts: list,
                   original_images: Optional[list] = None,
                   **kwargs) -> list:
        """
        批量评分
        
        Args:
            edited_images: 编辑后的图像列表
            original_descriptions: 原始图像描述列表
            edit_instructions: 编辑指令列表
            system_prompts: 系统prompt列表
            user_prompts: 用户prompt列表
            original_images: 原始图像列表（可选）
            
        Returns:
            评分列表
        """
        n = len(edited_images)
        if not all(len(lst) == n for lst in [original_descriptions, edit_instructions, 
                                              system_prompts, user_prompts]):
            raise ValueError("All input lists must have the same length")
        
        self.logger.info(f"Batch scoring {n} images via subprocess...")
        
        # 编码所有图像
        self.logger.info("Encoding images to base64...")
        images_b64 = [self._encode_image_to_base64(img) for img in edited_images]
        
        # 构建输入数据
        tasks = []
        for i in range(n):
            task = {
                'image_b64': images_b64[i],
                'system_prompt': system_prompts[i],
                'user_prompt': user_prompts[i]
            }
            tasks.append(task)
        
        input_data = {'tasks': tasks}
        
        # 调用子进程
        output_data = self._call_subprocess(input_data, timeout=1800)  # 30分钟超时
        
        # 返回分数
        scores = output_data.get('scores', [5.0] * n)
        if len(scores) != n:
            self.logger.warning(f"Expected {n} scores, got {len(scores)}, padding with 5.0")
            scores = scores + [5.0] * (n - len(scores))
        
        return scores[:n]
    
    def unload_from_gpu(self):
        """卸载模型（子进程模式下无需操作）"""
        self.logger.info("[Qwen3VLSubprocess] No need to unload (subprocess mode)")
    
    def load_to_gpu(self):
        """加载模型到GPU（子进程模式下无需操作）"""
        self.logger.info("[Qwen3VLSubprocess] No need to load (subprocess mode)")

