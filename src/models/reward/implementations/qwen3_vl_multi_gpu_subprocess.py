"""
Qwen3-VL Multi-GPU Subprocess Reward Model
使用数据并行：多个GPU各运行一个独立的评分进程
"""

import json
import tempfile
import subprocess
import time
import base64
from pathlib import Path
from typing import Any, Dict, List, Optional
from PIL import Image
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..base_reward import BaseRewardModel
from ....utils import setup_logger


class Qwen3VLMultiGPUSubprocessRewardModel(BaseRewardModel):
    """
    Multi-GPU版本的Qwen3-VL Subprocess Reward Model
    
    使用数据并行策略：
    - 每个GPU运行一个独立的评分进程
    - 任务按轮询方式分配到各个GPU
    - 所有GPU并行处理不同的图像
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.logger = setup_logger(self.__class__.__name__)
        
        # 初始化基本配置
        self.model_name = config.get("model_name", "Qwen/Qwen3-VL-30B-Instruct")
        self.device = config.get("device", "auto")
        self.dtype = config.get("dtype", "bfloat16")
        self.max_new_tokens = config.get("max_new_tokens", 128)
        self.use_batch_inference = config.get("use_batch_inference", True)
        self.batch_size = config.get("batch_size", 4)
        self.script_path = config.get("script_path", None)
        self.conda_env = config.get("conda_env", None)
        self.python_path = config.get("python_path", None)
        self.timeout = config.get("timeout", 600)
        
        # 多GPU配置
        device_ids = config.get("device_ids", None)
        if device_ids is None:
            # 默认使用所有可用GPU
            import torch
            if torch.cuda.is_available():
                self.device_ids = list(range(torch.cuda.device_count()))
            else:
                self.device_ids = [0]
        else:
            self.device_ids = device_ids
        
        self.num_gpus = len(self.device_ids)
        
        super().__init__(config)
    
    def _initialize(self):
        """初始化模型（实现BaseModel的抽象方法）"""
        # 检测脚本路径
        if self.script_path is None:
            current_dir = Path(__file__).parent.parent
            self.script_path = current_dir / "qwen3_vl_standalone.py"
        else:
            self.script_path = Path(self.script_path)
        
        if not self.script_path.exists():
            raise FileNotFoundError(f"Standalone script not found: {self.script_path}")
        
        self.logger.info(f"Initialized Qwen3-VL Multi-GPU Subprocess Reward Model")
        self.logger.info(f"  Model: {self.model_name}")
        self.logger.info(f"  GPUs: {self.device_ids} ({self.num_gpus} GPUs)")
        self.logger.info(f"  Script: {self.script_path}")
        if self.conda_env:
            self.logger.info(f"  Conda Env: {self.conda_env}")
        elif self.python_path:
            self.logger.info(f"  Python: {self.python_path}")
    
    def _encode_image(self, image: Image.Image) -> str:
        """将PIL图像编码为base64字符串"""
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    def _call_subprocess_single_gpu(self, tasks: List[Dict], gpu_id: int) -> List[float]:
        """
        在指定GPU上调用子进程进行评分
        
        Args:
            tasks: 评分任务列表
            gpu_id: 使用的GPU ID
            
        Returns:
            分数列表
        """
        if not tasks:
            return []
        
        # 创建临时文件
        input_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        output_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        
        try:
            # 写入输入数据
            input_data = {'tasks': tasks}
            json.dump(input_data, input_file)
            input_file.close()
            output_file.close()
            
            # 构建命令
            if self.conda_env:
                cmd = [
                    'conda', 'run', '-n', self.conda_env, '--no-capture-output',
                    'python', str(self.script_path)
                ]
            elif self.python_path:
                cmd = [self.python_path, str(self.script_path)]
            else:
                cmd = ['python', str(self.script_path)]
            
            # 添加参数（指定GPU）
            cmd.extend([
                '--input', input_file.name,
                '--output', output_file.name,
                '--model-name', self.model_name,
                '--device', f'cuda:{gpu_id}',  # 指定GPU
                '--dtype', self.dtype,
                '--batch-size', str(self.batch_size),
                '--max-new-tokens', str(self.max_new_tokens),
            ])
            
            if self.use_batch_inference:
                cmd.append('--use-batch-inference')
            
            # 执行子进程（使用Popen实时捕获输出）
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
                    # 添加GPU标识
                    print(f"[GPU {gpu_id}] {stderr_line.rstrip()}")
                    stderr_output.append(stderr_line)
                elif process.poll() is not None:
                    break
            
            # 获取剩余输出
            remaining_stderr = process.stderr.read()
            if remaining_stderr:
                for line in remaining_stderr.split('\n'):
                    if line.strip():
                        print(f"[GPU {gpu_id}] {line}")
                stderr_output.append(remaining_stderr)
            
            # 等待进程完成
            return_code = process.wait(timeout=self.timeout)
            
            # 检查返回码
            if return_code != 0:
                stderr_text = ''.join(stderr_output)
                raise RuntimeError(f"GPU {gpu_id} subprocess failed: {stderr_text}")
            
            # 读取输出
            with open(output_file.name, 'r') as f:
                output_data = json.load(f)
            
            if output_data.get('status') != 'success':
                raise RuntimeError(f"GPU {gpu_id} error: {output_data.get('error', 'Unknown')}")
            
            return output_data['scores']
        
        finally:
            # 清理临时文件
            Path(input_file.name).unlink(missing_ok=True)
            Path(output_file.name).unlink(missing_ok=True)
    
    def score(self, *args, **kwargs) -> float:
        """评分单张图像（不建议使用，建议使用batch_score）"""
        scores = self.batch_score([args], [kwargs])
        return scores[0] if scores else 0.0
    
    def batch_score(self,
                   edited_images: List[Image.Image],
                   original_descriptions: List[str],
                   edit_instructions: List[str],
                   system_prompts: List[str],
                   user_prompts: List[str],
                   original_images: Optional[List[Image.Image]] = None,
                   **kwargs) -> List[float]:
        """
        批量评分（多GPU并行）
        
        Args:
            edited_images: 编辑后的图像列表
            original_descriptions: 原始图像描述列表
            edit_instructions: 编辑指令列表
            system_prompts: 系统提示列表
            user_prompts: 用户提示列表
            original_images: 原始图像列表（可选）
            
        Returns:
            分数列表
        """
        n = len(edited_images)
        self.logger.info(f"Multi-GPU batch scoring {n} images across {self.num_gpus} GPUs...")
        
        # 编码所有图像为base64
        self.logger.info(f"Encoding images to base64...")
        start_time = time.time()
        image_b64s = [self._encode_image(img) for img in edited_images]
        encode_time = time.time() - start_time
        self.logger.info(f"Encoding completed in {encode_time:.2f}s")
        
        # 准备所有任务
        all_tasks = []
        for i in range(n):
            task = {
                'image_b64': image_b64s[i],
                'system_prompt': system_prompts[i],
                'user_prompt': user_prompts[i],
            }
            all_tasks.append(task)
        
        # 按GPU分配任务
        gpu_tasks = [[] for _ in range(self.num_gpus)]
        for i, task in enumerate(all_tasks):
            gpu_idx = i % self.num_gpus
            gpu_tasks[gpu_idx].append((i, task))  # 保存原始索引
        
        # 显示任务分配
        self.logger.info(f"Task allocation across {self.num_gpus} GPUs:")
        for gpu_idx, gpu_id in enumerate(self.device_ids):
            num_tasks = len(gpu_tasks[gpu_idx])
            self.logger.info(f"  GPU {gpu_id}: {num_tasks} tasks")
        
        # 并行执行
        scores = [0.0] * n  # 预分配结果列表
        
        with ThreadPoolExecutor(max_workers=self.num_gpus) as executor:
            futures = []
            for gpu_idx, gpu_id in enumerate(self.device_ids):
                if gpu_tasks[gpu_idx]:
                    # 提取任务（不包含索引）
                    tasks_for_gpu = [task for _, task in gpu_tasks[gpu_idx]]
                    indices_for_gpu = [idx for idx, _ in gpu_tasks[gpu_idx]]
                    
                    future = executor.submit(
                        self._call_subprocess_single_gpu,
                        tasks_for_gpu,
                        gpu_id
                    )
                    futures.append((future, indices_for_gpu))
            
            # 收集结果
            for future, indices in futures:
                try:
                    gpu_scores = future.result()
                    # 将结果放回正确的位置
                    for idx, score in zip(indices, gpu_scores):
                        scores[idx] = score
                except Exception as e:
                    self.logger.error(f"Error in GPU worker: {e}")
                    # 对失败的任务使用默认分数
                    for idx in indices:
                        scores[idx] = 5.0
        
        self.logger.info(f"Multi-GPU scoring completed!")
        return scores
    
    def load_to_gpu(self):
        """加载到GPU（多进程模式下不需要）"""
        self.logger.info("Multi-GPU subprocess mode: models are loaded on-demand")
    
    def unload_from_gpu(self):
        """从GPU卸载（多进程模式下不需要）"""
        self.logger.info("Multi-GPU subprocess mode: models are automatically unloaded")

