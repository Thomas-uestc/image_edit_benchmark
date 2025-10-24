#!/usr/bin/env python3
"""
Qwen3-VL Standalone评分脚本
用于在独立虚拟环境中运行Qwen3-VL模型

使用方法：
    python qwen3_vl_standalone.py --input input.json --output output.json
"""

import argparse
import json
import base64
import sys
from pathlib import Path
from io import BytesIO
from typing import List, Dict
import re

# 在新环境中导入Qwen3-VL
try:
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from PIL import Image
    import torch
except ImportError as e:
    print(f"Error: Failed to import required packages: {e}", file=sys.stderr, flush=True)
    print("Please install: pip install transformers pillow torch", file=sys.stderr, flush=True)
    sys.exit(1)


class Qwen3VLStandaloneScorer:
    """独立的Qwen3-VL评分器"""
    
    def __init__(self, model_name: str, device: str = "auto", dtype: str = "bfloat16"):
        """
        初始化模型
        
        Args:
            model_name: 模型名称或路径
            device: 设备（auto, cuda, cpu）
            dtype: 数据类型
        """
        print(f"[Qwen3VL-Standalone] Loading model: {model_name}", file=sys.stderr, flush=True)
        
        # 解析dtype
        if dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        elif dtype == "float16":
            torch_dtype = torch.float16
        elif dtype == "float32":
            torch_dtype = torch.float32
        else:
            torch_dtype = "auto"
        
        # 加载模型
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device
        )
        
        # 加载processor
        self.processor = AutoProcessor.from_pretrained(model_name)
        
        self.device = next(self.model.parameters()).device
        print(f"[Qwen3VL-Standalone] Model loaded on device: {self.device}", file=sys.stderr, flush=True)
    
    def decode_base64_image(self, base64_str: str) -> Image.Image:
        """解码base64图像"""
        image_data = base64.b64decode(base64_str)
        image = Image.open(BytesIO(image_data))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return image
    
    def extract_score(self, response: str) -> float:
        """从响应中提取分数"""
        # 清理响应（移除多余空白）
        response = response.strip()
        
        # 尝试多种模式
        patterns = [
            # 标准格式：Score: 8.500
            r'Score:\s*(\d+\.?\d*)',
            # 纯数字格式（模型可能只输出数字）
            r'^\s*(\d+\.\d+)\s*$',  # 精确匹配：开头^空白*数字.数字空白*结尾$
            r'^\s*(\d+)\s*$',        # 精确匹配：整数
            # 中文格式
            r'评分[:：]\s*(\d+\.?\d*)',
            r'分数[:：]\s*(\d+\.?\d*)',
            r'得分[:：]\s*(\d+\.?\d*)',
            r'(?:总分|综合评分)[:：]\s*(\d+\.?\d*)',
            # 宽松匹配：任何位置的数字
            r'(\d+\.\d+)',
            r'(\d+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                try:
                    score = float(match.group(1))
                    if 0 <= score <= 10:
                        return score
                except (ValueError, IndexError):
                    continue
        
        # 如果找不到，返回默认分数
        print(f"[Warning] Could not extract score from: '{response[:100]}'", file=sys.stderr, flush=True)
        return 5.0
    
    def score_single(self, image_b64: str, system_prompt: str, 
                    user_prompt: str, max_new_tokens: int = 128) -> float:
        """
        评分单张图像
        
        Args:
            image_b64: Base64编码的图像
            system_prompt: 系统提示
            user_prompt: 用户提示
            max_new_tokens: 最大生成token数
            
        Returns:
            评分
        """
        # 解码图像
        image = self.decode_base64_image(image_b64)
        
        # 构建messages
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": user_prompt}
                ]
            }
        ]
        
        # 准备输入
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.model.device)
        
        # 生成
        with torch.inference_mode():
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] 
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
        
        # 提取分数
        score = self.extract_score(output_text)
        return score
    
    def score_batch(self, tasks: List[Dict], batch_size: int = 4, 
                   max_new_tokens: int = 128, use_batch_inference: bool = True) -> List[float]:
        """
        批量评分
        
        Args:
            tasks: 任务列表，每个任务包含 image_b64, system_prompt, user_prompt
            batch_size: 批处理大小
            max_new_tokens: 最大生成token数
            use_batch_inference: 是否使用batch inference
            
        Returns:
            评分列表
        """
        n = len(tasks)
        
        if not use_batch_inference or batch_size == 1:
            # 串行处理
            scores = []
            for i, task in enumerate(tasks):
                score = self.score_single(
                    task['image_b64'],
                    task['system_prompt'],
                    task['user_prompt'],
                    max_new_tokens
                )
                scores.append(score)
                print(f"[Progress] {i+1}/{n} scored", file=sys.stderr, flush=True)
            return scores
        
        # Batch inference
        print(f"[Qwen3VL-Standalone] Batch scoring {n} images with batch_size={batch_size}", 
              file=sys.stderr, flush=True)
        
        # 设置padding_side
        original_padding_side = self.processor.tokenizer.padding_side
        self.processor.tokenizer.padding_side = 'left'
        
        all_scores = []
        
        # 打印评分开始信息
        print(f"\n{'='*70}", file=sys.stderr, flush=True)
        print(f"[Qwen3-VL Scoring] Starting batch scoring for {n} images", file=sys.stderr, flush=True)
        print(f"  Batch size: {batch_size}", file=sys.stderr, flush=True)
        print(f"  Total batches: {(n + batch_size - 1) // batch_size}", file=sys.stderr, flush=True)
        print(f"{'='*70}\n", file=sys.stderr, flush=True)
        
        try:
            # 分批处理
            for batch_start in range(0, n, batch_size):
                batch_end = min(batch_start + batch_size, n)
                batch_tasks = tasks[batch_start:batch_end]
                
                # 解码图像
                images = [self.decode_base64_image(t['image_b64']) for t in batch_tasks]
                
                # 构建batch messages
                batch_messages = []
                for task, image in zip(batch_tasks, images):
                    messages = [
                        {
                            "role": "system",
                            "content": [{"type": "text", "text": task['system_prompt']}]
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": image},
                                {"type": "text", "text": task['user_prompt']}
                            ]
                        }
                    ]
                    batch_messages.append(messages)
                
                # 准备输入
                inputs = self.processor.apply_chat_template(
                    batch_messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                    padding=True
                )
                inputs = inputs.to(self.model.device)
                
                # 生成
                with torch.inference_mode():
                    generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
                    generated_ids_trimmed = [
                        out_ids[len(in_ids):] 
                        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]
                    output_texts = self.processor.batch_decode(
                        generated_ids_trimmed,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False
                    )
                
                # 提取分数并打印详细信息
                batch_scores = []
                for i, (text, task) in enumerate(zip(output_texts, batch_tasks)):
                    score = self.extract_score(text)
                    batch_scores.append(score)
                    
                    # 打印每个样本的详细信息
                    global_idx = batch_start + i
                    print(f"  [Sample {global_idx:3d}] Score: {score:.2f} | Response: {text[:80]}...", 
                          file=sys.stderr, flush=True)
                
                all_scores.extend(batch_scores)
                
                # 打印批次统计
                avg_score = sum(batch_scores) / len(batch_scores)
                print(f"[Batch {batch_start//batch_size + 1}] Images {batch_start}-{batch_end-1} done, "
                      f"avg_score={avg_score:.3f}", 
                      file=sys.stderr, flush=True)
        
        finally:
            # 恢复padding_side
            self.processor.tokenizer.padding_side = original_padding_side
        
        # 打印评分总结
        if all_scores:
            print(f"\n{'='*70}", file=sys.stderr, flush=True)
            print(f"[Qwen3-VL Scoring] Completed!", file=sys.stderr, flush=True)
            print(f"  Total images: {len(all_scores)}", file=sys.stderr, flush=True)
            print(f"  Average score: {sum(all_scores)/len(all_scores):.3f}", file=sys.stderr, flush=True)
            print(f"  Min score: {min(all_scores):.3f}", file=sys.stderr, flush=True)
            print(f"  Max score: {max(all_scores):.3f}", file=sys.stderr, flush=True)
            print(f"{'='*70}\n", file=sys.stderr, flush=True)
        
        return all_scores


def main():
    parser = argparse.ArgumentParser(description="Qwen3-VL Standalone Scorer")
    parser.add_argument('--input', required=True, help='Input JSON file')
    parser.add_argument('--output', required=True, help='Output JSON file')
    parser.add_argument('--model-name', default='Qwen/Qwen3-VL-30B-Instruct', 
                       help='Model name or path')
    parser.add_argument('--device', default='auto', help='Device: auto, cuda, cpu')
    parser.add_argument('--dtype', default='bfloat16', 
                       help='Data type: bfloat16, float16, float32, auto')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--max-new-tokens', type=int, default=128, 
                       help='Max new tokens')
    parser.add_argument('--use-batch-inference', action='store_true', default=True,
                       help='Use batch inference')
    
    args = parser.parse_args()
    
    try:
        # 读取输入
        print(f"[Qwen3VL-Standalone] Reading input from: {args.input}", file=sys.stderr, flush=True)
        with open(args.input, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
        
        # 初始化模型
        scorer = Qwen3VLStandaloneScorer(
            model_name=args.model_name,
            device=args.device,
            dtype=args.dtype
        )
        
        # 评分
        tasks = input_data.get('tasks', [])
        print(f"[Qwen3VL-Standalone] Processing {len(tasks)} tasks...", file=sys.stderr, flush=True)
        
        scores = scorer.score_batch(
            tasks=tasks,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            use_batch_inference=args.use_batch_inference
        )
        
        # 写入输出
        output_data = {
            'scores': scores,
            'status': 'success',
            'num_tasks': len(tasks)
        }
        
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"[Qwen3VL-Standalone] Results written to: {args.output}", file=sys.stderr, flush=True)
        print(f"[Qwen3VL-Standalone] Average score: {sum(scores)/len(scores):.3f}", 
              file=sys.stderr, flush=True)
        
        sys.exit(0)
    
    except Exception as e:
        print(f"[ERROR] {str(e)}", file=sys.stderr, flush=True)
        import traceback
        traceback.print_exc(file=sys.stderr)
        
        # 写入错误信息
        error_data = {
            'status': 'error',
            'error': str(e),
            'scores': []
        }
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(error_data, f, ensure_ascii=False, indent=2)
        
        sys.exit(1)


if __name__ == '__main__':
    main()

