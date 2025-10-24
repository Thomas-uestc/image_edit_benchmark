"""
Main benchmark evaluation pipeline
主评测Pipeline
"""

import json
import importlib
from pathlib import Path
from typing import Dict, Any, Optional
from tqdm import tqdm
import logging

from .data import BenchmarkLoader, BenchmarkData, DataPair
from .models.diffusion.base_diffusion import BaseDiffusionModel
from .models.reward.base_reward import BaseRewardModel
from .evaluation import Scorer, Reporter
from .utils import decode_base64_image, save_image, setup_logger, PromptManager


class BenchmarkPipeline:
    """
    图像编辑Benchmark评测Pipeline
    
    整合所有模块，实现完整的评测流程：
    1. 加载benchmark数据
    2. 使用扩散模型编辑图像
    3. 使用reward模型评分
    4. 统计和生成报告
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化Pipeline
        
        Args:
            config: 配置字典
        """
        self.config = config
        
        # 设置日志
        log_config = config.get("logging", {})
        self.logger = setup_logger(
            name="benchmark_pipeline",
            level=log_config.get("level", "INFO"),
            log_file=log_config.get("log_file") if log_config.get("file_output") else None,
            console_output=log_config.get("console_output", True)
        )
        
        self.logger.info("="*60)
        self.logger.info("Initializing Benchmark Evaluation Pipeline")
        self.logger.info("="*60)
        
        # 创建输出目录
        self._setup_output_dirs()
        
        # 初始化各个组件
        self.data_loader = BenchmarkLoader(logger=self.logger)
        self.diffusion_model = self._load_diffusion_model()
        self.reward_model = self._load_reward_model()
        self.prompt_manager = PromptManager(config.get("prompts", {}))
        self.scorer = Scorer(
            metrics=config.get("evaluation", {}).get("metrics", ["mean", "std", "median"]),
            logger=self.logger
        )
        # 获取输出目录（兼容output_dir和results_dir两种配置）
        eval_config = self.config.get("evaluation", {})
        output_dir = eval_config.get("output_dir") or eval_config.get("results_dir", "outputs")
        
        self.reporter = Reporter(
            output_dir=output_dir,
            logger=self.logger
        )
        
        # 断点续传相关
        self.checkpoint_path = Path(config.get("evaluation", {}).get("checkpoint_path", "outputs/checkpoint.json"))
        self.resume_from_checkpoint = config.get("evaluation", {}).get("resume_from_checkpoint", False)
        self.checkpoint_data = self._load_checkpoint() if self.resume_from_checkpoint else {}
        
        self.logger.info("Pipeline initialized successfully")
    
    def _setup_output_dirs(self):
        """创建输出目录"""
        eval_config = self.config.get("evaluation", {})
        
        output_dir = Path(eval_config.get("output_dir", "outputs"))
        results_dir = Path(eval_config.get("results_dir", "outputs/results"))
        images_dir = Path(eval_config.get("images_dir", "outputs/images"))
        logs_dir = Path(eval_config.get("logs_dir", "outputs/logs"))
        
        for dir_path in [output_dir, results_dir, images_dir, logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Output directories created: {output_dir}")
    
    def _load_diffusion_model(self) -> BaseDiffusionModel:
        """动态加载扩散编辑模型"""
        model_config = self.config.get("diffusion_model", {})
        class_path = model_config.get("class_path")
        
        if not class_path:
            raise ValueError("diffusion_model.class_path not specified in config")
        
        self.logger.info(f"Loading diffusion model: {class_path}")
        
        # 动态导入模型类
        module_path, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        model_class = getattr(module, class_name)
        
        # 实例化模型
        model = model_class(model_config.get("params", {}))
        
        self.logger.info("Diffusion model loaded successfully")
        return model
    
    def _load_reward_model(self) -> BaseRewardModel:
        """动态加载Reward评分模型"""
        model_config = self.config.get("reward_model", {})
        class_path = model_config.get("class_path")
        
        if not class_path:
            raise ValueError("reward_model.class_path not specified in config")
        
        self.logger.info(f"Loading reward model: {class_path}")
        
        # 动态导入模型类
        module_path, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        model_class = getattr(module, class_name)
        
        # 实例化模型
        model = model_class(model_config.get("params", {}))
        
        self.logger.info("Reward model loaded successfully")
        return model
    
    def _load_checkpoint(self) -> Dict:
        """加载断点数据"""
        if self.checkpoint_path.exists():
            self.logger.info(f"Loading checkpoint from: {self.checkpoint_path}")
            with open(self.checkpoint_path, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_checkpoint(self, processed_pairs: Dict[str, list]):
        """保存断点数据"""
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.checkpoint_path, 'w') as f:
            json.dump(processed_pairs, f)
        self.logger.debug(f"Checkpoint saved to: {self.checkpoint_path}")
    
    def run(self) -> Dict[str, Any]:
        """
        运行完整的评测流程（两阶段处理优化）
        
        Returns:
            评测报告字典
        """
        self.logger.info("="*80)
        self.logger.info("Starting benchmark evaluation (Two-Stage Processing)")
        self.logger.info("="*80)
        
        # 1. 加载benchmark数据
        benchmark_data = self._load_benchmark_data()
        
        # 2. 初始化模型状态：确保Diffusion在GPU，Reward在CPU
        self.logger.info("\n" + "="*60)
        self.logger.info("[初始化] 设置模型状态")
        self.logger.info("="*60)
        self.diffusion_model.load_to_gpu()  # 确保Diffusion在GPU
        self.reward_model.unload_from_gpu()  # 确保Reward在CPU
        
        # 3. 按类别处理数据
        category_scores = {}
        
        for idx, category_name in enumerate(benchmark_data.category_names, 1):
            self.logger.info(f"\n{'#'*80}")
            self.logger.info(f"# 处理类别 [{idx}/{len(benchmark_data.category_names)}]: {category_name}")
            self.logger.info(f"{'#'*80}")
            
            category_data = benchmark_data.get_category(category_name)
            scores = self._process_category(category_data)
            category_scores[category_name] = scores
            
            # 更新CategoryData的scores
            category_data.scores = scores
            
            # 在处理下一个类别前，恢复模型状态
            if idx < len(benchmark_data.category_names):
                self.logger.info(f"\n{'='*60}")
                self.logger.info(f"[准备下一类别] 恢复模型状态：Diffusion → GPU, Reward → CPU")
                self.logger.info(f"{'='*60}")
                self.reward_model.unload_from_gpu()
                self.diffusion_model.load_to_gpu()
        
        # 4. 计算统计指标
        self.logger.info("\n" + "="*80)
        self.logger.info("Computing statistics...")
        self.logger.info("="*80)
        
        category_statistics = self.scorer.compute_all_statistics(category_scores)
        overall_statistics = self.scorer.compute_overall_statistics(category_scores)
        
        # 5. 生成报告
        self.logger.info("\n" + "="*60)
        self.logger.info("Generating report...")
        self.logger.info("="*60)
        
        metadata = {
            "benchmark_config": self.config.get("benchmark", {}),
            "diffusion_model": self.config.get("diffusion_model", {}),
            "reward_model": self.config.get("reward_model", {}),
            "total_pairs": benchmark_data.total_pairs,
            "categories": benchmark_data.category_names
        }
        
        report = self.reporter.generate_report(
            category_statistics=category_statistics,
            overall_statistics=overall_statistics,
            metadata=metadata
        )
        
        # 6. 保存报告
        json_path = self.reporter.save_report(report)
        md_path = self.reporter.save_markdown_report(report)
        
        self.logger.info("\n" + "="*80)
        self.logger.info("✓ Evaluation completed successfully!")
        self.logger.info("="*80)
        self.logger.info(f"JSON report: {json_path}")
        self.logger.info(f"Markdown report: {md_path}")
        self.logger.info("="*80)
        
        return report
    
    def _load_benchmark_data(self) -> BenchmarkData:
        """加载benchmark数据"""
        benchmark_config = self.config.get("benchmark", {})
        data_path = benchmark_config.get("data_path")
        categories = benchmark_config.get("categories", [])
        
        if not data_path:
            raise ValueError("benchmark.data_path not specified in config")
        
        if not categories:
            raise ValueError("benchmark.categories not specified in config")
        
        benchmark_data = self.data_loader.load(
            data_path=data_path,
            categories=categories,
            decode_images=False  # 按需解码以节省内存
        )
        
        return benchmark_data
    
    def _process_category(self, category_data) -> list:
        """
        处理单个类别的数据（两阶段处理优化）
        
        阶段1: 批量图像编辑（Diffusion Model在GPU）
        阶段2: 批量图像评分（Reward Model在GPU）
        
        Args:
            category_data: CategoryData对象
            
        Returns:
            评分列表
        """
        category_name = category_data.category_name
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"[阶段1/2] 开始批量图像编辑 - {category_name}")
        self.logger.info(f"{'='*60}")
        
        # ===== 阶段1: 批量图像编辑 =====
        
        # 准备数据：解码所有图像
        self.logger.info(f"解码原始图像...")
        for pair in category_data.data_pairs:
            if pair.original_image is None:
                pair.original_image = decode_base64_image(pair.original_image_b64)
        
        # 收集所有图像和指令
        original_images = [pair.original_image for pair in category_data.data_pairs]
        edit_instructions = [pair.edit_instruction for pair in category_data.data_pairs]
        
        # 使用batch_edit进行多GPU并行编辑
        self.logger.info(f"开始多GPU并行编辑 {len(original_images)} 张图像...")
        
        try:
            # 检查diffusion_model是否支持batch_edit
            if hasattr(self.diffusion_model, 'batch_edit'):
                # 多GPU并行批量编辑
                edited_images = self.diffusion_model.batch_edit(
                    images=original_images,
                    instructions=edit_instructions
                )
            else:
                # 回退到逐张处理（单GPU模型）
                self.logger.warning("Diffusion model不支持batch_edit，使用逐张处理")
                edited_images = []
                pbar = tqdm(zip(original_images, edit_instructions), 
                           total=len(original_images),
                           desc=f"[{category_name}] 编辑图像")
                for img, inst in pbar:
                    edited_img = self.diffusion_model.edit_image(img, inst)
                    edited_images.append(edited_img)
                pbar.close()
            
            # 将编辑后的图像分配回pair对象
            for pair, edited_image in zip(category_data.data_pairs, edited_images):
                pair.edited_image = edited_image
                
                # 保存编辑后的图像到磁盘（如果需要）
                if self.config.get("evaluation", {}).get("save_generated_images", False):
                    self._save_edited_image(pair, category_name)
        
        except Exception as e:
            self.logger.error(f"Error during batch editing: {e}")
            # 回退到逐张处理
            self.logger.info("回退到逐张处理模式...")
            pbar_edit = tqdm(category_data.data_pairs, desc=f"[{category_name}] 编辑图像")
            for pair in pbar_edit:
                try:
                    edited_image = self.diffusion_model.edit_image(
                        original_image=pair.original_image,
                        edit_instruction=pair.edit_instruction
                    )
                    pair.edited_image = edited_image
                    
                    if self.config.get("evaluation", {}).get("save_generated_images", False):
                        self._save_edited_image(pair, category_name)
                except Exception as e2:
                    self.logger.error(f"Error editing image for pair {pair.pair_id}: {e2}")
                    pair.edited_image = None
            pbar_edit.close()
        
        # ===== 模型切换：卸载Diffusion，加载Reward =====
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"[模型切换] 卸载Diffusion模型，加载Reward模型")
        self.logger.info(f"{'='*60}")
        
        self.diffusion_model.unload_from_gpu()
        self.reward_model.load_to_gpu()
        
        # ===== 阶段2: 批量图像评分 =====
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"[阶段2/2] 开始批量图像评分 - {category_name}")
        self.logger.info(f"{'='*60}")
        
        # 收集所有有效的待评分数据
        valid_pairs = []
        valid_indices = []
        edited_images = []
        original_descriptions = []
        edit_instructions = []
        system_prompts = []
        user_prompts = []
        original_images = []
        
        for idx, pair in enumerate(category_data.data_pairs):
            # 检查是否成功编辑
            if pair.edited_image is None:
                self.logger.warning(f"Pair {pair.pair_id} 没有编辑后的图像，跳过评分")
                continue
            
            try:
                # 获取该类别的prompt
                prompts = self.prompt_manager.get_full_prompt(
                    category=category_name,
                    original_description=pair.original_description,
                    edit_instruction=pair.edit_instruction
                )
                
                # 收集数据
                valid_pairs.append(pair)
                valid_indices.append(idx)
                edited_images.append(pair.edited_image)
                original_descriptions.append(pair.original_description)
                edit_instructions.append(pair.edit_instruction)
                system_prompts.append(prompts["system_prompt"])
                user_prompts.append(prompts["user_prompt"])
                original_images.append(pair.original_image)
                
            except Exception as e:
                self.logger.error(f"Error preparing pair {pair.pair_id} for scoring: {e}")
                continue
        
        # 使用batch inference评分
        scores = [0.0] * len(category_data.data_pairs)  # 初始化所有分数为0
        
        if valid_pairs:
            self.logger.info(f"[Qwen3VLRewardModel] 准备评分 {len(valid_pairs)} 张有效图像...")
            
            try:
                # 获取batch_size配置
                batch_size = self.config.get("reward_model", {}).get("params", {}).get("batch_size", 4)
                use_batch_inference = self.config.get("reward_model", {}).get("params", {}).get("use_batch_inference", True)
                
                # 批量评分
                batch_scores = self.reward_model.batch_score(
                    edited_images=edited_images,
                    original_descriptions=original_descriptions,
                    edit_instructions=edit_instructions,
                    system_prompts=system_prompts,
                    user_prompts=user_prompts,
                    original_images=original_images,
                    batch_size=batch_size,
                    use_batch_inference=use_batch_inference
                )
                
                # 将分数分配回对应的pair
                for pair, idx, score in zip(valid_pairs, valid_indices, batch_scores):
                    pair.score = score
                    scores[idx] = score
                    self.logger.debug(f"Pair {pair.pair_id}: score={score:.3f}")
                
                self.logger.info(f"✅ 评分完成，平均分: {sum(batch_scores)/len(batch_scores):.3f}")
                
            except Exception as e:
                self.logger.error(f"Error in batch scoring: {e}")
                self.logger.warning("Falling back to sequential scoring...")
                
                # 回退到逐个评分
                for pair, idx in zip(valid_pairs, valid_indices):
                    try:
                        prompts = self.prompt_manager.get_full_prompt(
                            category=category_name,
                            original_description=pair.original_description,
                            edit_instruction=pair.edit_instruction
                        )
                        
                        score = self.reward_model.score(
                            edited_image=pair.edited_image,
                            original_description=pair.original_description,
                            edit_instruction=pair.edit_instruction,
                            system_prompt=prompts["system_prompt"],
                            user_prompt=prompts["user_prompt"],
                            original_image=pair.original_image
                        )
                        
                        pair.score = score
                        scores[idx] = score
                        
                    except Exception as e2:
                        self.logger.error(f"Error scoring pair {pair.pair_id}: {e2}")
                        pair.score = 0.0
                        scores[idx] = 0.0
        else:
            self.logger.warning("没有有效的图像需要评分")
        
        # ===== 类别处理完成 =====
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"[完成] {category_name} - 共处理 {len(scores)} 个样本")
        self.logger.info(f"平均分: {sum(scores)/len(scores):.3f}")
        self.logger.info(f"{'='*60}\n")
        
        return scores
    
    def _save_edited_image(self, pair: DataPair, category_name: str):
        """保存编辑后的图像"""
        if pair.edited_image is None:
            return
        
        images_dir = Path(self.config["evaluation"]["images_dir"])
        category_dir = images_dir / category_name
        category_dir.mkdir(parents=True, exist_ok=True)
        
        image_path = category_dir / f"{pair.pair_id}.png"
        
        try:
            save_image(pair.edited_image, str(image_path))
            self.logger.debug(f"Saved edited image: {image_path}")
        except Exception as e:
            self.logger.warning(f"Failed to save image for {pair.pair_id}: {e}")
    
    def run_single_pair(self, 
                       original_image_b64: str,
                       edit_instruction: str,
                       original_description: str,
                       category: str) -> Dict[str, Any]:
        """
        对单个数据对进行评测（用于测试）
        
        Args:
            original_image_b64: 原始图像的base64编码
            edit_instruction: 编辑指令
            original_description: 原始图像描述
            category: 类别名称
            
        Returns:
            包含编辑图像和评分的字典
        """
        # 解码图像
        original_image = decode_base64_image(original_image_b64)
        
        # 编辑图像
        edited_image = self.diffusion_model.edit_image(
            original_image=original_image,
            edit_instruction=edit_instruction
        )
        
        # 获取prompt
        prompts = self.prompt_manager.get_full_prompt(
            category=category,
            original_description=original_description,
            edit_instruction=edit_instruction
        )
        
        # 评分
        score = self.reward_model.score(
            edited_image=edited_image,
            original_description=original_description,
            edit_instruction=edit_instruction,
            system_prompt=prompts["system_prompt"],
            user_prompt=prompts["user_prompt"],
            original_image=original_image
        )
        
        return {
            "edited_image": edited_image,
            "score": score,
            "prompts": prompts
        }
