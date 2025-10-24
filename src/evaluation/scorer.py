"""
Scorer for computing statistics
评分统计器
"""

import numpy as np
from typing import Dict, List, Optional
import logging


class Scorer:
    """
    评分统计器
    
    负责计算各类别的统计指标（平均分、标准差等）
    """
    
    def __init__(self, 
                 metrics: Optional[List[str]] = None,
                 logger: Optional[logging.Logger] = None):
        """
        初始化评分统计器
        
        Args:
            metrics: 需要计算的统计指标列表，默认为["mean", "std", "median"]
            logger: 日志记录器（可选）
        """
        self.metrics = metrics or ["mean", "std", "median", "min", "max"]
        self.logger = logger or logging.getLogger(__name__)
    
    def compute_category_statistics(self, 
                                    scores: List[float],
                                    category_name: str) -> Dict[str, float]:
        """
        计算单个类别的统计指标
        
        Args:
            scores: 该类别的所有评分列表
            category_name: 类别名称
            
        Returns:
            统计指标字典
        """
        if not scores:
            self.logger.warning(f"No scores for category '{category_name}'")
            return {}
        
        scores_array = np.array(scores)
        stats = {}
        
        if "mean" in self.metrics:
            stats["mean"] = float(np.mean(scores_array))
        
        if "std" in self.metrics:
            stats["std"] = float(np.std(scores_array))
        
        if "median" in self.metrics:
            stats["median"] = float(np.median(scores_array))
        
        if "min" in self.metrics:
            stats["min"] = float(np.min(scores_array))
        
        if "max" in self.metrics:
            stats["max"] = float(np.max(scores_array))
        
        if "count" in self.metrics:
            stats["count"] = len(scores)
        
        # 添加样本数量
        stats["num_samples"] = len(scores)
        
        return stats
    
    def compute_all_statistics(self, 
                              category_scores: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
        """
        计算所有类别的统计指标
        
        Args:
            category_scores: 字典，键为类别名，值为评分列表
            
        Returns:
            统计指标字典，格式为 {category_name: {metric: value}}
        """
        all_stats = {}
        
        for category, scores in category_scores.items():
            stats = self.compute_category_statistics(scores, category)
            all_stats[category] = stats
            
            self.logger.info(
                f"Category '{category}': "
                f"Mean={stats.get('mean', 0):.3f}, "
                f"Std={stats.get('std', 0):.3f}, "
                f"N={stats.get('num_samples', 0)}"
            )
        
        return all_stats
    
    def compute_overall_statistics(self,
                                   category_scores: Dict[str, List[float]]) -> Dict[str, float]:
        """
        计算整体统计指标（所有类别合并）
        
        Args:
            category_scores: 字典，键为类别名，值为评分列表
            
        Returns:
            整体统计指标字典
        """
        all_scores = []
        for scores in category_scores.values():
            all_scores.extend(scores)
        
        if not all_scores:
            self.logger.warning("No scores found for overall statistics")
            return {}
        
        overall_stats = self.compute_category_statistics(all_scores, "overall")
        
        self.logger.info(
            f"Overall statistics: "
            f"Mean={overall_stats.get('mean', 0):.3f}, "
            f"Std={overall_stats.get('std', 0):.3f}, "
            f"N={overall_stats.get('num_samples', 0)}"
        )
        
        return overall_stats
    
    def compute_weighted_average(self,
                                category_scores: Dict[str, List[float]]) -> float:
        """
        计算加权平均分（按类别样本数加权）
        
        Args:
            category_scores: 字典，键为类别名，值为评分列表
            
        Returns:
            加权平均分
        """
        total_score = 0
        total_count = 0
        
        for scores in category_scores.values():
            if scores:
                total_score += sum(scores)
                total_count += len(scores)
        
        if total_count == 0:
            return 0.0
        
        weighted_avg = total_score / total_count
        self.logger.info(f"Weighted average: {weighted_avg:.3f}")
        
        return weighted_avg


