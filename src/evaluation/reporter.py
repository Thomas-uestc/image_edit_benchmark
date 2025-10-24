"""
Reporter for generating evaluation reports
报告生成器
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging


class Reporter:
    """
    评测报告生成器
    
    负责生成和保存评测结果报告
    """
    
    def __init__(self, 
                 output_dir: str,
                 logger: Optional[logging.Logger] = None):
        """
        初始化报告生成器
        
        Args:
            output_dir: 输出目录
            logger: 日志记录器（可选）
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger or logging.getLogger(__name__)
    
    def generate_report(self,
                       category_statistics: Dict[str, Dict[str, float]],
                       overall_statistics: Dict[str, float],
                       metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        生成评测报告
        
        Args:
            category_statistics: 各类别统计指标
            overall_statistics: 整体统计指标
            metadata: 元数据（模型信息、配置等）
            
        Returns:
            报告字典
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "category_statistics": category_statistics,
            "overall_statistics": overall_statistics,
            "summary": self._generate_summary(category_statistics, overall_statistics),
            "metadata": metadata or {}
        }
        
        return report
    
    def _generate_summary(self,
                         category_statistics: Dict[str, Dict[str, float]],
                         overall_statistics: Dict[str, float]) -> Dict[str, Any]:
        """
        生成摘要信息
        
        Args:
            category_statistics: 各类别统计指标
            overall_statistics: 整体统计指标
            
        Returns:
            摘要字典
        """
        summary = {
            "num_categories": len(category_statistics),
            "total_samples": overall_statistics.get("num_samples", 0),
            "overall_mean": overall_statistics.get("mean", 0),
            "category_means": {}
        }
        
        for category, stats in category_statistics.items():
            summary["category_means"][category] = stats.get("mean", 0)
        
        # 找出最好和最差的类别
        if summary["category_means"]:
            best_category = max(summary["category_means"], key=summary["category_means"].get)
            worst_category = min(summary["category_means"], key=summary["category_means"].get)
            
            summary["best_category"] = {
                "name": best_category,
                "score": summary["category_means"][best_category]
            }
            summary["worst_category"] = {
                "name": worst_category,
                "score": summary["category_means"][worst_category]
            }
        
        return summary
    
    def save_report(self, 
                   report: Dict[str, Any],
                   filename: Optional[str] = None) -> str:
        """
        保存报告到JSON文件
        
        Args:
            report: 报告字典
            filename: 文件名（可选，默认使用时间戳）
            
        Returns:
            保存的文件路径
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"evaluation_report_{timestamp}.json"
        
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Report saved to: {filepath}")
        
        return str(filepath)
    
    def generate_markdown_report(self, 
                                report: Dict[str, Any]) -> str:
        """
        生成Markdown格式的报告
        
        Args:
            report: 报告字典
            
        Returns:
            Markdown格式的报告文本
        """
        md_lines = []
        
        # 标题
        md_lines.append("# Image Edit Benchmark Evaluation Report")
        md_lines.append("")
        md_lines.append(f"**Generated:** {report['timestamp']}")
        md_lines.append("")
        
        # 摘要
        summary = report.get("summary", {})
        md_lines.append("## Summary")
        md_lines.append("")
        md_lines.append(f"- **Total Samples:** {summary.get('total_samples', 0)}")
        md_lines.append(f"- **Number of Categories:** {summary.get('num_categories', 0)}")
        md_lines.append(f"- **Overall Mean Score:** {summary.get('overall_mean', 0):.3f}")
        md_lines.append("")
        
        if "best_category" in summary:
            md_lines.append(f"- **Best Category:** {summary['best_category']['name']} ({summary['best_category']['score']:.3f})")
        if "worst_category" in summary:
            md_lines.append(f"- **Worst Category:** {summary['worst_category']['name']} ({summary['worst_category']['score']:.3f})")
        md_lines.append("")
        
        # 各类别详细结果
        md_lines.append("## Category Results")
        md_lines.append("")
        
        category_stats = report.get("category_statistics", {})
        for category, stats in category_stats.items():
            md_lines.append(f"### {category}")
            md_lines.append("")
            md_lines.append(f"- **Mean:** {stats.get('mean', 0):.3f}")
            md_lines.append(f"- **Std:** {stats.get('std', 0):.3f}")
            md_lines.append(f"- **Median:** {stats.get('median', 0):.3f}")
            md_lines.append(f"- **Min:** {stats.get('min', 0):.3f}")
            md_lines.append(f"- **Max:** {stats.get('max', 0):.3f}")
            md_lines.append(f"- **Samples:** {stats.get('num_samples', 0)}")
            md_lines.append("")
        
        # 整体统计
        md_lines.append("## Overall Statistics")
        md_lines.append("")
        overall_stats = report.get("overall_statistics", {})
        md_lines.append(f"- **Mean:** {overall_stats.get('mean', 0):.3f}")
        md_lines.append(f"- **Std:** {overall_stats.get('std', 0):.3f}")
        md_lines.append(f"- **Median:** {overall_stats.get('median', 0):.3f}")
        md_lines.append(f"- **Min:** {overall_stats.get('min', 0):.3f}")
        md_lines.append(f"- **Max:** {overall_stats.get('max', 0):.3f}")
        md_lines.append("")
        
        return "\n".join(md_lines)
    
    def save_markdown_report(self,
                            report: Dict[str, Any],
                            filename: Optional[str] = None) -> str:
        """
        保存Markdown格式的报告
        
        Args:
            report: 报告字典
            filename: 文件名（可选）
            
        Returns:
            保存的文件路径
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"evaluation_report_{timestamp}.md"
        
        filepath = self.output_dir / filename
        
        md_content = self.generate_markdown_report(report)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        self.logger.info(f"Markdown report saved to: {filepath}")
        
        return str(filepath)


