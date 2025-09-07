"""
Comprehensive Comparison Framework for FHE vs Plain Text FL
Provides statistical analysis, benchmarking, and detailed comparison metrics
"""

import os
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

from src.core.base_pipeline import PipelineConfig, ExperimentResult, RoundResult
from src.strategies.plaintext_strategy import PlainTextFederatedLearningPipeline, EnhancedPlainTextPipeline
from src.strategies.true_fhe_strategy import TrueFHECKKSFederatedLearningPipeline, ClientSideFHECKKSPipeline

@dataclass
class ComparisonMetrics:
    """Comprehensive comparison metrics between FHE and Plain Text"""
    # Basic performance metrics
    accuracy_diff: float
    f1_score_diff: float
    precision_diff: float
    recall_diff: float
    
    # Timing metrics
    training_time_overhead: float  # Percentage increase
    aggregation_time_overhead: float
    total_time_overhead: float
    
    # Communication metrics
    communication_overhead: float  # Percentage increase
    bytes_transferred_overhead: float
    
    # Resource metrics
    memory_overhead: Optional[float] = None
    cpu_overhead: Optional[float] = None
    energy_overhead: Optional[float] = None
    
    # Statistical significance
    accuracy_p_value: Optional[float] = None
    f1_score_p_value: Optional[float] = None
    statistical_significance: Optional[bool] = None
    
    # Convergence analysis
    convergence_round_diff: Optional[int] = None
    convergence_rate_diff: Optional[float] = None
    
    # Privacy vs Performance trade-off
    privacy_score: float = 1.0  # FHE = 1.0, Plain Text = 0.0
    performance_score: float = 0.0  # Based on accuracy
    trade_off_ratio: Optional[float] = None

@dataclass
class ComparisonResult:
    """Complete comparison result between two experiments"""
    comparison_id: str
    timestamp: datetime
    
    # Experiment results
    plaintext_result: ExperimentResult
    fhe_result: ExperimentResult
    
    # Comparison metrics
    metrics: ComparisonMetrics
    
    # Detailed analysis
    round_by_round_comparison: List[Dict[str, Any]]
    statistical_analysis: Dict[str, Any]
    
    # Recommendations
    recommendations: List[str]
    conclusion: str

class FederatedLearningComparison:
    """
    Comprehensive comparison framework for FHE vs Plain Text FL
    Provides statistical analysis and detailed benchmarking
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.comparison_id = f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.results_dir = f"comparisons/{self.comparison_id}"
        
        # Create results directory
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Initialize pipelines
        self.plaintext_pipeline = PlainTextFederatedLearningPipeline(config)
        self.fhe_pipeline = ClientSideFHECKKSPipeline(config)
        
        print(f"🔬 Initialized FL Comparison Framework")
        print(f"📁 Results will be saved to: {self.results_dir}")
    
    def run_comprehensive_comparison(self, num_runs: int = 3) -> ComparisonResult:
        """
        Run comprehensive comparison with multiple runs for statistical significance
        """
        print(f"\n🚀 Starting Comprehensive FL Comparison")
        print(f"📊 Configuration: {self.config.rounds} rounds, {self.config.clients} clients")
        print(f"🔄 Running {num_runs} iterations for statistical significance")
        print("=" * 70)
        
        # Run multiple experiments for statistical significance
        plaintext_results = []
        fhe_results = []
        
        for run in range(num_runs):
            print(f"\n📈 Run {run + 1}/{num_runs}")
            
            # Run Plain Text experiment
            print("🔓 Running Plain Text FL...")
            plaintext_result = self.plaintext_pipeline.run_federated_learning()
            plaintext_results.append(plaintext_result)
            
            # Run FHE experiment
            print("🔐 Running FHE CKKS FL...")
            fhe_result = self.fhe_pipeline.run_federated_learning()
            fhe_results.append(fhe_result)
            
            print(f"✅ Run {run + 1} completed")
        
        # Calculate average results
        avg_plaintext = self._calculate_average_result(plaintext_results)
        avg_fhe = self._calculate_average_result(fhe_results)
        
        # Perform detailed comparison
        comparison_result = self._perform_detailed_comparison(avg_plaintext, avg_fhe, plaintext_results, fhe_results)
        
        # Save results
        self._save_comparison_results(comparison_result)
        
        # Generate visualizations
        self._generate_comparison_visualizations(comparison_result)
        
        return comparison_result
    
    def _calculate_average_result(self, results: List[ExperimentResult]) -> ExperimentResult:
        """Calculate average result from multiple runs"""
        if not results:
            raise ValueError("No results to average")
        
        # Average the metrics
        avg_final_accuracy = np.mean([r.final_accuracy for r in results])
        avg_best_accuracy = np.mean([r.best_accuracy for r in results])
        avg_accuracy_improvement = np.mean([r.accuracy_improvement for r in results])
        avg_training_time = np.mean([r.avg_training_time for r in results])
        avg_aggregation_time = np.mean([r.avg_aggregation_time for r in results])
        
        # Use the first result as template and update with averages
        avg_result = results[0]
        avg_result.final_accuracy = avg_final_accuracy
        avg_result.best_accuracy = avg_best_accuracy
        avg_result.accuracy_improvement = avg_accuracy_improvement
        avg_result.avg_training_time = avg_training_time
        avg_result.avg_aggregation_time = avg_aggregation_time
        
        # Average round results
        if results[0].round_results:
            num_rounds = len(results[0].round_results)
            avg_round_results = []
            
            for round_idx in range(num_rounds):
                round_accuracies = [r.round_results[round_idx].accuracy for r in results]
                round_f1_scores = [r.round_results[round_idx].f1_score for r in results]
                round_training_times = [r.round_results[round_idx].training_time for r in results]
                round_aggregation_times = [r.round_results[round_idx].aggregation_time for r in results]
                
                avg_round_result = results[0].round_results[round_idx]
                avg_round_result.accuracy = np.mean(round_accuracies)
                avg_round_result.f1_score = np.mean(round_f1_scores)
                avg_round_result.training_time = np.mean(round_training_times)
                avg_round_result.aggregation_time = np.mean(round_aggregation_times)
                
                avg_round_results.append(avg_round_result)
            
            avg_result.round_results = avg_round_results
        
        return avg_result
    
    def _perform_detailed_comparison(self, plaintext_result: ExperimentResult, 
                                   fhe_result: ExperimentResult,
                                   plaintext_runs: List[ExperimentResult],
                                   fhe_runs: List[ExperimentResult]) -> ComparisonResult:
        """Perform detailed statistical comparison"""
        print("\n📊 Performing Detailed Statistical Analysis...")
        
        # Calculate basic differences
        accuracy_diff = fhe_result.final_accuracy - plaintext_result.final_accuracy
        f1_score_diff = fhe_result.round_results[-1].f1_score - plaintext_result.round_results[-1].f1_score
        precision_diff = fhe_result.round_results[-1].precision - plaintext_result.round_results[-1].precision
        recall_diff = fhe_result.round_results[-1].recall - plaintext_result.round_results[-1].recall
        
        # Calculate timing overheads
        training_time_overhead = ((fhe_result.avg_training_time - plaintext_result.avg_training_time) / 
                                plaintext_result.avg_training_time) * 100
        aggregation_time_overhead = ((fhe_result.avg_aggregation_time - plaintext_result.avg_aggregation_time) / 
                                   plaintext_result.avg_aggregation_time) * 100
        total_time_overhead = ((fhe_result.total_duration - plaintext_result.total_duration) / 
                             plaintext_result.total_duration) * 100
        
        # Statistical significance testing
        accuracy_p_value = self._calculate_statistical_significance(
            [r.final_accuracy for r in plaintext_runs],
            [r.final_accuracy for r in fhe_runs]
        )
        
        f1_score_p_value = self._calculate_statistical_significance(
            [r.round_results[-1].f1_score for r in plaintext_runs],
            [r.round_results[-1].f1_score for r in fhe_runs]
        )
        
        # Create comparison metrics
        metrics = ComparisonMetrics(
            accuracy_diff=accuracy_diff,
            f1_score_diff=f1_score_diff,
            precision_diff=precision_diff,
            recall_diff=recall_diff,
            training_time_overhead=training_time_overhead,
            aggregation_time_overhead=aggregation_time_overhead,
            total_time_overhead=total_time_overhead,
            communication_overhead=0.0,  # Placeholder for communication overhead
            bytes_transferred_overhead=0.0,  # Placeholder for bytes overhead
            accuracy_p_value=accuracy_p_value,
            f1_score_p_value=f1_score_p_value,
            statistical_significance=accuracy_p_value < 0.05 if accuracy_p_value else False,
            privacy_score=1.0,  # FHE provides full privacy
            performance_score=fhe_result.final_accuracy
        )
        
        # Calculate trade-off ratio
        metrics.trade_off_ratio = metrics.privacy_score / metrics.performance_score
        
        # Round-by-round comparison
        round_comparison = self._create_round_by_round_comparison(plaintext_result, fhe_result)
        
        # Statistical analysis
        statistical_analysis = self._perform_statistical_analysis(plaintext_runs, fhe_runs)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(metrics, plaintext_result, fhe_result)
        
        # Generate conclusion
        conclusion = self._generate_conclusion(metrics, recommendations)
        
        return ComparisonResult(
            comparison_id=self.comparison_id,
            timestamp=datetime.now(),
            plaintext_result=plaintext_result,
            fhe_result=fhe_result,
            metrics=metrics,
            round_by_round_comparison=round_comparison,
            statistical_analysis=statistical_analysis,
            recommendations=recommendations,
            conclusion=conclusion
        )
    
    def _calculate_statistical_significance(self, group1: List[float], group2: List[float]) -> float:
        """Calculate statistical significance using t-test"""
        try:
            t_stat, p_value = stats.ttest_ind(group1, group2)
            return p_value
        except Exception as e:
            print(f"⚠️  Statistical test failed: {e}")
            return None
    
    def _create_round_by_round_comparison(self, plaintext_result: ExperimentResult, 
                                        fhe_result: ExperimentResult) -> List[Dict[str, Any]]:
        """Create detailed round-by-round comparison"""
        comparison = []
        
        for i, (pt_round, fhe_round) in enumerate(zip(plaintext_result.round_results, fhe_result.round_results)):
            round_comparison = {
                'round': i + 1,
                'plaintext_accuracy': pt_round.accuracy,
                'fhe_accuracy': fhe_round.accuracy,
                'accuracy_diff': fhe_round.accuracy - pt_round.accuracy,
                'plaintext_f1': pt_round.f1_score,
                'fhe_f1': fhe_round.f1_score,
                'f1_diff': fhe_round.f1_score - pt_round.f1_score,
                'plaintext_training_time': pt_round.training_time,
                'fhe_training_time': fhe_round.training_time,
                'training_time_overhead': ((fhe_round.training_time - pt_round.training_time) / 
                                         pt_round.training_time) * 100,
                'plaintext_aggregation_time': pt_round.aggregation_time,
                'fhe_aggregation_time': fhe_round.aggregation_time,
                'aggregation_time_overhead': ((fhe_round.aggregation_time - pt_round.aggregation_time) / 
                                            pt_round.aggregation_time) * 100
            }
            comparison.append(round_comparison)
        
        return comparison
    
    def _perform_statistical_analysis(self, plaintext_runs: List[ExperimentResult], 
                                    fhe_runs: List[ExperimentResult]) -> Dict[str, Any]:
        """Perform comprehensive statistical analysis"""
        analysis = {}
        
        # Extract metrics for analysis
        pt_accuracies = [r.final_accuracy for r in plaintext_runs]
        fhe_accuracies = [r.final_accuracy for r in fhe_runs]
        
        pt_f1_scores = [r.round_results[-1].f1_score for r in plaintext_runs]
        fhe_f1_scores = [r.round_results[-1].f1_score for r in fhe_runs]
        
        # Descriptive statistics
        analysis['plaintext_accuracy_stats'] = {
            'mean': np.mean(pt_accuracies),
            'std': np.std(pt_accuracies),
            'min': np.min(pt_accuracies),
            'max': np.max(pt_accuracies),
            'median': np.median(pt_accuracies)
        }
        
        analysis['fhe_accuracy_stats'] = {
            'mean': np.mean(fhe_accuracies),
            'std': np.std(fhe_accuracies),
            'min': np.min(fhe_accuracies),
            'max': np.max(fhe_accuracies),
            'median': np.median(fhe_accuracies)
        }
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(pt_accuracies) - 1) * np.var(pt_accuracies) + 
                             (len(fhe_accuracies) - 1) * np.var(fhe_accuracies)) / 
                            (len(pt_accuracies) + len(fhe_accuracies) - 2))
        
        cohens_d = (np.mean(fhe_accuracies) - np.mean(pt_accuracies)) / pooled_std
        
        analysis['effect_size'] = {
            'cohens_d': cohens_d,
            'interpretation': self._interpret_effect_size(cohens_d)
        }
        
        # Confidence intervals
        pt_ci = stats.t.interval(0.95, len(pt_accuracies)-1, 
                                loc=np.mean(pt_accuracies), 
                                scale=stats.sem(pt_accuracies))
        fhe_ci = stats.t.interval(0.95, len(fhe_accuracies)-1, 
                                 loc=np.mean(fhe_accuracies), 
                                 scale=stats.sem(fhe_accuracies))
        
        analysis['confidence_intervals'] = {
            'plaintext_accuracy_ci': pt_ci,
            'fhe_accuracy_ci': fhe_ci
        }
        
        return analysis
    
    def _interpret_effect_size(self, cohens_d: float) -> str:
        """Interpret Cohen's d effect size"""
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def _generate_recommendations(self, metrics: ComparisonMetrics, 
                                plaintext_result: ExperimentResult, 
                                fhe_result: ExperimentResult) -> List[str]:
        """Generate recommendations based on comparison results"""
        recommendations = []
        
        # Performance recommendations
        if metrics.accuracy_diff > 0.05:
            recommendations.append("✅ FHE CKKS shows superior accuracy performance - consider for production")
        elif metrics.accuracy_diff < -0.05:
            recommendations.append("⚠️  FHE CKKS shows accuracy degradation - evaluate privacy trade-offs")
        else:
            recommendations.append("📊 FHE CKKS maintains comparable accuracy to plain text")
        
        # Timing recommendations
        if metrics.training_time_overhead > 100:
            recommendations.append("⏱️  High training time overhead - consider optimization or hardware acceleration")
        elif metrics.training_time_overhead < 50:
            recommendations.append("⚡ Acceptable training time overhead for privacy benefits")
        
        # Statistical significance
        if metrics.statistical_significance:
            recommendations.append("📈 Results are statistically significant - reliable comparison")
        else:
            recommendations.append("📊 Consider more runs for statistical significance")
        
        # Privacy vs Performance trade-off
        if metrics.trade_off_ratio > 2.0:
            recommendations.append("🔒 Excellent privacy-to-performance ratio - ideal for sensitive data")
        elif metrics.trade_off_ratio > 1.0:
            recommendations.append("⚖️  Good privacy-to-performance balance - suitable for most use cases")
        else:
            recommendations.append("🤔 Consider if privacy benefits justify performance costs")
        
        # Use case recommendations
        if metrics.privacy_score == 1.0 and metrics.performance_score > 0.8:
            recommendations.append("🏆 FHE CKKS recommended for healthcare, finance, and sensitive applications")
        
        return recommendations
    
    def _generate_conclusion(self, metrics: ComparisonMetrics, recommendations: List[str]) -> str:
        """Generate overall conclusion"""
        conclusion_parts = []
        
        # Performance conclusion
        if metrics.accuracy_diff > 0:
            conclusion_parts.append(f"FHE CKKS achieves {metrics.accuracy_diff:.3f} higher accuracy")
        else:
            conclusion_parts.append(f"FHE CKKS shows {abs(metrics.accuracy_diff):.3f} accuracy reduction")
        
        # Timing conclusion
        conclusion_parts.append(f"with {metrics.training_time_overhead:.1f}% training time overhead")
        
        # Privacy conclusion
        conclusion_parts.append("while providing complete privacy protection")
        
        # Statistical conclusion
        if metrics.statistical_significance:
            conclusion_parts.append("(statistically significant results)")
        else:
            conclusion_parts.append("(results not statistically significant)")
        
        return " ".join(conclusion_parts) + "."
    
    def _save_comparison_results(self, result: ComparisonResult) -> None:
        """Save comprehensive comparison results"""
        # Save main comparison result
        result_dict = {
            'comparison_id': result.comparison_id,
            'timestamp': result.timestamp.isoformat(),
            'metrics': asdict(result.metrics),
            'recommendations': result.recommendations,
            'conclusion': result.conclusion,
            'statistical_analysis': result.statistical_analysis,
            'round_by_round_comparison': result.round_by_round_comparison
        }
        
        with open(f"{self.results_dir}/comparison_result.json", 'w') as f:
            json.dump(result_dict, f, indent=2)
        
        # Save individual experiment results
        with open(f"{self.results_dir}/plaintext_result.json", 'w') as f:
            json.dump(asdict(result.plaintext_result), f, indent=2, default=str)
        
        with open(f"{self.results_dir}/fhe_result.json", 'w') as f:
            json.dump(asdict(result.fhe_result), f, indent=2, default=str)
        
        print(f"💾 Comparison results saved to: {self.results_dir}")
    
    def _generate_comparison_visualizations(self, result: ComparisonResult) -> None:
        """Generate comprehensive comparison visualizations"""
        print("📊 Generating comparison visualizations...")
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('FHE CKKS vs Plain Text Federated Learning Comparison', fontsize=16, fontweight='bold')
        
        # 1. Accuracy comparison over rounds
        rounds = list(range(1, len(result.plaintext_result.round_results) + 1))
        pt_accuracies = [r.accuracy for r in result.plaintext_result.round_results]
        fhe_accuracies = [r.accuracy for r in result.fhe_result.round_results]
        
        axes[0, 0].plot(rounds, pt_accuracies, 'o-', label='Plain Text', linewidth=2, markersize=6)
        axes[0, 0].plot(rounds, fhe_accuracies, 's-', label='FHE CKKS', linewidth=2, markersize=6)
        axes[0, 0].set_xlabel('Federated Learning Round')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_title('Accuracy Over Rounds')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. F1 Score comparison
        pt_f1_scores = [r.f1_score for r in result.plaintext_result.round_results]
        fhe_f1_scores = [r.f1_score for r in result.fhe_result.round_results]
        
        axes[0, 1].plot(rounds, pt_f1_scores, 'o-', label='Plain Text', linewidth=2, markersize=6)
        axes[0, 1].plot(rounds, fhe_f1_scores, 's-', label='FHE CKKS', linewidth=2, markersize=6)
        axes[0, 1].set_xlabel('Federated Learning Round')
        axes[0, 1].set_ylabel('F1 Score')
        axes[0, 1].set_title('F1 Score Over Rounds')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Training time comparison
        pt_training_times = [r.training_time for r in result.plaintext_result.round_results]
        fhe_training_times = [r.training_time for r in result.fhe_result.round_results]
        
        axes[0, 2].plot(rounds, pt_training_times, 'o-', label='Plain Text', linewidth=2, markersize=6)
        axes[0, 2].plot(rounds, fhe_training_times, 's-', label='FHE CKKS', linewidth=2, markersize=6)
        axes[0, 2].set_xlabel('Federated Learning Round')
        axes[0, 2].set_ylabel('Training Time (seconds)')
        axes[0, 2].set_title('Training Time Over Rounds')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Performance metrics comparison (bar chart)
        metrics_names = ['Accuracy', 'F1 Score', 'Precision', 'Recall']
        pt_metrics = [
            result.plaintext_result.final_accuracy,
            result.plaintext_result.round_results[-1].f1_score,
            result.plaintext_result.round_results[-1].precision,
            result.plaintext_result.round_results[-1].recall
        ]
        fhe_metrics = [
            result.fhe_result.final_accuracy,
            result.fhe_result.round_results[-1].f1_score,
            result.fhe_result.round_results[-1].precision,
            result.fhe_result.round_results[-1].recall
        ]
        
        x = np.arange(len(metrics_names))
        width = 0.35
        
        axes[1, 0].bar(x - width/2, pt_metrics, width, label='Plain Text', alpha=0.8)
        axes[1, 0].bar(x + width/2, fhe_metrics, width, label='FHE CKKS', alpha=0.8)
        axes[1, 0].set_xlabel('Metrics')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].set_title('Final Performance Metrics Comparison')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(metrics_names)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Timing overhead analysis
        overhead_metrics = ['Training Time', 'Aggregation Time', 'Total Time']
        overhead_values = [
            result.metrics.training_time_overhead,
            result.metrics.aggregation_time_overhead,
            result.metrics.total_time_overhead
        ]
        
        colors = ['red' if x > 100 else 'orange' if x > 50 else 'green' for x in overhead_values]
        bars = axes[1, 1].bar(overhead_metrics, overhead_values, color=colors, alpha=0.7)
        axes[1, 1].set_ylabel('Overhead (%)')
        axes[1, 1].set_title('FHE CKKS Overhead Analysis')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, overhead_values):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                          f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 6. Privacy vs Performance trade-off
        privacy_scores = [0.0, 1.0]  # Plain Text = 0, FHE = 1
        performance_scores = [result.plaintext_result.final_accuracy, result.fhe_result.final_accuracy]
        
        axes[1, 2].scatter(privacy_scores, performance_scores, s=200, alpha=0.7, 
                          c=['blue', 'red'], edgecolors='black', linewidth=2)
        axes[1, 2].set_xlabel('Privacy Score (0=No Privacy, 1=Full Privacy)')
        axes[1, 2].set_ylabel('Performance Score (Accuracy)')
        axes[1, 2].set_title('Privacy vs Performance Trade-off')
        axes[1, 2].set_xlim(-0.1, 1.1)
        axes[1, 2].grid(True, alpha=0.3)
        
        # Add labels for points
        axes[1, 2].annotate('Plain Text', (privacy_scores[0], performance_scores[0]), 
                          xytext=(10, 10), textcoords='offset points', fontweight='bold')
        axes[1, 2].annotate('FHE CKKS', (privacy_scores[1], performance_scores[1]), 
                          xytext=(10, 10), textcoords='offset points', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/comprehensive_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Visualizations saved to: {self.results_dir}/comprehensive_comparison.png")
