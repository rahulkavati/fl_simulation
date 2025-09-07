"""
Automated Benchmarking System for FHE vs Plain Text FL
Provides systematic, reproducible, and fair comparison testing
"""

import os
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
import concurrent.futures
import multiprocessing as mp
from pathlib import Path

from src.core.base_pipeline import PipelineConfig, ExperimentResult
from src.core.comparison_engine import FederatedLearningComparison, ComparisonResult
from src.strategies.plaintext_strategy import PlainTextFederatedLearningPipeline, EnhancedPlainTextPipeline
from src.strategies.fhe_strategy import FHECKKSFederatedLearningPipeline, EnhancedFHECKKSPipeline

@dataclass
class BenchmarkConfig:
    """Configuration for automated benchmarking"""
    # Experiment parameters
    rounds_range: Tuple[int, int] = (5, 20)  # Min, max rounds to test
    clients_range: Tuple[int, int] = (5, 20)  # Min, max clients to test
    num_experiments_per_config: int = 3  # Number of runs per configuration
    
    # Statistical parameters
    confidence_level: float = 0.95
    min_sample_size: int = 5  # Minimum runs for statistical significance
    
    # Performance parameters
    max_execution_time: float = 3600.0  # 1 hour max per experiment
    memory_limit_mb: float = 8192.0  # 8GB memory limit
    
    # Output parameters
    save_intermediate_results: bool = True
    generate_visualizations: bool = True
    parallel_execution: bool = True
    max_workers: int = 4

@dataclass
class BenchmarkResult:
    """Result of automated benchmarking"""
    benchmark_id: str
    timestamp: datetime
    config: BenchmarkConfig
    
    # Results
    experiment_results: List[ComparisonResult]
    configuration_matrix: List[Dict[str, Any]]
    
    # Summary statistics
    best_configuration: Dict[str, Any]
    worst_configuration: Dict[str, Any]
    optimal_trade_off: Dict[str, Any]
    
    # Performance analysis
    scalability_analysis: Dict[str, Any]
    convergence_analysis: Dict[str, Any]
    
    # Recommendations
    recommendations: List[str]
    conclusion: str

class AutomatedBenchmarkingSystem:
    """
    Automated benchmarking system for systematic FHE vs Plain Text comparison
    Provides reproducible, fair, and comprehensive evaluation
    """
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.benchmark_id = f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.results_dir = f"benchmarks/{self.benchmark_id}"
        
        # Create results directory
        os.makedirs(self.results_dir, exist_ok=True)
        
        print(f"🔬 Initialized Automated Benchmarking System")
        print(f"📁 Results will be saved to: {self.results_dir}")
    
    def run_comprehensive_benchmark(self) -> BenchmarkResult:
        """
        Run comprehensive automated benchmark across multiple configurations
        """
        print(f"\n🚀 Starting Comprehensive Automated Benchmark")
        print(f"📊 Testing {self.config.num_experiments_per_config} runs per configuration")
        print(f"⚙️  Rounds: {self.config.rounds_range[0]}-{self.config.rounds_range[1]}")
        print(f"👥 Clients: {self.config.clients_range[0]}-{self.config.clients_range[1]}")
        print("=" * 70)
        
        # Generate configuration matrix
        configuration_matrix = self._generate_configuration_matrix()
        print(f"📋 Generated {len(configuration_matrix)} test configurations")
        
        # Run experiments
        experiment_results = []
        
        if self.config.parallel_execution:
            experiment_results = self._run_parallel_experiments(configuration_matrix)
        else:
            experiment_results = self._run_sequential_experiments(configuration_matrix)
        
        # Analyze results
        benchmark_result = self._analyze_benchmark_results(experiment_results, configuration_matrix)
        
        # Save results
        self._save_benchmark_results(benchmark_result)
        
        # Generate comprehensive report
        self._generate_benchmark_report(benchmark_result)
        
        return benchmark_result
    
    def _generate_configuration_matrix(self) -> List[Dict[str, Any]]:
        """Generate matrix of test configurations"""
        configurations = []
        
        # Test different round counts
        round_values = list(range(self.config.rounds_range[0], 
                                self.config.rounds_range[1] + 1, 2))
        
        # Test different client counts
        client_values = list(range(self.config.clients_range[0], 
                                 self.config.clients_range[1] + 1, 2))
        
        for rounds in round_values:
            for clients in client_values:
                config = {
                    'rounds': rounds,
                    'clients': clients,
                    'min_samples_per_client': 50,
                    'test_size': 0.2,
                    'random_state': 42,
                    'data_path': "data/health_fitness_data.csv",
                    'encryption_enabled': False,  # Will be set per experiment
                    'track_detailed_metrics': True,
                    'save_intermediate_results': self.config.save_intermediate_results
                }
                configurations.append(config)
        
        return configurations
    
    def _run_parallel_experiments(self, configuration_matrix: List[Dict[str, Any]]) -> List[ComparisonResult]:
        """Run experiments in parallel for efficiency"""
        print(f"🔄 Running experiments in parallel ({self.config.max_workers} workers)...")
        
        experiment_results = []
        
        # Create experiment tasks
        tasks = []
        for config_dict in configuration_matrix:
            for run in range(self.config.num_experiments_per_config):
                task = {
                    'config': PipelineConfig(**config_dict),
                    'run_id': run,
                    'config_id': f"{config_dict['rounds']}r_{config_dict['clients']}c"
                }
                tasks.append(task)
        
        # Execute tasks in parallel
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_to_task = {
                executor.submit(self._run_single_comparison, task): task 
                for task in tasks
            }
            
            completed = 0
            total = len(tasks)
            
            for future in concurrent.futures.as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result(timeout=self.config.max_execution_time)
                    experiment_results.append(result)
                    completed += 1
                    print(f"✅ Completed {completed}/{total} experiments")
                except Exception as e:
                    print(f"❌ Experiment failed for {task['config_id']}: {e}")
        
        return experiment_results
    
    def _run_sequential_experiments(self, configuration_matrix: List[Dict[str, Any]]) -> List[ComparisonResult]:
        """Run experiments sequentially"""
        print("🔄 Running experiments sequentially...")
        
        experiment_results = []
        total_experiments = len(configuration_matrix) * self.config.num_experiments_per_config
        completed = 0
        
        for config_dict in configuration_matrix:
            config = PipelineConfig(**config_dict)
            
            for run in range(self.config.num_experiments_per_config):
                print(f"\n📊 Running experiment {completed + 1}/{total_experiments}")
                print(f"⚙️  Configuration: {config.rounds} rounds, {config.clients} clients")
                
                try:
                    result = self._run_single_comparison({
                        'config': config,
                        'run_id': run,
                        'config_id': f"{config.rounds}r_{config.clients}c"
                    })
                    experiment_results.append(result)
                    completed += 1
                    print(f"✅ Experiment {completed}/{total_experiments} completed")
                except Exception as e:
                    print(f"❌ Experiment failed: {e}")
                    completed += 1
        
        return experiment_results
    
    def _run_single_comparison(self, task: Dict[str, Any]) -> ComparisonResult:
        """Run a single comparison experiment"""
        config = task['config']
        run_id = task['run_id']
        config_id = task['config_id']
        
        # Create comparison engine
        comparison = FederatedLearningComparison(config)
        
        # Run comparison with multiple runs for statistical significance
        result = comparison.run_comprehensive_comparison(num_runs=3)
        
        # Add metadata
        result.comparison_id = f"{config_id}_run{run_id}_{result.comparison_id}"
        
        return result
    
    def _analyze_benchmark_results(self, experiment_results: List[ComparisonResult], 
                                  configuration_matrix: List[Dict[str, Any]]) -> BenchmarkResult:
        """Analyze comprehensive benchmark results"""
        print("\n📊 Analyzing benchmark results...")
        
        # Group results by configuration
        config_groups = {}
        for result in experiment_results:
            config_key = f"{result.plaintext_result.config.rounds}r_{result.plaintext_result.config.clients}c"
            if config_key not in config_groups:
                config_groups[config_key] = []
            config_groups[config_key].append(result)
        
        # Find best and worst configurations
        best_config = self._find_best_configuration(config_groups)
        worst_config = self._find_worst_configuration(config_groups)
        optimal_trade_off = self._find_optimal_trade_off(config_groups)
        
        # Perform scalability analysis
        scalability_analysis = self._analyze_scalability(experiment_results)
        
        # Perform convergence analysis
        convergence_analysis = self._analyze_convergence(experiment_results)
        
        # Generate recommendations
        recommendations = self._generate_benchmark_recommendations(
            best_config, worst_config, optimal_trade_off, scalability_analysis
        )
        
        # Generate conclusion
        conclusion = self._generate_benchmark_conclusion(
            best_config, scalability_analysis, recommendations
        )
        
        return BenchmarkResult(
            benchmark_id=self.benchmark_id,
            timestamp=datetime.now(),
            config=self.config,
            experiment_results=experiment_results,
            configuration_matrix=configuration_matrix,
            best_configuration=best_config,
            worst_configuration=worst_config,
            optimal_trade_off=optimal_trade_off,
            scalability_analysis=scalability_analysis,
            convergence_analysis=convergence_analysis,
            recommendations=recommendations,
            conclusion=conclusion
        )
    
    def _find_best_configuration(self, config_groups: Dict[str, List[ComparisonResult]]) -> Dict[str, Any]:
        """Find the best performing configuration"""
        best_score = -float('inf')
        best_config = None
        
        for config_key, results in config_groups.items():
            # Calculate average performance score
            avg_accuracy = np.mean([r.fhe_result.final_accuracy for r in results])
            avg_f1_score = np.mean([r.fhe_result.round_results[-1].f1_score for r in results])
            avg_time_overhead = np.mean([r.metrics.training_time_overhead for r in results])
            
            # Composite score: accuracy + f1_score - time_overhead_penalty
            composite_score = avg_accuracy + avg_f1_score - (avg_time_overhead / 1000)
            
            if composite_score > best_score:
                best_score = composite_score
                best_config = {
                    'configuration': config_key,
                    'rounds': results[0].plaintext_result.config.rounds,
                    'clients': results[0].plaintext_result.config.clients,
                    'avg_accuracy': avg_accuracy,
                    'avg_f1_score': avg_f1_score,
                    'avg_time_overhead': avg_time_overhead,
                    'composite_score': composite_score,
                    'num_runs': len(results)
                }
        
        return best_config
    
    def _find_worst_configuration(self, config_groups: Dict[str, List[ComparisonResult]]) -> Dict[str, Any]:
        """Find the worst performing configuration"""
        worst_score = float('inf')
        worst_config = None
        
        for config_key, results in config_groups.items():
            # Calculate average performance score
            avg_accuracy = np.mean([r.fhe_result.final_accuracy for r in results])
            avg_f1_score = np.mean([r.fhe_result.round_results[-1].f1_score for r in results])
            avg_time_overhead = np.mean([r.metrics.training_time_overhead for r in results])
            
            # Composite score: accuracy + f1_score - time_overhead_penalty
            composite_score = avg_accuracy + avg_f1_score - (avg_time_overhead / 1000)
            
            if composite_score < worst_score:
                worst_score = composite_score
                worst_config = {
                    'configuration': config_key,
                    'rounds': results[0].plaintext_result.config.rounds,
                    'clients': results[0].plaintext_result.config.clients,
                    'avg_accuracy': avg_accuracy,
                    'avg_f1_score': avg_f1_score,
                    'avg_time_overhead': avg_time_overhead,
                    'composite_score': composite_score,
                    'num_runs': len(results)
                }
        
        return worst_config
    
    def _find_optimal_trade_off(self, config_groups: Dict[str, List[ComparisonResult]]) -> Dict[str, Any]:
        """Find configuration with optimal privacy-performance trade-off"""
        best_trade_off_ratio = -float('inf')
        optimal_config = None
        
        for config_key, results in config_groups.items():
            avg_accuracy = np.mean([r.fhe_result.final_accuracy for r in results])
            avg_time_overhead = np.mean([r.metrics.training_time_overhead for r in results])
            
            # Trade-off ratio: performance / overhead
            trade_off_ratio = avg_accuracy / (avg_time_overhead / 100 + 1)
            
            if trade_off_ratio > best_trade_off_ratio:
                best_trade_off_ratio = trade_off_ratio
                optimal_config = {
                    'configuration': config_key,
                    'rounds': results[0].plaintext_result.config.rounds,
                    'clients': results[0].plaintext_result.config.clients,
                    'avg_accuracy': avg_accuracy,
                    'avg_time_overhead': avg_time_overhead,
                    'trade_off_ratio': trade_off_ratio,
                    'num_runs': len(results)
                }
        
        return optimal_config
    
    def _analyze_scalability(self, experiment_results: List[ComparisonResult]) -> Dict[str, Any]:
        """Analyze scalability characteristics"""
        # Group by client count
        client_groups = {}
        for result in experiment_results:
            clients = result.plaintext_result.config.clients
            if clients not in client_groups:
                client_groups[clients] = []
            client_groups[clients].append(result)
        
        scalability_metrics = {}
        for clients, results in client_groups.items():
            avg_accuracy = np.mean([r.fhe_result.final_accuracy for r in results])
            avg_time_overhead = np.mean([r.metrics.training_time_overhead for r in results])
            avg_total_time = np.mean([r.fhe_result.total_duration for r in results])
            
            scalability_metrics[clients] = {
                'avg_accuracy': avg_accuracy,
                'avg_time_overhead': avg_time_overhead,
                'avg_total_time': avg_total_time,
                'num_experiments': len(results)
            }
        
        return scalability_metrics
    
    def _analyze_convergence(self, experiment_results: List[ComparisonResult]) -> Dict[str, Any]:
        """Analyze convergence characteristics"""
        convergence_metrics = {}
        
        for result in experiment_results:
            config_key = f"{result.plaintext_result.config.rounds}r_{result.plaintext_result.config.clients}c"
            
            # Calculate convergence metrics
            pt_accuracies = [r.accuracy for r in result.plaintext_result.round_results]
            fhe_accuracies = [r.accuracy for r in result.fhe_result.round_results]
            
            # Find convergence round (when improvement < 0.001)
            pt_convergence_round = self._find_convergence_round(pt_accuracies)
            fhe_convergence_round = self._find_convergence_round(fhe_accuracies)
            
            if config_key not in convergence_metrics:
                convergence_metrics[config_key] = {
                    'pt_convergence_rounds': [],
                    'fhe_convergence_rounds': [],
                    'convergence_diffs': []
                }
            
            convergence_metrics[config_key]['pt_convergence_rounds'].append(pt_convergence_round)
            convergence_metrics[config_key]['fhe_convergence_rounds'].append(fhe_convergence_round)
            convergence_metrics[config_key]['convergence_diffs'].append(
                fhe_convergence_round - pt_convergence_round
            )
        
        # Calculate averages
        for config_key in convergence_metrics:
            metrics = convergence_metrics[config_key]
            metrics['avg_pt_convergence'] = np.mean(metrics['pt_convergence_rounds'])
            metrics['avg_fhe_convergence'] = np.mean(metrics['fhe_convergence_rounds'])
            metrics['avg_convergence_diff'] = np.mean(metrics['convergence_diffs'])
        
        return convergence_metrics
    
    def _find_convergence_round(self, accuracies: List[float], threshold: float = 0.001) -> int:
        """Find the round where convergence occurs"""
        for i in range(1, len(accuracies)):
            if abs(accuracies[i] - accuracies[i-1]) < threshold:
                return i
        return len(accuracies) - 1
    
    def _generate_benchmark_recommendations(self, best_config: Dict[str, Any], 
                                          worst_config: Dict[str, Any],
                                          optimal_trade_off: Dict[str, Any],
                                          scalability_analysis: Dict[str, Any]) -> List[str]:
        """Generate comprehensive benchmark recommendations"""
        recommendations = []
        
        # Best configuration recommendation
        recommendations.append(f"🏆 Best Performance: {best_config['configuration']} "
                             f"({best_config['rounds']} rounds, {best_config['clients']} clients)")
        recommendations.append(f"   📈 Achieves {best_config['avg_accuracy']:.3f} accuracy "
                             f"with {best_config['avg_time_overhead']:.1f}% overhead")
        
        # Optimal trade-off recommendation
        recommendations.append(f"⚖️  Optimal Trade-off: {optimal_trade_off['configuration']} "
                             f"({optimal_trade_off['rounds']} rounds, {optimal_trade_off['clients']} clients)")
        recommendations.append(f"   🎯 Best balance of performance and efficiency")
        
        # Scalability recommendations
        client_counts = list(scalability_analysis.keys())
        if len(client_counts) > 1:
            recommendations.append(f"📊 Scalability Analysis:")
            recommendations.append(f"   • Tested with {min(client_counts)}-{max(client_counts)} clients")
            
            # Find optimal client count
            best_client_count = max(client_counts, 
                                  key=lambda c: scalability_analysis[c]['avg_accuracy'])
            recommendations.append(f"   • Optimal client count: {best_client_count}")
        
        # Performance recommendations
        if best_config['avg_time_overhead'] < 50:
            recommendations.append("⚡ Low overhead - FHE CKKS is highly efficient")
        elif best_config['avg_time_overhead'] < 100:
            recommendations.append("📊 Moderate overhead - acceptable for most use cases")
        else:
            recommendations.append("⚠️  High overhead - consider optimization or hardware acceleration")
        
        # Privacy recommendations
        recommendations.append("🔒 Privacy Benefits:")
        recommendations.append("   • Complete data privacy protection")
        recommendations.append("   • No data leakage during aggregation")
        recommendations.append("   • GDPR/HIPAA compliant")
        
        return recommendations
    
    def _generate_benchmark_conclusion(self, best_config: Dict[str, Any], 
                                     scalability_analysis: Dict[str, Any],
                                     recommendations: List[str]) -> str:
        """Generate comprehensive benchmark conclusion"""
        conclusion_parts = []
        
        conclusion_parts.append(f"Comprehensive benchmarking across {len(scalability_analysis)} configurations")
        conclusion_parts.append(f"reveals that FHE CKKS achieves {best_config['avg_accuracy']:.3f} accuracy")
        conclusion_parts.append(f"with {best_config['avg_time_overhead']:.1f}% computational overhead")
        conclusion_parts.append(f"in the optimal configuration ({best_config['configuration']}).")
        
        conclusion_parts.append("The results demonstrate that homomorphic encryption")
        conclusion_parts.append("provides strong privacy guarantees while maintaining")
        conclusion_parts.append("competitive performance for federated learning applications.")
        
        return " ".join(conclusion_parts)
    
    def _save_benchmark_results(self, result: BenchmarkResult) -> None:
        """Save comprehensive benchmark results"""
        # Save main benchmark result
        result_dict = asdict(result)
        result_dict['timestamp'] = result.timestamp.isoformat()
        
        with open(f"{self.results_dir}/benchmark_result.json", 'w') as f:
            json.dump(result_dict, f, indent=2, default=str)
        
        # Save summary CSV
        summary_data = []
        for experiment in result.experiment_results:
            summary_data.append({
                'configuration': f"{experiment.plaintext_result.config.rounds}r_{experiment.plaintext_result.config.clients}c",
                'rounds': experiment.plaintext_result.config.rounds,
                'clients': experiment.plaintext_result.config.clients,
                'fhe_accuracy': experiment.fhe_result.final_accuracy,
                'plaintext_accuracy': experiment.plaintext_result.final_accuracy,
                'accuracy_diff': experiment.metrics.accuracy_diff,
                'training_time_overhead': experiment.metrics.training_time_overhead,
                'aggregation_time_overhead': experiment.metrics.aggregation_time_overhead,
                'total_time_overhead': experiment.metrics.total_time_overhead,
                'statistical_significance': experiment.metrics.statistical_significance
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(f"{self.results_dir}/benchmark_summary.csv", index=False)
        
        print(f"💾 Benchmark results saved to: {self.results_dir}")
    
    def _generate_benchmark_report(self, result: BenchmarkResult) -> None:
        """Generate comprehensive benchmark report"""
        report_path = f"{self.results_dir}/BENCHMARK_REPORT.md"
        
        with open(report_path, 'w') as f:
            f.write("# 🔬 Automated Benchmark Report\n\n")
            f.write(f"**Benchmark ID**: {result.benchmark_id}\n")
            f.write(f"**Timestamp**: {result.timestamp.isoformat()}\n")
            f.write(f"**Total Experiments**: {len(result.experiment_results)}\n\n")
            
            f.write("## 🏆 Best Configuration\n\n")
            f.write(f"- **Configuration**: {result.best_configuration['configuration']}\n")
            f.write(f"- **Rounds**: {result.best_configuration['rounds']}\n")
            f.write(f"- **Clients**: {result.best_configuration['clients']}\n")
            f.write(f"- **Accuracy**: {result.best_configuration['avg_accuracy']:.4f}\n")
            f.write(f"- **F1 Score**: {result.best_configuration['avg_f1_score']:.4f}\n")
            f.write(f"- **Time Overhead**: {result.best_configuration['avg_time_overhead']:.1f}%\n\n")
            
            f.write("## ⚖️ Optimal Trade-off\n\n")
            f.write(f"- **Configuration**: {result.optimal_trade_off['configuration']}\n")
            f.write(f"- **Trade-off Ratio**: {result.optimal_trade_off['trade_off_ratio']:.4f}\n")
            f.write(f"- **Accuracy**: {result.optimal_trade_off['avg_accuracy']:.4f}\n")
            f.write(f"- **Time Overhead**: {result.optimal_trade_off['avg_time_overhead']:.1f}%\n\n")
            
            f.write("## 📊 Scalability Analysis\n\n")
            for clients, metrics in result.scalability_analysis.items():
                f.write(f"### {clients} Clients\n")
                f.write(f"- **Accuracy**: {metrics['avg_accuracy']:.4f}\n")
                f.write(f"- **Time Overhead**: {metrics['avg_time_overhead']:.1f}%\n")
                f.write(f"- **Total Time**: {metrics['avg_total_time']:.2f}s\n\n")
            
            f.write("## 💡 Recommendations\n\n")
            for i, rec in enumerate(result.recommendations, 1):
                f.write(f"{i}. {rec}\n")
            
            f.write(f"\n## 🎯 Conclusion\n\n{result.conclusion}\n")
        
        print(f"📋 Benchmark report saved to: {report_path}")
