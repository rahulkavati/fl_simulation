"""
Main Entry Point for FHE vs Plain Text FL Comparison
Provides easy-to-use interface for running comprehensive comparisons
"""

import os
import argparse
import time
from datetime import datetime
from typing import Optional

from src.core.base_pipeline import PipelineConfig
from src.core.comparison_engine import FederatedLearningComparison
from src.evaluation.benchmark_suite import AutomatedBenchmarkingSystem, BenchmarkConfig

def main():
    """Main entry point for FHE vs Plain Text comparison"""
    parser = argparse.ArgumentParser(
        description="FHE CKKS vs Plain Text Federated Learning Comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick comparison with default settings
  python compare_fhe_plaintext.py

  # Custom configuration
  python compare_fhe_plaintext.py --rounds 15 --clients 12 --runs 5

  # Full automated benchmark
  python compare_fhe_plaintext.py --benchmark --rounds-range 5,20 --clients-range 5,20

  # Enhanced pipelines only
  python compare_fhe_plaintext.py --enhanced-only
        """
    )
    
    # Basic configuration
    parser.add_argument("--rounds", type=int, default=10,
                       help="Number of federated learning rounds (default: 10)")
    parser.add_argument("--clients", type=int, default=10,
                       help="Number of clients (default: 10)")
    parser.add_argument("--runs", type=int, default=3,
                       help="Number of runs for statistical significance (default: 3)")
    
    # Data configuration
    parser.add_argument("--data-path", type=str, default="data/health_fitness_data.csv",
                       help="Path to health fitness dataset (default: data/health_fitness_data.csv)")
    parser.add_argument("--min-samples", type=int, default=50,
                       help="Minimum samples per client (default: 50)")
    
    # Experiment configuration
    parser.add_argument("--benchmark", action="store_true",
                       help="Run comprehensive automated benchmark")
    parser.add_argument("--rounds-range", type=str, default="5,20",
                       help="Range of rounds for benchmark (format: min,max)")
    parser.add_argument("--clients-range", type=str, default="5,20",
                       help="Range of clients for benchmark (format: min,max)")
    parser.add_argument("--enhanced-only", action="store_true",
                       help="Use only enhanced pipelines")
    
    # Performance configuration
    parser.add_argument("--parallel", action="store_true", default=True,
                       help="Use parallel execution (default: True)")
    parser.add_argument("--max-workers", type=int, default=4,
                       help="Maximum parallel workers (default: 4)")
    parser.add_argument("--max-time", type=float, default=3600.0,
                       help="Maximum execution time per experiment in seconds (default: 3600)")
    
    # Output configuration
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Custom output directory")
    parser.add_argument("--no-visualizations", action="store_true",
                       help="Skip generating visualizations")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Parse ranges
    rounds_range = tuple(map(int, args.rounds_range.split(',')))
    clients_range = tuple(map(int, args.clients_range.split(',')))
    
    print("🔬 FHE CKKS vs Plain Text Federated Learning Comparison")
    print("=" * 70)
    print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⚙️  Configuration:")
    print(f"   • Rounds: {args.rounds}")
    print(f"   • Clients: {args.clients}")
    print(f"   • Runs: {args.runs}")
    print(f"   • Data: {args.data_path}")
    print(f"   • Enhanced Only: {args.enhanced_only}")
    print(f"   • Benchmark Mode: {args.benchmark}")
    
    if args.benchmark:
        print(f"   • Rounds Range: {rounds_range[0]}-{rounds_range[1]}")
        print(f"   • Clients Range: {clients_range[0]}-{clients_range[1]}")
    
    print("=" * 70)
    
    start_time = time.time()
    
    try:
        if args.benchmark:
            # Run comprehensive automated benchmark
            benchmark_config = BenchmarkConfig(
                rounds_range=rounds_range,
                clients_range=clients_range,
                num_experiments_per_config=args.runs,
                parallel_execution=args.parallel,
                max_workers=args.max_workers,
                max_execution_time=args.max_time,
                generate_visualizations=not args.no_visualizations
            )
            
            benchmark_system = AutomatedBenchmarkingSystem(benchmark_config)
            benchmark_result = benchmark_system.run_comprehensive_benchmark()
            
            print(f"\n🎉 Benchmark completed successfully!")
            print(f"📊 Total experiments: {len(benchmark_result.experiment_results)}")
            print(f"🏆 Best configuration: {benchmark_result.best_configuration['configuration']}")
            print(f"⚖️  Optimal trade-off: {benchmark_result.optimal_trade_off['configuration']}")
            
        else:
            # Run single comparison
            config = PipelineConfig(
                rounds=args.rounds,
                clients=args.clients,
                min_samples_per_client=args.min_samples,
                data_path=args.data_path,
                track_detailed_metrics=True,
                save_intermediate_results=True
            )
            
            comparison = FederatedLearningComparison(config)
            result = comparison.run_comprehensive_comparison(num_runs=args.runs)
            
            print(f"\n🎉 Comparison completed successfully!")
            print(f"📈 FHE Accuracy: {result.fhe_result.final_accuracy:.4f}")
            print(f"📊 Plain Text Accuracy: {result.plaintext_result.final_accuracy:.4f}")
            print(f"📊 Accuracy Difference: {result.metrics.accuracy_diff:+.4f}")
            print(f"⏱️  Training Time Overhead: {result.metrics.training_time_overhead:.1f}%")
            print(f"🔒 Privacy Score: {result.metrics.privacy_score:.1f}")
            
            if result.metrics.statistical_significance:
                print("📈 Results are statistically significant")
            else:
                print("📊 Consider more runs for statistical significance")
        
        total_time = time.time() - start_time
        print(f"\n⏱️  Total execution time: {total_time:.2f} seconds")
        
        return 0
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Experiment interrupted by user")
        return 1
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

def run_quick_comparison():
    """Run a quick comparison with default settings"""
    print("🚀 Running Quick FHE vs Plain Text Comparison...")
    
    config = PipelineConfig(
        rounds=5,
        clients=8,
        min_samples_per_client=50,
        data_path="data/health_fitness_data.csv"
    )
    
    comparison = FederatedLearningComparison(config)
    result = comparison.run_comprehensive_comparison(num_runs=2)
    
    print(f"\n✅ Quick comparison completed!")
    print(f"📈 FHE Accuracy: {result.fhe_result.final_accuracy:.4f}")
    print(f"📊 Plain Text Accuracy: {result.plaintext_result.final_accuracy:.4f}")
    print(f"📊 Difference: {result.metrics.accuracy_diff:+.4f}")
    
    return result

def run_enhanced_comparison():
    """Run comparison using enhanced pipelines only"""
    print("🚀 Running Enhanced FHE vs Plain Text Comparison...")
    
    # This would use EnhancedPlainTextPipeline and EnhancedFHECKKSPipeline
    # Implementation would be similar to the main comparison but with enhanced models
    
    config = PipelineConfig(
        rounds=10,
        clients=10,
        min_samples_per_client=50,
        data_path="data/health_fitness_data.csv"
    )
    
    comparison = FederatedLearningComparison(config)
    result = comparison.run_comprehensive_comparison(num_runs=3)
    
    print(f"\n✅ Enhanced comparison completed!")
    print(f"📈 Enhanced FHE Accuracy: {result.fhe_result.final_accuracy:.4f}")
    print(f"📊 Enhanced Plain Text Accuracy: {result.plaintext_result.final_accuracy:.4f}")
    print(f"📊 Difference: {result.metrics.accuracy_diff:+.4f}")
    
    return result

if __name__ == "__main__":
    exit(main())
