#!/usr/bin/env python3
"""
Sequential vs Parallel FHE Federated Learning Comparison

This script compares the performance of:
1. Sequential FHE federated learning (original implementation)
2. Parallel FHE federated learning (edge device simulation)

Key Metrics Compared:
- Accuracy and model performance
- Timing performance (sequential vs parallel)
- Parallel efficiency and scaling
- Resource utilization

Usage:
    python compare_sequential_parallel_fl.py --rounds 10 --clients 20
    python compare_sequential_parallel_fl.py --rounds 5 --clients 10 --batch-size 2
"""

import argparse
import sys
import os
import time
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Any

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from federated_learning_pipeline import EnhancedFederatedLearningPipeline
from src.multiprocess import BatchCoordinator, BatchConfig
from src.fl import FLConfig, DataProcessor
from src.encryption import FHEConfig
from federated_learning_pipeline import EnhancedDataProcessor


def run_sequential_fhe_fl(clients_data: Dict[str, Any], fhe_config: FHEConfig, rounds: int) -> Dict[str, Any]:
    """Run sequential FHE federated learning"""
    print("🔄 Running Sequential FHE Federated Learning...")
    
    fl_config = FLConfig(rounds=rounds, clients=len(clients_data))
    pipeline = EnhancedFederatedLearningPipeline(fl_config, fhe_config)
    
    # Set the clients data directly
    pipeline.clients_data = clients_data
    
    start_time = time.time()
    round_results = pipeline.run_enhanced_federated_learning()
    total_time = time.time() - start_time
    
    # Calculate final statistics
    final_accuracy = round_results[-1]['accuracy']
    best_accuracy = max(r['accuracy'] for r in round_results)
    initial_accuracy = round_results[0]['accuracy']
    
    avg_encryption_time = np.mean([r['encryption_time'] for r in round_results])
    avg_aggregation_time = np.mean([r['aggregation_time'] for r in round_results])
    avg_round_time = np.mean([r['total_time'] for r in round_results])
    
    return {
        'type': 'sequential',
        'round_results': round_results,
        'final_stats': {
            'initial_accuracy': initial_accuracy,
            'final_accuracy': final_accuracy,
            'best_accuracy': best_accuracy,
            'accuracy_improvement': final_accuracy - initial_accuracy,
            'avg_encryption_time': avg_encryption_time,
            'avg_aggregation_time': avg_aggregation_time,
            'avg_round_time': avg_round_time,
            'total_pipeline_time': total_time,
            'total_rounds': len(round_results)
        }
    }


def run_parallel_fhe_fl(clients_data: Dict[str, Any], fhe_config: FHEConfig, 
                       rounds: int, batch_size: int, max_workers: int) -> Dict[str, Any]:
    """Run parallel FHE federated learning"""
    print("🚀 Running Parallel FHE Federated Learning...")
    
    batch_config = BatchConfig(
        batch_size=batch_size,
        max_workers=max_workers,
        timeout=30.0
    )
    
    coordinator = BatchCoordinator(batch_config)
    
    start_time = time.time()
    results = coordinator.run_parallel_federated_learning(
        clients_data=clients_data,
        fhe_config=fhe_config,
        rounds=rounds
    )
    total_time = time.time() - start_time
    
    # Add total pipeline time to final stats
    results['final_stats']['total_pipeline_time'] = total_time
    
    return {
        'type': 'parallel',
        'round_results': results['round_results'],
        'final_stats': results['final_stats'],
        'timing_stats': results['timing_stats'],
        'central_server_stats': results['central_server_stats']
    }


def compare_results(sequential_results: Dict[str, Any], parallel_results: Dict[str, Any]) -> Dict[str, Any]:
    """Compare sequential and parallel results"""
    seq_stats = sequential_results['final_stats']
    par_stats = parallel_results['final_stats']
    
    # Performance comparison
    performance_comparison = {
        'accuracy_difference': par_stats['final_accuracy'] - seq_stats['final_accuracy'],
        'best_accuracy_difference': par_stats['best_accuracy'] - seq_stats['best_accuracy'],
        'improvement_difference': par_stats['accuracy_improvement'] - seq_stats['accuracy_improvement']
    }
    
    # Timing comparison
    timing_comparison = {
        'total_time_speedup': seq_stats['total_pipeline_time'] / par_stats['total_pipeline_time'],
        'round_time_speedup': seq_stats['avg_round_time'] / par_stats['avg_round_time'],
        'encryption_time_ratio': par_stats['avg_encryption_time'] / seq_stats['avg_encryption_time'],
        'aggregation_time_ratio': par_stats['avg_aggregation_time'] / seq_stats['avg_aggregation_time']
    }
    
    # Parallel efficiency
    parallel_efficiency = {
        'avg_parallel_efficiency': par_stats.get('avg_parallel_efficiency', 0),
        'theoretical_max_efficiency': par_stats.get('avg_parallel_efficiency', 0) / 4.0,  # Assuming 4 workers
        'efficiency_percentage': (par_stats.get('avg_parallel_efficiency', 0) / 4.0) * 100
    }
    
    return {
        'performance_comparison': performance_comparison,
        'timing_comparison': timing_comparison,
        'parallel_efficiency': parallel_efficiency
    }


def print_comparison_summary(sequential_results: Dict[str, Any], 
                           parallel_results: Dict[str, Any], 
                           comparison: Dict[str, Any]):
    """Print detailed comparison summary"""
    print("\n" + "=" * 80)
    print("📊 SEQUENTIAL vs PARALLEL FHE FEDERATED LEARNING COMPARISON")
    print("=" * 80)
    
    seq_stats = sequential_results['final_stats']
    par_stats = parallel_results['final_stats']
    
    # Performance comparison
    print(f"🎯 Performance Comparison:")
    print(f"  Sequential Final Accuracy: {seq_stats['final_accuracy']:.4f} ({seq_stats['final_accuracy']*100:.2f}%)")
    print(f"  Parallel Final Accuracy:   {par_stats['final_accuracy']:.4f} ({par_stats['final_accuracy']*100:.2f}%)")
    print(f"  Accuracy Difference:       {comparison['performance_comparison']['accuracy_difference']:+.4f} ({comparison['performance_comparison']['accuracy_difference']*100:+.2f}%)")
    print()
    
    print(f"  Sequential Best Accuracy:  {seq_stats['best_accuracy']:.4f} ({seq_stats['best_accuracy']*100:.2f}%)")
    print(f"  Parallel Best Accuracy:    {par_stats['best_accuracy']:.4f} ({par_stats['best_accuracy']*100:.2f}%)")
    print(f"  Best Accuracy Difference: {comparison['performance_comparison']['best_accuracy_difference']:+.4f} ({comparison['performance_comparison']['best_accuracy_difference']*100:+.2f}%)")
    print()
    
    # Timing comparison
    print(f"⏱️ Timing Comparison:")
    print(f"  Sequential Total Time:     {seq_stats['total_pipeline_time']:.4f}s")
    print(f"  Parallel Total Time:       {par_stats['total_pipeline_time']:.4f}s")
    print(f"  Total Time Speedup:        {comparison['timing_comparison']['total_time_speedup']:.2f}x")
    print()
    
    print(f"  Sequential Avg Round Time: {seq_stats['avg_round_time']:.4f}s")
    print(f"  Parallel Avg Round Time:   {par_stats['avg_round_time']:.4f}s")
    print(f"  Round Time Speedup:        {comparison['timing_comparison']['round_time_speedup']:.2f}x")
    print()
    
    print(f"  Sequential Avg Encryption: {seq_stats['avg_encryption_time']:.4f}s")
    print(f"  Parallel Avg Encryption:   {par_stats['avg_encryption_time']:.4f}s")
    print(f"  Encryption Time Ratio:     {comparison['timing_comparison']['encryption_time_ratio']:.2f}x")
    print()
    
    print(f"  Sequential Avg Aggregation: {seq_stats['avg_aggregation_time']:.4f}s")
    print(f"  Parallel Avg Aggregation:  {par_stats['avg_aggregation_time']:.4f}s")
    print(f"  Aggregation Time Ratio:    {comparison['timing_comparison']['aggregation_time_ratio']:.2f}x")
    print()
    
    # Parallel efficiency
    print(f"🚀 Parallel Processing Efficiency:")
    print(f"  Average Parallel Efficiency: {comparison['parallel_efficiency']['avg_parallel_efficiency']:.2f}")
    print(f"  Theoretical Max Efficiency:  {comparison['parallel_efficiency']['theoretical_max_efficiency']:.2f}")
    print(f"  Efficiency Percentage:       {comparison['parallel_efficiency']['efficiency_percentage']:.1f}%")
    print()
    
    # Summary
    print(f"📈 Summary:")
    if comparison['timing_comparison']['total_time_speedup'] > 1.0:
        print(f"  ✅ Parallel processing is {comparison['timing_comparison']['total_time_speedup']:.2f}x faster")
    else:
        print(f"  ⚠️ Sequential processing is {1/comparison['timing_comparison']['total_time_speedup']:.2f}x faster")
    
    if abs(comparison['performance_comparison']['accuracy_difference']) < 0.01:
        print(f"  ✅ Model performance is equivalent (within 1%)")
    elif comparison['performance_comparison']['accuracy_difference'] > 0:
        print(f"  ✅ Parallel processing achieves better accuracy")
    else:
        print(f"  ⚠️ Sequential processing achieves better accuracy")
    
    if comparison['parallel_efficiency']['efficiency_percentage'] > 70:
        print(f"  ✅ High parallel efficiency ({comparison['parallel_efficiency']['efficiency_percentage']:.1f}%)")
    else:
        print(f"  ⚠️ Low parallel efficiency ({comparison['parallel_efficiency']['efficiency_percentage']:.1f}%)")
    
    print("=" * 80)


def main():
    """Main function for comparison script"""
    parser = argparse.ArgumentParser(
        description='Compare Sequential vs Parallel FHE Federated Learning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python compare_sequential_parallel_fl.py --rounds 10 --clients 20
  python compare_sequential_parallel_fl.py --rounds 5 --clients 10 --batch-size 2
  python compare_sequential_parallel_fl.py --rounds 15 --clients 30 --max-workers 3
        """
    )
    
    parser.add_argument(
        '--rounds', 
        type=int, 
        default=10, 
        help='Number of federated learning rounds (default: 10)'
    )
    
    parser.add_argument(
        '--clients', 
        type=int, 
        default=10, 
        help='Number of clients to simulate (default: 10)'
    )
    
    parser.add_argument(
        '--batch-size', 
        type=int, 
        default=4, 
        help='Number of concurrent workers per batch (default: 4)'
    )
    
    parser.add_argument(
        '--max-workers', 
        type=int, 
        default=4, 
        help='Maximum number of CPU cores to use (default: 4)'
    )
    
    parser.add_argument(
        '--polynomial-degree', 
        type=int, 
        default=8192,
        help='Polynomial degree for FHE CKKS (default: 8192)'
    )
    
    parser.add_argument(
        '--scale-bits', 
        type=int, 
        default=40,
        help='Scale bits for FHE CKKS (default: 40)'
    )
    
    parser.add_argument(
        '--verbose', 
        action='store_true', 
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    print("🔬 Sequential vs Parallel FHE Federated Learning Comparison")
    print("=" * 80)
    print(f"📊 Configuration:")
    print(f"  Rounds: {args.rounds}")
    print(f"  Clients: {args.clients}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Max Workers: {args.max_workers}")
    print(f"  Polynomial Degree: {args.polynomial_degree}")
    print(f"  Scale Bits: {args.scale_bits}")
    print(f"  CPU Cores: {os.cpu_count()}")
    print()
    
    try:
        # Create configurations
        fl_config = FLConfig(
            rounds=args.rounds,
            clients=args.clients
        )
        
        fhe_config = FHEConfig(
            polynomial_degree=args.polynomial_degree,
            scale_bits=args.scale_bits
        )
        
        # Load and preprocess data (shared between both approaches)
        print("📊 Loading and preprocessing data with enhanced feature engineering...")
        data_processor = EnhancedDataProcessor(fl_config)
        df, feature_columns = data_processor.load_health_fitness_data()
        
        # Create client datasets
        print("👥 Creating client datasets...")
        clients_data = data_processor.create_client_datasets(df)
        
        # Scale client data
        print("Scaling client data...")
        clients_data = data_processor.scale_client_data(clients_data)
        
        print(f"✅ Prepared {len(clients_data)} client datasets")
        print()
        
        # Run sequential FHE federated learning
        sequential_results = run_sequential_fhe_fl(clients_data, fhe_config, args.rounds)
        
        print()
        
        # Run parallel FHE federated learning
        parallel_results = run_parallel_fhe_fl(
            clients_data, fhe_config, args.rounds, args.batch_size, args.max_workers
        )
        
        # Compare results
        comparison = compare_results(sequential_results, parallel_results)
        
        # Print comparison summary
        print_comparison_summary(sequential_results, parallel_results, comparison)
        
        # Save comparison results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        comparison_file = f"performance_results/sequential_parallel_comparison_{args.clients}clients_{args.rounds}rounds_{timestamp}.json"
        
        os.makedirs("performance_results", exist_ok=True)
        
        save_results = {
            'comparison_info': {
                'timestamp': timestamp,
                'rounds': args.rounds,
                'clients': args.clients,
                'batch_size': args.batch_size,
                'max_workers': args.max_workers,
                'cpu_cores': os.cpu_count(),
                'polynomial_degree': args.polynomial_degree,
                'scale_bits': args.scale_bits
            },
            'sequential_results': sequential_results,
            'parallel_results': parallel_results,
            'comparison': comparison
        }
        
        with open(comparison_file, 'w') as f:
            json.dump(save_results, f, indent=2)
        
        print(f"💾 Comparison results saved to: {comparison_file}")
        
    except Exception as e:
        print(f"❌ Error running comparison: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
