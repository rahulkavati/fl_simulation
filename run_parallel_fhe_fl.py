#!/usr/bin/env python3
"""
Parallel FHE Federated Learning Runner

This script runs federated learning with true parallelization across edge devices:
- Each client runs as a separate process (simulating edge devices)
- Batch processing (2-4 concurrent workers) respects CPU/RAM limits
- Central server handles aggregation and global updates
- Measures sequential vs parallel timing performance

Usage:
    python run_parallel_fhe_fl.py --rounds 10 --clients 20 --batch-size 4
    python run_parallel_fhe_fl.py --rounds 5 --clients 10 --max-workers 2
"""

import argparse
import sys
import os
import time
import json
import numpy as np
from datetime import datetime

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.multiprocess import BatchCoordinator, BatchConfig, EdgeDeviceConfig
from src.fl import FLConfig, DataProcessor
from src.encryption import FHEConfig
from federated_learning_pipeline import EnhancedDataProcessor
from src.utils import print_final_summary


def main():
    """Main function for parallel FHE FL command line interface"""
    parser = argparse.ArgumentParser(
        description='Run Parallel FHE Federated Learning with Edge Device Simulation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_parallel_fhe_fl.py --rounds 10 --clients 20 --batch-size 4
  python run_parallel_fhe_fl.py --rounds 5 --clients 10 --max-workers 2
  python run_parallel_fhe_fl.py --rounds 15 --clients 30 --batch-size 3 --verbose
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
        '--timeout', 
        type=float, 
        default=30.0, 
        help='Timeout for process communication (default: 30.0s)'
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
    
    print("🚀 Starting Parallel FHE Federated Learning with Edge Device Simulation")
    print("=" * 80)
    print(f"📊 Configuration:")
    print(f"  Rounds: {args.rounds}")
    print(f"  Clients: {args.clients}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Max Workers: {args.max_workers}")
    print(f"  Timeout: {args.timeout}s")
    print(f"  Polynomial Degree: {args.polynomial_degree}")
    print(f"  Scale Bits: {args.scale_bits}")
    print(f"  Verbose: {args.verbose}")
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
        
        batch_config = BatchConfig(
            batch_size=args.batch_size,
            max_workers=args.max_workers,
            timeout=args.timeout
        )
        
        # Load and preprocess data using enhanced data engineering (same as sequential FHE)
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
        
        # Create batch coordinator
        coordinator = BatchCoordinator(batch_config)
        
        # Run parallel federated learning
        start_time = time.time()
        results = coordinator.run_parallel_federated_learning(
            clients_data=clients_data,
            fhe_config=fhe_config,
            rounds=args.rounds
        )
        total_time = time.time() - start_time
        
        # Print final summary
        print("=" * 80)
        print("📊 PARALLEL FHE FEDERATED LEARNING RESULTS")
        print("=" * 80)
        
        final_stats = results['final_stats']
        timing_stats = results['timing_stats']
        
        print(f"🎯 Performance Results:")
        print(f"  Initial Accuracy: {final_stats['initial_accuracy']:.4f} ({final_stats['initial_accuracy']*100:.2f}%)")
        print(f"  Final Accuracy: {final_stats['final_accuracy']:.4f} ({final_stats['final_accuracy']*100:.2f}%)")
        print(f"  Best Accuracy: {final_stats['best_accuracy']:.4f} ({final_stats['best_accuracy']*100:.2f}%)")
        print(f"  Accuracy Improvement: {final_stats['accuracy_improvement']:+.4f} ({final_stats['accuracy_improvement']*100:+.2f}%)")
        print()
        
        print(f"⏱️ Detailed Timing Results:")
        print(f"  Total Pipeline Time: {total_time:.4f}s")
        print(f"  Average Round Time: {final_stats['avg_round_time']:.4f}s")
        print(f"  Average Batch Time: {final_stats['avg_batch_time']:.4f}s")
        print()
        print(f"📱 Edge Device Timing:")
        print(f"  Average Training Time: {final_stats['avg_training_time']:.4f}s (per device)")
        print(f"  Average Encryption Time: {final_stats['avg_edge_encryption_time']:.4f}s (per device)")
        print(f"  Total Training Time: {final_stats['total_training_time']:.4f}s (all devices)")
        print(f"  Total Encryption Time: {final_stats['total_edge_encryption_time']:.4f}s (all devices)")
        print()
        print(f"🖥️ Server Timing:")
        print(f"  Average Aggregation Time: {final_stats['avg_server_aggregation_time']:.4f}s")
        print(f"  Average Internal Aggregation: {final_stats['avg_internal_aggregation_time']:.4f}s")
        print(f"  Average Global Update Time: {final_stats['avg_global_update_time']:.4f}s")
        print(f"  Average Evaluation Time: {final_stats['avg_evaluation_time']:.4f}s")
        print()
        print(f"🚀 Parallel Processing:")
        print(f"  Average Parallel Efficiency: {final_stats['avg_parallel_efficiency']:.2f}")
        print()
        
        print(f"🔧 Configuration:")
        print(f"  Total Clients: {args.clients}")
        print(f"  Batch Size: {args.batch_size}")
        print(f"  Max Workers: {args.max_workers}")
        print(f"  CPU Cores Available: {os.cpu_count()}")
        print(f"  Workers Used: {min(args.max_workers, os.cpu_count())}")
        print()
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"performance_results/parallel_fhe_fl_results_{args.clients}clients_{args.rounds}rounds_{timestamp}.json"
        
        os.makedirs("performance_results", exist_ok=True)
        
        # Prepare results for saving
        save_results = {
            'pipeline_info': {
                'type': 'parallel_fhe',
                'rounds': args.rounds,
                'clients': args.clients,
                'batch_size': args.batch_size,
                'max_workers': args.max_workers,
                'timestamp': timestamp,
                'total_samples': sum(len(X) for X, y in clients_data.values()),
                'one_class_clients': sum(1 for X, y in clients_data.values() if len(np.unique(y)) < 2),
                'cpu_cores': os.cpu_count()
            },
            'final_performance': {
                'accuracy': final_stats['final_accuracy'],
                'f1_score': results['round_results'][-1]['f1_score'],
                'precision': results['round_results'][-1]['precision'],
                'recall': results['round_results'][-1]['recall'],
                'auc': results['round_results'][-1]['auc'],
                'pr_auc': results['round_results'][-1]['pr_auc'],
                'mae': results['round_results'][-1]['mae'],
                'rmse': results['round_results'][-1]['rmse'],
                'ece': results['round_results'][-1]['ece'],
                'mce': results['round_results'][-1]['mce']
            },
            'performance_trends': {
                'initial_accuracy': final_stats['initial_accuracy'],
                'final_accuracy': final_stats['final_accuracy'],
                'accuracy_improvement': final_stats['accuracy_improvement'],
                'best_accuracy': final_stats['best_accuracy']
            },
            'timing_statistics': {
                'total_pipeline_time': total_time,
                'avg_round_time': final_stats['avg_round_time'],
                'avg_batch_time': final_stats['avg_batch_time'],
                # Edge device timing
                'avg_edge_encryption_time': final_stats['avg_edge_encryption_time'],
                'avg_training_time': final_stats['avg_training_time'],
                'total_edge_encryption_time': final_stats['total_edge_encryption_time'],
                'total_training_time': final_stats['total_training_time'],
                # Server timing
                'avg_server_aggregation_time': final_stats['avg_server_aggregation_time'],
                'avg_internal_aggregation_time': final_stats['avg_internal_aggregation_time'],
                'avg_global_update_time': final_stats['avg_global_update_time'],
                'avg_evaluation_time': final_stats['avg_evaluation_time'],
                # Backward compatibility
                'avg_encryption_time': final_stats['avg_encryption_time'],
                'avg_aggregation_time': final_stats['avg_aggregation_time'],
                # Parallel efficiency
                'avg_parallel_efficiency': final_stats['avg_parallel_efficiency'],
                'total_rounds': final_stats['total_rounds']
            },
            'parallel_processing': {
                'batch_size': args.batch_size,
                'max_workers': args.max_workers,
                'cpu_cores_available': os.cpu_count(),
                'workers_used': min(args.max_workers, os.cpu_count()),
                'parallel_efficiency': final_stats['avg_parallel_efficiency']
            },
            'round_details': results['round_results']
        }
        
        with open(results_file, 'w') as f:
            json.dump(save_results, f, indent=2)
        
        print(f"💾 Results saved to: {results_file}")
        
        # Performance assessment
        if final_stats['best_accuracy'] >= 0.95:
            print("🎯 SUCCESS! 95%+ accuracy achieved!")
        elif final_stats['best_accuracy'] >= 0.90:
            print("✅ GOOD! 90%+ accuracy achieved!")
        else:
            print(f"⚠️ Target not reached. Best accuracy: {final_stats['best_accuracy']*100:.2f}%")
            print("💡 Try increasing rounds or clients for better performance")
        
        print()
        print("🔒 TRUE END-TO-END ENCRYPTION: Model never decrypted during training")
        print("🚀 PARALLEL PROCESSING: True edge device simulation with batch processing")
        print("📊 SCALING ANALYSIS: Sequential vs parallel timing measurements")
        
    except Exception as e:
        print(f"❌ Error running parallel FHE federated learning: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
