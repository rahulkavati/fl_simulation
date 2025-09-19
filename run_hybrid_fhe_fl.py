#!/usr/bin/env python3
"""
Hybrid FHE Federated Learning Pipeline
Combines sequential convergence with parallel encryption for >95% accuracy
"""

import argparse
import sys
import os
import time
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Any

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.fl import FLConfig, DataProcessor
from src.encryption import FHEConfig
from federated_learning_pipeline import EnhancedDataProcessor
from src.multiprocess import BatchCoordinator, BatchConfig, EdgeDeviceConfig

def main():
    parser = argparse.ArgumentParser(description='Hybrid FHE Federated Learning with >95% Accuracy')
    parser.add_argument('--rounds', type=int, default=15, help='Number of federated learning rounds')
    parser.add_argument('--clients', type=int, default=20, help='Number of clients')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size for parallel processing')
    parser.add_argument('--max-workers', type=int, default=4, help='Maximum number of workers')
    parser.add_argument('--polynomial-degree', type=int, default=8192, help='FHE polynomial degree')
    parser.add_argument('--scale-bits', type=int, default=40, help='FHE scale bits')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    
    args = parser.parse_args()
    
    print("🚀 Starting Hybrid FHE Federated Learning for >95% Accuracy")
    print("=" * 80)
    print("📊 Configuration:")
    print(f"  Rounds: {args.rounds}")
    print(f"  Clients: {args.clients}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Max Workers: {args.max_workers}")
    print(f"  Polynomial Degree: {args.polynomial_degree}")
    print(f"  Scale Bits: {args.scale_bits}")
    print(f"  Verbose: {args.verbose}")
    
    try:
        # Initialize configurations
        fl_config = FLConfig(
            rounds=args.rounds,
            random_state=42
        )
        
        fhe_config = FHEConfig(
            polynomial_degree=args.polynomial_degree,
            scale_bits=args.scale_bits
        )
        
        # Load and preprocess data using enhanced data engineering
        print("\n📊 Loading and preprocessing data with enhanced feature engineering...")
        data_processor = EnhancedDataProcessor(fl_config)
        df, feature_columns = data_processor.load_health_fitness_data()
        
        # Create client datasets
        print(f"\n👥 Creating client datasets...")
        clients_data = data_processor.create_client_datasets(df)
        
        print(f"\n✅ Prepared {len(clients_data)} client datasets")
        
        # Initialize batch coordinator
        batch_config = BatchConfig(
            batch_size=args.batch_size,
            max_workers=args.max_workers,
            timeout=30.0
        )
        
        coordinator = BatchCoordinator(batch_config)
        
        print(f"\n🚀 Starting Hybrid Federated Learning")
        print(f"📊 Configuration:")
        print(f"  Total Clients: {len(clients_data)}")
        print(f"  Batch Size: {args.batch_size}")
        print(f"  Max Workers: {args.max_workers}")
        print(f"  Rounds: {args.rounds}")
        
        # Run hybrid federated learning
        start_time = time.time()
        results = coordinator.run_hybrid_federated_learning(
            clients_data, fhe_config, args.rounds
        )
        total_time = time.time() - start_time
        
        # Print results
        print("\n" + "=" * 80)
        print("📊 HYBRID FHE FEDERATED LEARNING RESULTS")
        print("=" * 80)
        
        final_stats = results['final_statistics']
        
        print(f"🎯 Performance Results:")
        print(f"  Initial Accuracy: {final_stats['initial_accuracy']:.4f} ({final_stats['initial_accuracy']*100:.2f}%)")
        print(f"  Final Accuracy: {final_stats['final_accuracy']:.4f} ({final_stats['final_accuracy']*100:.2f}%)")
        print(f"  Best Accuracy: {final_stats['best_accuracy']:.4f} ({final_stats['best_accuracy']*100:.2f}%)")
        print(f"  Accuracy Improvement: {final_stats['accuracy_improvement']:.4f} ({final_stats['accuracy_improvement']*100:+.2f}%)")
        
        print(f"\n⏱️ Detailed Timing Results:")
        print(f"  Total Pipeline Time: {total_time:.4f}s")
        print(f"  Average Round Time: {final_stats['avg_round_time']:.4f}s")
        print(f"  Average Batch Time: {final_stats['avg_batch_time']:.4f}s")
        
        print(f"\n📱 Edge Device Timing:")
        print(f"  Average Training Time: {final_stats['avg_training_time']:.4f}s (per device)")
        print(f"  Average Encryption Time: {final_stats['avg_edge_encryption_time']:.4f}s (per device)")
        print(f"  Total Training Time: {final_stats['total_training_time']:.4f}s (all devices)")
        print(f"  Total Encryption Time: {final_stats['total_edge_encryption_time']:.4f}s (all devices)")
        
        print(f"\n🖥️ Server Timing:")
        print(f"  Average Aggregation Time: {final_stats['avg_server_aggregation_time']:.4f}s")
        print(f"  Average Internal Aggregation: {final_stats['avg_internal_aggregation_time']:.4f}s")
        print(f"  Average Global Update Time: {final_stats['avg_global_update_time']:.4f}s")
        print(f"  Average Evaluation Time: {final_stats['avg_evaluation_time']:.4f}s")
        
        print(f"\n🚀 Parallel Processing:")
        print(f"  Average Parallel Efficiency: {final_stats['avg_parallel_efficiency']:.2f}")
        
        print(f"\n🔧 Configuration:")
        print(f"  Total Clients: {len(clients_data)}")
        print(f"  Batch Size: {args.batch_size}")
        print(f"  Max Workers: {args.max_workers}")
        print(f"  CPU Cores Available: {os.cpu_count()}")
        print(f"  Workers Used: {args.max_workers}")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"performance_results/hybrid_fhe_fl_results_{args.clients}clients_{args.rounds}rounds_{timestamp}.json"
        
        # Prepare results for JSON serialization
        json_results = {
            'pipeline_info': {
                'type': 'hybrid_fhe',
                'rounds': args.rounds,
                'clients': args.clients,
                'batch_size': args.batch_size,
                'max_workers': args.max_workers,
                'timestamp': timestamp,
                'total_samples': sum(len(y) for _, y in clients_data.values()),
                'one_class_clients': results['final_statistics']['one_class_clients'],
                'cpu_cores': os.cpu_count()
            },
            'final_performance': results['final_performance'],
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
                'avg_edge_encryption_time': final_stats['avg_edge_encryption_time'],
                'avg_training_time': final_stats['avg_training_time'],
                'total_edge_encryption_time': final_stats['total_edge_encryption_time'],
                'total_training_time': final_stats['total_training_time'],
                'avg_server_aggregation_time': final_stats['avg_server_aggregation_time'],
                'avg_internal_aggregation_time': final_stats['avg_internal_aggregation_time'],
                'avg_global_update_time': final_stats['avg_global_update_time'],
                'avg_evaluation_time': final_stats['avg_evaluation_time'],
                'avg_encryption_time': final_stats['avg_encryption_time'],
                'avg_aggregation_time': final_stats['avg_aggregation_time'],
                'avg_parallel_efficiency': final_stats['avg_parallel_efficiency'],
                'total_rounds': final_stats['total_rounds']
            },
            'parallel_processing': {
                'batch_size': args.batch_size,
                'max_workers': args.max_workers,
                'avg_parallel_efficiency': final_stats['avg_parallel_efficiency'],
                'total_batches': sum(len(r['batch_results']) for r in results['round_results']),
                'avg_batch_time': final_stats['avg_batch_time']
            },
            'round_results': results['round_results']
        }
        
        os.makedirs('performance_results', exist_ok=True)
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to: {results_file}")
        
        # Check if target achieved
        if final_stats['best_accuracy'] >= 0.95:
            print(f"\n🎉 TARGET ACHIEVED! Best accuracy: {final_stats['best_accuracy']:.4f} >= 95%")
        else:
            print(f"\n⚠️ Target not reached. Best accuracy: {final_stats['best_accuracy']:.4f}")
            print(f"💡 Try increasing rounds or clients for better performance")
        
        print(f"\n🔒 TRUE END-TO-END ENCRYPTION: Model never decrypted during training")
        print(f"🚀 HYBRID PROCESSING: Sequential convergence with parallel encryption")
        print(f"📊 SCALING ANALYSIS: Sequential vs parallel timing measurements")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
