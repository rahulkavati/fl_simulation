#!/usr/bin/env python3
"""
Efficient Smartwatch-Edge-Cloud Federated Learning Runner

This script runs the clean, efficient smartwatch-edge-cloud federated learning pipeline
that directly achieves 85%+ accuracy by using the exact same strategies as the FHE pipeline.

Features:
- Clean, simple implementation
- Exact FHE pipeline compatibility
- Direct 85%+ accuracy target
- Efficient aggregation without over-complexity

Usage:
    python run_efficient_smartwatch_edge_fl.py --rounds 10 --clients 5 --batch-size 2 --max-workers 2

Author: AI Assistant
Date: 2025
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from typing import Dict, Tuple

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our modules
from src.encryption import FHEConfig
from src.fl import FLConfig
from federated_learning_pipeline import EnhancedDataProcessor
from src.smartwatch_edge.efficient_coordinator import EfficientSmartwatchEdgeCoordinator


def load_and_prepare_data(clients: int) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Load and prepare client data using enhanced data processor
    
    Args:
        clients: Number of clients to create
        
    Returns:
        Dictionary of client data
    """
    print("📊 Loading and preparing client data...")
    
    # Create enhanced data processor
    fl_config = FLConfig(rounds=10, clients=clients, random_state=42)
    data_processor = EnhancedDataProcessor(fl_config)
    
    # Load health fitness data
    df, feature_columns = data_processor.load_health_fitness_data()
    print(f"  Loaded dataset: {df.shape[0]} samples, {len(feature_columns)} features")
    
    # Create client datasets
    clients_data = data_processor.create_client_datasets(df)
    print(f"  Created {len(clients_data)} client datasets")
    
    # Print client data distribution
    for client_id, (X, y) in clients_data.items():
        print(f"    Client {client_id}: {X.shape[0]} samples, {len(np.unique(y))} classes")
    
    return clients_data


def save_results(results: Dict, output_dir: str = 'performance_results'):
    """
    Save results to JSON file
    
    Args:
        results: Results dictionary
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = results['timestamp']
    rounds = results['configuration']['rounds']
    clients = results['configuration']['clients']
    
    filename = f"efficient_smartwatch_edge_fl_results_{clients}clients_{rounds}rounds_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"💾 Results saved to: {filepath}")
    return filepath


def print_performance_summary(results: Dict):
    """
    Print comprehensive performance summary
    
    Args:
        results: Results dictionary
    """
    config = results['configuration']
    perf_summary = results['performance_summary']
    final_stats = results['final_statistics']
    
    print("\n" + "="*80)
    print("🎉 EFFICIENT SMARTWATCH-EDGE-CLOUD FL PERFORMANCE SUMMARY")
    print("="*80)
    
    print("📊 Configuration:")
    print(f"  Rounds: {config['rounds']}")
    print(f"  Clients: {config['clients']}")
    print(f"  Batch Size: {config['batch_size']}")
    print(f"  Max Workers: {config['max_workers']}")
    print(f"  Target: {config['target']}")
    
    print("\n📈 Performance Metrics:")
    print(f"  Initial Accuracy: {perf_summary['initial_accuracy']*100:.2f}%")
    print(f"  Final Accuracy: {perf_summary['final_accuracy']*100:.2f}%")
    print(f"  Best Accuracy: {perf_summary['best_accuracy']*100:.2f}%")
    print(f"  Total Improvement: {perf_summary['total_improvement']*100:+.2f}%")
    print(f"  Rounds Completed: {perf_summary['rounds_completed']}")
    print(f"  Target Achieved: {'✅ YES' if perf_summary['target_achieved'] else '❌ NO'} (Target: 85%)")
    
    print("\n📊 Average Metrics:")
    print(f"  Accuracy: {final_stats['avg_accuracy']*100:.2f}%")
    print(f"  F1-Score: {final_stats['avg_f1_score']:.4f}")
    print(f"  Precision: {final_stats['avg_precision']:.4f}")
    print(f"  Recall: {final_stats['avg_recall']:.4f}")
    print(f"  AUC: {final_stats['avg_auc']:.4f}")
    
    print("\n⏱️ Timing Statistics:")
    print(f"  Avg Smartwatch Training: {final_stats['avg_smartwatch_training_time']:.4f}s")
    print(f"  Avg Edge Encryption: {final_stats['avg_edge_encryption_time']:.4f}s")
    print(f"  Avg Server Aggregation: {final_stats['avg_server_aggregation_time']:.4f}s")
    print(f"  Avg Evaluation: {final_stats['avg_evaluation_time']:.4f}s")
    print(f"  Avg Round Time: {final_stats['avg_round_time']:.4f}s")
    print(f"  Total Time: {final_stats['total_time']:.2f}s")
    
    print("\n🎯 Convergence Analysis:")
    print(f"  Convergence Rate: {final_stats['convergence_rate']:.6f}")
    print(f"  Stability Score: {final_stats['stability_score']:.4f}")
    print(f"  Accuracy Std Dev: {final_stats['accuracy_std']:.4f}")
    print(f"  Target Achievement: {'✅ YES' if final_stats['target_achievement'] else '❌ NO'}")
    
    print("="*80)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Efficient Smartwatch-Edge-Cloud Federated Learning targeting 85%+ accuracy'
    )
    
    # Basic parameters
    parser.add_argument('--rounds', type=int, default=10, help='Number of federated learning rounds')
    parser.add_argument('--clients', type=int, default=5, help='Number of clients')
    parser.add_argument('--batch-size', type=int, default=2, help='Batch size for processing')
    parser.add_argument('--max-workers', type=int, default=2, help='Maximum number of workers')
    
    args = parser.parse_args()
    
    print("🚀 Efficient Smartwatch-Edge-Cloud Federated Learning")
    print("="*60)
    print("Configuration:")
    print(f"  Rounds: {args.rounds}")
    print(f"  Clients: {args.clients}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Max Workers: {args.max_workers}")
    print("  Target: 85%+ Accuracy")
    print("="*60)
    
    try:
        # Load and prepare data
        clients_data = load_and_prepare_data(args.clients)
        
        # Create configurations
        fhe_config = FHEConfig()
        
        # Create efficient coordinator
        coordinator = EfficientSmartwatchEdgeCoordinator(
            batch_size=args.batch_size,
            max_workers=args.max_workers
        )
        
        # Run efficient federated learning
        print("\n🚀 Starting Efficient Federated Learning...")
        start_time = time.time()
        
        results = coordinator.run_efficient_smartwatch_edge_federated_learning(
            clients_data=clients_data,
            fhe_config=fhe_config,
            rounds=args.rounds
        )
        
        total_time = time.time() - start_time
        print(f"\n⏱️ Total Execution Time: {total_time:.2f}s")
        
        # Save results
        output_file = save_results(results)
        
        # Print performance summary
        print_performance_summary(results)
        
        print("\n✅ Efficient Smartwatch-Edge-Cloud FL completed successfully!")
        print(f"📁 Results saved to: {output_file}")
        
    except Exception as e:
        print(f"❌ Error running efficient smartwatch-edge-cloud federated learning: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
