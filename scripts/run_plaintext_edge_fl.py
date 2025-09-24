#!/usr/bin/env python3
"""
Plaintext Edge Device Federated Learning Runner

This script runs the plaintext edge device federated learning pipeline with:
- N clients performing local training
- N edge devices handling data processing and validation
- 1 cloud server performing aggregation and global updates

Architecture: Client → Edge → Cloud → Edge → Client
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from datetime import datetime

# Add the parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.plaintext_edge_fl import (
    PlaintextEdgeFederatedLearningCoordinator,
    PlaintextClientConfig,
    PlaintextEdgeDeviceConfig
)
from src.plaintext import PlaintextConfig
from src.fl import FLConfig, DataProcessor
from plaintext_federated_learning_pipeline import EnhancedDataProcessor


def main():
    """Main function to run plaintext edge device federated learning"""
    parser = argparse.ArgumentParser(description='Run Plaintext Edge Device Federated Learning')
    parser.add_argument('--rounds', type=int, default=10, help='Number of federated learning rounds')
    parser.add_argument('--clients', type=int, default=10, help='Number of clients')
    parser.add_argument('--patience', type=int, default=999, help='Early stopping patience')
    parser.add_argument('--output-dir', type=str, default='performance_results', help='Output directory')
    
    args = parser.parse_args()
    
    print("Plaintext Edge Device Federated Learning Pipeline")
    print("=" * 60)
    print("Architecture: N Clients + N Edge Devices + 1 Cloud Server")
    print("Data Flow: Client → Edge → Cloud → Edge → Client")
    print(f"Configuration: {args.clients} clients, {args.rounds} rounds")
    print()
    
    # Create output directory - Plaintext Edge FL specific folder
    results_dir = 'plaintext_edge_results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Initialize configurations
    fl_config = FLConfig(
        rounds=args.rounds,
        clients=args.clients
    )
    
    plaintext_config = PlaintextConfig(
        aggregation_method="federated_averaging",
        weight_by_samples=True,
        model_type="logistic_regression",
        feature_count=46,  # Updated to match EnhancedDataProcessor
        track_timing=True,
        verbose=True
    )
    
    # Initialize enhanced data processor (same as original plaintext FL)
    data_processor = EnhancedDataProcessor(fl_config)
    
    # Load and preprocess data
    print("=" * 60)
    print("STEP 1: Data Loading and Preprocessing")
    print("=" * 60)
    
    print("Loading Health Fitness Dataset with Enhanced Feature Engineering...")
    df, feature_columns = data_processor.load_health_fitness_data()
    
    print("Creating client datasets...")
    clients_data = data_processor.create_client_datasets(df)
    
    print("Scaling client data...")
    clients_data = data_processor.scale_client_data(clients_data)
    print(f"Scaled data for {len(clients_data)} clients")
    
    print(f"✅ Data loaded: {len(clients_data)} clients, {len(feature_columns)} features")
    
    # Initialize coordinator
    coordinator = PlaintextEdgeFederatedLearningCoordinator(fl_config, plaintext_config)
    
    # Run plaintext edge federated learning
    print("\n" + "=" * 60)
    print("STEP 2: Plaintext Edge Device Federated Learning")
    print("=" * 60)
    
    start_time = time.time()
    round_results = coordinator.run_plaintext_edge_federated_learning(clients_data)
    total_time = time.time() - start_time
    
    # Calculate final statistics
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    
    final_accuracy = round_results[-1]['accuracy'] if round_results else 0.0
    best_accuracy = max([r['accuracy'] for r in round_results]) if round_results else 0.0
    
    print(f"Final Accuracy: {final_accuracy*100:.2f}%")
    print(f"Best Accuracy: {best_accuracy*100:.2f}%")
    print(f"Total Rounds: {len(round_results)}")
    
    # Calculate comprehensive timing statistics
    total_time = time.time() - start_time
    avg_round_time = np.mean([r['round_time'] for r in round_results])
    
    # Calculate all timing metrics with proper separation
    total_client_training_time = sum([r.get('client_training_time', 0) for r in round_results])
    total_edge_processing_wall_time = sum([r.get('edge_processing_wall_time', 0) for r in round_results])
    total_pure_processing_time = sum([r.get('total_pure_processing_time', 0) for r in round_results])
    total_server_aggregation_time = sum([r.get('server_aggregation_time', 0) for r in round_results])
    total_internal_aggregation_time = sum([r.get('internal_aggregation_time', 0) for r in round_results])
    total_evaluation_time = sum([r.get('evaluation_time', 0) for r in round_results])
    
    # Calculate averages
    avg_client_training_time = total_client_training_time / len(round_results) if round_results else 0
    avg_edge_processing_wall_time = total_edge_processing_wall_time / len(round_results) if round_results else 0
    avg_pure_processing_time = total_pure_processing_time / len(round_results) if round_results else 0
    avg_server_aggregation_time = total_server_aggregation_time / len(round_results) if round_results else 0
    avg_internal_aggregation_time = total_internal_aggregation_time / len(round_results) if round_results else 0
    avg_evaluation_time = total_evaluation_time / len(round_results) if round_results else 0
    
    # Calculate average processing per client (should be constant)
    avg_processing_per_client = np.mean([r.get('avg_processing_per_client', 0) for r in round_results])
    
    print(f"\nTiming Statistics:")
    print(f"  Total Time: {total_time:.2f}s")
    print(f"  Average Round Time: {avg_round_time:.2f}s")
    print(f"  Average Client Training Time: {avg_client_training_time:.4f}s")
    print(f"  Average Edge Processing Wall Time: {avg_edge_processing_wall_time:.4f}s")
    print(f"  Average Pure Processing Time: {avg_pure_processing_time:.4f}s")
    print(f"  Average Processing per Client: {avg_processing_per_client:.6f}s")
    print(f"  Average Server Aggregation Time: {avg_server_aggregation_time:.4f}s")
    print(f"  Average Internal Aggregation Time: {avg_internal_aggregation_time:.4f}s")
    print(f"  Average Evaluation Time: {avg_evaluation_time:.4f}s")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(results_dir, f"plaintext_edge_fl_results_{args.clients}clients_{args.rounds}rounds_{timestamp}.json")
    
    results_data = {
        'pipeline_type': 'plaintext_edge_device_fl',
        'configuration': {
            'rounds': fl_config.rounds,
            'clients': fl_config.clients,
            'patience': args.patience
        },
        'results': round_results,
        'final_statistics': {
            'final_accuracy': final_accuracy,
            'best_accuracy': best_accuracy,
            'total_time': total_time,
            'avg_round_time': avg_round_time,
            'avg_client_training_time': avg_client_training_time,
            'avg_edge_processing_wall_time': avg_edge_processing_wall_time,
            'avg_pure_processing_time': avg_pure_processing_time,
            'avg_processing_per_client': avg_processing_per_client,
            'avg_server_aggregation_time': avg_server_aggregation_time,
            'avg_internal_aggregation_time': avg_internal_aggregation_time,
            'avg_evaluation_time': avg_evaluation_time
        },
        'timing_statistics': {
            'total_client_training_time': total_client_training_time,
            'total_edge_processing_wall_time': total_edge_processing_wall_time,
            'total_pure_processing_time': total_pure_processing_time,
            'total_server_aggregation_time': total_server_aggregation_time,
            'total_internal_aggregation_time': total_internal_aggregation_time,
            'total_evaluation_time': total_evaluation_time,
            'total_time': total_time,
            'avg_round_time': avg_round_time,
            'avg_client_training_time': avg_client_training_time,
            'avg_edge_processing_wall_time': avg_edge_processing_wall_time,
            'avg_pure_processing_time': avg_pure_processing_time,
            'avg_processing_per_client': avg_processing_per_client,
            'avg_server_aggregation_time': avg_server_aggregation_time,
            'avg_internal_aggregation_time': avg_internal_aggregation_time,
            'avg_evaluation_time': avg_evaluation_time
        },
        'architecture': {
            'clients': fl_config.clients,
            'edge_devices': fl_config.clients,
            'cloud_servers': 1,
            'data_flow': 'Client → Edge → Cloud → Edge → Client'
        }
    }
    
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"\n✅ Results saved to: {results_file}")
    
    return results_data


if __name__ == '__main__':
    try:
        results = main()
        print("\n🎉 Plaintext Edge Device Federated Learning completed successfully!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
