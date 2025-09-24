#!/usr/bin/env python3
"""
Command Line Interface for Edge Device Federated Learning

This script provides a command-line interface to run the edge device
federated learning pipeline with n clients + n edge devices architecture.

Usage:
    python run_edge_fl.py --rounds 10 --clients 20
    python run_edge_fl.py --rounds 5 --clients 10 --patience 3
"""

import argparse
import sys
import os
import time
import numpy as np

# Add the parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.edge_fl import EdgeFederatedLearningCoordinator
from src.fl import FLConfig
from src.encryption import FHEConfig
from src.utils import generate_structured_summary, save_results_to_folder, print_structured_summary


def main():
    """Main function for Edge FL command line interface"""
    parser = argparse.ArgumentParser(
        description='Run Edge Device Federated Learning Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_edge_fl.py --rounds 10 --clients 20
  python run_edge_fl.py --rounds 5 --clients 10 --patience 3
  python run_edge_fl.py --rounds 15 --clients 30 --verbose
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
        help='Number of clients (default: 10)'
    )
    
    parser.add_argument(
        '--patience', 
        type=int, 
        default=999, 
        help='Patience for convergence detection (default: 999)'
    )
    
    parser.add_argument(
        '--verbose', 
        action='store_true', 
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    print("Edge Device Federated Learning Pipeline")
    print("=" * 70)
    print("Architecture: N Clients + N Edge Devices + 1 Cloud Server")
    print("Data Flow: Client → Edge → Cloud → Edge → Client")
    print(f"Configuration: {args.clients} clients, {args.rounds} rounds")
    
    start_time = time.time()
    
    try:
        # Create configurations
        fl_config = FLConfig(rounds=args.rounds, clients=args.clients)
        fhe_config = FHEConfig()
        
        # Load and preprocess data (same as FHE pipeline)
        from federated_learning_pipeline import EnhancedDataProcessor
        
        print("\n" + "=" * 70)
        print("STEP 1: Data Loading and Preprocessing")
        print("=" * 70)
        
        # Load data using same processor as FHE pipeline
        data_processor = EnhancedDataProcessor(fl_config)
        df, feature_columns = data_processor.load_health_fitness_data()
        
        # Create client datasets
        print(f"\nCreating {fl_config.clients} client datasets...")
        clients_data = data_processor.create_client_datasets(df)
        
        # Scale client data
        print("Scaling client data...")
        clients_data = data_processor.scale_client_data(clients_data)
        
        print(f"✅ Data loaded: {len(clients_data)} clients, {feature_columns} features")
        
        # Create and run edge federated learning coordinator
        coordinator = EdgeFederatedLearningCoordinator(fl_config, fhe_config)
        
        print("\n" + "=" * 70)
        print("STEP 2: Edge Device Federated Learning")
        print("=" * 70)
        
        round_results = coordinator.run_edge_federated_learning(clients_data)
        
        # Prepare results - FHE Edge FL specific folder
        results_dir = 'fhe_edge_results'
        os.makedirs(results_dir, exist_ok=True)
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        
        # Print final summary
        print("\n" + "=" * 70)
        print("FINAL RESULTS")
        print("=" * 70)
        
        final_accuracy = round_results[-1]['accuracy']
        best_accuracy = max(r['accuracy'] for r in round_results)
        
        print(f"Final Accuracy: {final_accuracy*100:.2f}%")
        print(f"Best Accuracy: {best_accuracy*100:.2f}%")
        print(f"Total Rounds: {len(round_results)}")
        
        # Calculate comprehensive timing statistics
        total_time = time.time() - start_time
        avg_round_time = np.mean([r['round_time'] for r in round_results])
        
        # Calculate all timing metrics with proper separation
        total_client_training_time = sum([r.get('client_training_time', 0) for r in round_results])
        total_edge_encryption_wall_time = sum([r.get('edge_encryption_wall_time', 0) for r in round_results])
        total_pure_encryption_time = sum([r.get('total_pure_encryption_time', 0) for r in round_results])
        total_server_aggregation_time = sum([r.get('server_aggregation_time', 0) for r in round_results])
        total_internal_aggregation_time = sum([r.get('internal_aggregation_time', 0) for r in round_results])
        total_evaluation_time = sum([r.get('evaluation_time', 0) for r in round_results])
        
        # Calculate averages
        avg_client_training_time = total_client_training_time / len(round_results) if round_results else 0
        avg_edge_encryption_wall_time = total_edge_encryption_wall_time / len(round_results) if round_results else 0
        avg_pure_encryption_time = total_pure_encryption_time / len(round_results) if round_results else 0
        avg_server_aggregation_time = total_server_aggregation_time / len(round_results) if round_results else 0
        avg_internal_aggregation_time = total_internal_aggregation_time / len(round_results) if round_results else 0
        avg_evaluation_time = total_evaluation_time / len(round_results) if round_results else 0
        
        # Calculate average encryption per client (should be constant)
        avg_encryption_per_client = np.mean([r.get('avg_encryption_per_client', 0) for r in round_results])
        
        print(f"\nTiming Statistics:")
        print(f"  Total Time: {total_time:.2f}s")
        print(f"  Average Round Time: {avg_round_time:.2f}s")
        print(f"  Average Client Training Time: {avg_client_training_time:.4f}s")
        print(f"  Average Edge Encryption Wall Time: {avg_edge_encryption_wall_time:.4f}s")
        print(f"  Average Pure Encryption Time: {avg_pure_encryption_time:.4f}s")
        print(f"  Average Encryption per Client: {avg_encryption_per_client:.6f}s")
        print(f"  Average Server Aggregation Time: {avg_server_aggregation_time:.4f}s")
        print(f"  Average Internal Aggregation Time: {avg_internal_aggregation_time:.4f}s")
        print(f"  Average Evaluation Time: {avg_evaluation_time:.4f}s")
        
        # Save results
        results_data = {
            'pipeline_type': 'edge_device_fl',
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
                'avg_edge_encryption_wall_time': avg_edge_encryption_wall_time,
                'avg_pure_encryption_time': avg_pure_encryption_time,
                'avg_encryption_per_client': avg_encryption_per_client,
                'avg_server_aggregation_time': avg_server_aggregation_time,
                'avg_internal_aggregation_time': avg_internal_aggregation_time,
                'avg_evaluation_time': avg_evaluation_time
            },
            'timing_statistics': {
                'total_client_training_time': total_client_training_time,
                'total_edge_encryption_wall_time': total_edge_encryption_wall_time,
                'total_pure_encryption_time': total_pure_encryption_time,
                'total_server_aggregation_time': total_server_aggregation_time,
                'total_internal_aggregation_time': total_internal_aggregation_time,
                'total_evaluation_time': total_evaluation_time,
                'total_time': total_time,
                'avg_round_time': avg_round_time,
                'avg_client_training_time': avg_client_training_time,
                'avg_edge_encryption_wall_time': avg_edge_encryption_wall_time,
                'avg_pure_encryption_time': avg_pure_encryption_time,
                'avg_encryption_per_client': avg_encryption_per_client,
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
        
        results_file = f'{results_dir}/fhe_edge_fl_results_{fl_config.clients}clients_{fl_config.rounds}rounds_{timestamp}.json'
        
        with open(results_file, 'w') as f:
            import json
            json.dump(results_data, f, indent=2, default=str)
        
        print(f"\n✅ Results saved to: {results_file}")
        
    except Exception as e:
        print(f"❌ Error running edge federated learning: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
