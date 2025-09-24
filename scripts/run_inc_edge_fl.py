#!/usr/bin/env python3
"""
INC-Assisted Edge Device Federated Learning Runner

This script runs the INC-assisted edge device federated learning pipeline with:
- N clients performing local training
- N edge devices handling encryption/decryption
- M INCs performing intermediate aggregation
- 1 cloud server receiving pre-aggregated data

Architecture: Client → Edge → INC → Cloud → INC → Edge → Client
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

from src.inc_edge_fl import (
    INCFederatedLearningCoordinator,
    ClientConfig,
    EdgeDeviceConfig,
    INCConfig
)
from src.fl import FLConfig
from src.encryption import FHEConfig
from src.utils import generate_structured_summary, save_results_to_folder, print_structured_summary


def create_inc_configurations(num_clients: int, num_incs: int) -> list:
    """Create INC configurations based on number of clients and INCs"""
    inc_configs = []
    
    if num_incs == 1:
        # Single INC manages all edge devices
        inc_config = INCConfig(
            inc_id="inc_0",
            edge_device_ids=[f"edge_client_{i}" for i in range(num_clients)]
        )
        inc_configs.append(inc_config)
    else:
        # Multiple INCs - distribute edge devices evenly
        edges_per_inc = num_clients // num_incs
        remaining_edges = num_clients % num_incs
        
        edge_id_counter = 0
        for inc_id in range(num_incs):
            # Calculate number of edge devices for this INC
            edges_for_this_inc = edges_per_inc
            if inc_id < remaining_edges:
                edges_for_this_inc += 1
            
            # Assign edge device IDs
            edge_device_ids = []
            for _ in range(edges_for_this_inc):
                edge_device_ids.append(f"edge_client_{edge_id_counter}")
                edge_id_counter += 1
            
            inc_config = INCConfig(
                inc_id=f"inc_{inc_id}",
                edge_device_ids=edge_device_ids
            )
            inc_configs.append(inc_config)
    
    return inc_configs


def main():
    """Main function for INC Edge FL command line interface"""
    parser = argparse.ArgumentParser(
        description="Run INC-Assisted Edge Device Federated Learning Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_inc_edge_fl.py --clients 10 --rounds 10 --incs 1
  python run_inc_edge_fl.py --clients 20 --rounds 15 --incs 2
  python run_inc_edge_fl.py --clients 30 --rounds 20 --incs 3 --patience 5
        """
    )
    
    parser.add_argument('--clients', type=int, default=10, help='Number of clients (default: 10)')
    parser.add_argument('--rounds', type=int, default=10, help='Number of rounds (default: 10)')
    parser.add_argument('--incs', type=int, default=1, help='Number of INCs (default: 1)')
    parser.add_argument('--patience', type=int, default=999, help='Early stopping patience (default: 999)')
    parser.add_argument('--poly_modulus_degree', type=int, default=8192, help='FHE polynomial modulus degree (default: 8192)')
    parser.add_argument('--scale_bits', type=int, default=40, help='FHE scale bits (default: 40)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.clients < 1:
        print("Error: Number of clients must be at least 1")
        return 1
    
    if args.rounds < 1:
        print("Error: Number of rounds must be at least 1")
        return 1
    
    if args.incs < 1:
        print("Error: Number of INCs must be at least 1")
        return 1
    
    if args.incs > args.clients:
        print("Error: Number of INCs cannot exceed number of clients")
        return 1
    
    print("INC-Assisted Edge Device Federated Learning Pipeline")
    print("=" * 60)
    print(f"Architecture: N Clients + N Edge Devices + M INCs + 1 Cloud Server")
    print(f"Data Flow: Client → Edge → INC → Cloud → INC → Edge → Client")
    print(f"Configuration: {args.clients} clients, {args.rounds} rounds, {args.incs} INCs")
    
    # Initialize configurations
    fl_config = FLConfig(
        rounds=args.rounds,
        clients=args.clients
    )
    
    fhe_config = FHEConfig(
        polynomial_degree=args.poly_modulus_degree,
        coeff_mod_bit_sizes=[args.scale_bits, args.scale_bits, args.scale_bits, args.scale_bits],
        scale_bits=args.scale_bits
    )
    
    # Create INC configurations
    inc_configs = create_inc_configurations(args.clients, args.incs)
    
    print(f"\nINC Configuration:")
    for inc_config in inc_configs:
        print(f"  {inc_config.inc_id}: manages {len(inc_config.edge_device_ids)} edge devices")
    
    try:
        # Initialize data processor
        from federated_learning_pipeline import EnhancedDataProcessor
        data_processor = EnhancedDataProcessor(fl_config)
        
        print("\n" + "=" * 60)
        print("STEP 1: Data Loading and Preprocessing")
        print("=" * 60)
        
        # Load and preprocess data
        df, feature_columns = data_processor.load_health_fitness_data()
        clients_data = data_processor.create_client_datasets(df)
        
        # Scale client data
        print("Scaling client data...")
        clients_data = data_processor.scale_client_data(clients_data)
        
        print(f"✅ Data loaded: {len(clients_data)} clients, {len(feature_columns)} features")
        
        print("\n" + "=" * 60)
        print("STEP 2: INC-Assisted Edge Device Federated Learning")
        print("=" * 60)
        
        # Initialize coordinator
        coordinator = INCFederatedLearningCoordinator(fl_config, fhe_config)
        
        # Run federated learning
        start_time = time.time()
        round_results = coordinator.run_inc_federated_learning(clients_data, inc_configs)
        total_time = time.time() - start_time
        
        # Calculate final metrics
        final_accuracy = round_results[-1]['accuracy'] if round_results else 0.0
        best_accuracy = max([r['accuracy'] for r in round_results]) if round_results else 0.0
        
        # Calculate timing statistics
        total_client_training_time = sum([r['timing_statistics']['client_training_time'] for r in round_results])
        total_edge_encryption_wall_time = sum([r['timing_statistics']['edge_encryption_wall_time'] for r in round_results])
        total_pure_encryption_time = sum([r['timing_statistics']['total_pure_encryption_time'] for r in round_results])
        avg_encryption_per_client = np.mean([r['timing_statistics']['avg_encryption_per_client'] for r in round_results])
        total_inc_aggregation_time = sum([r['timing_statistics']['inc_aggregation_time'] for r in round_results])
        total_cloud_update_time = sum([r['timing_statistics']['cloud_update_time'] for r in round_results])
        total_inc_distribution_time = sum([r['timing_statistics']['inc_distribution_time'] for r in round_results])
        total_evaluation_time = sum([r['timing_statistics']['evaluation_time'] for r in round_results])
        
        print("\n" + "=" * 60)
        print("FINAL RESULTS")
        print("=" * 60)
        print(f"Final Accuracy: {final_accuracy*100:.2f}%")
        print(f"Best Accuracy: {best_accuracy*100:.2f}%")
        print(f"Total Rounds: {len(round_results)}")
        
        print(f"\nTiming Statistics:")
        print(f"  Total Time: {total_time:.2f}s")
        print(f"  Average Round Time: {total_time/len(round_results):.2f}s")
        print(f"  Average Client Training Time: {total_client_training_time/len(round_results):.4f}s")
        print(f"  Average Edge Encryption Wall Time: {total_edge_encryption_wall_time/len(round_results):.4f}s")
        print(f"  Average Pure Encryption Time: {total_pure_encryption_time/len(round_results):.4f}s")
        print(f"  Average Encryption per Client: {avg_encryption_per_client:.6f}s")
        print(f"  Average INC Aggregation Time: {total_inc_aggregation_time/len(round_results):.4f}s")
        print(f"  Average Cloud Update Time: {total_cloud_update_time/len(round_results):.4f}s")
        print(f"  Average INC Distribution Time: {total_inc_distribution_time/len(round_results):.4f}s")
        print(f"  Average Evaluation Time: {total_evaluation_time/len(round_results):.4f}s")
        
        # Prepare results for saving
        results = {
            'pipeline_type': 'inc_edge_fl',
            'configuration': {
                'clients': args.clients,
                'rounds': args.rounds,
                'incs': args.incs,
                'patience': args.patience,
                'poly_modulus_degree': args.poly_modulus_degree,
                'scale_bits': args.scale_bits
            },
            'inc_configuration': [
                {
                    'inc_id': inc.inc_id,
                    'edge_device_ids': inc.edge_device_ids,
                    'edge_count': len(inc.edge_device_ids)
                } for inc in inc_configs
            ],
            'round_results': round_results,
            'final_metrics': {
                'final_accuracy': final_accuracy,
                'best_accuracy': best_accuracy,
                'total_rounds': len(round_results)
            },
            'timing_statistics': {
                'total_time': total_time,
                'avg_round_time': total_time/len(round_results),
                'total_client_training_time': total_client_training_time,
                'total_edge_encryption_wall_time': total_edge_encryption_wall_time,
                'total_pure_encryption_time': total_pure_encryption_time,
                'avg_encryption_per_client': avg_encryption_per_client,
                'total_inc_aggregation_time': total_inc_aggregation_time,
                'total_cloud_update_time': total_cloud_update_time,
                'total_inc_distribution_time': total_inc_distribution_time,
                'total_evaluation_time': total_evaluation_time,
                'avg_client_training_time': total_client_training_time/len(round_results),
                'avg_edge_encryption_wall_time': total_edge_encryption_wall_time/len(round_results),
                'avg_pure_encryption_time': total_pure_encryption_time/len(round_results),
                'avg_inc_aggregation_time': total_inc_aggregation_time/len(round_results),
                'avg_cloud_update_time': total_cloud_update_time/len(round_results),
                'avg_inc_distribution_time': total_inc_distribution_time/len(round_results),
                'avg_evaluation_time': total_evaluation_time/len(round_results)
            },
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S')
        }
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"inc_edge_fl_results_{args.clients}clients_{args.rounds}rounds_{args.incs}incs_{timestamp}.json"
        
        # Create results directory if it doesn't exist
        results_dir = "inc_edge_results"
        os.makedirs(results_dir, exist_ok=True)
        
        filepath = os.path.join(results_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n✅ Results saved to: {filepath}")
        
        print(f"\n🎉 INC-Assisted Edge Device Federated Learning completed successfully!")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
