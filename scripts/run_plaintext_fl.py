#!/usr/bin/env python3
"""
Command Line Interface for Plaintext Federated Learning

This script provides a simple command-line interface to run the plaintext
federated learning pipeline with various configuration options.

Usage:
    python run_plaintext_fl.py --rounds 10 --clients 20
    python run_plaintext_fl.py --rounds 5 --clients 10 --patience 3
"""

import argparse
import sys
import os

# Add the parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from plaintext_federated_learning_pipeline import PlaintextFederatedLearningPipeline
from src.fl import FLConfig
from src.plaintext import PlaintextConfig
from src.utils import generate_structured_summary, save_results_to_folder, print_structured_summary


def main():
    """Main function for plaintext FL command line interface"""
    parser = argparse.ArgumentParser(
        description='Run Plaintext Federated Learning Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_plaintext_fl.py --rounds 10 --clients 20
  python run_plaintext_fl.py --rounds 5 --clients 10 --patience 3
  python run_plaintext_fl.py --rounds 15 --clients 30 --verbose
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
        '--patience', 
        type=int, 
        default=999, 
        help='Patience for convergence detection (default: 999 = disabled)'
    )
    
    parser.add_argument(
        '--verbose', 
        action='store_true', 
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--aggregation-method', 
        choices=['federated_averaging', 'equal_weighting'], 
        default='federated_averaging',
        help='Aggregation method to use (default: federated_averaging)'
    )
    
    args = parser.parse_args()
    
    print("Starting Plaintext Federated Learning Pipeline")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Rounds: {args.rounds}")
    print(f"  Clients: {args.clients}")
    print(f"  Patience: {args.patience}")
    print(f"  Aggregation: {args.aggregation_method}")
    print(f"  Verbose: {args.verbose}")
    print()
    
    try:
        # Create configurations
        fl_config = FLConfig(
            rounds=args.rounds,
            clients=args.clients
        )
        
        plaintext_config = PlaintextConfig(
            aggregation_method=args.aggregation_method,
            weight_by_samples=(args.aggregation_method == 'federated_averaging'),
            verbose=args.verbose
        )
        
        # Create and run pipeline
        pipeline = PlaintextFederatedLearningPipeline(fl_config, plaintext_config, patience=args.patience)
        results = pipeline.run_plaintext_federated_learning()
        
        # Generate structured summary
        summary = generate_structured_summary(
            results, 
            pipeline.clients_data, 
            'plaintext', 
            args.rounds, 
            args.clients
        )
        
        # Save results to organized folder
        saved_file = save_results_to_folder(summary, 'plaintext', args.rounds, args.clients)
        print(f"\nResults saved to: {saved_file}")
        
        # Print structured summary
        print_structured_summary(summary)
        
    except Exception as e:
        print(f"Error running plaintext federated learning: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
