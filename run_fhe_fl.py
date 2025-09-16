#!/usr/bin/env python3
"""
Command Line Interface for FHE Federated Learning

This script provides a simple command-line interface to run the FHE
federated learning pipeline with various configuration options.

Usage:
    python run_fhe_fl.py --rounds 10 --clients 20
    python run_fhe_fl.py --rounds 5 --clients 10 --patience 3
"""

import argparse
import sys
import os
import numpy as np

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from federated_learning_pipeline import EnhancedFederatedLearningPipeline
from src.fl import FLConfig
from src.encryption import FHEConfig
from src.utils import generate_structured_summary, save_results_to_folder, print_structured_summary


def main():
    """Main function for FHE FL command line interface"""
    parser = argparse.ArgumentParser(
        description='Run FHE Federated Learning Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_fhe_fl.py --rounds 10 --clients 20
  python run_fhe_fl.py --rounds 5 --clients 10 --patience 3
  python run_fhe_fl.py --rounds 15 --clients 30 --verbose
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
    
    args = parser.parse_args()
    
    print("Starting FHE Federated Learning Pipeline")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Rounds: {args.rounds}")
    print(f"  Clients: {args.clients}")
    print(f"  Patience: {args.patience}")
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
        
        # Create and run pipeline
        pipeline = EnhancedFederatedLearningPipeline(fl_config, fhe_config)
        results = pipeline.run_enhanced_federated_learning()
        
        # Generate structured summary
        summary = generate_structured_summary(
            results, 
            pipeline.clients_data, 
            'fhe', 
            args.rounds, 
            args.clients
        )
        
        # Save results to organized folder
        saved_file = save_results_to_folder(summary, 'fhe', args.rounds, args.clients)
        print(f"\nResults saved to: {saved_file}")
        
        # Print structured summary
        print_structured_summary(summary)
        
    except Exception as e:
        print(f"Error running FHE federated learning: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
