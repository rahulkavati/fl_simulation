#!/usr/bin/env python3
"""
Results Comparison Script

This script compares FHE and Plaintext federated learning results
from the organized folder structure.

Usage:
    python compare_results.py --clients 2 --rounds 1
    python compare_results.py --clients 5 --rounds 3
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any, List

def load_latest_results(pipeline_type: str, clients: int, rounds: int) -> Dict[str, Any]:
    """
    Load the latest results for a specific pipeline configuration
    
    Args:
        pipeline_type: 'fhe' or 'plaintext'
        clients: Number of clients
        rounds: Number of rounds
        
    Returns:
        Results dictionary or None if not found
    """
    results_dir = Path("results") / pipeline_type
    
    if not results_dir.exists():
        return None
    
    # Find JSON files matching the pattern
    json_files = list(results_dir.glob(f"{pipeline_type}_results_{clients}clients_{rounds}rounds_*.json"))
    
    if not json_files:
        return None
    
    # Sort by modification time and get the latest
    latest_file = max(json_files, key=lambda f: f.stat().st_mtime)
    
    with open(latest_file, 'r') as f:
        return json.load(f)

def print_comparison(fhe_results: Dict[str, Any], plaintext_results: Dict[str, Any]):
    """
    Print a side-by-side comparison of FHE and Plaintext results
    
    Args:
        fhe_results: FHE pipeline results
        plaintext_results: Plaintext pipeline results
    """
    if not fhe_results or not plaintext_results:
        print("Error: Missing results for comparison")
        return
    
    print("\n" + "="*80)
    print("FHE vs PLAINTEXT FEDERATED LEARNING COMPARISON")
    print("="*80)
    
    # Pipeline information comparison
    print(f"\nPIPELINE CONFIGURATION:")
    print(f"  Clients: {fhe_results['pipeline_info']['clients']}")
    print(f"  Rounds: {fhe_results['pipeline_info']['rounds']}")
    print(f"  Total Samples: {fhe_results['pipeline_info']['total_samples']}")
    print(f"  One-Class Clients: {fhe_results['pipeline_info']['one_class_clients']} ({fhe_results['pipeline_info']['one_class_percentage']:.1f}%)")
    
    # Performance comparison
    print(f"\nPERFORMANCE COMPARISON:")
    print(f"{'Metric':<20} {'FHE':<15} {'Plaintext':<15} {'Difference':<15}")
    print("-" * 65)
    
    metrics = [
        ('Accuracy', 'accuracy'),
        ('F1 Score', 'f1_score'),
        ('Precision', 'precision'),
        ('Recall', 'recall'),
        ('AUC', 'auc'),
        ('PR-AUC', 'pr_auc'),
        ('MAE', 'mae'),
        ('RMSE', 'rmse'),
        ('ECE', 'ece'),
        ('MCE', 'mce')
    ]
    
    for metric_name, metric_key in metrics:
        fhe_val = fhe_results['final_performance'][metric_key]
        plaintext_val = plaintext_results['final_performance'][metric_key]
        
        # Handle NaN values
        if str(fhe_val) == 'nan':
            fhe_val = 0.0
        if str(plaintext_val) == 'nan':
            plaintext_val = 0.0
            
        diff = plaintext_val - fhe_val
        
        print(f"{metric_name:<20} {fhe_val:<15.4f} {plaintext_val:<15.4f} {diff:+.4f}")
    
    # Timing comparison
    print(f"\nTIMING COMPARISON:")
    print(f"{'Metric':<25} {'FHE':<15} {'Plaintext':<15} {'Speedup':<15}")
    print("-" * 70)
    
    timing_metrics = [
        ('Total Time', 'total_time'),
        ('Avg Round Time', 'avg_round_time'),
        ('Encryption Time', 'total_encryption_time'),
        ('Aggregation Time', 'total_aggregation_time'),
        ('Evaluation Time', 'total_evaluation_time')
    ]
    
    for metric_name, metric_key in timing_metrics:
        fhe_val = fhe_results['timing_statistics'][metric_key]
        plaintext_val = plaintext_results['timing_statistics'][metric_key]
        
        speedup = fhe_val / plaintext_val if plaintext_val > 0 else float('inf')
        
        print(f"{metric_name:<25} {fhe_val:<15.4f}s {plaintext_val:<15.4f}s {speedup:<15.2f}x")
    
    # Performance trends comparison
    print(f"\nPERFORMANCE TRENDS:")
    print(f"{'Metric':<25} {'FHE':<15} {'Plaintext':<15} {'Difference':<15}")
    print("-" * 70)
    
    trend_metrics = [
        ('Initial Accuracy', 'initial_accuracy'),
        ('Final Accuracy', 'final_accuracy'),
        ('Accuracy Improvement', 'accuracy_improvement'),
        ('Best Accuracy', 'best_accuracy'),
        ('Final F1', 'final_f1'),
        ('Best F1', 'best_f1')
    ]
    
    for metric_name, metric_key in trend_metrics:
        fhe_val = fhe_results['performance_trends'][metric_key]
        plaintext_val = plaintext_results['performance_trends'][metric_key]
        diff = plaintext_val - fhe_val
        
        print(f"{metric_name:<25} {fhe_val:<15.4f} {plaintext_val:<15.4f} {diff:+.4f}")
    
    print("\n" + "="*80)

def main():
    """Main function for results comparison"""
    parser = argparse.ArgumentParser(
        description='Compare FHE and Plaintext Federated Learning Results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python compare_results.py --clients 2 --rounds 1
  python compare_results.py --clients 5 --rounds 3
  python compare_results.py --clients 10 --rounds 5
        """
    )
    
    parser.add_argument(
        '--clients',
        type=int,
        default=2,
        help='Number of clients to compare (default: 2)'
    )
    
    parser.add_argument(
        '--rounds',
        type=int,
        default=1,
        help='Number of rounds to compare (default: 1)'
    )
    
    args = parser.parse_args()
    
    print(f"Comparing FHE vs Plaintext results for {args.clients} clients, {args.rounds} rounds")
    
    # Load results
    fhe_results = load_latest_results('fhe', args.clients, args.rounds)
    plaintext_results = load_latest_results('plaintext', args.clients, args.rounds)
    
    if not fhe_results:
        print(f"❌ No FHE results found for {args.clients} clients, {args.rounds} rounds")
        print("   Run: python run_fhe_fl.py --clients {} --rounds {}".format(args.clients, args.rounds))
    
    if not plaintext_results:
        print(f"❌ No Plaintext results found for {args.clients} clients, {args.rounds} rounds")
        print("   Run: python run_plaintext_fl.py --clients {} --rounds {}".format(args.clients, args.rounds))
    
    if fhe_results and plaintext_results:
        print_comparison(fhe_results, plaintext_results)
    else:
        print("\nPlease run both pipelines first to enable comparison.")

if __name__ == "__main__":
    main()
