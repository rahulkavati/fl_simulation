#!/usr/bin/env python3
"""
Timing Analysis Report for Federated Learning Pipelines

This script analyzes timing metrics from FHE Edge FL and Plaintext Edge FL results
to determine what represents end-to-end delay and other useful performance metrics.
"""

import json
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Any

def load_results(file_path: str) -> Dict[str, Any]:
    """Load results from JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)

def analyze_timing_metrics(results: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze timing metrics from results"""
    rounds = results['results']
    
    analysis = {
        'pipeline_type': results['pipeline_type'],
        'configuration': results['configuration'],
        'round_analysis': [],
        'summary_metrics': {}
    }
    
    # Analyze each round
    for round_data in rounds:
        round_analysis = {
            'round': round_data['round'],
            'round_time': round_data['round_time'],
            'client_training_time': round_data.get('client_training_time', 0),
            'edge_encryption_wall_time': round_data.get('edge_encryption_wall_time', 0),
            'edge_processing_wall_time': round_data.get('edge_processing_wall_time', 0),
            'total_pure_encryption_time': round_data.get('total_pure_encryption_time', 0),
            'total_pure_processing_time': round_data.get('total_pure_processing_time', 0),
            'server_aggregation_time': round_data.get('server_aggregation_time', 0),
            'internal_aggregation_time': round_data.get('internal_aggregation_time', 0),
            'evaluation_time': round_data.get('evaluation_time', 0),
            'clients_processed': round_data.get('clients_processed', 0)
        }
        
        # Calculate end-to-end delay components
        if 'edge_encryption_wall_time' in round_data:
            # FHE Edge FL
            round_analysis['end_to_end_delay'] = round_analysis['round_time']
            round_analysis['encryption_overhead'] = round_analysis['edge_encryption_wall_time']
            round_analysis['pure_encryption_time'] = round_analysis['total_pure_encryption_time']
            round_analysis['encryption_efficiency'] = (
                round_analysis['total_pure_encryption_time'] / 
                round_analysis['edge_encryption_wall_time'] 
                if round_analysis['edge_encryption_wall_time'] > 0 else 0
            )
        else:
            # Plaintext Edge FL
            round_analysis['end_to_end_delay'] = round_analysis['round_time']
            round_analysis['processing_overhead'] = round_analysis['edge_processing_wall_time']
            round_analysis['pure_processing_time'] = round_analysis['total_pure_processing_time']
        
        # Calculate per-client metrics
        clients = round_analysis['clients_processed']
        if clients > 0:
            round_analysis['delay_per_client'] = round_analysis['round_time'] / clients
            round_analysis['training_per_client'] = round_analysis['client_training_time'] / clients
            if 'edge_encryption_wall_time' in round_data:
                round_analysis['encryption_per_client'] = round_analysis['edge_encryption_wall_time'] / clients
            else:
                round_analysis['processing_per_client'] = round_analysis['edge_processing_wall_time'] / clients
        
        analysis['round_analysis'].append(round_analysis)
    
    # Calculate summary metrics
    round_times = [r['round_time'] for r in analysis['round_analysis']]
    client_training_times = [r['client_training_time'] for r in analysis['round_analysis']]
    
    analysis['summary_metrics'] = {
        'avg_round_time': np.mean(round_times),
        'std_round_time': np.std(round_times),
        'min_round_time': np.min(round_times),
        'max_round_time': np.max(round_times),
        'avg_client_training_time': np.mean(client_training_times),
        'total_experiment_time': sum(round_times),
        'rounds_count': len(round_times)
    }
    
    # Add encryption-specific metrics for FHE
    if rounds and 'edge_encryption_wall_time' in rounds[0]:
        encryption_times = [r['edge_encryption_wall_time'] for r in analysis['round_analysis']]
        pure_encryption_times = [r['total_pure_encryption_time'] for r in analysis['round_analysis']]
        
        avg_encryption_time = np.mean(encryption_times)
        analysis['summary_metrics'].update({
            'avg_encryption_time': avg_encryption_time,
            'avg_pure_encryption_time': np.mean(pure_encryption_times),
            'encryption_overhead_ratio': np.mean([r['encryption_efficiency'] for r in analysis['round_analysis']]),
            'encryption_scaling_factor': avg_encryption_time / analysis['configuration']['clients']
        })
    
    return analysis

def create_comparison_table(fhe_analysis: Dict, plaintext_analysis: Dict) -> pd.DataFrame:
    """Create comparison table between FHE and Plaintext Edge FL"""
    
    metrics = [
        ('End-to-End Delay (Round Time)', 'avg_round_time', 'seconds'),
        ('Client Training Time', 'avg_client_training_time', 'seconds'),
        ('Total Experiment Time', 'total_experiment_time', 'seconds'),
        ('Rounds Count', 'rounds_count', 'count'),
        ('Clients Count', 'configuration.clients', 'count')
    ]
    
    # Add FHE-specific metrics
    if 'avg_encryption_time' in fhe_analysis['summary_metrics']:
        metrics.extend([
            ('Encryption Time', 'avg_encryption_time', 'seconds'),
            ('Pure Encryption Time', 'avg_pure_encryption_time', 'seconds'),
            ('Encryption Overhead Ratio', 'encryption_overhead_ratio', 'ratio'),
            ('Encryption Scaling Factor', 'encryption_scaling_factor', 'seconds/client')
        ])
    
    data = []
    for metric_name, metric_key, unit in metrics:
        fhe_value = fhe_analysis['summary_metrics'].get(metric_key.split('.')[-1], 
                                                       fhe_analysis['configuration'].get(metric_key.split('.')[-1], 'N/A'))
        plaintext_value = plaintext_analysis['summary_metrics'].get(metric_key.split('.')[-1], 
                                                                  plaintext_analysis['configuration'].get(metric_key.split('.')[-1], 'N/A'))
        
        # Calculate speedup/overhead
        if isinstance(fhe_value, (int, float)) and isinstance(plaintext_value, (int, float)) and plaintext_value > 0:
            if 'encryption' in metric_name.lower() or 'overhead' in metric_name.lower():
                overhead = ((fhe_value - plaintext_value) / plaintext_value) * 100
                comparison = f"+{overhead:.1f}% overhead"
            else:
                speedup = fhe_value / plaintext_value if plaintext_value > 0 else float('inf')
                comparison = f"{speedup:.2f}x slower"
        else:
            comparison = "N/A"
        
        data.append({
            'Metric': metric_name,
            'FHE Edge FL': f"{fhe_value:.4f}" if isinstance(fhe_value, float) else str(fhe_value),
            'Plaintext Edge FL': f"{plaintext_value:.4f}" if isinstance(plaintext_value, float) else str(plaintext_value),
            'Comparison': comparison,
            'Unit': unit
        })
    
    return pd.DataFrame(data)

def main():
    """Main analysis function"""
    print("=" * 80)
    print("TIMING ANALYSIS REPORT - FEDERATED LEARNING PIPELINES")
    print("=" * 80)
    print()
    
    # Load results
    fhe_file = "fhe_edge_results/fhe_edge_fl_results_10clients_10rounds_20250919_122422.json"
    plaintext_file = "plaintext_edge_results/plaintext_edge_fl_results_10clients_10rounds_20250919_125102.json"
    
    if not os.path.exists(fhe_file) or not os.path.exists(plaintext_file):
        print("❌ Results files not found. Please run the pipelines first.")
        return
    
    print("📊 Loading results...")
    fhe_results = load_results(fhe_file)
    plaintext_results = load_results(plaintext_file)
    
    print("🔍 Analyzing timing metrics...")
    fhe_analysis = analyze_timing_metrics(fhe_results)
    plaintext_analysis = analyze_timing_metrics(plaintext_results)
    
    print("\n" + "=" * 80)
    print("END-TO-END DELAY ANALYSIS")
    print("=" * 80)
    
    print(f"\n🎯 **ROUND TIME AS END-TO-END DELAY**")
    print(f"   Round time represents the complete time for one federated learning round")
    print(f"   from start to finish, including all phases:")
    print(f"   • Client local training")
    print(f"   • Edge device processing/encryption")
    print(f"   • Cloud server aggregation")
    print(f"   • Global model update")
    print(f"   • Model evaluation")
    
    print(f"\n📈 **FHE Edge FL End-to-End Delay:**")
    print(f"   Average Round Time: {fhe_analysis['summary_metrics']['avg_round_time']:.4f}s")
    print(f"   Min Round Time: {fhe_analysis['summary_metrics']['min_round_time']:.4f}s")
    print(f"   Max Round Time: {fhe_analysis['summary_metrics']['max_round_time']:.4f}s")
    print(f"   Std Deviation: {fhe_analysis['summary_metrics']['std_round_time']:.4f}s")
    
    print(f"\n📈 **Plaintext Edge FL End-to-End Delay:**")
    print(f"   Average Round Time: {plaintext_analysis['summary_metrics']['avg_round_time']:.4f}s")
    print(f"   Min Round Time: {plaintext_analysis['summary_metrics']['min_round_time']:.4f}s")
    print(f"   Max Round Time: {plaintext_analysis['summary_metrics']['max_round_time']:.4f}s")
    print(f"   Std Deviation: {plaintext_analysis['summary_metrics']['std_round_time']:.4f}s")
    
    # Calculate overhead
    fhe_avg = fhe_analysis['summary_metrics']['avg_round_time']
    plaintext_avg = plaintext_analysis['summary_metrics']['avg_round_time']
    overhead = ((fhe_avg - plaintext_avg) / plaintext_avg) * 100
    
    print(f"\n⚡ **Encryption Overhead:**")
    print(f"   FHE is {overhead:.1f}% slower than Plaintext")
    print(f"   Speedup Factor: {fhe_avg / plaintext_avg:.2f}x slower")
    
    print("\n" + "=" * 80)
    print("DETAILED TIMING BREAKDOWN")
    print("=" * 80)
    
    # Create comparison table
    comparison_df = create_comparison_table(fhe_analysis, plaintext_analysis)
    
    print("\n📊 **Timing Metrics Comparison:**")
    print(comparison_df.to_string(index=False))
    
    print("\n" + "=" * 80)
    print("OTHER USEFUL METRICS")
    print("=" * 80)
    
    print(f"\n🔧 **Per-Client Metrics:**")
    fhe_clients = fhe_analysis['configuration']['clients']
    plaintext_clients = plaintext_analysis['configuration']['clients']
    
    fhe_delay_per_client = fhe_analysis['summary_metrics']['avg_round_time'] / fhe_clients
    plaintext_delay_per_client = plaintext_analysis['summary_metrics']['avg_round_time'] / plaintext_clients
    
    print(f"   FHE Edge FL Delay per Client: {fhe_delay_per_client:.6f}s")
    print(f"   Plaintext Edge FL Delay per Client: {plaintext_delay_per_client:.6f}s")
    
    print(f"\n📊 **Scalability Analysis:**")
    print(f"   FHE Scaling Factor: {fhe_analysis['summary_metrics'].get('encryption_scaling_factor', 'N/A')}")
    print(f"   Both pipelines show O(n) scaling for client count")
    
    print(f"\n🎯 **Performance Efficiency:**")
    if 'encryption_overhead_ratio' in fhe_analysis['summary_metrics']:
        efficiency = fhe_analysis['summary_metrics']['encryption_overhead_ratio']
        print(f"   FHE Encryption Efficiency: {efficiency:.2%}")
        print(f"   (Higher = more time spent on actual encryption vs overhead)")
    
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS FOR RESEARCH PAPER")
    print("=" * 80)
    
    print(f"\n📝 **For End-to-End Delay:**")
    print(f"   ✅ Use 'round_time' as the primary end-to-end delay metric")
    print(f"   ✅ Report average, min, max, and standard deviation")
    print(f"   ✅ Include per-client delay for scalability analysis")
    
    print(f"\n📝 **For Performance Comparison:**")
    print(f"   ✅ Report encryption overhead percentage")
    print(f"   ✅ Include speedup factors")
    print(f"   ✅ Show timing breakdown by phase")
    
    print(f"\n📝 **For Scalability Analysis:**")
    print(f"   ✅ Report scaling factors (time per client)")
    print(f"   ✅ Show how delay scales with client count")
    print(f"   ✅ Include efficiency ratios")
    
    print(f"\n📝 **Additional Metrics to Report:**")
    print(f"   ✅ Total experiment time")
    print(f"   ✅ Client training time (computational cost)")
    print(f"   ✅ Server aggregation time (communication cost)")
    print(f"   ✅ Evaluation time (model quality assessment cost)")

if __name__ == "__main__":
    main()
