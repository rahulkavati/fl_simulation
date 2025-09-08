"""
Quick Results Viewer
"""

import json

# Load results
with open('enhanced_fhe_results_47_features_20250907_183029/enhanced_fhe_results_47_features.json', 'r') as f:
    results = json.load(f)

print('🎯 QUICK RESULTS SUMMARY:')
print('='*50)
print(f'📊 Final Accuracy: {results["final_performance"]["accuracy"]:.4f} ({results["final_performance"]["accuracy"]*100:.2f}%)')
print(f'📊 Final F1-Score: {results["final_performance"]["f1_score"]:.4f}')
print(f'📊 Final Precision: {results["final_performance"]["precision"]:.4f}')
print(f'📊 Final Recall: {results["final_performance"]["recall"]:.4f}')
print(f'⏱️  Total Time: {results["performance_metrics"]["total_time"]:.4f}s')
print(f'🔐 Encryption Time: {results["performance_metrics"]["total_encryption_time"]:.4f}s')
print(f'☁️  Aggregation Time: {results["performance_metrics"]["total_aggregation_time"]:.4f}s')
print(f'🔄 Rounds: {len(results["round_results"])}')
print(f'👥 Clients: {results["configuration"]["clients"]}')
print(f'📊 Features: {results["configuration"]["features"]}')
print()
print('📈 ROUND-BY-ROUND RESULTS:')
for round_result in results['round_results']:
    print(f'  Round {round_result["round"]}: Accuracy = {round_result["accuracy"]:.4f} ({round_result["accuracy"]*100:.2f}%)')
