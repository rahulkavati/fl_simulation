#!/usr/bin/env python3
"""
Example: FHE CKKS vs Plain Text Federated Learning Comparison
Demonstrates the usage of the comparison framework
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core.base_pipeline import PipelineConfig
from src.core.comparison_engine import FederatedLearningComparison

def run_example_comparison():
    """Run a simple example comparison"""
    print("🔬 FHE CKKS vs Plain Text FL Comparison Example")
    print("=" * 60)
    
    # Create configuration
    config = PipelineConfig(
        rounds=5,
        clients=8,
        min_samples_per_client=50,
        data_path="data/health_fitness_data.csv",
        track_detailed_metrics=True
    )
    
    print(f"⚙️  Configuration:")
    print(f"   • Rounds: {config.rounds}")
    print(f"   • Clients: {config.clients}")
    print(f"   • Data: {config.data_path}")
    print()
    
    try:
        # Create comparison engine
        comparison = FederatedLearningComparison(config)
        
        # Run comparison
        print("🚀 Running comparison...")
        result = comparison.run_comprehensive_comparison(num_runs=2)
        
        # Print results
        print("\n📊 Results Summary:")
        print(f"   • FHE Accuracy: {result.fhe_result.final_accuracy:.4f}")
        print(f"   • Plain Text Accuracy: {result.plaintext_result.final_accuracy:.4f}")
        print(f"   • Accuracy Difference: {result.metrics.accuracy_diff:+.4f}")
        print(f"   • Training Time Overhead: {result.metrics.training_time_overhead:.1f}%")
        print(f"   • Privacy Score: {result.metrics.privacy_score:.1f}")
        
        if result.metrics.statistical_significance:
            print("   • Statistical Significance: ✅ Yes")
        else:
            print("   • Statistical Significance: ❌ No")
        
        print("\n💡 Recommendations:")
        for i, rec in enumerate(result.recommendations[:3], 1):
            print(f"   {i}. {rec}")
        
        print(f"\n🎯 Conclusion: {result.conclusion}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

if __name__ == "__main__":
    result = run_example_comparison()
    
    if result:
        print("\n✅ Example completed successfully!")
        print("📁 Check the 'comparisons/' directory for detailed results")
    else:
        print("\n❌ Example failed")
        sys.exit(1)
