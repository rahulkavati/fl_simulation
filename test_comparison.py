#!/usr/bin/env python3
"""
Test script to run FHE CKKS vs Plain Text comparison
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core.base_pipeline import PipelineConfig
from src.core.comparison_engine import FederatedLearningComparison

def test_comparison():
    """Test the FHE vs Plain Text comparison"""
    print("🔬 Testing FHE CKKS vs Plain Text FL Comparison")
    print("=" * 60)
    
    try:
        # Create configuration
        config = PipelineConfig(
            rounds=3,  # Small number for testing
            clients=5,  # Small number for testing
            min_samples_per_client=50,
            data_path="data/health_fitness_data.csv",
            track_detailed_metrics=True
        )
        
        print(f"⚙️  Configuration:")
        print(f"   • Rounds: {config.rounds}")
        print(f"   • Clients: {config.clients}")
        print(f"   • Data: {config.data_path}")
        print()
        
        # Create comparison engine
        print("🚀 Creating comparison engine...")
        comparison = FederatedLearningComparison(config)
        
        # Run comparison
        print("🔄 Running comparison...")
        result = comparison.run_comprehensive_comparison(num_runs=1)  # Single run for testing
        
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
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = test_comparison()
    
    if result:
        print("\n✅ Test completed successfully!")
        print("📁 Check the 'comparisons/' directory for detailed results")
    else:
        print("\n❌ Test failed")
        sys.exit(1)
