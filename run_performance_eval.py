#!/usr/bin/env python3
"""
Quick Performance Evaluation Runner
Runs performance evaluation with optional custom naming
"""

import sys
import os
from datetime import datetime
from performance_evaluation import PerformanceEvaluator

def main():
    """Main function"""
    print("🚀 Quick Performance Evaluation Runner")
    print("=" * 50)
    
    # Check for custom name argument
    if len(sys.argv) > 1:
        custom_name = sys.argv[1]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"performance_results_{custom_name}_{timestamp}"
        
        print(f"📁 Custom output directory: {output_dir}")
        
        # Create custom evaluator
        evaluator = PerformanceEvaluator()
        evaluator.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Run evaluation
        evaluator.load_round_data()
        evaluator.load_summary_data()
        evaluator.create_accuracy_f1_chart()
        evaluator.create_computation_time_chart()
        evaluator.create_communication_size_chart()
        evaluator.create_end_to_end_delay_chart()
        evaluator.create_energy_consumption_chart()
        evaluator.generate_performance_report()
        
        print(f"\n🎉 Performance evaluation complete!")
        print(f"📁 Results saved to: {output_dir}/")
    else:
        # Run with default timestamp
        print("📁 Using default timestamped directory")
        evaluator = PerformanceEvaluator()
        evaluator.run_complete_evaluation()

if __name__ == "__main__":
    main()
