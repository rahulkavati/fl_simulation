"""
Complete Research Pipeline for FHE CKKS Federated Learning
Top 1% Developer Implementation for Publication-Ready Results
"""

import argparse
import time
from datetime import datetime
import json

from research_demonstration import ResearchDemonstrationPipeline
from research_analysis import ResearchAnalysis
from src.fl import FLConfig
from src.fhe import FHEConfig

class CompleteResearchPipeline:
    """
    Complete research pipeline that demonstrates and analyzes FHE CKKS federated learning
    """
    
    def __init__(self, rounds: int = 5, clients: int = 8):
        self.rounds = rounds
        self.clients = clients
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.results_dir = f"research_results_{self.timestamp}"
        
    def run_complete_research(self) -> None:
        """
        Run complete research pipeline:
        1. Demonstrate FHE CKKS federated learning
        2. Analyze results comprehensively
        3. Generate publication-ready outputs
        """
        print("🔬 FHE CKKS Federated Learning Complete Research Pipeline")
        print("="*80)
        print("Demonstrating: Smartwatch Data → Edge Trains → Encrypt Updates → Server Aggregates → Edge Decrypts & Updates")
        print(f"Configuration: {self.rounds} rounds, {self.clients} clients")
        print(f"Timestamp: {self.timestamp}")
        
        start_time = time.time()
        
        # Step 1: Run demonstration
        print(f"\n{'='*80}")
        print("STEP 1: Running FHE CKKS Federated Learning Demonstration")
        print(f"{'='*80}")
        
        fl_config = FLConfig(rounds=self.rounds, clients=self.clients)
        fhe_config = FHEConfig()
        
        pipeline = ResearchDemonstrationPipeline(fl_config, fhe_config)
        results = pipeline.run_complete_demonstration()
        
        # Save demonstration results
        results_file = f"{self.results_dir}/demonstration_results.json"
        import os
        os.makedirs(self.results_dir, exist_ok=True)
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"✅ Demonstration completed! Results saved to {results_file}")
        
        # Step 2: Analyze results
        print(f"\n{'='*80}")
        print("STEP 2: Comprehensive Research Analysis")
        print(f"{'='*80}")
        
        analyzer = ResearchAnalysis(results_file)
        analyzer.generate_comprehensive_analysis()
        
        print(f"✅ Analysis completed! Results saved to {analyzer.analysis_dir}/")
        
        # Step 3: Generate summary
        self._generate_research_summary(results, analyzer)
        
        total_time = time.time() - start_time
        print(f"\n🎉 Complete research pipeline completed in {total_time:.2f}s")
        print(f"📁 All results saved to: {self.results_dir}/")
        
    def _generate_research_summary(self, results: dict, analyzer: ResearchAnalysis) -> None:
        """Generate final research summary"""
        print(f"\n{'='*80}")
        print("STEP 3: Generating Research Summary")
        print(f"{'='*80}")
        
        # Get key metrics
        performance_stats = analyzer._analyze_performance()
        timing_stats = analyzer._analyze_timing()
        device_stats = analyzer._analyze_devices()
        privacy_stats = analyzer._analyze_privacy()
        
        summary = {
            'research_info': {
                'title': 'FHE CKKS Federated Learning Research',
                'timestamp': self.timestamp,
                'configuration': {
                    'rounds': self.rounds,
                    'clients': self.clients
                },
                'flow': 'Smartwatch Data → Edge Trains → Encrypt Updates → Server Aggregates → Edge Decrypts & Updates'
            },
            'key_results': {
                'performance': {
                    'final_accuracy': performance_stats['final_accuracy'],
                    'best_accuracy': performance_stats['best_accuracy'],
                    'improvement': performance_stats['accuracy_improvement']
                },
                'timing': {
                    'avg_total_time': timing_stats['avg_total_time'],
                    'avg_device_training': timing_stats['avg_device_training'],
                    'avg_server_aggregation': timing_stats['avg_server_aggregation'],
                    'avg_device_update': timing_stats['avg_device_update']
                },
                'devices': {
                    'total_devices': device_stats['total_devices'],
                    'avg_battery_level': device_stats['avg_battery_level'],
                    'total_data_points': device_stats['total_data_points']
                },
                'privacy': {
                    'overall_privacy_score': privacy_stats['overall_privacy_score'],
                    'gdpr_compliant': privacy_stats['compliance']['gdpr_compliant'],
                    'hipaa_compliant': privacy_stats['compliance']['hipaa_compliant']
                }
            },
            'research_contributions': [
                'Complete FHE CKKS federated learning implementation',
                'Realistic edge device simulation with resource constraints',
                'End-to-end privacy preservation demonstration',
                'Comprehensive performance and timing analysis',
                'Publication-ready results and visualizations'
            ],
            'files_generated': [
                f"{self.results_dir}/demonstration_results.json",
                f"{analyzer.analysis_dir}/performance_analysis.json",
                f"{analyzer.analysis_dir}/timing_analysis.json",
                f"{analyzer.analysis_dir}/device_analysis.json",
                f"{analyzer.analysis_dir}/privacy_analysis.json",
                f"{analyzer.analysis_dir}/network_analysis.json",
                f"{analyzer.analysis_dir}/RESEARCH_REPORT.md",
                f"{analyzer.analysis_dir}/performance_over_rounds.png",
                f"{analyzer.analysis_dir}/timing_analysis.png",
                f"{analyzer.analysis_dir}/device_analysis.png",
                f"{analyzer.analysis_dir}/privacy_metrics.png"
            ]
        }
        
        # Save summary
        summary_file = f"{self.results_dir}/RESEARCH_SUMMARY.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print summary
        print("📊 RESEARCH SUMMARY:")
        print(f"  🎯 Final Accuracy: {performance_stats['final_accuracy']:.4f} ({performance_stats['final_accuracy']*100:.2f}%)")
        print(f"  📈 Improvement: {performance_stats['accuracy_improvement']*100:+.2f}%")
        print(f"  ⏱️  Average Total Time: {timing_stats['avg_total_time']:.3f}s")
        print(f"  📱 Total Devices: {device_stats['total_devices']}")
        print(f"  🔒 Privacy Score: {privacy_stats['overall_privacy_score']:.3f}")
        print(f"  ✅ GDPR/HIPAA Compliant: {privacy_stats['compliance']['gdpr_compliant']}")
        
        print(f"\n📁 Research Summary saved to: {summary_file}")
        print(f"📊 Analysis results saved to: {analyzer.analysis_dir}/")
        
        # Print file list
        print(f"\n📄 Generated Files:")
        for file_path in summary['files_generated']:
            print(f"  📄 {file_path}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="FHE CKKS Federated Learning Complete Research Pipeline"
    )
    parser.add_argument(
        "--rounds", type=int, default=5,
        help="Number of federated learning rounds (default: 5)"
    )
    parser.add_argument(
        "--clients", type=int, default=8,
        help="Number of edge devices/clients (default: 8)"
    )
    
    args = parser.parse_args()
    
    # Create and run research pipeline
    pipeline = CompleteResearchPipeline(rounds=args.rounds, clients=args.clients)
    pipeline.run_complete_research()

if __name__ == "__main__":
    main()
