"""
FL Efficiency Metrics Analysis and Visualization

This script provides comprehensive analysis and visualization of FL efficiency metrics
stored in the metrics directory.
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import List, Dict
import numpy as np

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class FLMetricsAnalyzer:
    """Analyze and visualize FL efficiency metrics"""
    
    def __init__(self, metrics_dir: str = "metrics"):
        self.metrics_dir = metrics_dir
        self.metrics_data = None
        self.load_metrics()
    
    def load_metrics(self):
        """Load all metrics from the metrics directory"""
        if not os.path.exists(self.metrics_dir):
            print(f"Metrics directory {self.metrics_dir} not found!")
            return
        
        # Try to load CSV first (more efficient)
        csv_file = os.path.join(self.metrics_dir, "metrics_history.csv")
        if os.path.exists(csv_file):
            self.metrics_data = pd.read_csv(csv_file)
            print(f"Loaded {len(self.metrics_data)} experiments from CSV")
        else:
            # Fallback to individual JSON files
            self.load_from_json_files()
    
    def load_from_json_files(self):
        """Load metrics from individual JSON files"""
        metrics_list = []
        for fname in os.listdir(self.metrics_dir):
            if fname.endswith('.json') and fname != 'metrics_summary.json':
                filepath = os.path.join(self.metrics_dir, fname)
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                    metrics_list.append(data)
                except Exception as e:
                    print(f"Error loading {fname}: {e}")
        
        if metrics_list:
            self.metrics_data = pd.DataFrame(metrics_list)
            print(f"Loaded {len(self.metrics_data)} experiments from JSON files")
        else:
            print("No metrics data found!")
    
    def plot_accuracy_progression(self):
        """Plot accuracy improvement across experiments"""
        if self.metrics_data is None:
            print("No metrics data to plot!")
            return
        
        plt.figure(figsize=(12, 6))
        
        # Plot accuracy improvement
        plt.subplot(1, 2, 1)
        plt.scatter(self.metrics_data['num_clients'], self.metrics_data['final_accuracy'], 
                   alpha=0.7, s=100)
        plt.xlabel('Number of Clients')
        plt.ylabel('Final Accuracy')
        plt.title('Accuracy vs Number of Clients')
        plt.grid(True, alpha=0.3)
        
        # Plot accuracy improvement
        plt.subplot(1, 2, 2)
        plt.scatter(self.metrics_data['num_rounds'], self.metrics_data['accuracy_improvement'], 
                   alpha=0.7, s=100)
        plt.xlabel('Number of Rounds')
        plt.ylabel('Accuracy Improvement')
        plt.title('Accuracy Improvement vs Training Rounds')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.metrics_dir, 'accuracy_analysis.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_efficiency_metrics(self):
        """Plot various efficiency metrics"""
        if self.metrics_data is None:
            print("No metrics data to plot!")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Communication efficiency
        axes[0, 0].scatter(self.metrics_data['total_communication_rounds'], 
                           self.metrics_data['bytes_transferred']/1024, alpha=0.7)
        axes[0, 0].set_xlabel('Communication Rounds')
        axes[0, 0].set_ylabel('Bytes Transferred (KB)')
        axes[0, 0].set_title('Communication Efficiency')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Training time
        axes[0, 1].scatter(self.metrics_data['num_rounds'], 
                           self.metrics_data['total_training_time'], alpha=0.7)
        axes[0, 1].set_xlabel('Number of Rounds')
        axes[0, 1].set_ylabel('Total Training Time (s)')
        axes[0, 1].set_title('Training Time vs Rounds')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Memory usage
        axes[1, 0].scatter(self.metrics_data['num_clients'], 
                           self.metrics_data['memory_usage'], alpha=0.7)
        axes[1, 0].set_xlabel('Number of Clients')
        axes[1, 0].set_ylabel('Memory Usage (MB)')
        axes[1, 0].set_title('Memory Usage vs Clients')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Convergence analysis
        convergence_data = self.metrics_data[self.metrics_data['convergence_rounds'].notna()]
        if not convergence_data.empty:
            axes[1, 1].scatter(convergence_data['num_rounds'], 
                               convergence_data['convergence_rounds'], alpha=0.7)
            axes[1, 1].set_xlabel('Total Rounds')
            axes[1, 1].set_ylabel('Convergence Round')
            axes[1, 1].set_title('Convergence Analysis')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'No convergence data\navailable', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Convergence Analysis')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.metrics_dir, 'efficiency_metrics.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_convergence_trends(self):
        """Plot convergence trends across rounds for individual experiments"""
        if self.metrics_data is None:
            print("No metrics data to plot!")
            return
        
        # Find experiments with convergence data
        convergence_experiments = self.metrics_data[self.metrics_data['convergence_rounds'].notna()]
        
        if convergence_experiments.empty:
            print("No convergence data available for plotting!")
            return
        
        plt.figure(figsize=(12, 8))
        
        for idx, row in convergence_experiments.iterrows():
            # Plot weight change magnitude
            weight_changes = row['weight_change_magnitude']
            if isinstance(weight_changes, str):
                # Handle case where it's stored as string
                weight_changes = eval(weight_changes)
            
            rounds = range(1, len(weight_changes) + 1)
            plt.plot(rounds, weight_changes, marker='o', alpha=0.7, 
                    label=f"Exp {idx}: {row['num_clients']} clients, {row['num_rounds']} rounds")
        
        plt.xlabel('Training Round')
        plt.ylabel('Weight Change Magnitude')
        plt.title('Convergence Trends: Weight Changes Across Rounds')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.metrics_dir, 'convergence_trends.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_summary_report(self):
        """Generate a comprehensive summary report"""
        if self.metrics_data is None:
            print("No metrics data to analyze!")
            return
        
        print("\n" + "="*60)
        print("FL EFFICIENCY METRICS SUMMARY REPORT")
        print("="*60)
        
        # Basic statistics
        print(f"Total Experiments: {len(self.metrics_data)}")
        print(f"Date Range: {self.metrics_data['timestamp'].min()} to {self.metrics_data['timestamp'].max()}")
        
        # Performance metrics
        print(f"\nPERFORMANCE METRICS:")
        print(f"  Average Final Accuracy: {self.metrics_data['final_accuracy'].mean():.4f} ± {self.metrics_data['final_accuracy'].std():.4f}")
        print(f"  Average Accuracy Improvement: {self.metrics_data['accuracy_improvement'].mean():.4f} ± {self.metrics_data['accuracy_improvement'].std():.4f}")
        
        # Efficiency metrics
        print(f"\nEFFICIENCY METRICS:")
        print(f"  Average Training Time: {self.metrics_data['total_training_time'].mean():.2f}s ± {self.metrics_data['total_training_time'].std():.2f}s")
        print(f"  Average Communication Rounds: {self.metrics_data['total_communication_rounds'].mean():.1f} ± {self.metrics_data['total_communication_rounds'].std():.1f}")
        print(f"  Average Memory Usage: {self.metrics_data['memory_usage'].mean():.4f}MB ± {self.metrics_data['memory_usage'].std():.4f}MB")
        
        # Convergence analysis
        convergence_data = self.metrics_data[self.metrics_data['convergence_rounds'].notna()]
        if not convergence_data.empty:
            print(f"\nCONVERGENCE ANALYSIS:")
            print(f"  Experiments that converged: {len(convergence_data)}/{len(self.metrics_data)}")
            print(f"  Average convergence round: {convergence_data['convergence_rounds'].mean():.1f}")
        
        # Recommendations
        print(f"\nRECOMMENDATIONS:")
        best_acc_exp = self.metrics_data.loc[self.metrics_data['final_accuracy'].idxmax()]
        print(f"  Best performing experiment: {best_acc_exp['num_clients']} clients, {best_acc_exp['num_rounds']} rounds")
        print(f"  Achieved accuracy: {best_acc_exp['final_accuracy']:.4f}")
        
        print("="*60)
    
    def save_analysis_report(self):
        """Save analysis report to file"""
        if self.metrics_data is None:
            print("No metrics data to analyze!")
            return
        
        report = {
            "analysis_timestamp": datetime.now().isoformat(),
            "total_experiments": len(self.metrics_data),
            "summary_stats": {
                "accuracy": {
                    "mean": float(self.metrics_data['final_accuracy'].mean()),
                    "std": float(self.metrics_data['final_accuracy'].std()),
                    "min": float(self.metrics_data['final_accuracy'].min()),
                    "max": float(self.metrics_data['final_accuracy'].max())
                },
                "training_time": {
                    "mean": float(self.metrics_data['total_training_time'].mean()),
                    "std": float(self.metrics_data['total_training_time'].std())
                },
                "communication_rounds": {
                    "mean": float(self.metrics_data['total_communication_rounds'].mean()),
                    "std": float(self.metrics_data['total_communication_rounds'].std())
                }
            },
            "best_experiment": self.metrics_data.loc[self.metrics_data['final_accuracy'].idxmax()].to_dict()
        }
        
        report_file = os.path.join(self.metrics_dir, 'analysis_report.json')
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"Analysis report saved to {report_file}")

def main():
    """Main function to run the analysis"""
    print("FL Efficiency Metrics Analysis")
    print("="*40)
    
    analyzer = FLMetricsAnalyzer()
    
    if analyzer.metrics_data is not None:
        # Generate summary report
        analyzer.generate_summary_report()
        
        # Create visualizations
        print("\nGenerating visualizations...")
        analyzer.plot_accuracy_progression()
        analyzer.plot_efficiency_metrics()
        analyzer.plot_convergence_trends()
        
        # Save analysis report
        analyzer.save_analysis_report()
        
        print("\nAnalysis completed! Check the metrics directory for generated files.")
    else:
        print("No metrics data found. Run the simulation first to generate metrics.")

if __name__ == "__main__":
    main()
