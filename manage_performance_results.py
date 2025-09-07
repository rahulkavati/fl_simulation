#!/usr/bin/env python3
"""
Performance Results Manager
Manages timestamped performance evaluation results and provides comparison tools
"""

import os
import json
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns

class PerformanceResultsManager:
    """Manages timestamped performance evaluation results"""
    
    def __init__(self):
        self.results_dirs = self.find_performance_dirs()
        
    def find_performance_dirs(self) -> List[str]:
        """Find all performance results directories"""
        dirs = []
        for item in os.listdir('.'):
            if os.path.isdir(item) and item.startswith('performance_results_'):
                dirs.append(item)
        return sorted(dirs)
    
    def list_all_results(self) -> None:
        """List all available performance results"""
        print("📊 Available Performance Results")
        print("=" * 50)
        
        if not self.results_dirs:
            print("❌ No performance results found")
            return
        
        for i, dir_name in enumerate(self.results_dirs, 1):
            timestamp = dir_name.replace('performance_results_', '')
            try:
                dt = datetime.strptime(timestamp, '%Y%m%d_%H%M%S')
                formatted_time = dt.strftime('%Y-%m-%d %H:%M:%S')
            except:
                formatted_time = timestamp
            
            # Check if directory has results
            files = os.listdir(dir_name)
            chart_count = len([f for f in files if f.endswith('.png')])
            report_count = len([f for f in files if f.endswith('.md')])
            
            print(f"{i:2d}. {formatted_time}")
            print(f"    Directory: {dir_name}")
            print(f"    Charts: {chart_count}, Reports: {report_count}")
            print()
    
    def create_comparison_summary(self) -> None:
        """Create a summary comparing all performance results"""
        if len(self.results_dirs) < 2:
            print("❌ Need at least 2 performance results to create comparison")
            return
        
        print("📈 Creating Performance Comparison Summary")
        print("=" * 50)
        
        comparison_data = []
        
        for dir_name in self.results_dirs:
            timestamp = dir_name.replace('performance_results_', '')
            try:
                dt = datetime.strptime(timestamp, '%Y%m%d_%H%M%S')
                formatted_time = dt.strftime('%Y-%m-%d %H:%M:%S')
            except:
                formatted_time = timestamp
            
            # Try to extract metrics from the directory
            metrics = self.extract_metrics_from_dir(dir_name)
            if metrics:
                metrics['timestamp'] = formatted_time
                metrics['directory'] = dir_name
                comparison_data.append(metrics)
        
        if comparison_data:
            # Create comparison DataFrame
            df = pd.DataFrame(comparison_data)
            
            # Save comparison
            comparison_file = f"performance_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(comparison_file, index=False)
            
            print(f"✅ Comparison summary saved to: {comparison_file}")
            print("\n📊 Comparison Summary:")
            print(df.to_string(index=False))
            
            # Create comparison chart
            self.create_comparison_chart(df)
        else:
            print("❌ Could not extract metrics from directories")
    
    def extract_metrics_from_dir(self, dir_name: str) -> Dict[str, Any]:
        """Extract key metrics from a performance results directory"""
        metrics = {}
        
        # Try to read the evaluation report
        report_file = os.path.join(dir_name, 'PERFORMANCE_EVALUATION_REPORT.md')
        if os.path.exists(report_file):
            with open(report_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
                # Extract key metrics (simplified extraction)
                if 'Encrypted Accuracy' in content:
                    try:
                        # This is a simplified extraction - in practice you'd use regex
                        lines = content.split('\n')
                        for line in lines:
                            if 'Encrypted Accuracy' in line:
                                metrics['accuracy'] = float(line.split(':')[1].strip())
                            elif 'CKKS Overhead' in line:
                                metrics['computation_overhead'] = float(line.split('+')[1].split('%')[0].strip())
                            elif 'Expansion Factor' in line:
                                metrics['communication_expansion'] = float(line.split(':')[1].split('x')[0].strip())
                            elif 'Energy Overhead' in line:
                                metrics['energy_overhead'] = float(line.split('+')[1].split('%')[0].strip())
                    except:
                        pass
        
        return metrics
    
    def create_comparison_chart(self, df: pd.DataFrame) -> None:
        """Create comparison charts for different runs"""
        if df.empty:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy comparison
        if 'accuracy' in df.columns:
            axes[0, 0].plot(df['timestamp'], df['accuracy'], 'o-', linewidth=2, markersize=8)
            axes[0, 0].set_title('Accuracy Over Time', fontweight='bold')
            axes[0, 0].set_ylabel('Accuracy')
            axes[0, 0].tick_params(axis='x', rotation=45)
            axes[0, 0].grid(True, alpha=0.3)
        
        # Computation overhead comparison
        if 'computation_overhead' in df.columns:
            axes[0, 1].plot(df['timestamp'], df['computation_overhead'], 's-', linewidth=2, markersize=8, color='red')
            axes[0, 1].set_title('Computation Overhead Over Time', fontweight='bold')
            axes[0, 1].set_ylabel('Overhead (%)')
            axes[0, 1].tick_params(axis='x', rotation=45)
            axes[0, 1].grid(True, alpha=0.3)
        
        # Communication expansion comparison
        if 'communication_expansion' in df.columns:
            axes[1, 0].plot(df['timestamp'], df['communication_expansion'], '^-', linewidth=2, markersize=8, color='green')
            axes[1, 0].set_title('Communication Expansion Over Time', fontweight='bold')
            axes[1, 0].set_ylabel('Expansion Factor (x)')
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].grid(True, alpha=0.3)
        
        # Energy overhead comparison
        if 'energy_overhead' in df.columns:
            axes[1, 1].plot(df['timestamp'], df['energy_overhead'], 'd-', linewidth=2, markersize=8, color='orange')
            axes[1, 1].set_title('Energy Overhead Over Time', fontweight='bold')
            axes[1, 1].set_ylabel('Overhead (%)')
            axes[1, 1].tick_params(axis='x', rotation=45)
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save comparison chart
        comparison_chart = f"performance_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(comparison_chart, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Comparison chart saved to: {comparison_chart}")
    
    def cleanup_old_results(self, keep_days: int = 30) -> None:
        """Clean up old performance results (keep only recent ones)"""
        print(f"🧹 Cleaning up performance results older than {keep_days} days")
        print("=" * 50)
        
        cutoff_date = datetime.now().timestamp() - (keep_days * 24 * 60 * 60)
        removed_count = 0
        
        for dir_name in self.results_dirs:
            try:
                timestamp = dir_name.replace('performance_results_', '')
                dt = datetime.strptime(timestamp, '%Y%m%d_%H%M%S')
                
                if dt.timestamp() < cutoff_date:
                    print(f"🗑️ Removing old results: {dir_name}")
                    import shutil
                    shutil.rmtree(dir_name)
                    removed_count += 1
            except:
                print(f"⚠️ Could not parse timestamp for: {dir_name}")
        
        print(f"✅ Removed {removed_count} old performance result directories")
    
    def create_latest_summary(self) -> None:
        """Create a summary of the latest performance results"""
        if not self.results_dirs:
            print("❌ No performance results found")
            return
        
        latest_dir = self.results_dirs[-1]
        print(f"📊 Latest Performance Results Summary")
        print(f"Directory: {latest_dir}")
        print("=" * 50)
        
        # List files in latest directory
        files = os.listdir(latest_dir)
        print(f"Files in latest results:")
        for file in sorted(files):
            file_path = os.path.join(latest_dir, file)
            size = os.path.getsize(file_path)
            print(f"  - {file} ({size:,} bytes)")
        
        # Try to extract key metrics
        metrics = self.extract_metrics_from_dir(latest_dir)
        if metrics:
            print(f"\nKey Metrics:")
            for key, value in metrics.items():
                if isinstance(value, float):
                    print(f"  - {key}: {value:.3f}")
                else:
                    print(f"  - {key}: {value}")

def main():
    """Main function"""
    manager = PerformanceResultsManager()
    
    print("🔬 Performance Results Manager")
    print("=" * 40)
    
    while True:
        print("\nOptions:")
        print("1. List all performance results")
        print("2. Create comparison summary")
        print("3. Show latest results summary")
        print("4. Cleanup old results")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == '1':
            manager.list_all_results()
        elif choice == '2':
            manager.create_comparison_summary()
        elif choice == '3':
            manager.create_latest_summary()
        elif choice == '4':
            days = input("Enter number of days to keep (default 30): ").strip()
            try:
                keep_days = int(days) if days else 30
                manager.cleanup_old_results(keep_days)
            except ValueError:
                print("❌ Invalid number of days")
        elif choice == '5':
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please enter 1-5.")

if __name__ == "__main__":
    main()
