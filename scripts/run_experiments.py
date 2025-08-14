#!/usr/bin/env python3
"""
Run multiple FL experiments with different configurations

This script runs the FL simulation multiple times with different parameters
to generate comprehensive efficiency metrics for analysis.
"""

import os
import sys
import subprocess
import time
from datetime import datetime

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_experiment(num_rounds, num_clients=None):
    """Run a single FL experiment"""
    print(f"\n{'='*60}")
    print(f"Running Experiment: {num_rounds} rounds")
    print(f"{'='*60}")
    
    # Modify the simulation parameters
    sim_file = os.path.join(os.path.dirname(__file__), "..", "simulation", "client_simulation.py")
    
    # Read the current simulation file
    with open(sim_file, 'r') as f:
        content = f.read()
    
    # Find and replace NUM_ROUNDS
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if line.startswith('NUM_ROUNDS ='):
            lines[i] = f"NUM_ROUNDS = {num_rounds}"
            break
    
    # Write back
    with open(sim_file, 'w') as f:
        f.write('\n'.join(lines))
    
    # Run the simulation
    try:
        result = subprocess.run(
            ["python3", sim_file],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(sim_file)
        )
        
        if result.returncode == 0:
            print("✅ Experiment completed successfully!")
            # Extract key metrics from output
            for line in result.stdout.split('\n'):
                if 'Final Accuracy:' in line:
                    print(f"   {line.strip()}")
                elif 'Total training time:' in line:
                    print(f"   {line.strip()}")
        else:
            print("❌ Experiment failed!")
            print(f"Error: {result.stderr}")
            
    except Exception as e:
        print(f"❌ Error running experiment: {e}")
    
    # Restore original NUM_ROUNDS
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if line.startswith('NUM_ROUNDS ='):
            lines[i] = f"NUM_ROUNDS = 3"
            break
    
    with open(sim_file, 'w') as f:
        f.write('\n'.join(lines))

def main():
    """Run multiple experiments"""
    print("FL Experiment Runner")
    print("="*40)
    
    # Get current NUM_ROUNDS value
    sim_file = os.path.join(os.path.dirname(__file__), "..", "simulation", "client_simulation.py")
    with open(sim_file, 'r') as f:
        content = f.read()
        for line in content.split('\n'):
            if line.startswith('NUM_ROUNDS ='):
                NUM_ROUNDS = int(line.split('=')[1].strip())
                break
    
    print(f"Current NUM_ROUNDS: {NUM_ROUNDS}")
    
    # Run experiments with different numbers of rounds
    experiments = [2, 3, 4, 5]
    
    for num_rounds in experiments:
        run_experiment(num_rounds)
        time.sleep(1)  # Small delay between experiments
    
    print(f"\n{'='*60}")
    print("All experiments completed!")
    print("Run 'python3 visualize/metrics_analysis.py' to analyze results")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
