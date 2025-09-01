#!/usr/bin/env python3
"""
Complete Federated Learning Pipeline Runner
Executes all steps from data generation to global update automatically

Usage: python run_complete_pipeline.py [--rounds N] [--clients N] [--cleanup]
"""

import argparse
import os
import sys
import subprocess
import time
import json
import shutil
from pathlib import Path
from typing import List, Dict, Optional

def print_step(step_num: int, total_steps: int, description: str):
    """Print a formatted step header"""
    print(f"\n{'='*60}")
    print(f"STEP {step_num}/{total_steps}: {description}")
    print(f"{'='*60}")

def print_success(message: str):
    """Print a success message"""
    print(f"✅ {message}")

def print_error(message: str):
    """Print an error message"""
    print(f"❌ {message}")

def print_info(message: str):
    """Print an info message"""
    print(f"ℹ️  {message}")

def run_command(cmd: str, description: str) -> bool:
    """Run a command and return success status"""
    print(f"\n🔄 {description}")
    print(f"Command: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
        print_success(f"{description} completed successfully")
        if result.stdout.strip():
            print(f"Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"{description} failed")
        print(f"Error: {e.stderr.strip()}")
        return False

def create_directories():
    """Create necessary directories"""
    directories = [
        "data/clients",
        "updates/json",
        "updates/encrypted", 
        "Sriven/outbox",
        "federated_artifacts/global",
        "metrics"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print_success(f"Created directory: {directory}")

def cleanup_outputs():
    """Clean up all output directories"""
    cleanup_dirs = [
        "updates",
        "Sriven/outbox", 
        "federated_artifacts",
        "metrics",
        "data/clients"  # Also clean up data files to force regeneration
    ]
    
    for directory in cleanup_dirs:
        if os.path.exists(directory):
            shutil.rmtree(directory)
            print_success(f"Cleaned up: {directory}")

def check_prerequisites() -> bool:
    """Check if all required files and dependencies exist"""
    print_info("Checking prerequisites...")
    
    # Check required Python packages
    required_packages = ['tenseal', 'torch', 'numpy', 'pandas', 'sklearn']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print_error(f"Missing packages: {', '.join(missing_packages)}")
        print_info("Install with: pip install " + " ".join(missing_packages))
        return False
    
    # Check required files
    required_files = [
        "data/simulate_health_data.py",
        "simulation/client_simulation.py", 
        "Huzaif/prepare_ctx.py",
        "Huzaif/encrypt_update.py",
        "Sriven/smart_switch_tenseal.py",
        "cloud/global_update.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print_error(f"Missing files: {', '.join(missing_files)}")
        return False
    
    print_success("All prerequisites satisfied")
    return True

def step1_generate_data(num_clients: int = 5) -> bool:
    """Step 1: Generate synthetic health data"""
    print_step(1, 6, "Generate Synthetic Health Data")
    
    # Check if data already exists
    if os.path.exists("data/clients/client_0.csv"):
        print_info("Health data already exists, skipping generation")
        return True
    
    cmd = f"python data/simulate_health_data.py --clients {num_clients}"
    return run_command(cmd, f"Generate synthetic health data for {num_clients} clients")

def step2_prepare_encryption_context() -> bool:
    """Step 2: Prepare TenSEAL encryption context"""
    print_step(2, 6, "Prepare Encryption Context")
    
    # Check if keys already exist
    if os.path.exists("Huzaif/keys/params.ctx.b64") and os.path.exists("Huzaif/keys/secret.ctx"):
        print_info("Encryption keys already exist, skipping generation")
        return True
    
    cmd = "python Huzaif/prepare_ctx.py"
    return run_command(cmd, "Generate TenSEAL CKKS encryption context")

def step3_run_client_simulation(num_rounds: int = 3, num_clients: int = 5) -> bool:
    """Step 3: Run federated learning client simulation"""
    print_step(3, 6, "Run Client Simulation")
    
    cmd = f"python simulation/client_simulation.py --rounds {num_rounds} --clients {num_clients}"
    return run_command(cmd, f"Run federated learning simulation for {num_rounds} rounds with {num_clients} clients")

def step4_encrypt_updates() -> bool:
    """Step 4: Encrypt all client updates"""
    print_step(4, 6, "Encrypt Client Updates")
    
    # Find all JSON update files
    json_dir = "updates/json"
    if not os.path.exists(json_dir):
        print_error("No client updates found. Run client simulation first.")
        return False
    
    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
    if not json_files:
        print_error("No JSON update files found")
        return False
    
    print_info(f"Found {len(json_files)} update files to encrypt")
    
    success_count = 0
    for json_file in json_files:
        input_path = os.path.join(json_dir, json_file)
        output_file = f"enc_{json_file}"
        output_path = os.path.join("updates/encrypted", output_file)
        
        cmd = f"python Huzaif/encrypt_update.py --in {input_path} --out {output_path} --ctx Huzaif/keys/secret.ctx"
        
        if run_command(cmd, f"Encrypt {json_file}"):
            success_count += 1
    
    print_success(f"Encrypted {success_count}/{len(json_files)} update files")
    return success_count == len(json_files)

def step5_aggregate_updates() -> bool:
    """Step 5: Aggregate encrypted updates by round"""
    print_step(5, 6, "Aggregate Encrypted Updates")
    
    # Check if encrypted updates exist
    encrypted_dir = "updates/encrypted"
    if not os.path.exists(encrypted_dir):
        print_error("No encrypted updates found. Run encryption step first.")
        return False
    
    encrypted_files = [f for f in os.listdir(encrypted_dir) if f.endswith('.json')]
    if not encrypted_files:
        print_error("No encrypted update files found")
        return False
    
    print_info(f"Found {len(encrypted_files)} encrypted update files")
    
    # Group files by round
    round_files = {}
    for file in encrypted_files:
        # Extract round number from filename (e.g., enc_client_0_round_1.json -> round 1)
        if '_round_' in file:
            round_num = file.split('_round_')[1].split('.')[0]
            if round_num not in round_files:
                round_files[round_num] = []
            round_files[round_num].append(file)
    
    print_info(f"Grouped into {len(round_files)} rounds")
    
    # Aggregate each round separately
    success_count = 0
    for round_num, files in round_files.items():
        print_info(f"Aggregating round {round_num} with {len(files)} files")
        
        # Create temporary directory for this round
        temp_dir = f"updates/encrypted_round_{round_num}"
        os.makedirs(temp_dir, exist_ok=True)
        
        # Copy files for this round to temp directory
        for file in files:
            src = os.path.join(encrypted_dir, file)
            dst = os.path.join(temp_dir, file)
            shutil.copy2(src, dst)
        
        # Run aggregation for this round
        cmd = f"python Sriven/smart_switch_tenseal.py --fedl_dir {temp_dir} --ctx_b64 Huzaif/keys/params.ctx.b64 --out_dir Sriven/outbox"
        
        if run_command(cmd, f"Aggregate round {round_num}"):
            success_count += 1
        
        # Clean up temp directory
        shutil.rmtree(temp_dir)
    
    print_success(f"Aggregated {success_count}/{len(round_files)} rounds")
    return success_count == len(round_files)

def step6_global_model_update() -> bool:
    """Step 6: Update global model with aggregated results"""
    print_step(6, 6, "Update Global Model")
    
    # Check if aggregated results exist
    outbox_dir = "Sriven/outbox"
    if not os.path.exists(outbox_dir):
        print_error("No aggregated results found. Run aggregation step first.")
        return False
    
    agg_files = [f for f in os.listdir(outbox_dir) if f.startswith('agg_round_') and f.endswith('.json')]
    if not agg_files:
        print_error("No aggregation files found")
        return False
    
    print_info(f"Found {len(agg_files)} aggregation files")
    
    # Import and run global update directly
    try:
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from cloud.global_update import CloudServer, load_encrypted_aggregation
        
        # Initialize cloud server
        cloud = CloudServer()
        
        # Update global model for each round
        success_count = 0
        for agg_file in sorted(agg_files):
            agg_path = os.path.join(outbox_dir, agg_file)
            print_info(f"Processing {agg_file}")
            
            try:
                # Load and apply encrypted aggregation
                encrypted_agg = load_encrypted_aggregation(agg_path)
                cloud.update_global_model_encrypted(encrypted_agg)
                success_count += 1
                print_success(f"Updated global model with {agg_file}")
            except Exception as e:
                print_error(f"Failed to update with {agg_file}: {str(e)}")
        
        print_success(f"Updated global model for {success_count}/{len(agg_files)} rounds")
        return success_count > 0
        
    except Exception as e:
        print_error(f"Failed to import or run global update: {str(e)}")
        return False

def display_results():
    """Display final results and summary"""
    print(f"\n{'='*60}")
    print("🎉 PIPELINE EXECUTION COMPLETE")
    print(f"{'='*60}")
    
    # Check outputs
    outputs = {
        "Health Data": "data/clients/",
        "Client Updates": "updates/json/",
        "Encrypted Updates": "updates/encrypted/",
        "Aggregated Results": "Sriven/outbox/",
        "Global Model": "federated_artifacts/global/"
    }
    
    print("\n📊 Generated Outputs:")
    for name, path in outputs.items():
        if os.path.exists(path):
            files = len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])
            print(f"  ✅ {name}: {files} files in {path}")
        else:
            print(f"  ❌ {name}: Not found")
    
    # Show global model artifacts
    global_dir = "federated_artifacts/global"
    if os.path.exists(global_dir):
        print(f"\n🏆 Global Model Artifacts:")
        for file in os.listdir(global_dir):
            if file.endswith('.npz'):
                print(f"  📁 {file}")
    
    print(f"\n🚀 Next Steps:")
    print(f"  • Run: python test_global_update.py (to test the global model)")
    print(f"  • Check: federated_artifacts/global/ (for model snapshots)")
    print(f"  • Review: metrics/ (for performance metrics)")

def main():
    """Main pipeline execution function"""
    parser = argparse.ArgumentParser(description="Complete Federated Learning Pipeline Runner")
    parser.add_argument("--rounds", type=int, default=3, help="Number of federated learning rounds")
    parser.add_argument("--clients", type=int, default=5, help="Number of clients")
    parser.add_argument("--cleanup", action="store_true", help="Clean up outputs before running")
    parser.add_argument("--skip-checks", action="store_true", help="Skip prerequisite checks")
    
    args = parser.parse_args()
    
    print("🔐 Secure Federated Learning Pipeline")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  • Rounds: {args.rounds}")
    print(f"  • Clients: {args.clients}")
    print(f"  • Cleanup: {args.cleanup}")
    
    # Cleanup if requested
    if args.cleanup:
        print_info("Cleaning up previous outputs...")
        cleanup_outputs()
    
    # Create directories
    create_directories()
    
    # Check prerequisites
    if not args.skip_checks and not check_prerequisites():
        print_error("Prerequisites not met. Exiting.")
        sys.exit(1)
    
    # Execute pipeline steps
    steps = [
        lambda: step1_generate_data(args.clients),
        lambda: step2_prepare_encryption_context(),
        lambda: step3_run_client_simulation(args.rounds, args.clients),
        lambda: step4_encrypt_updates(),
        lambda: step5_aggregate_updates(),
        lambda: step6_global_model_update()
    ]
    
    start_time = time.time()
    success_count = 0
    
    for i, step_func in enumerate(steps, 1):
        if step_func():
            success_count += 1
        else:
            print_error(f"Step {i} failed. Pipeline stopped.")
            break
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Display results
    print(f"\n⏱️  Pipeline Execution Time: {execution_time:.2f} seconds")
    print(f"✅ Successful Steps: {success_count}/{len(steps)}")
    
    if success_count == len(steps):
        display_results()
    else:
        print_error("Pipeline execution incomplete. Check errors above.")

if __name__ == "__main__":
    main()
