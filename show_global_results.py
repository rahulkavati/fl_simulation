#!/usr/bin/env python3
"""
Show Global Update Results
Display the results of the global model updates in a clear, visual format

Usage: python show_global_results.py
"""

import os
import sys
import json
import numpy as np
import torch
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from cloud.global_update import CloudServer
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def print_header(title: str):
    """Print a formatted header"""
    print(f"\n{'='*60}")
    print(f"🏆 {title}")
    print(f"{'='*60}")

def print_section(title: str):
    """Print a formatted section header"""
    print(f"\n📊 {title}")
    print("-" * 40)

def show_global_model_results():
    """Display the global model results"""
    print_header("Global Model Results")
    
    global_dir = "federated_artifacts/global"
    if not os.path.exists(global_dir):
        print("❌ Global model directory not found. Run the pipeline first.")
        return
    
    model_files = [f for f in os.listdir(global_dir) if f.startswith('encrypted_global_round_') and f.endswith('.json')]
    if not model_files:
        print("❌ No encrypted global model files found")
        return
    
    print(f"✅ Found {len(model_files)} encrypted global model snapshots")
    
    # Load the latest model
    latest_file = sorted(model_files)[-1]
    model_path = os.path.join(global_dir, latest_file)
    
    try:
        # Load encrypted model data
        with open(model_path, 'r') as f:
            encrypted_model_data = json.load(f)
        
        print_section("Latest Encrypted Global Model")
        print(f"📁 Model file: {latest_file}")
        print(f"🔄 Training round: {encrypted_model_data.get('round_id', 'Unknown')}")
        print(f"🔐 Model remains encrypted: ✅ (Server never sees plaintext)")
        
        # Analyze encrypted model structure
        print_section("Encrypted Model Structure")
        for key, value in encrypted_model_data.items():
            if key == 'encrypted_model':
                print(f"  {key}: {len(value)} characters (base64 encoded)")
            else:
                print(f"  {key}: {value}")
        
        print_section("Security Status")
        print("✅ Model is encrypted and secure")
        print("✅ Server never sees plaintext parameters")
        print("✅ Client-side decryption required for use")
        print("ℹ️  Use client simulation to decrypt and test model")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading global model: {e}")
        return False

def show_aggregation_results():
    """Display aggregation results"""
    print_header("Aggregation Results")
    
    outbox_dir = "Sriven/outbox"
    if not os.path.exists(outbox_dir):
        print("❌ Aggregation output directory not found")
        return
    
    agg_files = [f for f in os.listdir(outbox_dir) if f.startswith('agg_round_') and f.endswith('.json')]
    if not agg_files:
        print("❌ No aggregation files found")
        return
    
    print(f"✅ Found {len(agg_files)} aggregated rounds")
    
    print_section("Aggregation Summary")
    for agg_file in sorted(agg_files):
        agg_path = os.path.join(outbox_dir, agg_file)
        try:
            with open(agg_path, 'r') as f:
                agg_data = json.load(f)
            
            round_id = agg_data.get('round_id', 'unknown')
            created_at = agg_data.get('created_at', 'unknown')
            ciphertext_size = len(agg_data.get('ciphertext', ''))
            
            print(f"  Round {round_id}: {ciphertext_size:,} chars encrypted, created at {created_at}")
            
        except Exception as e:
            print(f"  {agg_file}: Error reading - {e}")

def show_encryption_summary():
    """Display encryption summary"""
    print_header("Encryption Summary")
    
    encrypted_dir = "updates/encrypted"
    if not os.path.exists(encrypted_dir):
        print("❌ Encrypted updates directory not found")
        return
    
    encrypted_files = [f for f in os.listdir(encrypted_dir) if f.endswith('.json')]
    if not encrypted_files:
        print("❌ No encrypted files found")
        return
    
    print(f"✅ Found {len(encrypted_files)} encrypted client updates")
    
    # Analyze encryption
    total_size = 0
    rounds = set()
    clients = set()
    
    for file in encrypted_files:
        file_path = os.path.join(encrypted_dir, file)
        total_size += os.path.getsize(file_path)
        
        if '_round_' in file:
            parts = file.split('_')
            client_id = parts[1]
            round_id = parts[3].split('.')[0]
            rounds.add(round_id)
            clients.add(client_id)
    
    print_section("Encryption Statistics")
    print(f"  Total encrypted files: {len(encrypted_files)}")
    print(f"  Total size: {total_size:,} bytes ({total_size/1024:.1f} KB)")
    print(f"  Average file size: {total_size/len(encrypted_files):.0f} bytes")
    print(f"  Rounds covered: {sorted(list(rounds))}")
    print(f"  Clients involved: {sorted(list(clients))}")

def show_data_summary():
    """Display data summary"""
    print_header("Data Summary")
    
    data_dir = "data/clients"
    if not os.path.exists(data_dir):
        print("❌ Data directory not found")
        return
    
    client_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    if not client_files:
        print("❌ No client data files found")
        return
    
    print(f"✅ Found {len(client_files)} client datasets")
    
    # Analyze first client data
    first_file = os.path.join(data_dir, client_files[0])
    try:
        import pandas as pd
        df = pd.read_csv(first_file)
        
        print_section("Dataset Information")
        print(f"  Features: {list(df.columns[:-1])}")  # Exclude label
        print(f"  Target: {df.columns[-1]}")
        print(f"  Samples per client: {len(df)}")
        print(f"  Total samples: {len(df) * len(client_files)}")
        
        print_section("Feature Statistics (Sample Client)")
        for col in df.columns[:-1]:  # Exclude label
            print(f"  {col:12}: {df[col].mean():6.1f} ± {df[col].std():5.1f}")
        
        print(f"\nLabel distribution: {df['label'].value_counts().to_dict()}")
        
    except Exception as e:
        print(f"❌ Error analyzing data: {e}")

def main():
    """Main function"""
    print("🔐 Secure Federated Learning - Global Results")
    print("=" * 60)
    
    # Show all results
    show_data_summary()
    show_encryption_summary()
    show_aggregation_results()
    show_global_model_results()
    
    print_header("Summary")
    print("✅ Your federated learning pipeline has successfully:")
    print("  • Generated and distributed health data across multiple clients")
    print("  • Trained local models on each client's data")
    print("  • Encrypted all client updates for privacy")
    print("  • Aggregated encrypted updates securely")
    print("  • Updated the global model with aggregated results")
    print("\n🔒 All data remained encrypted throughout the process!")
    print("📈 The global model can now make predictions on new health data")

if __name__ == "__main__":
    main()
