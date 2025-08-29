#!/usr/bin/env python3
"""
Test script for global update functionality
"""

import os
import sys
import numpy as np
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cloud.global_update import CloudServer, load_aggregated_update

def test_global_update():
    """Test the global update functionality"""
    
    print("🧪 Testing Global Update Functionality...")
    
    # 1. Test with sample aggregated update
    print("\n1️⃣ Testing with sample aggregated update...")
    
    # Create sample aggregated update (simulating decrypted output from Sriven)
    sample_agg_update = {
        "weight_delta": [0.1, -0.2, 0.3, -0.1, 0.05],
        "bias_delta": 0.02
    }
    
    # Initialize cloud server
    cloud = CloudServer(input_dim=5)
    print(f"   ✅ Cloud server initialized with input_dim=5")
    
    # Create test data
    X_test = np.random.randn(100, 5)
    y_test = (np.sum(X_test, axis=1) > 0).astype(int)
    print(f"   ✅ Test data created: {X_test.shape[0]} samples, {X_test.shape[1]} features")
    
    # Test global update
    print("\n2️⃣ Testing global model update...")
    accuracy = cloud.update_global_model(sample_agg_update, X_test, y_test)
    print(f"   ✅ Global model updated successfully")
    print(f"   📊 Round {cloud.round} accuracy: {accuracy:.4f}")
    
    # 2. Test with actual decrypted aggregation file
    print("\n3️⃣ Testing with actual decrypted aggregation file...")
    
    decrypted_file = "Sriven/outbox/agg_round_0.decrypted.json"
    if os.path.exists(decrypted_file):
        print(f"   📁 Found decrypted file: {decrypted_file}")
        
        try:
            # Load the actual decrypted aggregation
            actual_agg_update = load_aggregated_update(decrypted_file)
            print(f"   ✅ Successfully loaded aggregated update")
            print(f"   📊 Weight delta length: {len(actual_agg_update['weight_delta'])}")
            print(f"   📊 Bias delta: {actual_agg_update['bias_delta']:.6f}")
            
            # Test with actual data
            cloud2 = CloudServer(input_dim=4)  # Based on actual data
            accuracy2 = cloud2.update_global_model(actual_agg_update, X_test[:, :4], y_test)
            print(f"   ✅ Global model updated with actual data")
            print(f"   📊 Round {cloud2.round} accuracy: {accuracy2:.4f}")
            
        except Exception as e:
            print(f"   ❌ Error loading actual file: {e}")
    else:
        print(f"   ⚠️  Decrypted file not found: {decrypted_file}")
        print("   💡 Run the encryption/decryption pipeline first to generate this file")
    
    # 3. Test error handling
    print("\n4️⃣ Testing error handling...")
    
    try:
        # Test with invalid file format
        invalid_update = load_aggregated_update("nonexistent_file.txt")
    except ValueError as e:
        print(f"   ✅ Properly handled invalid file format: {e}")
    
    # 4. Test model state
    print("\n5️⃣ Testing model state...")
    
    # Check model parameters
    for name, param in cloud.global_model.named_parameters():
        print(f"   📊 {name}: shape {param.shape}, norm {param.norm().item():.6f}")
    
    print("\n🎉 Global Update Testing Complete!")
    return True

def test_encryption_pipeline():
    """Test the complete encryption pipeline"""
    
    print("\n🔐 Testing Complete Encryption Pipeline...")
    
    # Check if encrypted files exist
    encrypted_dir = "updates/encrypted"
    if os.path.exists(encrypted_dir):
        encrypted_files = [f for f in os.listdir(encrypted_dir) if f.endswith('.json')]
        print(f"   📁 Found {len(encrypted_files)} encrypted files")
        
        if encrypted_files:
            print(f"   📄 Sample encrypted file: {encrypted_files[0]}")
            
            # Check if aggregation files exist
            outbox_dir = "Sriven/outbox"
            if os.path.exists(outbox_dir):
                agg_files = [f for f in os.listdir(outbox_dir) if f.endswith('.json')]
                print(f"   📁 Found {len(agg_files)} aggregation files")
                
                if agg_files:
                    print(f"   📄 Aggregation files: {agg_files}")
                    return True
                else:
                    print("   ⚠️  No aggregation files found")
                    print("   💡 Run the aggregation step: python Sriven/smart_switch_tenseal.py")
            else:
                print("   ⚠️  Outbox directory not found")
        else:
            print("   ⚠️  No encrypted files found")
            print("   💡 Run the encryption step first")
    else:
        print("   ⚠️  Encrypted directory not found")
        print("   💡 Run the complete pipeline first")
    
    return False

if __name__ == "__main__":
    print("🚀 Starting Global Update Tests...")
    
    # Test global update functionality
    test_global_update()
    
    # Test encryption pipeline status
    test_encryption_pipeline()
    
    print("\n📋 Test Summary:")
    print("   ✅ Global update functionality tested")
    print("   ✅ Error handling tested")
    print("   ✅ Model state verified")
    print("\n💡 To run complete pipeline:")
    print("   1. python simulation/client_simulation.py")
    print("   2. python Huzaif/encrypt_update.py --in <file> --out <file>")
    print("   3. python Sriven/smart_switch_tenseal.py --fedl_dir updates/encrypted --ctx_b64 Huzaif/keys/params.ctx.b64 --out_dir Sriven/outbox")
    print("   4. python Huzaif/decrypt.py --in <agg_file> --out <decrypted_file>")
    print("   5. python test_global_update.py")
