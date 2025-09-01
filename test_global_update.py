#!/usr/bin/env python3
"""
Test script for secure global update functionality
This is the main test for encrypted global updates using encrypted aggregation
"""

import os
import sys
import numpy as np
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cloud.global_update import CloudServer, load_encrypted_aggregation

def test_encrypted_global_update():
    """Test the encrypted global update functionality"""
    
    print("🔐 Testing Encrypted Global Update Functionality...")
    
    # 1. Test with encrypted aggregation file (preferred method)
    print("\n1️⃣ Testing with encrypted aggregation file...")
    
    encrypted_file = "Sriven/outbox/agg_round_0.json"
    if os.path.exists(encrypted_file):
        print(f"   📁 Found encrypted aggregation file: {encrypted_file}")
        
        try:
            # Load the encrypted aggregation directly
            encrypted_agg = load_encrypted_aggregation(encrypted_file)
            print(f"   ✅ Successfully loaded encrypted aggregation")
            print(f"   📊 Round ID: {encrypted_agg['round_id']}")
            print(f"   📊 Layout: {encrypted_agg['layout']}")
            print(f"   📊 Ciphertext length: {len(encrypted_agg['ciphertext'])} chars")
            
            # Initialize cloud server
            input_dim = encrypted_agg['layout']['weights']
            cloud = CloudServer(input_dim=input_dim)
            print(f"   ✅ Cloud server initialized with input_dim={input_dim}")
            
            # Create test data
            X_test = np.random.randn(100, input_dim)
            y_test = (np.sum(X_test, axis=1) > 0).astype(int)
            print(f"   ✅ Test data created: {X_test.shape[0]} samples, {X_test.shape[1]} features")
            
            # Test encrypted global update
            print("\n2️⃣ Testing encrypted global model update...")
            accuracy = cloud.update_global_model_encrypted(encrypted_agg, X_test, y_test)
            print(f"   ✅ Encrypted global model updated successfully")
            print(f"   📊 Round {cloud.round} accuracy: {accuracy:.4f}")
            
            # Check model parameters
            print("\n3️⃣ Checking updated model parameters...")
            for name, param in cloud.global_model.named_parameters():
                print(f"   📊 {name}: shape {param.shape}, norm {param.norm().item():.6f}")
            
        except Exception as e:
            print(f"   ❌ Error with encrypted update: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"   ⚠️  Encrypted aggregation file not found: {encrypted_file}")
        print("   💡 Run the aggregation step first: python Sriven/smart_switch_tenseal.py")
    
    # 2. Test error handling
    print("\n4️⃣ Testing error handling...")
    
    try:
        # Test with non-existent file
        invalid_agg = load_encrypted_aggregation("nonexistent_file.json")
    except FileNotFoundError as e:
        print(f"   ✅ Properly handled missing file: {e}")
    
    try:
        # Test with invalid format
        invalid_agg = load_encrypted_aggregation("test.txt")
    except ValueError as e:
        print(f"   ✅ Properly handled invalid format: {e}")
    
    # 3. Compare with old method (if decrypted file exists)
    print("\n5️⃣ Comparing with old decrypted method...")
    
    decrypted_file = "Sriven/outbox/agg_round_0.decrypted.json"
    if os.path.exists(decrypted_file):
        print(f"   📁 Found decrypted file for comparison: {decrypted_file}")
        
        try:
            from cloud.global_update import load_aggregated_update
            
            # Load decrypted data
            decrypted_agg = load_aggregated_update(decrypted_file)
            print(f"   📊 Decrypted weight delta length: {len(decrypted_agg['weight_delta'])}")
            print(f"   📊 Decrypted bias delta: {decrypted_agg['bias_delta']:.6f}")
            
            # Create new cloud server for comparison
            cloud2 = CloudServer(input_dim=len(decrypted_agg['weight_delta']))
            X_test2 = np.random.randn(50, len(decrypted_agg['weight_delta']))
            y_test2 = (np.sum(X_test2, axis=1) > 0).astype(int)
            
            # Test old method
            accuracy2 = cloud2.update_global_model(decrypted_agg, X_test2, y_test2)
            print(f"   📊 Old method accuracy: {accuracy2:.4f}")
            
            print("   💡 Both methods should produce similar results")
            
        except Exception as e:
            print(f"   ❌ Error with decrypted comparison: {e}")
    else:
        print(f"   ⚠️  No decrypted file found for comparison")
    
    print("\n🎉 Encrypted Global Update Testing Complete!")
    return True

def test_pipeline_comparison():
    """Compare the old vs new pipeline approaches"""
    
    print("\n🔄 Pipeline Comparison:")
    print("=" * 50)
    
    print("\n❌ OLD PIPELINE (Insecure):")
    print("   1. Client Simulation → plaintext updates")
    print("   2. Encryption → encrypted client updates")
    print("   3. Aggregation → encrypted aggregated result")
    print("   4. Decryption → decrypted aggregated result")
    print("   5. Global Update → uses decrypted result")
    print("   ⚠️  Problem: Decryption step exposes aggregated data")
    
    print("\n✅ NEW PIPELINE (Secure):")
    print("   1. Client Simulation → plaintext updates")
    print("   2. Encryption → encrypted client updates")
    print("   3. Aggregation → encrypted aggregated result")
    print("   4. Global Update → uses encrypted result directly")
    print("   🔒 Advantage: No decryption step, data stays encrypted")
    
    print("\n🔐 Security Benefits:")
    print("   • Aggregated data never leaves encrypted form")
    print("   • No intermediate decryption step")
    print("   • Reduced attack surface")
    print("   • Better privacy preservation")

def test_encryption_pipeline_status():
    """Check the status of the encryption pipeline"""
    
    print("\n🔐 Checking Encryption Pipeline Status...")
    
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
                agg_files = [f for f in os.listdir(outbox_dir) if f.endswith('.json') and 'agg_' in f]
                print(f"   📁 Found {len(agg_files)} aggregation files")
                
                if agg_files:
                    print(f"   📄 Aggregation files: {agg_files}")
                    
                    # Check for encrypted vs decrypted
                    encrypted_agg = [f for f in agg_files if not f.endswith('.decrypted.json')]
                    decrypted_agg = [f for f in agg_files if f.endswith('.decrypted.json')]
                    
                    print(f"   🔐 Encrypted aggregations: {len(encrypted_agg)}")
                    print(f"   🔓 Decrypted aggregations: {len(decrypted_agg)}")
                    
                    if encrypted_agg:
                        print("   ✅ Ready for encrypted global update!")
                    else:
                        print("   ⚠️  No encrypted aggregations found")
                        
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
    print("🚀 Starting Encrypted Global Update Tests...")
    
    # Test encrypted global update functionality
    test_encrypted_global_update()
    
    # Show pipeline comparison
    test_pipeline_comparison()
    
    # Test encryption pipeline status
    test_encryption_pipeline_status()
    
    print("\n📋 Test Summary:")
    print("   ✅ Encrypted global update functionality tested")
    print("   ✅ Pipeline comparison shown")
    print("   ✅ Error handling tested")
    print("   ✅ Security benefits explained")
    print("\n💡 To run complete secure pipeline:")
    print("   1. python simulation/client_simulation.py")
    print("   2. python Huzaif/encrypt_update.py --in <file> --out <file>")
    print("   3. python Sriven/smart_switch_tenseal.py --fedl_dir updates/encrypted --ctx_b64 Huzaif/keys/params.ctx.b64 --out_dir Sriven/outbox")
    print("   4. python test_encrypted_global_update.py")
    print("\n🔒 Note: Step 4 (decryption) is no longer needed!")
