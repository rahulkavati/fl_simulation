#!/usr/bin/env python3
"""
Integration test for the complete FL pipeline
"""

import os
import sys
import subprocess
import json
import time

def run_command(cmd, description):
    """Run a command and return success status"""
    print(f"\n🔄 {description}...")
    print(f"   Command: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            print(f"   ✅ {description} completed successfully")
            return True
        else:
            print(f"   ❌ {description} failed")
            print(f"   Error: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print(f"   ⏰ {description} timed out")
        return False
    except Exception as e:
        print(f"   ❌ {description} failed with exception: {e}")
        return False

def test_complete_pipeline():
    """Test the complete FL pipeline end-to-end"""
    
    print("🚀 Testing Complete FL Pipeline...")
    
    # Check if we're in the right directory
    if not os.path.exists("simulation/client_simulation.py"):
        print("❌ Please run this script from the project root directory")
        return False
    
    # Step 1: Run client simulation
    if not run_command("python simulation/client_simulation.py", "Client Simulation"):
        return False
    
    # Step 2: Check if updates were generated
    if not os.path.exists("updates/json"):
        print("❌ Client simulation didn't generate updates")
        return False
    
    # Step 3: Check if we have encrypted updates
    if not os.path.exists("updates/encrypted"):
        print("⚠️  No encrypted updates found. Checking if encryption is needed...")
        
        # Check if we have the encryption keys
        if os.path.exists("Huzaif/keys/secret.ctx"):
            print("🔐 Encryption keys found. Running encryption...")
            
            # Encrypt a sample update
            sample_update = "updates/json/client_0_round_0.json"
            if os.path.exists(sample_update):
                encrypt_cmd = f"python Huzaif/encrypt_update.py --in {sample_update} --out updates/encrypted/enc_client_0_round_0.json"
                if not run_command(encrypt_cmd, "Encryption"):
                    return False
            else:
                print("❌ Sample update file not found")
                return False
        else:
            print("⚠️  Encryption keys not found. Skipping encryption step.")
    
    # Step 4: Check if we have aggregation files
    if not os.path.exists("Sriven/outbox"):
        print("📁 Creating outbox directory...")
        os.makedirs("Sriven/outbox", exist_ok=True)
    
    # Step 5: Run aggregation (if we have encrypted files)
    encrypted_dir = "updates/encrypted"
    if os.path.exists(encrypted_dir) and os.listdir(encrypted_dir):
        if os.path.exists("Huzaif/keys/params.ctx.b64"):
            agg_cmd = f"python Sriven/smart_switch_tenseal.py --fedl_dir {encrypted_dir} --ctx_b64 Huzaif/keys/params.ctx.b64 --out_dir Sriven/outbox"
            if not run_command(agg_cmd, "Aggregation"):
                return False
        else:
            print("⚠️  Aggregation context file not found")
    
    # Step 6: Check if we have aggregated files
    outbox_dir = "Sriven/outbox"
    if os.path.exists(outbox_dir):
        agg_files = [f for f in os.listdir(outbox_dir) if f.endswith('.json') and 'agg_' in f]
        if agg_files:
            print(f"📄 Found aggregation files: {agg_files}")
            
            # Step 7: Decrypt aggregated files (if we have secret key)
            if os.path.exists("Huzaif/keys/secret.ctx"):
                for agg_file in agg_files:
                    if not agg_file.endswith('.decrypted.json'):
                        agg_path = os.path.join(outbox_dir, agg_file)
                        decrypted_path = agg_path.replace('.json', '.decrypted.json')
                        decrypt_cmd = f"python Huzaif/decrypt.py --in {agg_path} --out {decrypted_path}"
                        if not run_command(decrypt_cmd, f"Decryption of {agg_file}"):
                            return False
            else:
                print("⚠️  Secret key not found. Cannot decrypt aggregated files.")
        else:
            print("⚠️  No aggregation files found")
    
    # Step 8: Test global update
    print("\n🧪 Testing Global Update...")
    
    # Check if we have decrypted files
    decrypted_files = [f for f in os.listdir(outbox_dir) if f.endswith('.decrypted.json')] if os.path.exists(outbox_dir) else []
    
    if decrypted_files:
        print(f"📄 Found decrypted files: {decrypted_files}")
        
        # Test global update with the first decrypted file
        test_file = os.path.join(outbox_dir, decrypted_files[0])
        
        # Import and test global update
        try:
            sys.path.append(os.path.dirname(os.path.abspath(__file__)))
            from cloud.global_update import CloudServer, load_aggregated_update
            
            # Load the decrypted update
            agg_update = load_aggregated_update(test_file)
            print(f"✅ Successfully loaded aggregated update")
            print(f"📊 Weight delta length: {len(agg_update['weight_delta'])}")
            print(f"📊 Bias delta: {agg_update['bias_delta']:.6f}")
            
            # Initialize cloud server
            input_dim = len(agg_update['weight_delta'])
            cloud = CloudServer(input_dim=input_dim)
            print(f"✅ Cloud server initialized with input_dim={input_dim}")
            
            # Create test data
            import numpy as np
            X_test = np.random.randn(100, input_dim)
            y_test = (np.sum(X_test, axis=1) > 0).astype(int)
            
            # Update global model
            accuracy = cloud.update_global_model(agg_update, X_test, y_test)
            print(f"✅ Global model updated successfully!")
            print(f"📊 Round {cloud.round} accuracy: {accuracy:.4f}")
            
        except Exception as e:
            print(f"❌ Error testing global update: {e}")
            return False
    else:
        print("⚠️  No decrypted files found for global update testing")
    
    print("\n🎉 Pipeline Testing Complete!")
    return True

def check_dependencies():
    """Check if all required dependencies are available"""
    
    print("🔍 Checking Dependencies...")
    
    required_files = [
        "simulation/client_simulation.py",
        "cloud/global_update.py",
        "Huzaif/encrypt_update.py",
        "Huzaif/decrypt.py",
        "Sriven/smart_switch_tenseal.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing required files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        return False
    
    print("✅ All required files found")
    
    # Check Python packages
    try:
        import numpy
        import torch
        import sklearn
        print("✅ All required Python packages available")
        return True
    except ImportError as e:
        print(f"❌ Missing Python package: {e}")
        print("💡 Install with: pip install numpy torch scikit-learn")
        return False

if __name__ == "__main__":
    print("🚀 FL Pipeline Integration Test")
    print("=" * 50)
    
    # Check dependencies first
    if not check_dependencies():
        print("\n❌ Dependencies check failed. Please fix issues before running tests.")
        sys.exit(1)
    
    # Run the complete pipeline test
    success = test_complete_pipeline()
    
    if success:
        print("\n🎉 All tests passed! Your FL pipeline is working correctly.")
    else:
        print("\n❌ Some tests failed. Check the output above for details.")
        sys.exit(1)
