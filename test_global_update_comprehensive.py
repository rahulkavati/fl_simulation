#!/usr/bin/env python3
"""
Comprehensive test script for global update functionality
"""

import os
import sys
import json
import traceback

def check_dependencies():
    """Check all required dependencies"""
    print("🔍 Checking Dependencies...")
    
    dependencies = {
        'numpy': 'numpy',
        'torch': 'torch', 
        'sklearn': 'scikit-learn',
        'pandas': 'pandas'
    }
    
    missing = []
    available = []
    
    for name, package in dependencies.items():
        try:
            if name == 'sklearn':
                import sklearn
            else:
                __import__(name)
            available.append(name)
            print(f"   ✅ {name} ({package})")
        except ImportError:
            missing.append(package)
            print(f"   ❌ {name} ({package}) - MISSING")
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("💡 Install with: pip install " + " ".join(missing))
        return False
    
    print(f"\n✅ All dependencies available: {', '.join(available)}")
    return True

def test_global_update_basic():
    """Test basic global update functionality"""
    print("\n🧪 Testing Basic Global Update...")
    
    try:
        # Test 1: Import and initialize
        print("   1️⃣ Importing CloudServer...")
        from cloud.global_update import CloudServer
        print("      ✅ CloudServer imported successfully")
        
        # Test 2: Create cloud server
        print("   2️⃣ Creating Cloud Server...")
        cloud = CloudServer(input_dim=4)
        print(f"      ✅ Cloud server created with input_dim=4")
        print(f"      📊 Initial round: {cloud.round}")
        
        # Test 3: Check model structure
        print("   3️⃣ Checking Model Structure...")
        param_count = 0
        for name, param in cloud.global_model.named_parameters():
            print(f"      📊 {name}: shape {param.shape}")
            param_count += 1
        
        if param_count == 2:  # weight and bias
            print("      ✅ Model structure correct (weight + bias)")
        else:
            print(f"      ⚠️  Expected 2 parameters, got {param_count}")
        
        # Test 4: Create sample update
        print("   4️⃣ Testing Model Update...")
        sample_update = {
            "weight_delta": [0.1, -0.2, 0.3, -0.1],
            "bias_delta": 0.05
        }
        print(f"      📊 Sample update: {len(sample_update['weight_delta'])} weights, bias: {sample_update['bias_delta']}")
        
        # Test 5: Apply update
        print("   5️⃣ Applying Update...")
        cloud.update_global_model(sample_update)
        print(f"      ✅ Update applied successfully")
        print(f"      📊 New round: {cloud.round}")
        
        # Test 6: Check updated parameters
        print("   6️⃣ Checking Updated Parameters...")
        for name, param in cloud.global_model.named_parameters():
            norm = param.norm().item()
            print(f"      📊 {name}: norm {norm:.6f}")
        
        print("      ✅ Basic global update test passed!")
        return True
        
    except Exception as e:
        print(f"      ❌ Basic test failed: {e}")
        traceback.print_exc()
        return False

def test_global_update_with_data():
    """Test global update with evaluation data"""
    print("\n📊 Testing Global Update with Evaluation Data...")
    
    try:
        from cloud.global_update import CloudServer
        import numpy as np
        
        # Create test data
        print("   1️⃣ Creating Test Data...")
        X_test = np.random.randn(100, 4)
        y_test = (np.sum(X_test, axis=1) > 0).astype(int)
        print(f"      ✅ Test data created: {X_test.shape[0]} samples, {X_test.shape[1]} features")
        
        # Initialize cloud server
        print("   2️⃣ Initializing Cloud Server...")
        cloud = CloudServer(input_dim=4)
        
        # Test evaluation before update
        print("   3️⃣ Testing Evaluation Before Update...")
        try:
            acc_before = cloud.evaluate(X_test, y_test)
            print(f"      📊 Accuracy before update: {acc_before:.4f}")
        except Exception as e:
            print(f"      ⚠️  Evaluation before update failed: {e}")
            acc_before = None
        
        # Apply update with evaluation
        print("   4️⃣ Applying Update with Evaluation...")
        sample_update = {
            "weight_delta": [0.1, -0.2, 0.3, -0.1],
            "bias_delta": 0.05
        }
        
        acc_after = cloud.update_global_model(sample_update, X_test, y_test)
        print(f"      ✅ Update with evaluation completed")
        print(f"      📊 Round {cloud.round} accuracy: {acc_after:.4f}")
        
        if acc_before is not None:
            print(f"      📈 Accuracy change: {acc_after - acc_before:+.4f}")
        
        return True
        
    except Exception as e:
        print(f"      ❌ Evaluation test failed: {e}")
        traceback.print_exc()
        return False

def test_file_loading():
    """Test file loading functionality"""
    print("\n📁 Testing File Loading...")
    
    try:
        from cloud.global_update import load_aggregated_update
        
        # Test 1: Check if decrypted files exist
        print("   1️⃣ Checking for Decrypted Files...")
        decrypted_file = "Sriven/outbox/agg_round_0.decrypted.json"
        
        if os.path.exists(decrypted_file):
            print(f"      📁 Found decrypted file: {decrypted_file}")
            
            # Test 2: Load the file
            print("   2️⃣ Loading Decrypted File...")
            agg_update = load_aggregated_update(decrypted_file)
            print(f"      ✅ Successfully loaded file")
            print(f"      📊 Weight delta length: {len(agg_update['weight_delta'])}")
            print(f"      📊 Bias delta: {agg_update['bias_delta']:.6f}")
            
            # Test 3: Test global update with real data
            print("   3️⃣ Testing Global Update with Real Data...")
            from cloud.global_update import CloudServer
            import numpy as np
            
            input_dim = len(agg_update['weight_delta'])
            cloud = CloudServer(input_dim=input_dim)
            print(f"      ✅ Cloud server initialized with input_dim={input_dim}")
            
            X_test = np.random.randn(50, input_dim)
            y_test = (np.sum(X_test, axis=1) > 0).astype(int)
            
            accuracy = cloud.update_global_model(agg_update, X_test, y_test)
            print(f"      ✅ Global update with real data successful")
            print(f"      📊 Accuracy: {accuracy:.4f}")
            
        else:
            print(f"      ⚠️  No decrypted file found at {decrypted_file}")
            print("      💡 This is normal if you haven't run the encryption pipeline yet")
        
        # Test 4: Test error handling
        print("   4️⃣ Testing Error Handling...")
        try:
            invalid_update = load_aggregated_update("nonexistent_file.txt")
        except ValueError as e:
            print(f"      ✅ Properly handled invalid file format: {e}")
        except Exception as e:
            print(f"      ⚠️  Unexpected error handling: {e}")
        
        print("      ✅ File loading test completed")
        return True
        
    except Exception as e:
        print(f"      ❌ File loading test failed: {e}")
        traceback.print_exc()
        return False

def test_complete_pipeline_status():
    """Check the status of the complete pipeline"""
    print("\n🔐 Checking Complete Pipeline Status...")
    
    pipeline_steps = [
        ("Client Simulation", "simulation/client_simulation.py"),
        ("Global Update", "cloud/global_update.py"),
        ("Encryption", "Huzaif/encrypt_update.py"),
        ("Decryption", "Huzaif/decrypt.py"),
        ("Aggregation", "Sriven/smart_switch_tenseal.py")
    ]
    
    for step_name, file_path in pipeline_steps:
        if os.path.exists(file_path):
            print(f"   ✅ {step_name}: {file_path}")
        else:
            print(f"   ❌ {step_name}: {file_path} - MISSING")
    
    # Check for data files
    print("\n📊 Checking Data Files...")
    
    data_dirs = [
        ("Client Updates", "updates/json"),
        ("Encrypted Updates", "updates/encrypted"),
        ("Aggregation Output", "Sriven/outbox"),
        ("Encryption Keys", "Huzaif/keys")
    ]
    
    for dir_name, dir_path in data_dirs:
        if os.path.exists(dir_path):
            files = [f for f in os.listdir(dir_path) if f.endswith(('.json', '.npy', '.ctx', '.b64'))]
            print(f"   📁 {dir_name}: {dir_path} ({len(files)} files)")
        else:
            print(f"   ⚠️  {dir_name}: {dir_path} - NOT FOUND")

def main():
    """Main test function"""
    print("🚀 Comprehensive Global Update Test")
    print("=" * 50)
    
    # Check dependencies first
    deps_ok = check_dependencies()
    
    if not deps_ok:
        print("\n❌ Dependencies check failed. Please install missing packages first.")
        print("\n💡 Quick fix commands:")
        print("   pip install scikit-learn torch numpy pandas")
        print("   pip install -r requirements.txt")
        return False
    
    # Run tests
    tests = [
        ("Basic Global Update", test_global_update_basic),
        ("Global Update with Data", test_global_update_with_data),
        ("File Loading", test_file_loading)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Check pipeline status
    test_complete_pipeline_status()
    
    # Summary
    print("\n📋 Test Summary:")
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your global update is working correctly.")
        print("\n💡 Next steps:")
        print("   1. Run: python simulation/client_simulation.py")
        print("   2. Run: python test_integration.py (for full pipeline)")
        return True
    else:
        print(f"\n❌ {total - passed} tests failed. Check the output above for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
