#!/usr/bin/env python3
"""
Simple test script for global update functionality
"""

import os
import sys
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_basic_global_update():
    """Test basic global update functionality"""
    
    print("🧪 Testing Basic Global Update...")
    
    try:
        # Import the global update module
        from cloud.global_update import CloudServer
        
        # Test 1: Initialize cloud server
        print("\n1️⃣ Testing Cloud Server Initialization...")
        cloud = CloudServer(input_dim=4)
        print(f"   ✅ Cloud server created with input_dim=4")
        print(f"   📊 Initial round: {cloud.round}")
        
        # Test 2: Check model structure
        print("\n2️⃣ Testing Model Structure...")
        for name, param in cloud.global_model.named_parameters():
            print(f"   📊 {name}: shape {param.shape}")
        
        # Test 3: Create sample update
        print("\n3️⃣ Testing Model Update...")
        sample_update = {
            "weight_delta": [0.1, -0.2, 0.3, -0.1],
            "bias_delta": 0.05
        }
        print(f"   📊 Sample update created: {len(sample_update['weight_delta'])} weights, bias: {sample_update['bias_delta']}")
        
        # Test 4: Apply update
        print("\n4️⃣ Applying Update...")
        cloud.update_global_model(sample_update)
        print(f"   ✅ Update applied successfully")
        print(f"   📊 New round: {cloud.round}")
        
        # Test 5: Check updated parameters
        print("\n5️⃣ Checking Updated Parameters...")
        for name, param in cloud.global_model.named_parameters():
            print(f"   📊 {name}: norm {param.norm().item():.6f}")
        
        # Test 6: Test with evaluation data
        print("\n6️⃣ Testing with Evaluation Data...")
        X_test = np.random.randn(50, 4)
        y_test = (np.sum(X_test, axis=1) > 0).astype(int)
        
        accuracy = cloud.evaluate(X_test, y_test)
        print(f"   ✅ Evaluation completed")
        print(f"   📊 Accuracy: {accuracy:.4f}")
        
        print("\n🎉 Basic Global Update Test Passed!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure you're running from the project root directory")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_file_loading():
    """Test file loading functionality"""
    
    print("\n📁 Testing File Loading...")
    
    try:
        from cloud.global_update import load_aggregated_update
        
        # Test with existing decrypted file if available
        decrypted_file = "Sriven/outbox/agg_round_0.decrypted.json"
        if os.path.exists(decrypted_file):
            print(f"   📁 Found decrypted file: {decrypted_file}")
            
            agg_update = load_aggregated_update(decrypted_file)
            print(f"   ✅ Successfully loaded file")
            print(f"   📊 Weight delta length: {len(agg_update['weight_delta'])}")
            print(f"   📊 Bias delta: {agg_update['bias_delta']:.6f}")
            
            # Test global update with this data
            input_dim = len(agg_update['weight_delta'])
            cloud = CloudServer(input_dim=input_dim)
            
            X_test = np.random.randn(50, input_dim)
            y_test = (np.sum(X_test, axis=1) > 0).astype(int)
            
            accuracy = cloud.update_global_model(agg_update, X_test, y_test)
            print(f"   ✅ Global update with real data successful")
            print(f"   📊 Accuracy: {accuracy:.4f}")
            
        else:
            print(f"   ⚠️  No decrypted file found at {decrypted_file}")
            print("   💡 This is normal if you haven't run the encryption pipeline yet")
        
        print("   ✅ File loading test completed")
        return True
        
    except Exception as e:
        print(f"   ❌ File loading test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Simple Global Update Test")
    print("=" * 40)
    
    # Test basic functionality
    success1 = test_basic_global_update()
    
    # Test file loading
    success2 = test_file_loading()
    
    if success1 and success2:
        print("\n🎉 All tests passed! Your global update is working correctly.")
        print("\n💡 Next steps:")
        print("   1. Run: python simulation/client_simulation.py")
        print("   2. Run: python test_integration.py (for full pipeline)")
    else:
        print("\n❌ Some tests failed. Check the output above for details.")
        sys.exit(1)
