#!/usr/bin/env python3
"""
Simple step-by-step test of global update functionality
"""

import os
import sys
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_step_by_step():
    """Test global update step by step"""
    
    print("🚀 Step-by-Step Global Update Test")
    print("=" * 40)
    
    try:
        # Step 1: Import CloudServer
        print("\n1️⃣ Importing CloudServer...")
        from cloud.global_update import CloudServer
        print("   ✅ CloudServer imported successfully")
        
        # Step 2: Create cloud server
        print("\n2️⃣ Creating Cloud Server...")
        cloud = CloudServer(input_dim=4)
        print(f"   ✅ Cloud server created with input_dim=4")
        print(f"   📊 Initial round: {cloud.round}")
        
        # Step 3: Check initial model state
        print("\n3️⃣ Checking Initial Model State...")
        for name, param in cloud.global_model.named_parameters():
            print(f"   📊 {name}: shape {param.shape}, norm {param.norm().item():.6f}")
        
        # Step 4: Create test data
        print("\n4️⃣ Creating Test Data...")
        X_test = np.random.randn(100, 4)
        y_test = (np.sum(X_test, axis=1) > 0).astype(int)
        print(f"   ✅ Test data created: {X_test.shape[0]} samples, {X_test.shape[1]} features")
        
        # Step 5: Test initial evaluation
        print("\n5️⃣ Testing Initial Evaluation...")
        initial_acc = cloud.evaluate(X_test, y_test)
        print(f"   📊 Initial accuracy: {initial_acc:.4f}")
        
        # Step 6: Apply first update
        print("\n6️⃣ Applying First Update...")
        update1 = {
            "weight_delta": [0.1, -0.2, 0.3, -0.1],
            "bias_delta": 0.05
        }
        acc1 = cloud.update_global_model(update1, X_test, y_test)
        print(f"   📊 Round {cloud.round} accuracy: {acc1:.4f}")
        print(f"   📈 Accuracy change: {acc1 - initial_acc:+.4f}")
        
        # Step 7: Apply second update
        print("\n7️⃣ Applying Second Update...")
        update2 = {
            "weight_delta": [-0.05, 0.1, -0.15, 0.08],
            "bias_delta": -0.02
        }
        acc2 = cloud.update_global_model(update2, X_test, y_test)
        print(f"   📊 Round {cloud.round} accuracy: {acc2:.4f}")
        print(f"   📈 Accuracy change: {acc2 - acc1:+.4f}")
        
        # Step 8: Check final model state
        print("\n8️⃣ Checking Final Model State...")
        for name, param in cloud.global_model.named_parameters():
            print(f"   📊 {name}: norm {param.norm().item():.6f}")
        
        # Step 9: Test with real aggregated data if available
        print("\n9️⃣ Testing with Real Aggregated Data...")
        decrypted_file = "Sriven/outbox/agg_round_0.decrypted.json"
        
        if os.path.exists(decrypted_file):
            print(f"   📁 Found real aggregated data: {decrypted_file}")
            
            from cloud.global_update import load_aggregated_update
            
            # Load real data
            real_update = load_aggregated_update(decrypted_file)
            print(f"   📊 Real update - Weight delta length: {len(real_update['weight_delta'])}")
            print(f"   📊 Real update - Bias delta: {real_update['bias_delta']:.6f}")
            
            # Create new cloud server for real data
            real_cloud = CloudServer(input_dim=len(real_update['weight_delta']))
            print(f"   ✅ New cloud server created for real data")
            
            # Test with real data
            X_real = np.random.randn(50, len(real_update['weight_delta']))
            y_real = (np.sum(X_real, axis=1) > 0).astype(int)
            
            real_acc = real_cloud.update_global_model(real_update, X_real, y_real)
            print(f"   📊 Real data accuracy: {real_acc:.4f}")
            
        else:
            print(f"   ⚠️  No real aggregated data found at {decrypted_file}")
            print("   💡 This is normal if you haven't run the encryption pipeline yet")
        
        print("\n🎉 Step-by-step test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_step_by_step()
    
    if success:
        print("\n💡 Next steps:")
        print("   1. Run: python simulation/client_simulation.py")
        print("   2. Run: python test_global_update_comprehensive.py")
        print("   3. Run: python cloud/global_update.py")
    else:
        print("\n❌ Test failed. Check the output above for details.")
        sys.exit(1)
