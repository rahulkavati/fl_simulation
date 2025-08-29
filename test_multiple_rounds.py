#!/usr/bin/env python3
"""
Test script to demonstrate multiple rounds of global updates
"""

import os
import sys
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_multiple_rounds():
    """Test global update with multiple rounds"""
    
    print("🚀 Testing Multiple Rounds of Global Updates")
    print("=" * 50)
    
    try:
        from cloud.global_update import CloudServer
        
        # Create cloud server
        cloud = CloudServer(input_dim=4)
        print(f"✅ Cloud server created with input_dim=4")
        print(f"📊 Initial round: {cloud.round}")
        
        # Create test data
        X_test = np.random.randn(100, 4)
        y_test = (np.sum(X_test, axis=1) > 0).astype(int)
        print(f"✅ Test data created: {X_test.shape[0]} samples, {X_test.shape[1]} features")
        
        # Test initial evaluation
        initial_acc = cloud.evaluate(X_test, y_test)
        print(f"📊 Initial accuracy: {initial_acc:.4f}")
        
        # Simulate multiple rounds
        rounds = [
            {
                "name": "Round 1",
                "weight_delta": [0.1, -0.2, 0.3, -0.1],
                "bias_delta": 0.05
            },
            {
                "name": "Round 2", 
                "weight_delta": [-0.05, 0.1, -0.15, 0.08],
                "bias_delta": -0.02
            },
            {
                "name": "Round 3",
                "weight_delta": [0.08, -0.12, 0.06, -0.09],
                "bias_delta": 0.03
            },
            {
                "name": "Round 4",
                "weight_delta": [-0.03, 0.07, -0.11, 0.14],
                "bias_delta": -0.01
            }
        ]
        
        print(f"\n🔄 Applying {len(rounds)} rounds of updates...")
        
        accuracies = [initial_acc]
        for i, round_data in enumerate(rounds):
            print(f"\n{round_data['name']}:")
            
            # Apply update
            acc = cloud.update_global_model(round_data, X_test, y_test)
            accuracies.append(acc)
            
            # Show progress
            print(f"   📊 Round {cloud.round} accuracy: {acc:.4f}")
            if i > 0:
                change = acc - accuracies[i]
                print(f"   📈 Change from previous: {change:+.4f}")
            
            # Show model state
            for name, param in cloud.global_model.named_parameters():
                norm = param.norm().item()
                print(f"   📊 {name} norm: {norm:.6f}")
        
        # Summary
        print(f"\n📋 Multiple Rounds Summary:")
        print(f"   📊 Initial accuracy: {accuracies[0]:.4f}")
        print(f"   📊 Final accuracy: {accuracies[-1]:.4f}")
        print(f"   📈 Total improvement: {accuracies[-1] - accuracies[0]:+.4f}")
        print(f"   🔄 Total rounds completed: {cloud.round}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def show_current_pipeline_status():
    """Show what's currently available in the pipeline"""
    
    print(f"\n🔐 Current Pipeline Status:")
    print("=" * 40)
    
    # Check client updates
    json_dir = "updates/json"
    if os.path.exists(json_dir):
        json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
        rounds = set()
        for f in json_files:
            if 'round_' in f:
                round_num = f.split('round_')[1].split('.')[0]
                rounds.add(int(round_num))
        
        print(f"   📁 Client Updates: {len(json_files)} files")
        print(f"   🔄 Available rounds: {sorted(rounds)}")
    
    # Check encrypted updates
    encrypted_dir = "updates/encrypted"
    if os.path.exists(encrypted_dir):
        encrypted_files = [f for f in os.listdir(encrypted_dir) if f.endswith('.json')]
        print(f"   🔐 Encrypted Updates: {len(encrypted_files)} files")
        if encrypted_files:
            print(f"      📄 Files: {encrypted_files}")
    
    # Check aggregated results
    outbox_dir = "Sriven/outbox"
    if os.path.exists(outbox_dir):
        agg_files = [f for f in os.listdir(outbox_dir) if f.endswith('.json')]
        print(f"   📦 Aggregated Results: {len(agg_files)} files")
        if agg_files:
            print(f"      📄 Files: {agg_files}")
    
    print(f"\n💡 To get multiple rounds in global update:")
    print(f"   1. Encrypt all client updates (rounds 0-4)")
    print(f"   2. Aggregate encrypted updates for each round")
    print(f"   3. Decrypt aggregated results for each round")
    print(f"   4. Feed each round to global update")

if __name__ == "__main__":
    print("🚀 Multiple Rounds Global Update Test")
    print("=" * 50)
    
    # Test multiple rounds
    success = test_multiple_rounds()
    
    # Show current pipeline status
    show_current_pipeline_status()
    
    if success:
        print(f"\n🎉 Multiple rounds test completed!")
        print(f"💡 This shows how global update works with multiple rounds")
    else:
        print(f"\n❌ Test failed. Check the output above for details.")
        sys.exit(1)
