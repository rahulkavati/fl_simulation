#!/usr/bin/env python3
"""
Demonstration script showing the difference between secure and insecure global updates
"""

import os
import sys
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def demonstrate_pipeline_difference():
    """Demonstrate the difference between secure and insecure approaches"""
    
    print("🔐 Global Update Security Demonstration")
    print("=" * 50)
    
    # Check if we have the necessary files
    encrypted_file = "Sriven/outbox/agg_round_0.json"
    decrypted_file = "Sriven/outbox/agg_round_0.decrypted.json"
    
    print("\n📁 Checking Available Files:")
    print(f"   🔐 Encrypted aggregation: {'✅ Found' if os.path.exists(encrypted_file) else '❌ Not found'}")
    print(f"   🔓 Decrypted aggregation: {'✅ Found' if os.path.exists(decrypted_file) else '❌ Not found'}")
    
    if not os.path.exists(encrypted_file):
        print("\n⚠️  No encrypted aggregation found. Run the pipeline first:")
        print("   1. python simulation/client_simulation.py")
        print("   2. python Huzaif/encrypt_update.py --in <file> --out <file>")
        print("   3. python Sriven/smart_switch_tenseal.py --fedl_dir updates/encrypted --ctx_b64 Huzaif/keys/params.ctx.b64 --out_dir Sriven/outbox")
        return
    
    print("\n🔄 Pipeline Comparison:")
    print("=" * 30)
    
    print("\n❌ INSECURE APPROACH (Current):")
    print("   1. Client Simulation → plaintext updates")
    print("   2. Encryption → encrypted client updates")
    print("   3. Aggregation → encrypted aggregated result")
    print("   4. Decryption → decrypted aggregated result ← VULNERABILITY")
    print("   5. Global Update → uses decrypted result")
    print("   ⚠️  Problem: Step 4 exposes aggregated data in plaintext")
    
    print("\n✅ SECURE APPROACH (Recommended):")
    print("   1. Client Simulation → plaintext updates")
    print("   2. Encryption → encrypted client updates")
    print("   3. Aggregation → encrypted aggregated result")
    print("   4. Global Update → uses encrypted result directly ← SECURE")
    print("   🔒 Advantage: No decryption step, data stays encrypted")
    
    print("\n🔐 Security Benefits of Secure Approach:")
    print("   • Aggregated data never leaves encrypted form")
    print("   • No intermediate decryption step")
    print("   • Reduced attack surface")
    print("   • Better privacy preservation")
    print("   • Easier compliance demonstration")
    
    print("\n💻 Code Comparison:")
    print("=" * 20)
    
    print("\n❌ Insecure Code:")
    print("   from cloud.global_update import CloudServer, load_aggregated_update")
    print("   decrypted_agg = load_aggregated_update('agg_round_0.decrypted.json')")
    print("   cloud.update_global_model(decrypted_agg, X_test, y_test)")
    
    print("\n✅ Secure Code:")
    print("   from cloud.global_update import CloudServer, load_encrypted_aggregation")
    print("   encrypted_agg = load_encrypted_aggregation('agg_round_0.json')")
    print("   cloud.update_global_model_encrypted(encrypted_agg, X_test, y_test)")
    
    print("\n🚀 Migration Steps:")
    print("=" * 20)
    print("   1. Replace load_aggregated_update() with load_encrypted_aggregation()")
    print("   2. Replace update_global_model() with update_global_model_encrypted()")
    print("   3. Remove decryption step from pipeline")
    print("   4. Update documentation and tests")
    print("   5. Train team on secure practices")
    
    print("\n📋 Action Items:")
    print("=" * 15)
    print("   ✅ Use test_encrypted_global_update.py for testing")
    print("   ✅ Update pipeline scripts to skip decryption")
    print("   ✅ Modify existing code to use encrypted methods")
    print("   ✅ Update documentation to reflect secure approach")
    print("   ✅ Consider removing decryption step entirely")

def test_both_approaches():
    """Test both approaches if files are available"""
    
    print("\n🧪 Testing Both Approaches:")
    print("=" * 30)
    
    encrypted_file = "Sriven/outbox/agg_round_0.json"
    decrypted_file = "Sriven/outbox/agg_round_0.decrypted.json"
    
    if os.path.exists(encrypted_file) and os.path.exists(decrypted_file):
        print("   📁 Both files available - testing both approaches...")
        
        try:
            from cloud.global_update import CloudServer, load_encrypted_aggregation, load_aggregated_update
            
            # Test secure approach
            print("\n🔐 Testing Secure Approach:")
            encrypted_agg = load_encrypted_aggregation(encrypted_file)
            input_dim = encrypted_agg['layout']['weights']
            cloud_secure = CloudServer(input_dim=input_dim)
            
            X_test = np.random.randn(50, input_dim)
            y_test = (np.sum(X_test, axis=1) > 0).astype(int)
            
            acc_secure = cloud_secure.update_global_model_encrypted(encrypted_agg, X_test, y_test)
            print(f"   📊 Secure approach accuracy: {acc_secure:.4f}")
            
            # Test insecure approach
            print("\n🔓 Testing Insecure Approach:")
            decrypted_agg = load_aggregated_update(decrypted_file)
            cloud_insecure = CloudServer(input_dim=len(decrypted_agg['weight_delta']))
            
            X_test2 = np.random.randn(50, len(decrypted_agg['weight_delta']))
            y_test2 = (np.sum(X_test2, axis=1) > 0).astype(int)
            
            acc_insecure = cloud_insecure.update_global_model(decrypted_agg, X_test2, y_test2)
            print(f"   📊 Insecure approach accuracy: {acc_insecure:.4f}")
            
            print(f"\n💡 Results should be similar, but secure approach is safer!")
            
        except Exception as e:
            print(f"   ❌ Error testing approaches: {e}")
    else:
        print("   ⚠️  Need both encrypted and decrypted files to test both approaches")

if __name__ == "__main__":
    demonstrate_pipeline_difference()
    test_both_approaches()
    
    print("\n🎯 Recommendation:")
    print("   Use the secure approach (test_encrypted_global_update.py) for all future development!")
