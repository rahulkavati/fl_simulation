"""
Simple Test for Real FHE CKKS Implementation
Tests the core functionality without complex data loading
"""

import numpy as np
from src.real_fhe_ckks import RealFHEConfig, RealFHEEncryption, RealEncryptedModel

def test_real_fhe_simple():
    """Simple test of real FHE implementation"""
    print("🧪 Testing Real FHE CKKS Implementation (Simple)")
    print("="*60)
    
    # Create configuration
    config = RealFHEConfig()
    
    # Initialize FHE encryption
    fhe_encryption = RealFHEEncryption(config)
    
    # Create simple test data
    test_weights = np.array([0.1, -0.2, 0.3, -0.4, 0.5])
    test_bias = 0.6
    
    print(f"\n📊 Test Data:")
    print(f"  Weights: {test_weights}")
    print(f"  Bias: {test_bias}")
    
    # Test encryption
    print(f"\n🔐 Testing Encryption:")
    encrypted_model = RealEncryptedModel(test_weights, test_bias, fhe_encryption.context)
    
    # Test decryption
    print(f"\n🔓 Testing Decryption:")
    decrypted_weights, decrypted_bias = encrypted_model.decrypt_for_evaluation()
    
    print(f"  Original weights: {test_weights}")
    print(f"  Decrypted weights: {decrypted_weights}")
    print(f"  Original bias: {test_bias}")
    print(f"  Decrypted bias: {decrypted_bias}")
    
    # Test aggregation
    print(f"\n🔄 Testing Encrypted Aggregation:")
    test_updates = []
    for i in range(3):
        update = np.random.normal(0, 0.1, 6)  # 5 weights + 1 bias
        encrypted_update, _ = fhe_encryption.encrypt_model_update(update)
        test_updates.append(encrypted_update)
    
    sample_counts = [100, 150, 200]
    aggregated_update, agg_time = fhe_encryption.aggregate_encrypted_updates(
        test_updates, sample_counts
    )
    
    print(f"  Aggregated {len(test_updates)} updates")
    print(f"  Aggregation time: {agg_time:.3f}s")
    
    # Performance analysis
    print(f"\n📈 Performance Analysis:")
    performance_metrics = encrypted_model.get_performance_metrics()
    for metric, value in performance_metrics.items():
        print(f"  {metric}: {value}")
    
    print(f"\n✅ Real FHE CKKS implementation test completed!")
    return True

if __name__ == "__main__":
    test_real_fhe_simple()
