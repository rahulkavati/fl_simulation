# 🔐 Secure Global Update Guide

## Problem Statement

The original federated learning pipeline had a security vulnerability:

```
❌ OLD PIPELINE (Insecure):
1. Client Simulation → plaintext updates
2. Encryption → encrypted client updates  
3. Aggregation → encrypted aggregated result
4. Decryption → decrypted aggregated result  ← SECURITY HOLE
5. Global Update → uses decrypted result
```

**Issue**: Step 4 (decryption) exposes the aggregated data in plaintext, defeating the purpose of encryption.

## Solution: Encrypted Global Updates

The new secure pipeline eliminates the decryption step:

```
✅ NEW PIPELINE (Secure):
1. Client Simulation → plaintext updates
2. Encryption → encrypted client updates
3. Aggregation → encrypted aggregated result
4. Global Update → uses encrypted result directly  ← SECURE
```

## Implementation Details

### 1. Enhanced CloudServer Class

The `CloudServer` class now supports two methods:

#### Method 1: Traditional (for backward compatibility)
```python
def update_global_model(self, aggregated_update, X_test=None, y_test=None):
    # Works with decrypted/plaintext aggregated updates
    self.round += 1
    self.apply_update(aggregated_update)
    self.save_snapshot(aggregated_update)
    # ... evaluation code
```

#### Method 2: Secure (preferred)
```python
def update_global_model_encrypted(self, encrypted_aggregation, X_test=None, y_test=None, ctx_path="Huzaif/keys/secret.ctx"):
    # Works directly with encrypted aggregation from Sriven
    self.round += 1
    update = self.apply_encrypted_update(encrypted_aggregation, ctx_path)
    self.save_snapshot(update)
    # ... evaluation code
```

### 2. New Helper Functions

#### Load Encrypted Aggregation
```python
def load_encrypted_aggregation(path):
    """
    Load encrypted aggregation from Sriven's output.
    This is the preferred method for secure global updates.
    """
    if not path.endswith(".json"):
        raise ValueError("Encrypted aggregation must be in JSON format")
    
    with open(path, "r") as f:
        data = json.load(f)
    
    # Validate required fields
    required_fields = ["ciphertext", "layout", "round_id"]
    for field in required_fields:
        if field not in data:
            raise ValueError(f"Missing required field: {field}")
    
    return data
```

### 3. Internal Decryption Process

The `apply_encrypted_update` method handles decryption internally:

```python
def apply_encrypted_update(self, encrypted_aggregation, ctx_path="Huzaif/keys/secret.ctx"):
    # Load TenSEAL context with secret key
    with open(ctx_path, "rb") as f:
        ctx = ts.context_from(f.read())
    
    # Extract ciphertext and layout
    ciphertext_b64 = encrypted_aggregation["ciphertext"]
    layout = encrypted_aggregation["layout"]
    
    # Decode and reconstruct the vector
    ct_bytes = base64.b64decode(ciphertext_b64)
    ct_vec = ts.ckks_vector_from(ctx, ct_bytes)
    
    # Decrypt to get the aggregated deltas
    values = ct_vec.decrypt()  # list[float]
    
    # Extract weights and bias based on layout
    w_len = int(layout["weights"])
    b_len = int(layout["bias"])
    
    weight_delta = values[:w_len]
    bias_delta = values[w_len]
    
    # Create update dictionary and apply
    update = {
        "weight_delta": weight_delta,
        "bias_delta": float(bias_delta),
        "round_id": encrypted_aggregation.get("round_id", self.round),
        "ctx_ref": encrypted_aggregation.get("ctx_ref"),
        "created_at": encrypted_aggregation.get("created_at")
    }
    
    self.apply_update(update)
    return update
```

## Usage Examples

### Secure Global Update (Recommended)

```python
from cloud.global_update import CloudServer, load_encrypted_aggregation
import numpy as np

# Load encrypted aggregation directly
encrypted_file = "Sriven/outbox/agg_round_0.json"
encrypted_agg = load_encrypted_aggregation(encrypted_file)

# Initialize cloud server
input_dim = encrypted_agg['layout']['weights']
cloud = CloudServer(input_dim=input_dim)

# Create test data
X_test = np.random.randn(100, input_dim)
y_test = (np.sum(X_test, axis=1) > 0).astype(int)

# Update global model securely
accuracy = cloud.update_global_model_encrypted(encrypted_agg, X_test, y_test)
print(f"Round {cloud.round} accuracy: {accuracy:.4f}")
```

### Traditional Global Update (Legacy)

```python
from cloud.global_update import CloudServer, load_aggregated_update

# Load decrypted aggregation (not recommended)
decrypted_file = "Sriven/outbox/agg_round_0.decrypted.json"
decrypted_agg = load_aggregated_update(decrypted_file)

# Initialize cloud server
input_dim = len(decrypted_agg['weight_delta'])
cloud = CloudServer(input_dim=input_dim)

# Update global model (less secure)
accuracy = cloud.update_global_model(decrypted_agg, X_test, y_test)
```

## Security Benefits

### 1. **No Intermediate Decryption**
- Aggregated data never exists in plaintext outside the cloud server
- Eliminates the security vulnerability of step 4 in the old pipeline

### 2. **Reduced Attack Surface**
- Fewer files containing sensitive data
- No decrypted aggregation files stored on disk
- Decryption only happens internally within the cloud server

### 3. **Better Privacy Preservation**
- Aggregated client contributions remain encrypted throughout the pipeline
- Only the final model parameters are exposed (which is necessary for model usage)

### 4. **Compliance Benefits**
- Easier to demonstrate data protection compliance
- Clear audit trail of encrypted data handling
- Reduced risk of data breaches

## Migration Guide

### For Existing Code

1. **Replace decrypted file loading**:
   ```python
   # Old
   decrypted_agg = load_aggregated_update("agg_round_0.decrypted.json")
   cloud.update_global_model(decrypted_agg, X_test, y_test)
   
   # New
   encrypted_agg = load_encrypted_aggregation("agg_round_0.json")
   cloud.update_global_model_encrypted(encrypted_agg, X_test, y_test)
   ```

2. **Update pipeline scripts**:
   - Remove decryption step from automation scripts
   - Update documentation to reflect new secure pipeline

3. **Update tests**:
   - Use `test_encrypted_global_update.py` instead of `test_global_update.py`
   - Test both methods for backward compatibility

### For New Code

Always use the encrypted method:

```python
# Always prefer this approach
cloud.update_global_model_encrypted(encrypted_agg, X_test, y_test)
```

## Testing

### Run Secure Tests
```bash
python test_encrypted_global_update.py
```

### Complete Secure Pipeline
```bash
# 1. Generate client updates
python simulation/client_simulation.py

# 2. Encrypt client updates
python Huzaif/encrypt_update.py --in updates/json/client_0_round_0.json --out updates/encrypted/enc_client_0_round_0.json

# 3. Aggregate encrypted updates
python Sriven/smart_switch_tenseal.py --fedl_dir updates/encrypted --ctx_b64 Huzaif/keys/params.ctx.b64 --out_dir Sriven/outbox

# 4. Update global model securely (NO DECRYPTION NEEDED)
python test_encrypted_global_update.py
```

## Best Practices

### 1. **Always Use Encrypted Method**
- Prefer `update_global_model_encrypted()` over `update_global_model()`
- Only use decrypted method for testing or legacy compatibility

### 2. **Secure Key Management**
- Ensure TenSEAL context with secret key is properly secured
- Use appropriate file permissions for key files
- Consider key rotation strategies

### 3. **Error Handling**
- Always handle TenSEAL import errors gracefully
- Validate encrypted aggregation format before processing
- Provide clear error messages for debugging

### 4. **Monitoring and Logging**
- Log successful encrypted updates
- Monitor for decryption failures
- Track model performance with encrypted updates

### 5. **Documentation**
- Document the security benefits of the new approach
- Update all pipeline documentation
- Train team members on secure practices

## Troubleshooting

### Common Issues

1. **TenSEAL Not Available**
   ```
   Warning: TenSEAL not available. Encrypted updates will not work.
   ```
   **Solution**: Install TenSEAL: `pip install tenseal`

2. **Missing Secret Key**
   ```
   RuntimeError: Context does not have secret key for decryption
   ```
   **Solution**: Ensure `Huzaif/keys/secret.ctx` exists and contains secret key

3. **Invalid Aggregation Format**
   ```
   ValueError: Missing required field: ciphertext
   ```
   **Solution**: Verify aggregation file format matches expected schema

4. **Layout Mismatch**
   ```
   ValueError: Expected bias length 1, got 2
   ```
   **Solution**: Check that aggregation layout matches model architecture

## Conclusion

The new encrypted global update approach provides significant security improvements:

- ✅ **Eliminates decryption step vulnerability**
- ✅ **Maintains data encryption throughout pipeline**
- ✅ **Reduces attack surface**
- ✅ **Improves privacy preservation**
- ✅ **Enables compliance benefits**

By using `update_global_model_encrypted()` and `load_encrypted_aggregation()`, you can ensure that your federated learning pipeline maintains the highest level of security while still providing the same functionality as the original approach.
