import os
import json
import numpy as np
import torch
from sklearn.metrics import accuracy_score
import base64

try:
    import tenseal as ts
except ImportError:
    print("Warning: TenSEAL not available. Encrypted updates will not work.")
    ts = None


class CloudServer:
    def __init__(self, input_dim, save_dir="federated_artifacts/global"):
        self.global_model = torch.nn.Linear(input_dim, 1, bias=True)
        self.round = 0
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def apply_update(self, aggregated_update):
        with torch.no_grad():
            # Map parameter names to the expected keys in aggregated_update
            param_mapping = {
                'weight': 'weight_delta',
                'bias': 'bias_delta'
            }
            
            for name, param in self.global_model.named_parameters():
                if name in param_mapping:
                    key = param_mapping[name]
                    if key in aggregated_update:
                        delta = aggregated_update[key]
                        # Ensure delta has the right shape
                        if name == 'weight':
                            delta = np.array(delta).reshape(param.shape)
                        else:
                            delta = np.array(delta)
                        param.add_(torch.tensor(delta, dtype=torch.float32))
                    else:
                        print(f"Warning: {key} not found in aggregated_update")
                else:
                    print(f"Warning: Unknown parameter name: {name}")

    def apply_encrypted_update(self, encrypted_aggregation, ctx_path="Huzaif/keys/secret.ctx"):
        """
        Apply encrypted aggregated update directly without decryption.
        This method works with the encrypted output from Sriven's aggregation.
        
        Args:
            encrypted_aggregation: Dict containing encrypted aggregation data
            ctx_path: Path to TenSEAL context with secret key
        """
        if ts is None:
            raise RuntimeError("TenSEAL not available. Cannot process encrypted updates.")
        
        try:
            # Load TenSEAL context with secret key
            with open(ctx_path, "rb") as f:
                ctx = ts.context_from(f.read())
            
            if not ctx.has_secret_key():
                raise RuntimeError("Context does not have secret key for decryption")
            
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
            
            if b_len != 1:
                raise ValueError(f"Expected bias length 1, got {b_len}")
            if len(values) < w_len + b_len:
                raise ValueError(f"Ciphertext length {len(values)} < layout size {w_len + b_len}")
            
            # Extract weight and bias deltas
            weight_delta = values[:w_len]
            bias_delta = values[w_len]  # single float
            
            # Create update dictionary
            update = {
                "weight_delta": weight_delta,
                "bias_delta": float(bias_delta),
                "round_id": encrypted_aggregation.get("round_id", self.round),
                "ctx_ref": encrypted_aggregation.get("ctx_ref"),
                "created_at": encrypted_aggregation.get("created_at")
            }
            
            # Apply the update
            self.apply_update(update)
            print(f"[Cloud] Applied encrypted update for round {update['round_id']}")
            
            return update
            
        except Exception as e:
            print(f"Error applying encrypted update: {e}")
            raise

    def evaluate(self, X_test, y_test):
        with torch.no_grad():
            logits = self.global_model(torch.tensor(X_test, dtype=torch.float32))
            preds = (torch.sigmoid(logits).numpy() > 0.5).astype(int)
            acc = accuracy_score(y_test, preds)
        return acc

    def save_snapshot(self, aggregated_update):
        np.savez(
            os.path.join(self.save_dir, f"global_round_{self.round}.npz"),
            **aggregated_update
        )
    
    def load_snapshot(self, snapshot_path):
        """
        Load a model snapshot from file
        
        Args:
            snapshot_path: Path to the snapshot file (.npz format)
        """
        if not snapshot_path.endswith('.npz'):
            raise ValueError("Snapshot must be in .npz format")
        
        # Load the snapshot data
        snapshot_data = np.load(snapshot_path)
        
        # Extract weight and bias deltas
        weight_delta = snapshot_data['weight_delta']
        bias_delta = snapshot_data['bias_delta']
        
        # Apply the deltas to the current model
        with torch.no_grad():
            self.global_model.weight.data += torch.tensor(weight_delta, dtype=torch.float32).reshape_as(self.global_model.weight)
            self.global_model.bias.data += torch.tensor(bias_delta, dtype=torch.float32).reshape_as(self.global_model.bias)
        
        # Extract round information if available
        if 'round_id' in snapshot_data:
            self.round = int(snapshot_data['round_id'])
        
        print(f"[Cloud] Loaded snapshot from {snapshot_path}")

    def update_global_model(self, aggregated_update, X_test=None, y_test=None):
        self.round += 1
        self.apply_update(aggregated_update)
        self.save_snapshot(aggregated_update)

        if X_test is not None and y_test is not None:
            acc = self.evaluate(X_test, y_test)
            print(f"[Cloud] Round {self.round}: Accuracy = {acc:.4f}")
            return acc
        else:
            print(f"[Cloud] Round {self.round}: Update applied.")
            return None

    def update_global_model_encrypted(self, encrypted_aggregation, X_test=None, y_test=None, ctx_path="Huzaif/keys/secret.ctx"):
        """
        Update global model using encrypted aggregation directly.
        This is the preferred method for secure global updates.
        
        Args:
            encrypted_aggregation: Encrypted aggregation from Sriven
            X_test: Test features for evaluation
            y_test: Test labels for evaluation
            ctx_path: Path to TenSEAL context with secret key
        """
        self.round += 1
        
        # Apply encrypted update
        update = self.apply_encrypted_update(encrypted_aggregation, ctx_path)
        
        # Save snapshot with the decrypted update data
        self.save_snapshot(update)

        if X_test is not None and y_test is not None:
            acc = self.evaluate(X_test, y_test)
            print(f"[Cloud] Round {self.round}: Accuracy = {acc:.4f}")
            return acc
        else:
            print(f"[Cloud] Round {self.round}: Encrypted update applied.")
            return None


def load_aggregated_update(path):
    """
    Expect .npz or .json (from Sriven after decryption).
    """
    if path.endswith(".npz"):
        data = np.load(path)
        return {k: data[k] for k in data.files}
    elif path.endswith(".json"):
        with open(path, "r") as f:
            return json.load(f)
    else:
        raise ValueError("Unsupported file format: must be .npz or .json")


def load_encrypted_aggregation(path):
    """
    Load encrypted aggregation from Sriven's output.
    This is the preferred method for secure global updates.
    
    Args:
        path: Path to encrypted aggregation JSON file
        
    Returns:
        Dict containing encrypted aggregation data
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


if __name__ == "__main__":
    # Simulated test data
    X_test = np.random.randn(100, 5)
    y_test = (np.sum(X_test, axis=1) > 0).astype(int)

    cloud = CloudServer(input_dim=5)

    # Create sample aggregated update for testing
    sample_agg_update = {
        "weight_delta": [0.1, -0.2, 0.3, -0.1, 0.05],
        "bias_delta": 0.02
    }

    cloud.update_global_model(sample_agg_update, X_test, y_test)
