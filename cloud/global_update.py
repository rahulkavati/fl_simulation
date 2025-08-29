import os
import json
import numpy as np
import torch
from sklearn.metrics import accuracy_score


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
