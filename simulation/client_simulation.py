import os
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
import warnings

# Suppress convergence warnings if they still occur
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# Paths
DATA_DIR = "data/clients"      # Where simulate_health_data.py saved client CSVs
OUTPUT_JSON = "updates/json"
OUTPUT_NPY = "updates/numpy"

NUM_ROUNDS = 3

# ---------- Helpers ----------
def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=4)

def save_npy(array, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, array)

# ---------- Load client data ----------
def load_client_data():
    clients = {}
    for fname in os.listdir(DATA_DIR):
        if fname.endswith(".csv"):
            cid = fname.replace(".csv", "")
            arr = np.loadtxt(os.path.join(DATA_DIR, fname), delimiter=",", skiprows=1)
            X, y = arr[:, :-1], arr[:, -1]
            clients[cid] = (X, y)
    print(f"Loaded data for {len(clients)} clients")
    return clients

# ---------- Main simulation ----------
def main():
    print("Starting Federated Learning Simulation...")
    clients_data = load_client_data()

    # Initialize global model (random start)
    sample_X, sample_y = next(iter(clients_data.values()))
    global_model = LogisticRegression(
        penalty=None, fit_intercept=True, solver="lbfgs", max_iter=1000, warm_start=True, random_state=42
    )
    global_model.fit(sample_X[:10], sample_y[:10])  # dummy fit to set shapes
    print(f"Initialized global model with {sample_X.shape[1]} features")

    for rnd in range(NUM_ROUNDS):
        print(f"\n--- Round {rnd + 1}/{NUM_ROUNDS} ---")
        base_coef = np.copy(global_model.coef_)
        base_intercept = np.copy(global_model.intercept_)

        for cid, (X_train, y_train) in clients_data.items():
            print(f"  Training client {cid} with {len(X_train)} samples...")
            local_model = LogisticRegression(
                penalty=None, fit_intercept=True, solver="lbfgs", max_iter=1000, warm_start=True, random_state=42
            )
            local_model.classes_ = np.array([0, 1])
            local_model.coef_ = np.copy(base_coef)
            local_model.intercept_ = np.copy(base_intercept)

            local_model.fit(X_train, y_train)

            weight_delta = (local_model.coef_ - base_coef).flatten().tolist()
            bias_delta = (local_model.intercept_ - base_intercept).item()

            # Save JSON
            json_path = os.path.join(OUTPUT_JSON, f"{cid}_round_{rnd}.json")
            save_json({
                "client_id": cid,
                "round_id": rnd,
                "weight_delta": weight_delta,
                "bias_delta": bias_delta,
                "num_samples": len(X_train)
            }, json_path)

            # Save NumPy
            npy_path = os.path.join(OUTPUT_NPY, f"{cid}_round_{rnd}.npy")
            save_npy(np.array(weight_delta + [bias_delta]), npy_path)

        # Simple FedAvg (plaintext)
        print("  Aggregating client updates...")
        all_w = []
        all_b = []
        for cid in clients_data:
            arr = np.load(os.path.join(OUTPUT_NPY, f"{cid}_round_{rnd}.npy"))
            all_w.append(arr[:-1])
            all_b.append(arr[-1])
        avg_w = np.mean(all_w, axis=0)
        avg_b = np.mean(all_b)
        global_model.coef_ = base_coef + avg_w.reshape(1, -1)
        global_model.intercept_ = base_intercept + avg_b
        
        print(f"  Global model updated. Weight norm: {np.linalg.norm(global_model.coef_):.4f}")
    
    print("\nSimulation completed successfully!")
    print(f"Final global model - Weight norm: {np.linalg.norm(global_model.coef_):.4f}")
    print(f"Bias: {global_model.intercept_[0]:.4f}")

if __name__ == "__main__":
    main()