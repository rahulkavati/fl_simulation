import os
import sys
import json
import numpy as np
import time
import argparse
from sklearn.linear_model import LogisticRegression
import warnings

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.efficiency_metrics import FLEfficiencyCalculator

# Suppress convergence warnings if they still occur
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# Paths
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "clients")
OUTPUT_JSON = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "updates", "json")
OUTPUT_NPY = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "updates", "numpy")

# Default values
DEFAULT_ROUNDS = 3
DEFAULT_CLIENTS = 5

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
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Federated Learning Client Simulation")
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS, help=f"Number of federated learning rounds (default: {DEFAULT_ROUNDS})")
    parser.add_argument("--clients", type=int, default=DEFAULT_CLIENTS, help=f"Number of clients to simulate (default: {DEFAULT_CLIENTS})")
    
    args = parser.parse_args()
    
    print("Starting Federated Learning Simulation...")
    print(f"Configuration: {args.rounds} rounds, {args.clients} clients")
    start_time = time.time()
    
    # Initialize efficiency calculator
    efficiency_calc = FLEfficiencyCalculator(DATA_DIR, "updates")
    
    clients_data = load_client_data()
    
    # Limit clients if specified
    if args.clients < len(clients_data):
        client_keys = list(clients_data.keys())[:args.clients]
        clients_data = {k: clients_data[k] for k in client_keys}
        print(f"Limited to {args.clients} clients: {client_keys}")

    # Initialize global model (random start)
    sample_X, sample_y = next(iter(clients_data.values()))
    global_model = LogisticRegression(
        penalty=None, fit_intercept=True, solver="lbfgs", max_iter=1000, warm_start=True, random_state=42
    )
    global_model.fit(sample_X[:10], sample_y[:10])  # dummy fit to set shapes
    print(f"Initialized global model with {sample_X.shape[1]} features")

    for rnd in range(args.rounds):
        print(f"\n--- Round {rnd + 1}/{args.rounds} ---")
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
    
    total_training_time = time.time() - start_time
    
    print("\nSimulation completed successfully!")
    print(f"Final global model - Weight norm: {np.linalg.norm(global_model.coef_):.4f}")
    print(f"Bias: {global_model.intercept_[0]:.4f}")
    print(f"Total training time: {total_training_time:.2f} seconds")
    
    # Calculate and save efficiency metrics
    print("\nCalculating efficiency metrics...")
    metrics = efficiency_calc.calculate_efficiency_metrics(
        clients_data, global_model, args.rounds, total_training_time
    )
    
    # Save metrics with experiment name
    experiment_name = f"fl_simulation_{args.rounds}rounds_{len(clients_data)}clients"
    efficiency_calc.save_metrics(metrics, experiment_name)
    
    # Display key metrics
    print("\n" + "="*50)
    print("FL EFFICIENCY METRICS SUMMARY")
    print("="*50)
    print(f"Communication Rounds: {metrics.total_communication_rounds}")
    print(f"Bytes Transferred: {metrics.bytes_transferred/1024:.2f} KB")
    print(f"Final Accuracy: {metrics.final_accuracy:.4f}")
    print(f"Accuracy Improvement: {metrics.accuracy_improvement:.4f}")
    print(f"Convergence Round: {metrics.convergence_rounds or 'Not reached'}")
    print(f"Memory Usage: {metrics.memory_usage:.4f} MB")
    print("="*50)

if __name__ == "__main__":
    main()