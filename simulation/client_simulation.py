import os
import sys
import json
import numpy as np
import time
import argparse
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import warnings
import base64

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
DEFAULT_ROUNDS = 10  # Increased for better convergence
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
    """
    Load client data from health fitness dataset
    COMMENTED OUT: Data generation is no longer needed - using real health_fitness_dataset.csv
    """
    # OLD CODE (COMMENTED OUT - NO LONGER NEEDED):
    # clients = {}
    # all_X = []
    # 
    # # First pass: collect all data for scaling
    # for fname in os.listdir(DATA_DIR):
    #     if fname.endswith(".csv"):
    #         cid = fname.replace(".csv", "")
    #         arr = np.loadtxt(os.path.join(DATA_DIR, fname), delimiter=",", skiprows=1)
    #         X, y = arr[:, :-1], arr[:, -1]
    #         clients[cid] = (X, y)
    #         all_X.append(X)
    # 
    # # Fit scaler on all data
    # if all_X:
    #     scaler = StandardScaler()
    #     scaler.fit(np.vstack(all_X))
    #     
    #     # Apply scaling to each client's data
    #     for cid in clients:
    #         X, y = clients[cid]
    #         x_scaled = scaler.transform(X)
    #         clients[cid] = (x_scaled, y)
    # 
    # print(f"Loaded and scaled data for {len(clients)} clients")
    # return clients
    
    # NEW CODE: Load from health fitness dataset
    print("Loading health fitness dataset...")
    
    # Import the health fitness data loading function
    try:
        from health_fitness_fl_comparison import load_health_fitness_data, create_client_datasets, scale_client_data
        import pandas as pd
        
        # Load the health fitness dataset
        df, feature_columns = load_health_fitness_data()
        if df is None:
            print("Failed to load health fitness dataset")
            return {}
        
        # Create client datasets
        clients_data = create_client_datasets(df, feature_columns, DEFAULT_CLIENTS)
        if not clients_data:
            print("Failed to create client datasets")
            return {}
        
        # Scale the data
        scaled_clients, scaler = scale_client_data(clients_data)
        
        print(f"Loaded health fitness data for {len(scaled_clients)} clients")
        return scaled_clients
        
    except ImportError:
        print("Could not import health fitness data loading functions")
        print("Please run: python health_fitness_fl_comparison.py")
        return {}

# ---------- Load and decrypt global model ----------
def load_and_decrypt_global_model(round_id=None, ctx_path="Huzaif/keys/secret.ctx"):
    """
    Load and decrypt the encrypted global model on client side (MORE SECURE)
    
    Args:
        round_id: Specific round to load (None for latest)
        ctx_path: Path to TenSEAL context with secret key
    """
    global_model_dir = "updates/global_model"
    
    if not os.path.exists(global_model_dir):
        print("No global model directory found, using random initialization")
        return None
    
    # Find encrypted global model files
    model_files = [f for f in os.listdir(global_model_dir) 
                   if f.startswith('encrypted_global_model_round_') and f.endswith('.json')]
    
    if not model_files:
        print("No encrypted global model files found, using random initialization")
        return None
    
    # Select model file
    if round_id is not None:
        target_file = f"encrypted_global_model_round_{round_id}.json"
        if target_file not in model_files:
            print(f"Encrypted global model for round {round_id} not found, using latest")
            round_id = None
    
    if round_id is None:
        # Use latest round
        round_numbers = [int(f.split('_round_')[1].split('.')[0]) for f in model_files]
        latest_round = max(round_numbers)
        target_file = f"encrypted_global_model_round_{latest_round}.json"
    
    # Load encrypted global model
    model_path = os.path.join(global_model_dir, target_file)
    with open(model_path, "r") as f:
        encrypted_model_data = json.load(f)
    
    try:
        # Import TenSEAL
        import tenseal as ts
        
        # Load TenSEAL context with secret key
        with open(ctx_path, "rb") as f:
            ctx = ts.context_from(f.read())
        
        if not ctx.has_secret_key():
            raise RuntimeError("Context does not have secret key for decryption")
        
        # Decode and reconstruct the encrypted model
        encrypted_model_b64 = encrypted_model_data["encrypted_model"]
        encrypted_model_bytes = base64.b64decode(encrypted_model_b64)
        encrypted_model = ts.ckks_vector_from(ctx, encrypted_model_bytes)
        
        # Decrypt the model parameters
        decrypted_params = encrypted_model.decrypt()
        
        # Extract weights and bias
        input_dim = encrypted_model_data["input_dim"]
        output_dim = encrypted_model_data["output_dim"]
        
        weights = np.array(decrypted_params[:input_dim * output_dim]).reshape(output_dim, input_dim)
        bias = np.array(decrypted_params[input_dim * output_dim:])
        
        print(f"[Client] Decrypted global model from round {encrypted_model_data['round_id']}")
        print(f"[Client] Model parameters: {input_dim} features, {output_dim} outputs")
        print(f"[Client] Weight norm: {np.linalg.norm(weights):.4f}, Bias: {bias[0]:.4f}")
        
        return {
            "round_id": encrypted_model_data["round_id"],
            "weights": weights,
            "bias": bias,
            "model_type": encrypted_model_data["model_type"]
        }
        
    except Exception as e:
        print(f"Error decrypting global model: {e}")
        return None

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

    # Initialize global model (try to load from encrypted distributed model, otherwise random start)
    sample_X, sample_y = next(iter(clients_data.values()))
    
    # Try to load and decrypt the latest encrypted global model
    global_model_data = load_and_decrypt_global_model()
    
    if global_model_data is not None:
        # Initialize with decrypted global model
        weights = global_model_data["weights"]
        bias = global_model_data["bias"]
        
        global_model = LogisticRegression(
            penalty=None, fit_intercept=True, solver="lbfgs", max_iter=5000, warm_start=True, random_state=42
        )
        global_model.fit(sample_X[:10], sample_y[:10])  # dummy fit to set shapes
        global_model.coef_ = weights
        global_model.intercept_ = bias
        
        print(f"Initialized global model from decrypted distributed model (round {global_model_data['round_id']})")
    else:
        # Initialize with proper training on sample data
        global_model = LogisticRegression(
            penalty=None, fit_intercept=True, solver="lbfgs", max_iter=5000, warm_start=True, random_state=42
        )
        # Use more samples for better initialization
        init_samples = min(50, len(sample_X))
        global_model.fit(sample_X[:init_samples], sample_y[:init_samples])
        print(f"Initialized global model with {init_samples} samples")
    
    print(f"Global model has {sample_X.shape[1]} features")

    for rnd in range(args.rounds):
        print(f"\n--- Round {rnd + 1}/{args.rounds} ---")
        base_coef = np.copy(global_model.coef_)
        base_intercept = np.copy(global_model.intercept_)

        for cid, (X_train, y_train) in clients_data.items():
            print(f"  Training client {cid} with {len(X_train)} samples...")
            local_model = LogisticRegression(
                penalty=None, fit_intercept=True, solver="lbfgs", max_iter=5000, warm_start=True, random_state=42
            )
            local_model.classes_ = np.array([0, 1])
            local_model.coef_ = np.copy(base_coef)
            local_model.intercept_ = np.copy(base_intercept)

            # Train with better convergence settings
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

        # Weighted FedAvg based on sample count (plaintext)
        print("  Aggregating client updates...")
        all_w = []
        all_b = []
        sample_counts = []
        
        for cid in clients_data:
            arr = np.load(os.path.join(OUTPUT_NPY, f"{cid}_round_{rnd}.npy"))
            all_w.append(arr[:-1])
            all_b.append(arr[-1])
            sample_counts.append(len(clients_data[cid][0]))
        
        # Weighted average based on sample count
        total_samples = sum(sample_counts)
        weights = np.array(sample_counts) / total_samples
        
        avg_w = np.average(all_w, axis=0, weights=weights)
        avg_b = np.average(all_b, weights=weights)
        
        global_model.coef_ = base_coef + avg_w.reshape(1, -1)
        global_model.intercept_ = base_intercept + avg_b
        
        # Evaluate model performance
        all_x_test = []
        all_y_test = []
        for cid, (X, y) in clients_data.items():
            all_x_test.append(X)
            all_y_test.append(y)
        
        x_test = np.vstack(all_x_test)
        y_test = np.concatenate(all_y_test)
        
        y_pred = global_model.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(f"  Global model updated. Weight norm: {np.linalg.norm(global_model.coef_):.4f}")
        print(f"  Round {rnd + 1} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
    
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