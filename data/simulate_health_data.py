import os
import sys
import argparse
import numpy as np
import pandas as pd

# Default values
DEFAULT_CLIENTS = 5
SAMPLES_PER_CLIENT = 200
OUTPUT_DIR = "data/clients"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_client_data(client_id):
    # Synthetic smartwatch metrics
    heart_rate = np.random.normal(75, 10, SAMPLES_PER_CLIENT)  # bpm
    steps = np.random.normal(100, 30, SAMPLES_PER_CLIENT)      # steps/min
    calories = np.random.normal(4, 1, SAMPLES_PER_CLIENT)      # kcal/min
    sleep_hours = np.random.normal(7, 1.5, SAMPLES_PER_CLIENT) # hours/night
    
    # Label: Healthy if moderately active and rested
    label = ((heart_rate > 65) & (steps > 80) & (sleep_hours > 6)).astype(int)

    data = np.column_stack((heart_rate, steps, calories, sleep_hours, label))
    df = pd.DataFrame(data, columns=["heart_rate", "steps", "calories", "sleep_hours", "label"])
    
    output_path = os.path.join(OUTPUT_DIR, f"client_{client_id}.csv")
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate synthetic health data for federated learning")
    parser.add_argument("--clients", type=int, default=DEFAULT_CLIENTS, help=f"Number of clients to generate data for (default: {DEFAULT_CLIENTS})")
    
    args = parser.parse_args()
    
    print(f"Generating synthetic health data for {args.clients} clients...")
    
    for cid in range(args.clients):
        generate_client_data(cid)
    print(f"Generated {args.clients} synthetic client datasets in '{OUTPUT_DIR}'")
