#!/usr/bin/env python3
"""
Federated Learning Results Evaluation Script
Manually evaluate and display results from the complete pipeline

Usage: python evaluate_results.py [--detailed] [--compare] [--test-model]
"""

import argparse
import os
import sys
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from cloud.global_update import CloudServer, load_encrypted_aggregation
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.linear_model import LogisticRegression
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure all dependencies are installed: pip install tenseal torch numpy pandas sklearn")
    sys.exit(1)

def print_header(title: str):
    """Print a formatted header"""
    print(f"\n{'='*60}")
    print(f"🔍 {title}")
    print(f"{'='*60}")

def print_section(title: str):
    """Print a formatted section header"""
    print(f"\n📋 {title}")
    print("-" * 40)

def print_success(message: str):
    """Print a success message"""
    print(f"✅ {message}")

def print_error(message: str):
    """Print an error message"""
    print(f"❌ {message}")

def print_info(message: str):
    """Print an info message"""
    print(f"ℹ️  {message}")

def check_file_exists(filepath: str, description: str) -> bool:
    """Check if a file exists and print status"""
    if os.path.exists(filepath):
        size = os.path.getsize(filepath)
        print_success(f"{description}: {filepath} ({size:,} bytes)")
        return True
    else:
        print_error(f"{description}: {filepath} (NOT FOUND)")
        return False

def evaluate_data_generation():
    """Evaluate the data generation step"""
    print_header("Data Generation Evaluation")
    
    data_dir = "data/clients"
    if not os.path.exists(data_dir):
        print_error("Data directory not found. Run the pipeline first.")
        return False
    
    client_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    print_info(f"Found {len(client_files)} client data files")
    
    if not client_files:
        print_error("No client data files found")
        return False
    
    # Check first client file in detail
    first_file = os.path.join(data_dir, client_files[0])
    try:
        df = pd.read_csv(first_file)
        print_success(f"Sample data from {client_files[0]}:")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Sample data:")
        print(df.head(3).to_string(index=False))
        
        # Check data quality
        print_section("Data Quality Check")
        print(f"  Heart rate range: {df['heart_rate'].min():.1f} - {df['heart_rate'].max():.1f} bpm")
        print(f"  Steps range: {df['steps'].min():.1f} - {df['steps'].max():.1f} steps/min")
        print(f"  Calories range: {df['calories'].min():.1f} - {df['calories'].max():.1f} kcal/min")
        print(f"  Sleep range: {df['sleep_hours'].min():.1f} - {df['sleep_hours'].max():.1f} hours")
        print(f"  Label distribution: {df['label'].value_counts().to_dict()}")
        
        return True
    except Exception as e:
        print_error(f"Error reading data file: {e}")
        return False

def evaluate_client_updates():
    """Evaluate the client simulation step"""
    print_header("Client Updates Evaluation")
    
    json_dir = "updates/json"
    if not os.path.exists(json_dir):
        print_error("Client updates directory not found. Run the pipeline first.")
        return False
    
    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
    print_info(f"Found {len(json_files)} client update files")
    
    if not json_files:
        print_error("No client update files found")
        return False
    
    # Analyze update structure
    print_section("Update Structure Analysis")
    sample_file = os.path.join(json_dir, json_files[0])
    try:
        with open(sample_file, 'r') as f:
            sample_update = json.load(f)
        
        print_success(f"Sample update structure from {json_files[0]}:")
        for key, value in sample_update.items():
            if key == 'weight_delta':
                print(f"  {key}: {len(value)} weights (first 3: {value[:3]})")
            elif key == 'bias_delta':
                print(f"  {key}: {value}")
            else:
                print(f"  {key}: {value}")
        
        # Check all rounds
        rounds = set()
        clients = set()
        for file in json_files:
            if '_round_' in file:
                parts = file.split('_')
                client_id = parts[1]
                round_id = parts[3].split('.')[0]
                rounds.add(round_id)
                clients.add(client_id)
        
        print_section("Coverage Analysis")
        print(f"  Rounds: {sorted(list(rounds))}")
        print(f"  Clients: {sorted(list(clients))}")
        print(f"  Total updates: {len(json_files)}")
        print(f"  Expected updates: {len(rounds) * len(clients)}")
        
        return True
    except Exception as e:
        print_error(f"Error analyzing updates: {e}")
        return False

def evaluate_encryption():
    """Evaluate the encryption step"""
    print_header("Encryption Evaluation")
    
    encrypted_dir = "updates/encrypted"
    if not os.path.exists(encrypted_dir):
        print_error("Encrypted updates directory not found. Run the pipeline first.")
        return False
    
    encrypted_files = [f for f in os.listdir(encrypted_dir) if f.endswith('.json')]
    print_info(f"Found {len(encrypted_files)} encrypted update files")
    
    if not encrypted_files:
        print_error("No encrypted update files found")
        return False
    
    # Check encryption structure
    print_section("Encryption Structure Analysis")
    sample_file = os.path.join(encrypted_dir, encrypted_files[0])
    try:
        with open(sample_file, 'r') as f:
            sample_encrypted = json.load(f)
        
        print_success(f"Sample encrypted update structure from {encrypted_files[0]}:")
        for key, value in sample_encrypted.items():
            if key == 'ciphertext':
                print(f"  {key}: {len(value)} characters (base64 encoded)")
            elif key == 'layout':
                print(f"  {key}: {value}")
            else:
                print(f"  {key}: {value}")
        
        # Check encryption consistency
        print_section("Encryption Consistency")
        layouts = set()
        ctx_refs = set()
        for file in encrypted_files[:5]:  # Check first 5 files
            with open(os.path.join(encrypted_dir, file), 'r') as f:
                data = json.load(f)
                layouts.add(json.dumps(data.get('layout', {}), sort_keys=True))
                ctx_refs.add(data.get('ctx_ref', 'unknown'))
        
        print(f"  Layout consistency: {'✅' if len(layouts) == 1 else '❌'} ({len(layouts)} different layouts)")
        print(f"  Context reference consistency: {'✅' if len(ctx_refs) == 1 else '❌'} ({len(ctx_refs)} different refs)")
        
        return True
    except Exception as e:
        print_error(f"Error analyzing encryption: {e}")
        return False

def evaluate_aggregation():
    """Evaluate the aggregation step"""
    print_header("Aggregation Evaluation")
    
    outbox_dir = "Sriven/outbox"
    if not os.path.exists(outbox_dir):
        print_error("Aggregation output directory not found. Run the pipeline first.")
        return False
    
    agg_files = [f for f in os.listdir(outbox_dir) if f.startswith('agg_round_') and f.endswith('.json')]
    print_info(f"Found {len(agg_files)} aggregation files")
    
    if not agg_files:
        print_error("No aggregation files found")
        return False
    
    # Analyze aggregation structure
    print_section("Aggregation Structure Analysis")
    sample_file = os.path.join(outbox_dir, agg_files[0])
    try:
        with open(sample_file, 'r') as f:
            sample_agg = json.load(f)
        
        print_success(f"Sample aggregation structure from {agg_files[0]}:")
        for key, value in sample_agg.items():
            if key == 'ciphertext':
                print(f"  {key}: {len(value)} characters (base64 encoded)")
            elif key == 'layout':
                print(f"  {key}: {value}")
            else:
                print(f"  {key}: {value}")
        
        # Check aggregation coverage
        print_section("Aggregation Coverage")
        rounds = []
        for file in agg_files:
            round_id = file.split('_round_')[1].split('.')[0]
            rounds.append(round_id)
        
        print(f"  Aggregated rounds: {sorted(rounds)}")
        print(f"  Total aggregated rounds: {len(rounds)}")
        
        return True
    except Exception as e:
        print_error(f"Error analyzing aggregation: {e}")
        return False

def evaluate_global_model():
    """Evaluate the global model update step"""
    print_header("Global Model Evaluation")
    
    global_dir = "federated_artifacts/global"
    if not os.path.exists(global_dir):
        print_error("Global model directory not found. Run the pipeline first.")
        return False
    
    model_files = [f for f in os.listdir(global_dir) if f.endswith('.npz')]
    print_info(f"Found {len(model_files)} global model snapshots")
    
    if not model_files:
        print_error("No global model files found")
        return False
    
    # Load and analyze the latest model
    print_section("Latest Global Model Analysis")
    latest_file = sorted(model_files)[-1]
    model_path = os.path.join(global_dir, latest_file)
    
    try:
        # Load the model
        cloud = CloudServer(input_dim=4)
        cloud.load_snapshot(model_path)
        
        print_success(f"Loaded global model from {latest_file}")
        
        # Get model parameters
        with torch.no_grad():
            weights = cloud.global_model.weight.data.numpy().flatten()
            bias = cloud.global_model.bias.data.numpy().flatten()
        
        print_section("Model Parameters")
        print(f"  Weight shape: {cloud.global_model.weight.shape}")
        print(f"  Bias shape: {cloud.global_model.bias.shape}")
        print(f"  Weights: {weights}")
        print(f"  Bias: {bias}")
        print(f"  Weight norm: {np.linalg.norm(weights):.4f}")
        
        # Test model on sample data
        print_section("Model Testing")
        test_data = np.array([
            [75, 100, 4, 7],    # Normal values
            [90, 150, 6, 8],    # High activity
            [60, 50, 2, 5]      # Low activity
        ])
        
        with torch.no_grad():
            test_tensor = torch.tensor(test_data, dtype=torch.float32)
            predictions = torch.sigmoid(cloud.global_model(test_tensor))
        
        print_success("Sample predictions:")
        for i, (data, pred) in enumerate(zip(test_data, predictions)):
            print(f"  Input {i+1}: {data} -> Prediction: {pred.item():.4f}")
        
        return True
    except Exception as e:
        print_error(f"Error analyzing global model: {e}")
        return False

def test_model_performance():
    """Test the global model performance on health data"""
    print_header("Model Performance Testing")
    
    # Load test data
    data_dir = "data/clients"
    if not os.path.exists(data_dir):
        print_error("No test data available")
        return False
    
    # Combine all client data for testing
    all_data = []
    all_labels = []
    
    for i in range(5):  # Assuming 5 clients
        client_file = os.path.join(data_dir, f"client_{i}.csv")
        if os.path.exists(client_file):
            df = pd.read_csv(client_file)
            features = df[["heart_rate", "steps", "calories", "sleep_hours"]].values
            labels = df["label"].values
            all_data.append(features)
            all_labels.append(labels)
    
    if not all_data:
        print_error("No test data found")
        return False
    
    X_test = np.vstack(all_data)
    y_test = np.concatenate(all_labels)
    
    print_success(f"Test dataset: {X_test.shape[0]} samples, {X_test.shape[1]} features")
    
    # Load global model
    global_dir = "federated_artifacts/global"
    model_files = [f for f in os.listdir(global_dir) if f.endswith('.npz')]
    
    if not model_files:
        print_error("No global model found")
        return False
    
    latest_file = sorted(model_files)[-1]
    model_path = os.path.join(global_dir, latest_file)
    
    try:
        cloud = CloudServer(input_dim=4)
        cloud.load_snapshot(model_path)
        
        # Make predictions
        with torch.no_grad():
            test_tensor = torch.tensor(X_test, dtype=torch.float32)
            logits = cloud.global_model(test_tensor)
            probabilities = torch.sigmoid(logits)
            predictions = (probabilities > 0.5).float().numpy().flatten()
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, predictions)
        
        print_section("Performance Metrics")
        print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  Total samples: {len(y_test)}")
        print(f"  Positive samples: {sum(y_test)}")
        print(f"  Negative samples: {len(y_test) - sum(y_test)}")
        
        # Detailed classification report
        print_section("Detailed Classification Report")
        report = classification_report(y_test, predictions, target_names=['Unhealthy', 'Healthy'])
        print(report)
        
        return True
    except Exception as e:
        print_error(f"Error testing model performance: {e}")
        return False

def compare_with_baseline():
    """Compare federated learning results with centralized training"""
    print_header("Comparison with Centralized Training")
    
    # Load all data
    data_dir = "data/clients"
    if not os.path.exists(data_dir):
        print_error("No data available for comparison")
        return False
    
    all_data = []
    all_labels = []
    
    for i in range(5):
        client_file = os.path.join(data_dir, f"client_{i}.csv")
        if os.path.exists(client_file):
            df = pd.read_csv(client_file)
            features = df[["heart_rate", "steps", "calories", "sleep_hours"]].values
            labels = df["label"].values
            all_data.append(features)
            all_labels.append(labels)
    
    if not all_data:
        print_error("No data found for comparison")
        return False
    
    X_combined = np.vstack(all_data)
    y_combined = np.concatenate(all_labels)
    
    print_success(f"Combined dataset: {X_combined.shape[0]} samples")
    
    # Train centralized model
    print_section("Training Centralized Model")
    centralized_model = LogisticRegression(random_state=42, max_iter=1000)
    centralized_model.fit(X_combined, y_combined)
    
    # Test centralized model
    y_pred_centralized = centralized_model.predict(X_combined)
    accuracy_centralized = accuracy_score(y_combined, y_pred_centralized)
    
    print(f"  Centralized model accuracy: {accuracy_centralized:.4f}")
    print(f"  Centralized model weights: {centralized_model.coef_.flatten()}")
    print(f"  Centralized model bias: {centralized_model.intercept_[0]:.4f}")
    
    # Load federated model
    global_dir = "federated_artifacts/global"
    model_files = [f for f in os.listdir(global_dir) if f.endswith('.npz')]
    
    if model_files:
        latest_file = sorted(model_files)[-1]
        model_path = os.path.join(global_dir, latest_file)
        
        cloud = CloudServer(input_dim=4)
        cloud.load_snapshot(model_path)
        
        with torch.no_grad():
            test_tensor = torch.tensor(X_combined, dtype=torch.float32)
            logits = cloud.global_model(test_tensor)
            probabilities = torch.sigmoid(logits)
            y_pred_federated = (probabilities > 0.5).float().numpy().flatten()
        
        accuracy_federated = accuracy_score(y_combined, y_pred_federated)
        
        print_section("Comparison Results")
        print(f"  Federated model accuracy: {accuracy_federated:.4f}")
        print(f"  Centralized model accuracy: {accuracy_centralized:.4f}")
        print(f"  Accuracy difference: {accuracy_federated - accuracy_centralized:.4f}")
        
        # Compare predictions
        agreement = np.mean(y_pred_federated == y_pred_centralized)
        print(f"  Prediction agreement: {agreement:.4f} ({agreement*100:.2f}%)")
        
        return True
    else:
        print_error("No federated model found for comparison")
        return False

def display_metrics():
    """Display performance metrics from the pipeline"""
    print_header("Performance Metrics")
    
    metrics_dir = "metrics"
    if not os.path.exists(metrics_dir):
        print_error("Metrics directory not found")
        return False
    
    metrics_files = [f for f in os.listdir(metrics_dir) if f.endswith('.json')]
    
    if not metrics_files:
        print_error("No metrics files found")
        return False
    
    # Load and display metrics
    for metrics_file in metrics_files:
        metrics_path = os.path.join(metrics_dir, metrics_file)
        try:
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            
            print_section(f"Metrics from {metrics_file}")
            for key, value in metrics.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
        except Exception as e:
            print_error(f"Error reading metrics file {metrics_file}: {e}")

def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description="Federated Learning Results Evaluation")
    parser.add_argument("--detailed", action="store_true", help="Show detailed analysis")
    parser.add_argument("--compare", action="store_true", help="Compare with centralized training")
    parser.add_argument("--test-model", action="store_true", help="Test model performance")
    parser.add_argument("--metrics", action="store_true", help="Show performance metrics")
    
    args = parser.parse_args()
    
    print("🔍 Federated Learning Results Evaluation")
    print("=" * 60)
    
    # Run evaluations
    evaluations = [
        ("Data Generation", evaluate_data_generation),
        ("Client Updates", evaluate_client_updates),
        ("Encryption", evaluate_encryption),
        ("Aggregation", evaluate_aggregation),
        ("Global Model", evaluate_global_model),
    ]
    
    success_count = 0
    for name, eval_func in evaluations:
        try:
            if eval_func():
                success_count += 1
        except Exception as e:
            print_error(f"Error in {name} evaluation: {e}")
    
    # Optional evaluations
    if args.test_model:
        print("\n" + "="*60)
        test_model_performance()
    
    if args.compare:
        print("\n" + "="*60)
        compare_with_baseline()
    
    if args.metrics:
        print("\n" + "="*60)
        display_metrics()
    
    # Summary
    print_header("Evaluation Summary")
    print(f"✅ Successful evaluations: {success_count}/{len(evaluations)}")
    
    if success_count == len(evaluations):
        print_success("All pipeline components are working correctly!")
        print_info("Your federated learning pipeline has successfully:")
        print("  • Generated synthetic health data")
        print("  • Trained models on individual clients")
        print("  • Encrypted client updates")
        print("  • Aggregated encrypted updates")
        print("  • Updated the global model securely")
    else:
        print_error("Some pipeline components have issues. Check the errors above.")

if __name__ == "__main__":
    main()
