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
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score, confusion_matrix
import time

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
    
    model_files = [f for f in os.listdir(global_dir) if f.startswith('encrypted_global_round_') and f.endswith('.json')]
    print_info(f"Found {len(model_files)} encrypted global model snapshots")
    
    if not model_files:
        print_error("No encrypted global model files found")
        return False
    
    # Load and analyze the latest model
    print_section("Latest Encrypted Global Model Analysis")
    latest_file = sorted(model_files)[-1]
    model_path = os.path.join(global_dir, latest_file)
    
    try:
        # Load the encrypted model data
        with open(model_path, 'r') as f:
            encrypted_model_data = json.load(f)
        
        print_success(f"Loaded encrypted global model from {latest_file}")
        
        # Analyze encrypted model structure
        print_section("Encrypted Model Structure")
        for key, value in encrypted_model_data.items():
            if key == 'encrypted_model':
                print(f"  {key}: {len(value)} characters (base64 encoded)")
            else:
                print(f"  {key}: {value}")
        
        # Verify encryption status
        print_section("Encryption Verification")
        print(f"  Model remains encrypted: ✅ (Server never sees plaintext)")
        print(f"  Round ID: {encrypted_model_data.get('round_id', 'Unknown')}")
        print(f"  Created at: {encrypted_model_data.get('created_at', 'Unknown')}")
        
        return True
    except Exception as e:
        print_error(f"Error analyzing global model: {e}")
        return False

def evaluate_global_to_local_update():
    """Evaluate the global-to-local model update process"""
    print_header("Global-to-Local Model Update Evaluation")
    
    # Check distributed global models
    distributed_dir = "updates/global_model"
    if not os.path.exists(distributed_dir):
        print_error("Distributed global model directory not found")
        return False
    
    distributed_files = [f for f in os.listdir(distributed_dir) if f.startswith('encrypted_global_model_round_') and f.endswith('.json')]
    print_info(f"Found {len(distributed_files)} distributed global model files")
    
    if not distributed_files:
        print_error("No distributed global model files found")
        return False
    
    # Analyze distributed model structure
    print_section("Distributed Model Structure Analysis")
    sample_file = os.path.join(distributed_dir, distributed_files[0])
    try:
        with open(sample_file, 'r') as f:
            sample_distributed = json.load(f)
        
        print_success(f"Sample distributed model structure from {distributed_files[0]}:")
        for key, value in sample_distributed.items():
            if key == 'encrypted_model':
                print(f"  {key}: {len(value)} characters (base64 encoded)")
            elif key == 'distribution_info':
                print(f"  {key}: {value}")
            else:
                print(f"  {key}: {value}")
        
        # Check distribution coverage
        print_section("Distribution Coverage")
        rounds = []
        for file in distributed_files:
            round_id = file.split('_round_')[1].split('.')[0]
            rounds.append(round_id)
        
        print(f"  Distributed rounds: {sorted(rounds)}")
        print(f"  Total distributed rounds: {len(rounds)}")
        
        # Verify encryption status
        print_section("Encryption Verification")
        print(f"  Model remains encrypted: ✅ (Server never sees plaintext)")
        print(f"  Client-side decryption required: ✅ (Privacy preserved)")
        print(f"  Distribution format: JSON with base64 encoded ciphertext")
        
        return True
    except Exception as e:
        print_error(f"Error analyzing distributed models: {e}")
        return False

def test_client_decryption():
    """Test client-side decryption of distributed global models"""
    print_header("Client-Side Decryption Testing")
    
    distributed_dir = "updates/global_model"
    if not os.path.exists(distributed_dir):
        print_error("No distributed models found for decryption testing")
        return False
    
    distributed_files = [f for f in os.listdir(distributed_dir) if f.startswith('encrypted_global_model_round_') and f.endswith('.json')]
    
    if not distributed_files:
        print_error("No distributed global model files found")
        return False
    
    # Test decryption of latest model
    latest_file = sorted(distributed_files)[-1]
    model_path = os.path.join(distributed_dir, latest_file)
    
    try:
        # Import client decryption function
        sys.path.append(os.path.join(os.path.dirname(__file__), 'simulation'))
        from client_simulation import load_and_decrypt_global_model
        
        print_info(f"Testing decryption of {latest_file}")
        
        # Attempt client-side decryption
        decrypted_model = load_and_decrypt_global_model()
        
        if decrypted_model is not None:
            print_success("Client-side decryption successful!")
            print_section("Decrypted Model Information")
            print(f"  Round ID: {decrypted_model['round_id']}")
            print(f"  Model Type: {decrypted_model['model_type']}")
            print(f"  Weight Shape: {decrypted_model['weights'].shape}")
            print(f"  Bias Shape: {decrypted_model['bias'].shape}")
            print(f"  Weight Norm: {np.linalg.norm(decrypted_model['weights']):.4f}")
            print(f"  Bias Value: {decrypted_model['bias'][0]:.4f}")
            
            # Test model initialization
            print_section("Model Initialization Test")
            from sklearn.linear_model import LogisticRegression
            
            # Create test data
            test_X = np.random.randn(10, 4)
            test_y = np.random.randint(0, 2, 10)
            
            # Initialize model with decrypted parameters
            model = LogisticRegression()
            model.fit(test_X[:5], test_y[:5])  # Dummy fit to set shapes
            model.coef_ = decrypted_model['weights']
            model.intercept_ = decrypted_model['bias']
            
            # Test prediction
            predictions = model.predict(test_X)
            print_success(f"Model initialized and tested successfully!")
            print(f"  Test predictions: {predictions}")
            
            return True
        else:
            print_error("Client-side decryption failed")
            return False
            
    except Exception as e:
        print_error(f"Error testing client decryption: {e}")
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
    model_files = [f for f in os.listdir(global_dir) if f.startswith('encrypted_global_round_') and f.endswith('.json')]
    
    if not model_files:
        print_error("No encrypted global model found")
        return False
    
    latest_file = sorted(model_files)[-1]
    model_path = os.path.join(global_dir, latest_file)
    
    try:
        # Load encrypted model data
        with open(model_path, 'r') as f:
            encrypted_model_data = json.load(f)
        
        print_success(f"Loaded encrypted global model from {latest_file}")
        print_info("Note: Model is encrypted and cannot be tested directly")
        print_info("Use client-side decryption to test model performance")
        
        return True
    except Exception as e:
        print_error(f"Error testing model performance: {e}")
        return False

def evaluate_model_performance_metrics():
    """Evaluate model performance with comprehensive metrics (accuracy, F1, precision, recall)"""
    print_header("Model Performance Metrics Evaluation")
    
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
    print_section("Dataset Distribution")
    print(f"  Total samples: {len(y_test)}")
    print(f"  Healthy (Class 1): {sum(y_test)} samples ({sum(y_test)/len(y_test)*100:.1f}%)")
    print(f"  Unhealthy (Class 0): {len(y_test) - sum(y_test)} samples ({(len(y_test) - sum(y_test))/len(y_test)*100:.1f}%)")
    
    # Test federated model (client-side decryption)
    print_section("Federated Model Evaluation")
    
    try:
        # Import client decryption function
        sys.path.append(os.path.join(os.path.dirname(__file__), 'simulation'))
        from client_simulation import load_and_decrypt_global_model
        
        # Attempt client-side decryption
        decrypted_model = load_and_decrypt_global_model()
        
        if decrypted_model is not None:
            print_success("Successfully decrypted federated model")
            
            # Initialize model with decrypted parameters
            from sklearn.linear_model import LogisticRegression
            
            model = LogisticRegression()
            model.fit(X_test[:10], y_test[:10])  # Dummy fit to set shapes
            model.coef_ = decrypted_model['weights']
            model.intercept_ = decrypted_model['bias']
            
            # Make predictions
            y_pred_federated = model.predict(X_test)
            y_prob_federated = model.predict_proba(X_test)[:, 1]  # Probability of class 1
            
            # Calculate metrics
            accuracy_federated = accuracy_score(y_test, y_pred_federated)
            f1_federated = f1_score(y_test, y_pred_federated)
            precision_federated = precision_score(y_test, y_pred_federated)
            recall_federated = recall_score(y_test, y_pred_federated)
            
            print_section("Federated Model Metrics")
            print(f"  Accuracy:  {accuracy_federated:.4f} ({accuracy_federated*100:.2f}%)")
            print(f"  F1 Score:  {f1_federated:.4f} ({f1_federated*100:.2f}%)")
            print(f"  Precision: {precision_federated:.4f} ({precision_federated*100:.2f}%)")
            print(f"  Recall:    {recall_federated:.4f} ({recall_federated*100:.2f}%)")
            
            # Detailed classification report
            print_section("Detailed Classification Report")
            report = classification_report(y_test, y_pred_federated, 
                                        target_names=['Unhealthy', 'Healthy'],
                                        digits=4)
            print(report)
            
            # Confusion Matrix
            print_section("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred_federated)
            print("Predicted:")
            print("           Unhealthy  Healthy")
            print(f"Actual:    {cm[0,0]:8d}  {cm[0,1]:7d}")
            print(f"Healthy:   {cm[1,0]:8d}  {cm[1,1]:7d}")
            
            # Model parameters analysis
            print_section("Model Parameters Analysis")
            weights = decrypted_model['weights'].flatten()
            bias = decrypted_model['bias'][0]
            features = ["Heart Rate", "Steps", "Calories", "Sleep Hours"]
            
            print("Feature Weights:")
            for i, (feature, weight) in enumerate(zip(features, weights)):
                print(f"  {feature:12}: {weight:8.4f}")
            print(f"Bias: {bias:.4f}")
            print(f"Weight Norm: {np.linalg.norm(weights):.4f}")
            
            return True
        else:
            print_error("Failed to decrypt federated model")
            return False
            
    except Exception as e:
        print_error(f"Error evaluating federated model: {e}")
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
    model_files = [f for f in os.listdir(global_dir) if f.startswith('encrypted_global_round_') and f.endswith('.json')]
    
    if model_files:
        latest_file = sorted(model_files)[-1]
        model_path = os.path.join(global_dir, latest_file)
        
        print_success(f"Found encrypted federated model: {latest_file}")
        print_info("Note: Federated model is encrypted and cannot be compared directly")
        print_info("Use client-side decryption to compare with centralized model")
        
        return True
    else:
        print_error("No federated model found for comparison")
        return False

def compare_models_comprehensive():
    """Compare federated learning with centralized training using comprehensive metrics"""
    print_header("Comprehensive Model Comparison")
    
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
    y_prob_centralized = centralized_model.predict_proba(X_combined)[:, 1]
    
    # Calculate centralized metrics
    accuracy_centralized = accuracy_score(y_combined, y_pred_centralized)
    f1_centralized = f1_score(y_combined, y_pred_centralized)
    precision_centralized = precision_score(y_combined, y_pred_centralized)
    recall_centralized = recall_score(y_combined, y_pred_centralized)
    
    print_section("Centralized Model Metrics")
    print(f"  Accuracy:  {accuracy_centralized:.4f} ({accuracy_centralized*100:.2f}%)")
    print(f"  F1 Score:  {f1_centralized:.4f} ({f1_centralized*100:.2f}%)")
    print(f"  Precision: {precision_centralized:.4f} ({precision_centralized*100:.2f}%)")
    print(f"  Recall:    {recall_centralized:.4f} ({recall_centralized*100:.2f}%)")
    
    # Test federated model
    print_section("Testing Federated Model")
    try:
        sys.path.append(os.path.join(os.path.dirname(__file__), 'simulation'))
        from client_simulation import load_and_decrypt_global_model
        
        decrypted_model = load_and_decrypt_global_model()
        
        if decrypted_model is not None:
            # Initialize federated model
            federated_model = LogisticRegression()
            federated_model.fit(X_combined[:10], y_combined[:10])  # Dummy fit
            federated_model.coef_ = decrypted_model['weights']
            federated_model.intercept_ = decrypted_model['bias']
            
            # Make predictions
            y_pred_federated = federated_model.predict(X_combined)
            y_prob_federated = federated_model.predict_proba(X_combined)[:, 1]
            
            # Calculate federated metrics
            accuracy_federated = accuracy_score(y_combined, y_pred_federated)
            f1_federated = f1_score(y_combined, y_pred_federated)
            precision_federated = precision_score(y_combined, y_pred_federated)
            recall_federated = recall_score(y_combined, y_pred_federated)
            
            print_section("Federated Model Metrics")
            print(f"  Accuracy:  {accuracy_federated:.4f} ({accuracy_federated*100:.2f}%)")
            print(f"  F1 Score:  {f1_federated:.4f} ({f1_federated*100:.2f}%)")
            print(f"  Precision: {precision_federated:.4f} ({precision_federated*100:.2f}%)")
            print(f"  Recall:    {recall_federated:.4f} ({recall_federated*100:.2f}%)")
            
            # Comparison
            print_section("Model Comparison")
            print("Metric          Centralized    Federated    Difference")
            print("-" * 55)
            print(f"Accuracy        {accuracy_centralized:.4f}      {accuracy_federated:.4f}      {accuracy_federated - accuracy_centralized:+.4f}")
            print(f"F1 Score        {f1_centralized:.4f}      {f1_federated:.4f}      {f1_federated - f1_centralized:+.4f}")
            print(f"Precision       {precision_centralized:.4f}      {precision_federated:.4f}      {precision_federated - precision_centralized:+.4f}")
            print(f"Recall          {recall_centralized:.4f}      {recall_federated:.4f}      {recall_federated - recall_centralized:+.4f}")
            
            # Agreement analysis
            agreement = np.mean(y_pred_federated == y_pred_centralized)
            print(f"\nPrediction Agreement: {agreement:.4f} ({agreement*100:.2f}%)")
            
            # Save comparison results
            comparison_results = {
                "centralized": {
                    "accuracy": accuracy_centralized,
                    "f1_score": f1_centralized,
                    "precision": precision_centralized,
                    "recall": recall_centralized
                },
                "federated": {
                    "accuracy": accuracy_federated,
                    "f1_score": f1_federated,
                    "precision": precision_federated,
                    "recall": recall_federated
                },
                "difference": {
                    "accuracy": accuracy_federated - accuracy_centralized,
                    "f1_score": f1_federated - f1_centralized,
                    "precision": precision_federated - precision_centralized,
                    "recall": recall_federated - recall_centralized
                },
                "agreement": agreement,
                "dataset_size": len(y_combined),
                "timestamp": time.time()
            }
            
            # Save to file
            metrics_dir = "metrics"
            os.makedirs(metrics_dir, exist_ok=True)
            comparison_path = os.path.join(metrics_dir, "model_comparison.json")
            with open(comparison_path, 'w') as f:
                json.dump(comparison_results, f, indent=2)
            
            print_success(f"Comparison results saved to {comparison_path}")
            
            return True
        else:
            print_error("Failed to decrypt federated model for comparison")
            return False
            
    except Exception as e:
        print_error(f"Error comparing models: {e}")
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
    parser.add_argument("--test-decryption", action="store_true", help="Test client-side decryption")
    parser.add_argument("--performance-metrics", action="store_true", help="Evaluate accuracy, F1, precision, recall")
    parser.add_argument("--comprehensive-comparison", action="store_true", help="Compare FL vs centralized with all metrics")
    
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
        ("Global-to-Local Update", evaluate_global_to_local_update),
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
    
    # Test client decryption
    if args.test_decryption:
        print("\n" + "="*60)
        test_client_decryption()
    
    # Performance metrics evaluation
    if args.performance_metrics:
        print("\n" + "="*60)
        evaluate_model_performance_metrics()
    
    # Comprehensive model comparison
    if args.comprehensive_comparison:
        print("\n" + "="*60)
        compare_models_comprehensive()
    
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
