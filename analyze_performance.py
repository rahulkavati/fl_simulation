#!/usr/bin/env python3
"""
Federated Learning Performance Analysis
Comprehensive analysis of model performance metrics including accuracy, F1 score, precision, and recall

Usage: python analyze_performance.py [--federated] [--centralized] [--compare] [--detailed]
"""

import argparse
import os
import sys
import json
import numpy as np
import pandas as pd
import time
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure all dependencies are installed: pip install scikit-learn numpy pandas")
    sys.exit(1)

def print_header(title: str):
    """Print a formatted header"""
    print(f"\n{'='*70}")
    print(f"📊 {title}")
    print(f"{'='*70}")

def print_section(title: str):
    """Print a formatted section header"""
    print(f"\n🔍 {title}")
    print("-" * 50)

def print_success(message: str):
    """Print a success message"""
    print(f"✅ {message}")

def print_error(message: str):
    """Print an error message"""
    print(f"❌ {message}")

def print_info(message: str):
    """Print an info message"""
    print(f"ℹ️  {message}")

def print_metric(name: str, value: float, description: str = ""):
    """Print a metric with description"""
    print(f"  {name:12}: {value:.4f} ({value*100:.2f}%)")
    if description:
        print(f"              {description}")

def load_and_prepare_data():
    """Load and prepare data for analysis"""
    print_header("Data Preparation")
    
    data_dir = "data/clients"
    if not os.path.exists(data_dir):
        print_error("Data directory not found. Run the pipeline first.")
        return None, None
    
    # Load all client data
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
        print_error("No data found")
        return None, None
    
    X = np.vstack(all_data)
    y = np.concatenate(all_labels)
    
    print_success(f"Loaded dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print_section("Dataset Statistics")
    print(f"  Total samples: {len(y)}")
    print(f"  Healthy (Class 1): {sum(y)} samples ({sum(y)/len(y)*100:.1f}%)")
    print(f"  Unhealthy (Class 0): {len(y) - sum(y)} samples ({(len(y) - sum(y))/len(y)*100:.1f}%)")
    
    # Check for class imbalance
    imbalance_ratio = max(sum(y), len(y) - sum(y)) / min(sum(y), len(y) - sum(y))
    print(f"  Class imbalance ratio: {imbalance_ratio:.2f}")
    
    if imbalance_ratio > 2.0:
        print_info("⚠️  Significant class imbalance detected - F1 score will be more reliable than accuracy")
    
    return X, y

def explain_metrics():
    """Explain the importance of different metrics"""
    print_header("Understanding Model Performance Metrics")
    
    print_section("Accuracy")
    print("🎯 What it measures: Overall correctness of predictions")
    print("📊 Formula: (True Positives + True Negatives) / Total Predictions")
    print("💡 When to use: Good for balanced datasets")
    print("⚠️  Limitation: Can be misleading with imbalanced data")
    print("🏥 Health context: Overall ability to correctly classify health status")
    
    print_section("Precision")
    print("🎯 What it measures: Accuracy of positive predictions")
    print("📊 Formula: True Positives / (True Positives + False Positives)")
    print("💡 When to use: Important when false positives are costly")
    print("🏥 Health context: When we predict 'healthy', how often are we right?")
    print("   (Avoiding false alarms)")
    
    print_section("Recall (Sensitivity)")
    print("🎯 What it measures: Ability to find all positive cases")
    print("📊 Formula: True Positives / (True Positives + False Negatives)")
    print("💡 When to use: Important when missing positive cases is costly")
    print("🏥 Health context: How many actual healthy people did we identify?")
    print("   (Avoiding missed diagnoses)")
    
    print_section("F1 Score")
    print("🎯 What it measures: Harmonic mean of precision and recall")
    print("📊 Formula: 2 × (Precision × Recall) / (Precision + Recall)")
    print("💡 When to use: Best overall metric for imbalanced datasets")
    print("🏥 Health context: Balanced measure of prediction quality")
    print("   (Balances false alarms vs missed cases)")
    
    print_section("Why These Matter for Federated Learning")
    print("🔐 Privacy: FL preserves data privacy while learning")
    print("📈 Performance: Need to ensure FL doesn't sacrifice accuracy")
    print("⚖️  Balance: F1 score helps evaluate performance on imbalanced health data")
    print("🏥 Clinical: High precision avoids unnecessary interventions")
    print("🏥 Clinical: High recall ensures we don't miss health issues")

def evaluate_federated_model(X, y):
    """Evaluate the federated learning model"""
    print_header("Federated Learning Model Evaluation")
    
    try:
        # Import client decryption function
        sys.path.append(os.path.join(os.path.dirname(__file__), 'simulation'))
        from client_simulation import load_and_decrypt_global_model
        
        # Attempt client-side decryption
        decrypted_model = load_and_decrypt_global_model()
        
        if decrypted_model is None:
            print_error("Failed to decrypt federated model")
            return None
        
        print_success("Successfully decrypted federated model")
        print(f"  Round ID: {decrypted_model['round_id']}")
        print(f"  Model Type: {decrypted_model['model_type']}")
        
        # Initialize model with decrypted parameters
        model = LogisticRegression()
        model.fit(X[:10], y[:10])  # Dummy fit to set shapes
        model.coef_ = decrypted_model['weights']
        model.intercept_ = decrypted_model['bias']
        
        # Make predictions
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        
        print_section("Performance Metrics")
        print_metric("Accuracy", accuracy, "Overall correctness")
        print_metric("F1 Score", f1, "Balanced precision and recall")
        print_metric("Precision", precision, "Accuracy of positive predictions")
        print_metric("Recall", recall, "Ability to find all positive cases")
        
        # Confusion Matrix
        print_section("Confusion Matrix")
        cm = confusion_matrix(y, y_pred)
        print("Predicted:")
        print("           Unhealthy  Healthy")
        print(f"Actual:    {cm[0,0]:8d}  {cm[0,1]:7d}")
        print(f"Healthy:   {cm[1,0]:8d}  {cm[1,1]:7d}")
        
        # Detailed analysis
        print_section("Detailed Analysis")
        tn, fp, fn, tp = cm.ravel()
        print(f"  True Negatives (TN): {tn} - Correctly identified unhealthy")
        print(f"  False Positives (FP): {fp} - Incorrectly identified as healthy")
        print(f"  False Negatives (FN): {fn} - Missed healthy cases")
        print(f"  True Positives (TP): {tp} - Correctly identified healthy")
        
        # Model parameters
        print_section("Model Parameters")
        weights = decrypted_model['weights'].flatten()
        bias = decrypted_model['bias'][0]
        features = ["Heart Rate", "Steps", "Calories", "Sleep Hours"]
        
        print("Feature Weights:")
        for i, (feature, weight) in enumerate(zip(features, weights)):
            print(f"  {feature:12}: {weight:8.4f}")
        print(f"Bias: {bias:.4f}")
        print(f"Weight Norm: {np.linalg.norm(weights):.4f}")
        
        return {
            "accuracy": accuracy,
            "f1_score": f1,
            "precision": precision,
            "recall": recall,
            "confusion_matrix": cm.tolist(),
            "model_params": {
                "weights": weights.tolist(),
                "bias": bias,
                "weight_norm": float(np.linalg.norm(weights))
            }
        }
        
    except Exception as e:
        print_error(f"Error evaluating federated model: {e}")
        return None

def evaluate_centralized_model(X, y):
    """Evaluate centralized training model"""
    print_header("Centralized Training Model Evaluation")
    
    # Split data for proper evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    print_success(f"Training set: {len(y_train)} samples")
    print_success(f"Test set: {len(y_test)} samples")
    
    # Train centralized model
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    print_section("Performance Metrics")
    print_metric("Accuracy", accuracy, "Overall correctness")
    print_metric("F1 Score", f1, "Balanced precision and recall")
    print_metric("Precision", precision, "Accuracy of positive predictions")
    print_metric("Recall", recall, "Ability to find all positive cases")
    
    # Confusion Matrix
    print_section("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    print("Predicted:")
    print("           Unhealthy  Healthy")
    print(f"Actual:    {cm[0,0]:8d}  {cm[0,1]:7d}")
    print(f"Healthy:   {cm[1,0]:8d}  {cm[1,1]:7d}")
    
    # Model parameters
    print_section("Model Parameters")
    weights = model.coef_.flatten()
    bias = model.intercept_[0]
    features = ["Heart Rate", "Steps", "Calories", "Sleep Hours"]
    
    print("Feature Weights:")
    for i, (feature, weight) in enumerate(zip(features, weights)):
        print(f"  {feature:12}: {weight:8.4f}")
    print(f"Bias: {bias:.4f}")
    print(f"Weight Norm: {np.linalg.norm(weights):.4f}")
    
    return {
        "accuracy": accuracy,
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
        "confusion_matrix": cm.tolist(),
        "model_params": {
            "weights": weights.tolist(),
            "bias": bias,
            "weight_norm": float(np.linalg.norm(weights))
        }
    }

def compare_models(federated_results, centralized_results):
    """Compare federated and centralized models"""
    print_header("Model Comparison: Federated vs Centralized")
    
    if federated_results is None or centralized_results is None:
        print_error("Cannot compare models - missing results")
        return
    
    print_section("Performance Comparison")
    print("Metric          Federated    Centralized  Difference")
    print("-" * 55)
    
    metrics = ["accuracy", "f1_score", "precision", "recall"]
    metric_names = ["Accuracy", "F1 Score", "Precision", "Recall"]
    
    for metric, name in zip(metrics, metric_names):
        fed_val = federated_results[metric]
        cen_val = centralized_results[metric]
        diff = fed_val - cen_val
        
        print(f"{name:12}  {fed_val:.4f}      {cen_val:.4f}      {diff:+.4f}")
        
        # Interpretation
        if abs(diff) < 0.01:
            print(f"              {'='*20} Similar performance")
        elif diff > 0:
            print(f"              {'='*20} Federated performs better")
        else:
            print(f"              {'='*20} Centralized performs better")
    
    print_section("Interpretation")
    print("🔐 Privacy vs Performance Trade-off:")
    print("  • Federated Learning: Preserves data privacy")
    print("  • Centralized Training: Potentially better performance")
    print("  • Goal: Minimize performance gap while maintaining privacy")
    
    # Save comparison results
    comparison_results = {
        "federated": federated_results,
        "centralized": centralized_results,
        "comparison": {
            "accuracy_diff": federated_results["accuracy"] - centralized_results["accuracy"],
            "f1_diff": federated_results["f1_score"] - centralized_results["f1_score"],
            "precision_diff": federated_results["precision"] - centralized_results["precision"],
            "recall_diff": federated_results["recall"] - centralized_results["recall"]
        },
        "timestamp": time.time()
    }
    
    # Save to file
    metrics_dir = "metrics"
    os.makedirs(metrics_dir, exist_ok=True)
    comparison_path = os.path.join(metrics_dir, "performance_comparison.json")
    with open(comparison_path, 'w') as f:
        json.dump(comparison_results, f, indent=2)
    
    print_success(f"Comparison results saved to {comparison_path}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Federated Learning Performance Analysis")
    parser.add_argument("--explain", action="store_true", help="Explain metrics and their importance")
    parser.add_argument("--federated", action="store_true", help="Evaluate federated learning model")
    parser.add_argument("--centralized", action="store_true", help="Evaluate centralized training model")
    parser.add_argument("--compare", action="store_true", help="Compare federated vs centralized")
    parser.add_argument("--detailed", action="store_true", help="Show detailed analysis")
    
    args = parser.parse_args()
    
    print("📊 Federated Learning Performance Analysis")
    print("=" * 70)
    
    # Explain metrics if requested
    if args.explain:
        explain_metrics()
        return
    
    # Load data
    X, y = load_and_prepare_data()
    if X is None:
        return
    
    federated_results = None
    centralized_results = None
    
    # Evaluate federated model
    if args.federated or args.compare:
        federated_results = evaluate_federated_model(X, y)
    
    # Evaluate centralized model
    if args.centralized or args.compare:
        centralized_results = evaluate_centralized_model(X, y)
    
    # Compare models
    if args.compare:
        compare_models(federated_results, centralized_results)
    
    # Summary
    print_header("Analysis Summary")
    print("✅ Performance analysis completed!")
    print("📊 Key metrics evaluated: Accuracy, F1 Score, Precision, Recall")
    print("🔐 Privacy preserved in federated learning")
    print("🏥 Health classification performance assessed")
    
    if not any([args.federated, args.centralized, args.compare, args.explain]):
        print_info("Use --help to see available options")
        print_info("Recommended: python analyze_performance.py --explain --compare")

if __name__ == "__main__":
    main()
