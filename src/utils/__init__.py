"""
Utilities Module for Federated Learning

This module provides common utilities and helper functions for the federated learning pipeline:
- Enhanced metrics calculation (PR-AUC, calibration metrics)
- Subject-disjoint data splits
- Structured summary generation and saving
- Final summary printing

Key Functions:
1. calculate_enhanced_metrics: Calculate comprehensive evaluation metrics
2. create_subject_disjoint_splits: Create train/test splits by participant
3. generate_structured_summary: Generate structured result summaries
4. save_results_to_folder: Save results to organized folder structure
5. print_structured_summary: Print formatted summaries

Author: AI Assistant
Date: 2025
"""

import os
import json
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, mean_absolute_error, mean_squared_error,
    average_precision_score
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import matplotlib.pyplot as plt

def calculate_enhanced_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                              y_pred_proba: np.ndarray) -> Dict[str, float]:
    """
    Calculate comprehensive evaluation metrics including calibration metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities for positive class
        
    Returns:
        Dictionary containing all metrics
    """
    # Basic classification metrics
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred_proba)
    
    # Regression metrics (treating as regression problem)
    mae = mean_absolute_error(y_true, y_pred_proba)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred_proba))
    
    # PR-AUC (better for imbalanced datasets)
    pr_auc = average_precision_score(y_true, y_pred_proba)
    
    # Calibration metrics
    try:
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_pred_proba, n_bins=10
        )
        
        # Calculate ECE and MCE
        bin_boundaries = np.linspace(0, 1, 11)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        mce = 0
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_pred_proba > bin_lower) & (y_pred_proba <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_pred_proba[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                mce = max(mce, np.abs(avg_confidence_in_bin - accuracy_in_bin))
        
    except Exception as e:
        print(f"Warning: Calibration calculation failed: {e}")
        ece = float('nan')
        mce = float('nan')
    
    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'auc': auc,
        'pr_auc': pr_auc,
        'ece': ece,
        'mce': mce,
        'mae': mae,
        'rmse': rmse
    }

def create_subject_disjoint_splits(df: pd.DataFrame, test_ratio: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create subject-disjoint train/test splits to prevent data leakage
    
    Args:
        df: DataFrame with participant_id column
        test_ratio: Ratio of participants to use for testing
        
    Returns:
        Tuple of (train_df, test_df)
    """
    unique_participants = df['participant_id'].unique()
    n_test_participants = int(len(unique_participants) * test_ratio)
    
    np.random.seed(42)
    test_participants = np.random.choice(
        unique_participants, 
        size=n_test_participants, 
        replace=False
    )
    
    test_df = df[df['participant_id'].isin(test_participants)]
    train_df = df[~df['participant_id'].isin(test_participants)]
    
    print(f"Subject-disjoint splits:")
    print(f"  Train participants: {len(unique_participants) - n_test_participants}")
    print(f"  Test participants: {n_test_participants}")
    print(f"  Train samples: {len(train_df)}")
    print(f"  Test samples: {len(test_df)}")
    
    return train_df, test_df

def generate_structured_summary(round_results: List[Dict[str, Any]], 
                               clients_data: Dict[str, Any],
                               pipeline_type: str,
                               rounds: int,
                               clients: int) -> Dict[str, Any]:
    """
    Generate a structured summary of the federated learning results
    
    Args:
        round_results: List of round results
        clients_data: Dictionary of client data
        pipeline_type: Type of pipeline ('fhe' or 'plaintext')
        rounds: Number of rounds
        clients: Number of clients
        
    Returns:
        Structured summary dictionary
    """
    if not round_results:
        return {}
    
    final_result = round_results[-1]
    
    # Calculate statistics
    total_samples = sum(len(y) for _, y in clients_data.values())
    one_class_clients = sum(1 for _, y in clients_data.values() if len(np.unique(y)) < 2)
    
    # Performance trends
    accuracies = [r['accuracy'] for r in round_results]
    f1_scores = [r['f1_score'] for r in round_results]
    
    # Timing statistics
    total_encryption_time = sum(r.get('encryption_time', 0) for r in round_results)
    total_aggregation_time = sum(r.get('aggregation_time', 0) for r in round_results)
    total_evaluation_time = sum(r.get('evaluation_time', 0) for r in round_results)
    total_time = sum(r.get('total_time', 0) for r in round_results)
    
    summary = {
        'pipeline_info': {
            'type': pipeline_type,
            'rounds': rounds,
            'clients': clients,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_samples': total_samples,
            'one_class_clients': one_class_clients,
            'one_class_percentage': (one_class_clients / clients) * 100 if clients > 0 else 0
        },
        'final_performance': {
            'accuracy': final_result['accuracy'],
            'f1_score': final_result['f1_score'],
            'precision': final_result['precision'],
            'recall': final_result['recall'],
            'auc': final_result['auc'],
            'pr_auc': final_result.get('pr_auc', 0),
            'mae': final_result['mae'],
            'rmse': final_result['rmse'],
            'ece': final_result.get('ece', float('nan')),
            'mce': final_result.get('mce', float('nan'))
        },
        'performance_trends': {
            'initial_accuracy': accuracies[0] if accuracies else 0,
            'final_accuracy': accuracies[-1] if accuracies else 0,
            'accuracy_improvement': accuracies[-1] - accuracies[0] if len(accuracies) > 1 else 0,
            'best_accuracy': max(accuracies) if accuracies else 0,
            'final_f1': f1_scores[-1] if f1_scores else 0,
            'best_f1': max(f1_scores) if f1_scores else 0
        },
        'timing_statistics': {
            'total_encryption_time': total_encryption_time,
            'total_aggregation_time': total_aggregation_time,
            'total_evaluation_time': total_evaluation_time,
            'total_time': total_time,
            'avg_round_time': total_time / rounds if rounds > 0 else 0,
            'avg_encryption_time': total_encryption_time / rounds if rounds > 0 else 0,
            'avg_aggregation_time': total_aggregation_time / rounds if rounds > 0 else 0
        },
        'round_details': round_results
    }
    
    return summary

def save_results_to_folder(summary: Dict[str, Any], 
                          pipeline_type: str,
                          rounds: int,
                          clients: int) -> str:
    """
    Save results to organized folder structure
    
    Args:
        summary: Structured summary dictionary
        pipeline_type: Type of pipeline ('fhe' or 'plaintext')
        rounds: Number of rounds
        clients: Number of clients
        
    Returns:
        Path to saved file
    """
    # Create folder structure: results/{pipeline_type}/
    base_dir = Path("results")
    pipeline_dir = base_dir / pipeline_type
    
    # Create directories if they don't exist
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{pipeline_type}_results_{clients}clients_{rounds}rounds_{timestamp}.json"
    filepath = pipeline_dir / filename
    
    # Save summary as JSON
    with open(filepath, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    return str(filepath)

def print_structured_summary(summary: Dict[str, Any]):
    """
    Print a structured summary of the federated learning results
    
    Args:
        summary: Structured summary dictionary
    """
    if not summary:
        print("No summary to display")
        return
    
    print("\n" + "="*70)
    print(f"{summary['pipeline_info']['type'].upper()} FEDERATED LEARNING RESULTS")
    print("="*70)
    
    # Pipeline info
    print(f"\nPIPELINE INFORMATION:")
    print(f"  Type: {summary['pipeline_info']['type']}")
    print(f"  Rounds: {summary['pipeline_info']['rounds']}")
    print(f"  Clients: {summary['pipeline_info']['clients']}")
    print(f"  Total Samples: {summary['pipeline_info']['total_samples']}")
    print(f"  One-Class Clients: {summary['pipeline_info']['one_class_clients']} ({summary['pipeline_info']['one_class_percentage']:.1f}%)")
    print(f"  Timestamp: {summary['pipeline_info']['timestamp']}")
    
    # Final performance
    print(f"\nFINAL PERFORMANCE:")
    print(f"  Accuracy: {summary['final_performance']['accuracy']:.4f} ({summary['final_performance']['accuracy']*100:.2f}%)")
    print(f"  F1 Score: {summary['final_performance']['f1_score']:.4f}")
    print(f"  Precision: {summary['final_performance']['precision']:.4f}")
    print(f"  Recall: {summary['final_performance']['recall']:.4f}")
    print(f"  AUC: {summary['final_performance']['auc']:.4f}")
    print(f"  PR-AUC: {summary['final_performance']['pr_auc']:.4f}")
    print(f"  MAE: {summary['final_performance']['mae']:.4f}")
    print(f"  RMSE: {summary['final_performance']['rmse']:.4f}")
    print(f"  ECE: {summary['final_performance']['ece']:.4f}")
    print(f"  MCE: {summary['final_performance']['mce']:.4f}")
    
    # Performance trends
    print(f"\nPERFORMANCE TRENDS:")
    print(f"  Initial Accuracy: {summary['performance_trends']['initial_accuracy']:.4f}")
    print(f"  Final Accuracy: {summary['performance_trends']['final_accuracy']:.4f}")
    print(f"  Accuracy Improvement: {summary['performance_trends']['accuracy_improvement']:.4f}")
    print(f"  Best Accuracy: {summary['performance_trends']['best_accuracy']:.4f}")
    print(f"  Final F1: {summary['performance_trends']['final_f1']:.4f}")
    print(f"  Best F1: {summary['performance_trends']['best_f1']:.4f}")
    
    # Timing statistics
    print(f"\nTIMING STATISTICS:")
    print(f"  Total Encryption Time: {summary['timing_statistics']['total_encryption_time']:.4f}s")
    print(f"  Total Aggregation Time: {summary['timing_statistics']['total_aggregation_time']:.4f}s")
    print(f"  Total Evaluation Time: {summary['timing_statistics']['total_evaluation_time']:.4f}s")
    print(f"  Total Time: {summary['timing_statistics']['total_time']:.4f}s")
    print(f"  Average Round Time: {summary['timing_statistics']['avg_round_time']:.4f}s")
    print(f"  Average Encryption Time: {summary['timing_statistics']['avg_encryption_time']:.4f}s")
    print(f"  Average Aggregation Time: {summary['timing_statistics']['avg_aggregation_time']:.4f}s")
    
    print("\n" + "="*70)

# Legacy function for backward compatibility
def print_final_summary(round_results: List[Dict[str, Any]], clients_data: Dict[str, Any]):
    """
    Legacy function for backward compatibility
    """
    if not round_results:
        print("No results to summarize")
        return
    
    # Generate a simple summary for backward compatibility
    summary = generate_structured_summary(
        round_results, clients_data, 'legacy', len(round_results), len(clients_data)
    )
    print_structured_summary(summary)