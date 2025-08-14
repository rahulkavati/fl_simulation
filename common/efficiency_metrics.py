"""
Federated Learning Efficiency Metrics

This module provides comprehensive metrics for evaluating FL performance:
- Communication efficiency
- Training efficiency  
- Model convergence
- Resource utilization
- Performance metrics
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression

@dataclass
class FLEfficiencyMetrics:
    """Container for all FL efficiency metrics"""
    # Simulation metadata
    timestamp: str
    num_clients: int
    num_rounds: int
    total_samples: int
    
    # Communication efficiency
    total_communication_rounds: int
    bytes_transferred: float  # Estimated
    communication_overhead: float  # Percentage
    
    # Training efficiency
    total_training_time: float  # seconds
    avg_training_time_per_round: float
    convergence_rounds: Optional[int]  # When model stabilizes
    
    # Model performance
    initial_accuracy: float
    final_accuracy: float
    accuracy_improvement: float
    final_weight_norm: float
    final_bias: float
    
    # Resource utilization
    memory_usage: float  # MB
    cpu_utilization: float  # Percentage
    
    # Convergence metrics
    weight_change_magnitude: List[float]  # Per round
    loss_reduction: List[float]  # Per round
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)
    
    def save_to_file(self, filepath: str):
        """Save metrics to JSON file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            f.write(self.to_json())

class FLEfficiencyCalculator:
    """Calculate various FL efficiency metrics"""
    
    def __init__(self, data_dir: str, updates_dir: str):
        self.data_dir = data_dir
        self.updates_dir = updates_dir
        self.metrics_history = []
    
    def calculate_communication_efficiency(self, num_clients: int, num_rounds: int) -> Tuple[int, float, float]:
        """Calculate communication-related metrics"""
        total_rounds = num_clients * num_rounds
        # Estimate bytes transferred (weights + bias + metadata)
        bytes_per_update = 1024  # Rough estimate
        total_bytes = total_rounds * bytes_per_update
        overhead = (total_bytes / (1024 * 1024)) * 100  # Convert to MB and percentage
        
        return total_rounds, total_bytes, overhead
    
    def calculate_model_performance(self, global_model: LogisticRegression, 
                                  clients_data: Dict) -> Tuple[float, float, float]:
        """Calculate model accuracy and performance metrics"""
        # Calculate accuracy on all client data
        all_predictions = []
        all_true_labels = []
        
        for X, y in clients_data.values():
            pred = global_model.predict(X)
            all_predictions.extend(pred)
            all_true_labels.extend(y)
        
        final_accuracy = accuracy_score(all_true_labels, all_predictions)
        
        # For initial accuracy, we'd need to track it during training
        # For now, estimate based on random initialization
        initial_accuracy = 0.5  # Random baseline for binary classification
        
        improvement = final_accuracy - initial_accuracy
        
        return initial_accuracy, final_accuracy, improvement
    
    def calculate_convergence_metrics(self, updates_dir: str, num_rounds: int) -> Tuple[List[float], List[float]]:
        """Calculate convergence-related metrics across rounds"""
        weight_changes = []
        loss_reductions = []
        
        for round_id in range(num_rounds):
            # Load all client updates for this round
            round_updates = []
            for fname in os.listdir(updates_dir):
                if fname.endswith(f"_round_{round_id}.npy"):
                    arr = np.load(os.path.join(updates_dir, fname))
                    round_updates.append(arr)
            
            if round_updates:
                # Calculate average weight change magnitude for this round
                avg_change = np.mean([np.linalg.norm(update[:-1]) for update in round_updates])
                weight_changes.append(avg_change)
                
                # Estimate loss reduction (simplified)
                loss_reductions.append(max(0, 1.0 - avg_change))  # Simplified metric
        
        return weight_changes, loss_reductions
    
    def calculate_efficiency_metrics(self, clients_data: Dict, global_model: LogisticRegression,
                                  num_rounds: int, training_time: float = None) -> FLEfficiencyMetrics:
        """Calculate comprehensive FL efficiency metrics"""
        
        # Basic metadata
        timestamp = datetime.now().isoformat()
        num_clients = len(clients_data)
        total_samples = sum(len(X) for X, _ in clients_data.values())
        
        # Communication efficiency
        comm_rounds, bytes_transferred, comm_overhead = self.calculate_communication_efficiency(
            num_clients, num_rounds
        )
        
        # Training efficiency
        total_training_time = training_time or 0.0
        avg_training_time = total_training_time / num_rounds if num_rounds > 0 else 0.0
        
        # Model performance
        init_acc, final_acc, improvement = self.calculate_model_performance(global_model, clients_data)
        
        # Convergence metrics
        weight_changes, loss_reductions = self.calculate_convergence_metrics(
            os.path.join(self.updates_dir, "numpy"), num_rounds
        )
        
        # Resource utilization (estimated)
        memory_usage = len(global_model.coef_.flatten()) * 8 / (1024 * 1024)  # MB
        cpu_utilization = 80.0  # Estimated percentage
        
        # Determine convergence round
        convergence_round = None
        if len(weight_changes) > 2:
            # Simple heuristic: when weight changes become small
            threshold = 0.01
            for i, change in enumerate(weight_changes):
                if change < threshold:
                    convergence_round = i
                    break
        
        metrics = FLEfficiencyMetrics(
            timestamp=timestamp,
            num_clients=num_clients,
            num_rounds=num_rounds,
            total_samples=total_samples,
            total_communication_rounds=comm_rounds,
            bytes_transferred=bytes_transferred,
            communication_overhead=comm_overhead,
            total_training_time=total_training_time,
            avg_training_time_per_round=avg_training_time,
            convergence_rounds=convergence_round,
            initial_accuracy=init_acc,
            final_accuracy=final_acc,
            accuracy_improvement=improvement,
            final_weight_norm=float(np.linalg.norm(global_model.coef_)),
            final_bias=float(global_model.intercept_[0]),
            memory_usage=memory_usage,
            cpu_utilization=cpu_utilization,
            weight_change_magnitude=weight_changes,
            loss_reduction=loss_reductions
        )
        
        return metrics
    
    def save_metrics(self, metrics: FLEfficiencyMetrics, experiment_name: str = None):
        """Save metrics to centralized storage"""
        if experiment_name is None:
            experiment_name = f"fl_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create metrics directory if it doesn't exist
        metrics_dir = "metrics"
        os.makedirs(metrics_dir, exist_ok=True)
        
        # Save individual experiment metrics
        experiment_file = os.path.join(metrics_dir, f"{experiment_name}.json")
        metrics.save_to_file(experiment_file)
        
        # Save to metrics history
        self.metrics_history.append(metrics)
        
        # Update aggregated metrics file
        self.save_aggregated_metrics(metrics_dir)
        
        print(f"Metrics saved to {experiment_file}")
    
    def save_aggregated_metrics(self, metrics_dir: str):
        """Save aggregated metrics across all experiments"""
        if not self.metrics_history:
            return
        
        # Create summary
        summary = {
            "total_experiments": len(self.metrics_history),
            "last_updated": datetime.now().isoformat(),
            "experiments": [m.to_dict() for m in self.metrics_history]
        }
        
        # Save summary
        summary_file = os.path.join(metrics_dir, "metrics_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Create CSV for easy analysis
        df = pd.DataFrame([m.to_dict() for m in self.metrics_history])
        csv_file = os.path.join(metrics_dir, "metrics_history.csv")
        df.to_csv(csv_file, index=False)
        
        print(f"Aggregated metrics saved to {summary_file} and {csv_file}")

def load_metrics_from_file(filepath: str) -> FLEfficiencyMetrics:
    """Load metrics from JSON file"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Convert lists back to proper types
    if 'weight_change_magnitude' in data:
        data['weight_change_magnitude'] = [float(x) for x in data['weight_change_magnitude']]
    if 'loss_reduction' in data:
        data['loss_reduction'] = [float(x) for x in data['loss_reduction']]
    
    return FLEfficiencyMetrics(**data)
