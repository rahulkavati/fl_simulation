"""
Plaintext Aggregation Module for Federated Learning

This module provides plaintext aggregation capabilities for federated learning,
ensuring that model updates are processed in plaintext without any encryption.

Key Features:
- Plaintext model representation
- Standard federated averaging
- No encryption overhead
- Fast aggregation operations
- Compatible with standard ML libraries

Architecture:
1. PlaintextModel: Plaintext model representation
2. PlaintextAggregator: Plaintext aggregation operations
3. PlaintextConfig: Configuration for plaintext operations

Author: AI Assistant
Date: 2025
"""

import time
import numpy as np
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass


@dataclass
class PlaintextConfig:
    """
    Configuration for plaintext federated learning operations
    
    This configuration class defines parameters for plaintext aggregation
    and model operations without any encryption requirements.
    """
    # Aggregation parameters
    aggregation_method: str = "federated_averaging"  # Method for aggregating updates
    weight_by_samples: bool = True                   # Whether to weight by sample count
    
    # Model parameters
    model_type: str = "logistic_regression"         # Type of model being used
    feature_count: int = 47                         # Number of features in the model
    
    # Performance tracking
    track_timing: bool = True                       # Whether to track operation timing
    verbose: bool = False                           # Whether to print detailed logs


class PlaintextModel:
    """
    Plaintext model representation for federated learning
    
    This class represents a model in plaintext format, storing weights and bias
    as standard NumPy arrays without any encryption.
    """
    
    def __init__(self, weights: np.ndarray, bias: float):
        """
        Initialize a plaintext model
        
        Args:
            weights: Model weights as NumPy array
            bias: Model bias as float
        """
        self.weights = weights
        self.bias = bias
        self.is_encrypted = False  # Always False for plaintext
        
        # Store original values for reference
        self._original_weights = weights.copy()
        self._original_bias = bias
    
    def get_weights(self) -> np.ndarray:
        """Get model weights"""
        return self.weights
    
    def get_bias(self) -> float:
        """Get model bias"""
        return self.bias
    
    def update(self, new_weights: np.ndarray, new_bias: float):
        """
        Update model parameters
        
        Args:
            new_weights: New model weights
            new_bias: New model bias
        """
        self.weights = new_weights
        self.bias = new_bias
    
    def get_model_update(self) -> np.ndarray:
        """
        Get model update as a single vector
        
        Returns:
            Combined weights and bias as a single vector
        """
        return np.concatenate([self.weights, [self.bias]])
    
    def from_model_update(self, model_update: np.ndarray):
        """
        Set model parameters from a model update vector
        
        Args:
            model_update: Combined weights and bias vector
        """
        self.weights = model_update[:-1]  # All except last element
        self.bias = float(model_update[-1])  # Last element
    
    def copy(self) -> 'PlaintextModel':
        """Create a copy of this model"""
        return PlaintextModel(self.weights.copy(), self.bias)
    
    def __str__(self) -> str:
        """String representation of the model"""
        return f"PlaintextModel(weights_shape={self.weights.shape}, bias={self.bias:.4f})"


class PlaintextAggregator:
    """
    Plaintext aggregation operations for federated learning
    
    This class provides methods for aggregating model updates in plaintext,
    using standard federated averaging techniques.
    """
    
    def __init__(self, config: PlaintextConfig = None):
        """
        Initialize the plaintext aggregator
        
        Args:
            config: Configuration for plaintext operations
        """
        self.config = config or PlaintextConfig()
        self.aggregation_history = []
    
    def aggregate_updates(self, updates: List[np.ndarray], 
                         sample_counts: List[int]) -> Tuple[np.ndarray, float]:
        """
        Aggregate multiple model updates using federated averaging
        
        Args:
            updates: List of model update vectors
            sample_counts: List of sample counts for each update
            
        Returns:
            Tuple of (aggregated_update, aggregation_time)
        """
        start_time = time.time()
        
        if not updates:
            raise ValueError("No updates provided for aggregation")
        
        if len(updates) != len(sample_counts):
            raise ValueError("Number of updates must match number of sample counts")
        
        # Convert to numpy arrays for efficient computation
        updates_array = np.array(updates)
        sample_counts_array = np.array(sample_counts)
        
        if self.config.weight_by_samples:
            # Weighted federated averaging
            total_samples = np.sum(sample_counts_array)
            weights = sample_counts_array / total_samples
            
            # Perform weighted average
            aggregated_update = np.average(updates_array, axis=0, weights=weights)
        else:
            # Simple average (equal weights)
            aggregated_update = np.mean(updates_array, axis=0)
        
        aggregation_time = time.time() - start_time
        
        # Store aggregation history
        self.aggregation_history.append({
            'timestamp': time.time(),
            'num_updates': len(updates),
            'total_samples': np.sum(sample_counts_array),
            'aggregation_time': aggregation_time,
            'method': self.config.aggregation_method
        })
        
        if self.config.verbose:
            print(f"Plaintext aggregation completed in {aggregation_time:.4f}s")
            print(f"  Updates: {len(updates)}, Total samples: {np.sum(sample_counts_array)}")
        
        return aggregated_update, aggregation_time
    
    def aggregate_models(self, models: List[PlaintextModel], 
                        sample_counts: List[int]) -> PlaintextModel:
        """
        Aggregate multiple plaintext models
        
        Args:
            models: List of plaintext models
            sample_counts: List of sample counts for each model
            
        Returns:
            Aggregated plaintext model
        """
        # Extract model updates
        updates = [model.get_model_update() for model in models]
        
        # Aggregate updates
        aggregated_update, _ = self.aggregate_updates(updates, sample_counts)
        
        # Create new model from aggregated update
        aggregated_model = PlaintextModel(
            weights=aggregated_update[:-1],
            bias=float(aggregated_update[-1])
        )
        
        return aggregated_model
    
    def get_aggregation_stats(self) -> Dict[str, Any]:
        """
        Get statistics about aggregation operations
        
        Returns:
            Dictionary containing aggregation statistics
        """
        if not self.aggregation_history:
            return {
                'total_aggregations': 0,
                'avg_aggregation_time': 0.0,
                'total_updates_processed': 0,
                'total_samples_processed': 0
            }
        
        total_aggregations = len(self.aggregation_history)
        avg_aggregation_time = np.mean([h['aggregation_time'] for h in self.aggregation_history])
        total_updates_processed = sum(h['num_updates'] for h in self.aggregation_history)
        total_samples_processed = sum(h['total_samples'] for h in self.aggregation_history)
        
        return {
            'total_aggregations': total_aggregations,
            'avg_aggregation_time': avg_aggregation_time,
            'total_updates_processed': total_updates_processed,
            'total_samples_processed': total_samples_processed,
            'aggregation_method': self.config.aggregation_method
        }
    
    def reset_history(self):
        """Reset aggregation history"""
        self.aggregation_history = []
    
    def __str__(self) -> str:
        """String representation of the aggregator"""
        stats = self.get_aggregation_stats()
        return f"PlaintextAggregator(method={self.config.aggregation_method}, " \
               f"aggregations={stats['total_aggregations']})"


class PlaintextManager:
    """
    High-level interface for plaintext federated learning operations
    
    This class provides a simplified interface for plaintext operations,
    similar to the EncryptionManager but without any encryption.
    """
    
    def __init__(self, config: PlaintextConfig = None):
        """
        Initialize the plaintext manager
        
        Args:
            config: Configuration for plaintext operations
        """
        self.config = config or PlaintextConfig()
        self.aggregator = PlaintextAggregator(config)
        self.operation_history = []
    
    def create_model(self, weights: np.ndarray, bias: float) -> PlaintextModel:
        """
        Create a new plaintext model
        
        Args:
            weights: Model weights
            bias: Model bias
            
        Returns:
            New plaintext model
        """
        return PlaintextModel(weights, bias)
    
    def aggregate_updates(self, updates: List[np.ndarray], 
                         sample_counts: List[int]) -> Tuple[np.ndarray, float]:
        """
        Aggregate model updates
        
        Args:
            updates: List of model update vectors
            sample_counts: List of sample counts
            
        Returns:
            Tuple of (aggregated_update, aggregation_time)
        """
        return self.aggregator.aggregate_updates(updates, sample_counts)
    
    def update_global_model(self, global_model: PlaintextModel, 
                           aggregated_update: np.ndarray):
        """
        Update global model with aggregated update
        
        Args:
            global_model: Global model to update
            aggregated_update: Aggregated update vector
        """
        global_model.from_model_update(aggregated_update)
        
        # Log operation
        self.operation_history.append({
            'timestamp': time.time(),
            'operation': 'update_global_model',
            'update_shape': aggregated_update.shape
        })
    
    def get_operation_stats(self) -> Dict[str, Any]:
        """
        Get statistics about operations performed
        
        Returns:
            Dictionary containing operation statistics
        """
        aggregation_stats = self.aggregator.get_aggregation_stats()
        
        return {
            'aggregation_stats': aggregation_stats,
            'total_operations': len(self.operation_history),
            'config': {
                'aggregation_method': self.config.aggregation_method,
                'weight_by_samples': self.config.weight_by_samples,
                'model_type': self.config.model_type
            }
        }
    
    def reset_history(self):
        """Reset operation history"""
        self.operation_history = []
        self.aggregator.reset_history()
    
    def __str__(self) -> str:
        """String representation of the manager"""
        return f"PlaintextManager(config={self.config.aggregation_method})"
