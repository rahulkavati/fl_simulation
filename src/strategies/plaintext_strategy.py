"""
Plain Text Federated Learning Strategy
Implements standard federated learning without encryption for baseline comparison
"""

import time
import numpy as np
from typing import Dict, List, Tuple, Any
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.core.base_pipeline import BaseFederatedLearningPipeline, PipelineConfig, RoundResult

class PlainTextFederatedLearningPipeline(BaseFederatedLearningPipeline):
    """
    Plain Text Federated Learning Implementation
    Provides baseline performance for comparison with FHE approaches
    """
    
    def __init__(self, config: PipelineConfig):
        super().__init__(config)
        self.config.encryption_enabled = False
        self.model_params = {
            'penalty': 'l2',
            'C': 1.0,
            'fit_intercept': True,
            'solver': 'lbfgs',
            'max_iter': 1000,
            'class_weight': 'balanced',
            'random_state': config.random_state
        }
    
    def get_pipeline_type(self) -> str:
        return "PLAINTEXT"
    
    def initialize_global_model(self) -> LogisticRegression:
        """Initialize plain text global model"""
        print("🔓 Initializing Plain Text Global Model...")
        
        # Create initial model with random weights
        model = LogisticRegression(**self.model_params)
        
        # Initialize with dummy data to set up the model structure
        first_client_data = next(iter(self.clients_data.values()))
        X_client, y_client = first_client_data
        dummy_X = np.random.normal(0, 1, (10, X_client.shape[1]))
        dummy_y = np.random.randint(0, 2, 10)
        model.fit(dummy_X, dummy_y)
        
        print(f"✅ Plain text global model initialized")
        return model
    
    def train_local_model(self, client_id: str, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Train local model for a client"""
        # Create local model
        local_model = LogisticRegression(**self.model_params)
        local_model.fit(X, y)
        
        # Extract model parameters
        update = {
            'weights': local_model.coef_.flatten(),
            'bias': local_model.intercept_[0],
            'sample_count': len(X),
            'client_id': client_id
        }
        
        return update
    
    def aggregate_updates(self, local_updates: List[Dict[str, Any]], sample_counts: List[int]) -> Dict[str, Any]:
        """Aggregate local updates using federated averaging"""
        print("🔄 Aggregating Plain Text Updates...")
        
        if not local_updates:
            return {'weights': np.array([]), 'bias': 0.0}
        
        # Calculate weighted average based on sample counts
        total_samples = sum(sample_counts)
        weights = np.array(sample_counts) / total_samples
        
        # Separate weights and biases
        all_weights = []
        all_biases = []
        
        for update in local_updates:
            all_weights.append(update['weights'])
            all_biases.append(update['bias'])
        
        # Federated averaging
        aggregated_weights = np.average(all_weights, axis=0, weights=weights)
        aggregated_bias = np.average(all_biases, weights=weights)
        
        print(f"✅ Aggregated {len(local_updates)} plain text updates")
        
        return {
            'weights': aggregated_weights,
            'bias': aggregated_bias,
            'total_samples': total_samples
        }
    
    def update_global_model(self, aggregated_update: Dict[str, Any]) -> None:
        """Update global model with aggregated update"""
        if self.global_model is None:
            return
        
        # Update model parameters directly
        self.global_model.coef_ = aggregated_update['weights'].reshape(1, -1)
        self.global_model.intercept_ = np.array([aggregated_update['bias']])
        
        print("✅ Global model updated with plain text parameters")
    
    def evaluate_model(self, model: LogisticRegression, test_data: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
        """Evaluate model performance"""
        X_test, y_test = test_data
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred)
        }
        
        return metrics
    
    def _extract_model_state(self) -> Tuple[np.ndarray, float]:
        """Extract final model weights and bias"""
        if self.global_model is None:
            return np.array([]), 0.0
        
        return self.global_model.coef_.flatten(), self.global_model.intercept_[0]

class EnhancedPlainTextPipeline(PlainTextFederatedLearningPipeline):
    """
    Enhanced Plain Text Pipeline with ensemble methods
    Provides more sophisticated baseline for fair comparison
    """
    
    def __init__(self, config: PipelineConfig):
        super().__init__(config)
        self.use_ensemble = True
    
    def get_pipeline_type(self) -> str:
        return "PLAINTEXT_ENHANCED"
    
    def initialize_global_model(self) -> VotingClassifier:
        """Initialize enhanced ensemble model"""
        print("🔓 Initializing Enhanced Plain Text Global Model...")
        
        # Create ensemble model
        lr = LogisticRegression(**self.model_params)
        rf = RandomForestClassifier(
            n_estimators=50,
            max_depth=10,
            random_state=self.config.random_state,
            class_weight='balanced'
        )
        
        ensemble = VotingClassifier(
            estimators=[('lr', lr), ('rf', rf)],
            voting='soft'
        )
        
        # Initialize with dummy data
        dummy_X = np.random.normal(0, 1, (10, len(next(iter(self.clients_data.values()))[0].shape[1])))
        dummy_y = np.random.randint(0, 2, 10)
        ensemble.fit(dummy_X, dummy_y)
        
        print(f"✅ Enhanced plain text ensemble model initialized")
        return ensemble
    
    def train_local_model(self, client_id: str, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Train enhanced local model"""
        # Create ensemble model
        lr = LogisticRegression(**self.model_params)
        rf = RandomForestClassifier(
            n_estimators=50,
            max_depth=10,
            random_state=self.config.random_state,
            class_weight='balanced'
        )
        
        ensemble = VotingClassifier(
            estimators=[('lr', lr), ('rf', rf)],
            voting='soft'
        )
        
        ensemble.fit(X, y)
        
        # Extract parameters from logistic regression component
        lr_component = ensemble.estimators_[0]
        
        update = {
            'weights': lr_component.coef_.flatten(),
            'bias': lr_component.intercept_[0],
            'sample_count': len(X),
            'client_id': client_id,
            'ensemble_model': ensemble  # Store full model for advanced aggregation
        }
        
        return update
    
    def aggregate_updates(self, local_updates: List[Dict[str, Any]], sample_counts: List[int]) -> Dict[str, Any]:
        """Enhanced aggregation for ensemble models"""
        print("🔄 Aggregating Enhanced Plain Text Updates...")
        
        if not local_updates:
            return {'weights': np.array([]), 'bias': 0.0}
        
        # Use standard federated averaging for logistic regression components
        total_samples = sum(sample_counts)
        weights = np.array(sample_counts) / total_samples
        
        all_weights = []
        all_biases = []
        
        for update in local_updates:
            all_weights.append(update['weights'])
            all_biases.append(update['bias'])
        
        aggregated_weights = np.average(all_weights, axis=0, weights=weights)
        aggregated_bias = np.average(all_biases, weights=weights)
        
        print(f"✅ Aggregated {len(local_updates)} enhanced plain text updates")
        
        return {
            'weights': aggregated_weights,
            'bias': aggregated_bias,
            'total_samples': total_samples
        }
    
    def update_global_model(self, aggregated_update: Dict[str, Any]) -> None:
        """Update enhanced global model"""
        if self.global_model is None:
            return
        
        # Update the logistic regression component
        lr_component = self.global_model.estimators_[0]
        lr_component.coef_ = aggregated_update['weights'].reshape(1, -1)
        lr_component.intercept_ = np.array([aggregated_update['bias']])
        
        print("✅ Enhanced global model updated with plain text parameters")
    
    def evaluate_model(self, model: VotingClassifier, test_data: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
        """Evaluate ensemble model performance"""
        X_test, y_test = test_data
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred)
        }
        
        return metrics
    
    def _extract_model_state(self) -> Tuple[np.ndarray, float]:
        """Extract final model weights and bias from ensemble"""
        if self.global_model is None:
            return np.array([]), 0.0
        
        lr_component = self.global_model.estimators_[0]
        return lr_component.coef_.flatten(), lr_component.intercept_[0]
