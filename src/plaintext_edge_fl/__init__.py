"""
Plaintext Edge Device Federated Learning Pipeline

This module implements n clients + n edge devices architecture for plaintext federated learning where:
- Clients perform local training only
- Edge devices handle data processing and validation
- Central server performs aggregation and global updates

The architecture provides:
- Same n clients + n edge devices structure as FHE version
- Plaintext operations (no encryption overhead)
- Data processing and validation at edge devices
- Fast execution with edge device benefits
"""

import time
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass

from src.plaintext import PlaintextConfig, PlaintextModel, PlaintextManager
from src.fl import FLConfig, DataProcessor
from src.utils import calculate_enhanced_metrics


@dataclass
class PlaintextClientConfig:
    """Configuration for a plaintext federated learning client"""
    client_id: str
    client_data: Tuple[np.ndarray, np.ndarray]
    global_model_weights: Optional[np.ndarray] = None
    global_model_bias: Optional[float] = None


@dataclass
class PlaintextEdgeDeviceConfig:
    """Configuration for a plaintext edge device"""
    edge_device_id: str
    client_id: str
    plaintext_config: PlaintextConfig


class PlaintextClientProcess:
    """
    Plaintext client process that performs local training only
    
    This client:
    1. Receives global model from edge device
    2. Trains locally using same strategies as plaintext pipeline
    3. Sends plaintext updates to edge device
    """
    
    def __init__(self, config: PlaintextClientConfig):
        self.client_id = config.client_id
        self.client_data = config.client_data
        self.global_model_weights = config.global_model_weights
        self.global_model_bias = config.global_model_bias
        
        # One-class handling parameters (same as plaintext pipeline)
        self.l2_regularization = 1e-3
        self.fedprox_mu = 0.01
        self.laplace_smoothing = 0.1
        self.min_sample_weight = 10
    
    def is_one_class_client(self, y: np.ndarray) -> bool:
        """Check if client has only one class"""
        return len(np.unique(y)) == 1
    
    def train_client_with_strategy(self, X: np.ndarray, y: np.ndarray, strategy: str = "combined") -> Dict[str, Any]:
        """Train client with specified strategy (same as original plaintext pipeline)"""
        from sklearn.linear_model import LogisticRegression
        
        # Check if this is a one-class client
        is_one_class = self.is_one_class_client(y)
        
        if not is_one_class:
            # Normal training for multi-class clients
            model = LogisticRegression(
                solver='liblinear',
                max_iter=5000,
                random_state=42
            )
            model.fit(X, y)
            strategy_used = "normal"
        else:
            # One-class client handling
            if strategy == "laplace":
                X_smoothed, y_smoothed = self.apply_laplace_smoothing(X, y)
                model = LogisticRegression(
                    solver='liblinear',
                    max_iter=5000,
                    random_state=42
                )
                model.fit(X_smoothed, y_smoothed)
                strategy_used = "laplace"
            elif strategy == "warm_start":
                model = self.create_warm_start_model(X, y)
                model.fit(X, y)
                strategy_used = "warm_start"
            elif strategy == "fedprox":
                model = LogisticRegression(
                    solver='liblinear',
                    max_iter=5000,
                    random_state=42
                )
                model = self.apply_fedprox_regularization(model, X, y)
                strategy_used = "fedprox"
            elif strategy == "combined":
                # Use Laplace smoothing for one-class clients
                X_smoothed, y_smoothed = self.apply_laplace_smoothing(X, y)
                model = LogisticRegression(
                    solver='liblinear',
                    max_iter=5000,
                    random_state=42
                )
                model.fit(X_smoothed, y_smoothed)
                strategy_used = "combined"
            else:
                # Fallback to normal training
                model = LogisticRegression(
                    solver='liblinear',
                    max_iter=5000,
                    random_state=42
                )
                model.fit(X, y)
                strategy_used = "fallback"
        
        return {
            'weights': model.coef_.flatten(),
            'bias': model.intercept_[0],
            'strategy': strategy_used
        }
    
    def apply_laplace_smoothing(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply Laplace smoothing to one-class client data (same as original plaintext FL)"""
        unique_class = np.unique(y)[0]
        n_samples = len(y)
        n_features = X.shape[1]
        
        # Generate synthetic samples for the opposite class
        opposite_class = 1 - unique_class
        synthetic_samples = max(1, int(n_samples * self.laplace_smoothing))
        
        # Generate synthetic data based on existing data distribution
        synthetic_X = np.random.normal(np.mean(X, axis=0), np.std(X, axis=0), (synthetic_samples, n_features))
        synthetic_y = np.full(synthetic_samples, opposite_class)
        
        # Combine original and synthetic data
        X_combined = np.vstack([X, synthetic_X])
        y_combined = np.hstack([y, synthetic_y])
        
        return X_combined, y_combined
    
    def create_warm_start_model(self, X: np.ndarray, y: np.ndarray):
        """Create model with warm start using global model (same as original plaintext FL)"""
        from sklearn.linear_model import LogisticRegression
        
        model = LogisticRegression(
            solver='liblinear',
            max_iter=5000,
            random_state=42
        )
        
        # Use global model as warm start if available
        if self.global_model_weights is not None and self.global_model_bias is not None:
            model.coef_ = self.global_model_weights.reshape(1, -1)
            model.intercept_ = np.array([self.global_model_bias])
        
        return model
    
    def apply_fedprox_regularization(self, model, X: np.ndarray, y: np.ndarray):
        """Apply FedProx regularization (same as original plaintext FL)"""
        # Use global model for FedProx if available
        if self.global_model_weights is not None and self.global_model_bias is not None:
            # Create model with FedProx regularization
            from sklearn.linear_model import LogisticRegression
            
            model = LogisticRegression(
                solver='liblinear',
                max_iter=5000,
                random_state=42,
                C=1.0/(self.l2_regularization + self.fedprox_mu)
            )
            
            # Set initial parameters
            model.coef_ = self.global_model_weights.reshape(1, -1)
            model.intercept_ = np.array([self.global_model_bias])
        
        return model
    
    def _train_with_laplace_smoothing(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Train with Laplace smoothing for one-class clients"""
        from sklearn.linear_model import LogisticRegression
        
        # For one-class clients, create synthetic data with Laplace smoothing
        unique_class = np.unique(y)[0]
        n_samples = len(y)
        
        # Create synthetic data with both classes
        # Add small amount of opposite class data
        n_features = X.shape[1]
        synthetic_samples = int(n_samples * self.laplace_smoothing)
        
        if synthetic_samples > 0:
            # Generate synthetic samples for the opposite class
            opposite_class = 1 - unique_class
            
            # Use global model weights if available for synthetic data generation
            if self.global_model_weights is not None:
                # Generate synthetic data based on global model
                synthetic_X = np.random.normal(0, 0.1, (synthetic_samples, n_features))
                # Add some variation based on existing data
                synthetic_X += np.mean(X, axis=0) + np.random.normal(0, 0.05, (synthetic_samples, n_features))
            else:
                # Generate synthetic data based on existing data distribution
                synthetic_X = np.random.normal(np.mean(X, axis=0), np.std(X, axis=0), (synthetic_samples, n_features))
            
            synthetic_y = np.full(synthetic_samples, opposite_class)
            
            # Combine original and synthetic data
            X_combined = np.vstack([X, synthetic_X])
            y_combined = np.hstack([y, synthetic_y])
            
            # Create sample weights (original data gets higher weight)
            sample_weights = np.ones(len(y_combined))
            sample_weights[:n_samples] = 1.0  # Original data
            sample_weights[n_samples:] = self.laplace_smoothing  # Synthetic data
            
        else:
            # Fallback: use original data with modified parameters
            X_combined = X
            y_combined = y
            sample_weights = np.ones(len(y))
        
        # Train model
        model = LogisticRegression(
            solver='liblinear',
            max_iter=5000,
            random_state=42,
            C=1.0/self.l2_regularization
        )
        
        model.fit(X_combined, y_combined, sample_weight=sample_weights)
        
        return {
            'weights': model.coef_.flatten(),
            'bias': model.intercept_[0],
            'strategy': 'laplace'
        }
    
    def _train_with_warm_start(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Train with warm start using global model"""
        from sklearn.linear_model import LogisticRegression
        
        # Use global model as warm start if available
        if self.global_model_weights is not None and self.global_model_bias is not None:
            # Create model with warm start
            model = LogisticRegression(
                solver='liblinear',
                max_iter=5000,
                random_state=42,
                C=1.0/self.l2_regularization,
                warm_start=True
            )
            
            # Set initial parameters
            model.coef_ = self.global_model_weights.reshape(1, -1)
            model.intercept_ = np.array([self.global_model_bias])
            
            model.fit(X, y)
            
            return {
                'weights': model.coef_.flatten(),
                'bias': model.intercept_[0],
                'strategy': 'warm_start'
            }
        else:
            return self._train_basic(X, y)
    
    def _train_with_fedprox(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Train with FedProx regularization"""
        from sklearn.linear_model import LogisticRegression
        
        # Use global model for FedProx if available
        if self.global_model_weights is not None and self.global_model_bias is not None:
            # Create model with FedProx
            model = LogisticRegression(
                solver='liblinear',
                max_iter=5000,
                random_state=42,
                C=1.0/(self.l2_regularization + self.fedprox_mu)
            )
            
            # Set initial parameters
            model.coef_ = self.global_model_weights.reshape(1, -1)
            model.intercept_ = np.array([self.global_model_bias])
            
            model.fit(X, y)
            
            return {
                'weights': model.coef_.flatten(),
                'bias': model.intercept_[0],
                'strategy': 'fedprox'
            }
        else:
            return self._train_basic(X, y)
    
    def _train_basic(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Basic training without special strategies"""
        from sklearn.linear_model import LogisticRegression
        
        model = LogisticRegression(
            solver='liblinear',
            max_iter=5000,
            random_state=42,
            C=1.0/self.l2_regularization
        )
        
        model.fit(X, y)
        
        return {
            'weights': model.coef_.flatten(),
            'bias': model.intercept_[0],
            'strategy': 'basic'
        }
    
    def process_round(self, round_id: int) -> Dict[str, Any]:
        """Process one round of local training"""
        print(f"    📱 Client {self.client_id}: Training locally...")
        
        X, y = self.client_data
        
        # Train with appropriate strategy
        result = self.train_client_with_strategy(X, y)
        
        return {
            'client_id': self.client_id,
            'round_id': round_id,
            'weights': result['weights'],
            'bias': result['bias'],
            'sample_count': len(X),
            'strategy': result['strategy']
        }


class PlaintextEdgeDeviceProcess:
    """
    Plaintext edge device process that handles data processing and validation
    
    This edge device:
    1. Receives plaintext updates from client
    2. Validates and processes data
    3. Sends processed updates to server
    4. Receives global model from server
    5. Validates and sends global model to client
    """
    
    def __init__(self, config: PlaintextEdgeDeviceConfig, shared_plaintext_manager: PlaintextManager = None):
        self.edge_device_id = config.edge_device_id
        self.client_id = config.client_id
        
        # Use shared plaintext manager for consistency
        if shared_plaintext_manager is not None:
            self.plaintext_manager = shared_plaintext_manager
        else:
            self.plaintext_manager = PlaintextManager(config.plaintext_config)
    
    def process_client_update(self, weights: np.ndarray, bias: float) -> Tuple[np.ndarray, float, float]:
        """Process client update (validation and formatting)"""
        # Combine weights and bias
        model_update = np.concatenate([weights, [bias]])
        
        # Validate data
        if np.any(np.isnan(model_update)) or np.any(np.isinf(model_update)):
            print(f"    ⚠️ Edge {self.edge_device_id}: Invalid data detected, using zeros")
            model_update = np.zeros_like(model_update)
        
        # Separate weights and bias
        processed_weights = model_update[:-1]
        processed_bias = float(model_update[-1])
        
        return processed_weights, processed_bias, 0.0  # processing_time
    
    def process_round(self, weights: np.ndarray, bias: float, round_id: int) -> Dict[str, Any]:
        """Process one round of edge device operations"""
        print(f"    🔧 Edge {self.edge_device_id}: Processing update...")
        
        # Process client update
        processed_weights, processed_bias, processing_time = self.process_client_update(weights, bias)
        
        return {
            'client_id': self.client_id,
            'round_id': round_id,
            'processed_weights': processed_weights,
            'processed_bias': processed_bias,
            'processing_time': processing_time
        }


class PlaintextCloudServerProcess:
    """
    Plaintext cloud server process that handles aggregation and global updates
    
    This server:
    1. Receives processed updates from edge devices
    2. Aggregates updates in plaintext
    3. Updates global model
    4. Evaluates model performance
    """
    
    def __init__(self, plaintext_config: PlaintextConfig, feature_count: int):
        self.plaintext_manager = PlaintextManager(plaintext_config)
        self.feature_count = feature_count
        
        # Global model reference for client synchronization
        self.global_model_weights = None
        self.global_model_bias = None
        
        # Initialize global model
        self._initialize_global_model()
    
    def _initialize_global_model(self):
        """Initialize global model"""
        print("  🔧 Initializing global model...")
        
        # Create enhanced initialization data
        np.random.seed(42)
        n_samples = 200
        n_features = self.feature_count
        
        X = np.random.randn(n_samples, n_features)
        y = np.random.choice([0, 1], size=n_samples, p=[0.5, 0.5])
        
        from sklearn.linear_model import LogisticRegression
        initial_model = LogisticRegression(solver='liblinear', max_iter=5000, random_state=42)
        initial_model.fit(X, y)
        
        initial_weights = initial_model.coef_.flatten()
        initial_bias = initial_model.intercept_[0]
        
        # Create plaintext global model
        self.plaintext_global_model = self.plaintext_manager.create_model(initial_weights, initial_bias)
        
        # Store global model reference
        self.global_model_weights = initial_weights
        self.global_model_bias = initial_bias
        
        print(f"  ✅ Global model initialized with {n_features} weights")
    
    def aggregate_updates(self, processed_updates: List[Dict[str, Any]], sample_counts: List[int]) -> Tuple[np.ndarray, float]:
        """Aggregate processed updates in plaintext"""
        print("  🔄 Aggregating processed updates...")
        
        # Extract weights and bias from processed updates
        weights_list = [update['processed_weights'] for update in processed_updates]
        bias_list = [update['processed_bias'] for update in processed_updates]
        
        # Weighted aggregation
        total_samples = sum(sample_counts)
        aggregated_weights = np.zeros_like(weights_list[0])
        aggregated_bias = 0.0
        
        for weights, bias, sample_count in zip(weights_list, bias_list, sample_counts):
            weight_factor = sample_count / total_samples
            aggregated_weights += weights * weight_factor
            aggregated_bias += bias * weight_factor
        
        # Combine into single vector
        aggregated_update = np.concatenate([aggregated_weights, [aggregated_bias]])
        
        print("  ✅ Aggregation completed - result in PLAINTEXT")
        return aggregated_update, 0.0  # aggregation_time
    
    def update_global_model(self, aggregated_update: np.ndarray):
        """Update global model with aggregated update"""
        print("  🔄 Updating global model...")
        
        # Update plaintext global model
        self.plaintext_global_model.from_model_update(aggregated_update)
        
        # Update global model reference
        self.global_model_weights = aggregated_update[:-1]
        self.global_model_bias = float(aggregated_update[-1])
        
        print("  ✅ Global model updated - remains PLAINTEXT")
    
    def evaluate_model(self, clients_data: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> Dict[str, float]:
        """Evaluate model performance"""
        print("  📊 Evaluating model...")
        
        # Use global model reference for evaluation
        if self.global_model_weights is not None:
            global_weights = self.global_model_weights
            global_bias = self.global_model_bias
        else:
            global_weights = self.plaintext_global_model.get_weights()
            global_bias = self.plaintext_global_model.get_bias()
        
        # Create test data from all clients
        X_test_list = []
        y_test_list = []
        
        for client_id, (X, y) in clients_data.items():
            X_test_list.append(X)
            y_test_list.append(y)
        
        X_test = np.vstack(X_test_list)
        y_test = np.hstack(y_test_list)
        
        # Create model for evaluation
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(solver='liblinear', max_iter=5000, random_state=42)
        
        # Set model parameters
        model.coef_ = global_weights.reshape(1, -1)
        model.intercept_ = np.array([global_bias])
        model.classes_ = np.array([0, 1])  # Set classes for prediction
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = calculate_enhanced_metrics(y_test, y_pred, y_pred_proba)
        
        print(f"  ✅ Evaluation completed - Accuracy: {metrics['accuracy']*100:.2f}%")
        return metrics


class PlaintextEdgeFederatedLearningCoordinator:
    """
    Coordinator for plaintext edge device federated learning
    
    This coordinator orchestrates:
    1. Client local training
    2. Edge device data processing
    3. Cloud server aggregation
    4. Global model updates
    5. Edge device validation
    6. Client model synchronization
    """
    
    def __init__(self, fl_config: FLConfig, plaintext_config: PlaintextConfig):
        self.config = fl_config
        self.plaintext_config = plaintext_config
        
        # Performance tracking
        self.performance_history = []
        self.best_accuracy = 0.0
    
    def run_plaintext_edge_federated_learning(self, clients_data: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> List[Dict[str, Any]]:
        """Run plaintext edge federated learning with n clients + n edge devices"""
        print("\n" + "=" * 70)
        print("PLAINTEXT EDGE DEVICE FEDERATED LEARNING")
        print("=" * 70)
        print(f"Architecture: {len(clients_data)} clients + {len(clients_data)} edge devices + 1 cloud server")
        print("Data Flow: Client → Edge → Cloud → Edge → Client")
        
        # Initialize components
        client_configs = []
        edge_device_configs = []
        
        # Use shared plaintext manager for consistency
        feature_count = next(iter(clients_data.values()))[0].shape[1]
        cloud_server = PlaintextCloudServerProcess(self.plaintext_config, feature_count)
        shared_plaintext_manager = cloud_server.plaintext_manager
        
        for client_id, client_data in clients_data.items():
            # Create client config
            client_config = PlaintextClientConfig(
                client_id=client_id,
                client_data=client_data,
                global_model_weights=cloud_server.global_model_weights,
                global_model_bias=cloud_server.global_model_bias
            )
            client_configs.append(client_config)
            
            # Create edge device config
            edge_device_config = PlaintextEdgeDeviceConfig(
                edge_device_id=f"edge_{client_id}",
                client_id=client_id,
                plaintext_config=self.plaintext_config
            )
            edge_device_configs.append(edge_device_config)
        
        results = []
        
        # Run federated learning rounds
        for round_num in range(1, self.config.rounds + 1):
            print(f"\n🔄 Round {round_num}/{self.config.rounds}")
            round_start = time.time()
            
            # Phase 1: Client Local Training
            print("  📱 Phase 1: Client Local Training")
            client_training_start = time.time()
            client_results = []
            
            for config in client_configs:
                client = PlaintextClientProcess(config)
                result = client.process_round(round_num)
                client_results.append(result)
            client_training_time = time.time() - client_training_start
            
            # Phase 2: Edge Device Processing
            print("  🔧 Phase 2: Edge Device Processing")
            edge_processing_start = time.time()
            edge_results = []
            individual_processing_times = []
            
            for i, config in enumerate(edge_device_configs):
                edge_device = PlaintextEdgeDeviceProcess(config, shared_plaintext_manager)
                client_result = client_results[i]
                
                result = edge_device.process_round(
                    client_result['weights'],
                    client_result['bias'],
                    round_num
                )
                edge_results.append(result)
                individual_processing_times.append(result.get('processing_time', 0.0))
            
            # Calculate proper timing metrics for sequential processing
            edge_processing_wall_time = time.time() - edge_processing_start
            total_pure_processing_time = sum(individual_processing_times)
            avg_processing_per_client = np.mean(individual_processing_times) if individual_processing_times else 0.0
            
            # Phase 3: Cloud Server Aggregation
            print("  ☁️ Phase 3: Cloud Server Aggregation")
            server_aggregation_start = time.time()
            
            # Align results by client_id
            client_map = {r['client_id']: r for r in client_results}
            edge_map = {r['client_id']: r for r in edge_results}
            common_ids = sorted(set(client_map) & set(edge_map))
            
            processed_updates = [edge_map[cid] for cid in common_ids]
            sample_counts = [client_map[cid]['sample_count'] for cid in common_ids]
            
            # Aggregate processed updates
            aggregated_update, internal_aggregation_time = cloud_server.aggregate_updates(
                processed_updates, sample_counts
            )
            
            # Update global model
            cloud_server.update_global_model(aggregated_update)
            
            server_aggregation_time = time.time() - server_aggregation_start
            
            # Phase 4: Global Model Synchronization
            print("  🔓 Phase 4: Global Model Synchronization")
            
            # Use cloud server's global model reference
            if cloud_server.global_model_weights is not None:
                global_weights = cloud_server.global_model_weights
                global_bias = cloud_server.global_model_bias
                
                for i, config in enumerate(client_configs):
                    client_configs[i].global_model_weights = global_weights.copy()
                    client_configs[i].global_model_bias = global_bias
            
            # Phase 5: Model Evaluation
            print("  📊 Phase 5: Model Evaluation")
            evaluation_start = time.time()
            metrics = cloud_server.evaluate_model(clients_data)
            evaluation_time = time.time() - evaluation_start
            
            # Calculate timing
            round_time = time.time() - round_start
            
            # Store results with comprehensive timing statistics
            round_result = {
                'round': round_num,
                'accuracy': metrics['accuracy'],
                'f1_score': metrics['f1_score'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'roc_auc': metrics['auc'],
                'pr_auc': metrics['pr_auc'],
                'mae': metrics['mae'],
                'rmse': metrics['rmse'],
                'ece': metrics['ece'],
                'mce': metrics['mce'],
                'round_time': round_time,
                'client_training_time': client_training_time,
                'edge_processing_wall_time': edge_processing_wall_time,
                'total_pure_processing_time': total_pure_processing_time,
                'avg_processing_per_client': avg_processing_per_client,
                'server_aggregation_time': server_aggregation_time,
                'internal_aggregation_time': internal_aggregation_time,
                'evaluation_time': evaluation_time,
                'clients_processed': len(common_ids)
            }
            
            results.append(round_result)
            
            # Update best accuracy
            if metrics['accuracy'] > self.best_accuracy:
                self.best_accuracy = metrics['accuracy']
            
            print(f"  ✅ Round {round_num} completed - Accuracy: {metrics['accuracy']*100:.2f}%")
        
        return results
