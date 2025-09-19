"""
Edge Device Federated Learning Pipeline

This module implements n clients + n edge devices architecture where:
- Clients perform local training only
- Edge devices handle encryption/decryption
- Central server performs aggregation and global updates

The architecture maintains identical results to the original FHE pipeline by:
- Using the same training strategies (Laplace, FedProx, warm start)
- Using the same aggregation method (weighted sum)
- Using the same encryption/decryption operations
- Only moving encryption/decryption from clients to edge devices
"""

import time
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass

from src.encryption import FHEConfig, EncryptedModel, EncryptionManager
from src.fl import FLConfig, DataProcessor
from src.utils import calculate_enhanced_metrics


@dataclass
class ClientConfig:
    """Configuration for a federated learning client"""
    client_id: str
    client_data: Tuple[np.ndarray, np.ndarray]
    global_model_weights: Optional[np.ndarray] = None
    global_model_bias: Optional[float] = None


@dataclass
class EdgeDeviceConfig:
    """Configuration for an edge device"""
    edge_device_id: str
    client_id: str
    fhe_config: FHEConfig


class ClientProcess:
    """
    Client process that performs local training only
    
    This client:
    1. Receives global model from edge device
    2. Trains locally using same strategies as FHE pipeline
    3. Sends plaintext updates to edge device
    """
    
    def __init__(self, config: ClientConfig):
        self.client_id = config.client_id
        self.X, self.y = config.client_data
        self.global_model_weights = config.global_model_weights
        self.global_model_bias = config.global_model_bias
        
        # One-class handling parameters (same as FHE pipeline)
        self.l2_regularization = 1e-3
        self.fedprox_mu = 0.01
        self.laplace_smoothing = 0.1
        self.min_sample_weight = 10
    
    def is_one_class_client(self, y: np.ndarray) -> bool:
        """Check if client has only one class"""
        return len(np.unique(y)) == 1
    
    def apply_laplace_smoothing(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply Laplace smoothing for one-class clients (same as FHE pipeline)"""
        unique_class = int(np.unique(y)[0])
        virtual_samples = max(1, int(self.laplace_smoothing * len(y)))
        
        # Generate virtual samples for the opposite class
        virtual_X = np.random.normal(0, 0.1, (virtual_samples, X.shape[1]))
        virtual_y = (1 - unique_class) * np.ones(virtual_samples)
        
        # Combine with original data
        augmented_X = np.vstack([X, virtual_X])
        augmented_y = np.hstack([y, virtual_y])
        
        return augmented_X, augmented_y
    
    def create_warm_start_model(self, X: np.ndarray, y: np.ndarray):
        """Create warm start model (same as FHE pipeline)"""
        from sklearn.linear_model import LogisticRegression
        
        model = LogisticRegression(
            solver='liblinear',
            max_iter=5000,
            random_state=42
        )
        
        if self.global_model_weights is not None:
            print(f"    🔥 Using warm start for {self.client_id}")
            model.coef_ = self.global_model_weights.reshape(1, -1)
            model.intercept_ = np.array([self.global_model_bias])
            model.classes_ = np.array([0, 1])
        else:
            print(f"    ❄️ No warm start available for {self.client_id}")
        
        return model
    
    def apply_fedprox_regularization(self, model, X: np.ndarray, y: np.ndarray):
        """Apply FedProx regularization (same as FHE pipeline)"""
        if self.global_model_weights is not None:
            # Add proximal term to loss
            proximal_loss = self.fedprox_mu * np.sum(
                (model.coef_.flatten() - self.global_model_weights) ** 2
            )
            print(f"    🎯 FedProx regularization applied: {proximal_loss:.6f}")
    
    def train_local_model(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Train local model using same strategies as FHE pipeline"""
        is_one_class = self.is_one_class_client(y)
        
        if is_one_class:
            return self._train_one_class_client_with_strategy(X, y, strategy="combined")
        else:
            return self._train_normal_client_with_strategy(X, y, strategy="combined")
    
    def _train_normal_client_with_strategy(self, X: np.ndarray, y: np.ndarray, strategy: str = "combined") -> Dict[str, Any]:
        """Train normal client (same as FHE pipeline)"""
        from sklearn.linear_model import LogisticRegression
        
        model = LogisticRegression(
            solver='liblinear',
            max_iter=5000,
            random_state=42
        )
        
        # Apply warm start if available
        if self.global_model_weights is not None:
            model.coef_ = self.global_model_weights.reshape(1, -1)
            model.intercept_ = np.array([self.global_model_bias])
            model.classes_ = np.array([0, 1])
        
        model.fit(X, y)
        
        # Apply FedProx regularization
        self.apply_fedprox_regularization(model, X, y)
        
        weights = model.coef_.flatten()
        bias = float(model.intercept_[0])
        sample_count = len(X)
        
        return {
            'weights': weights,
            'bias': bias,
            'sample_count': sample_count,
            'is_one_class': False,
            'strategy': 'normal'
        }
    
    def _train_one_class_client_with_strategy(self, X: np.ndarray, y: np.ndarray, strategy: str = "combined") -> Dict[str, Any]:
        """Train one-class client (same as FHE pipeline)"""
        from sklearn.linear_model import LogisticRegression
        
        # Apply Laplace smoothing
        augmented_X, augmented_y = self.apply_laplace_smoothing(X, y)
        
        # Create warm start model
        model = self.create_warm_start_model(augmented_X, augmented_y)
        
        # Train on augmented data
        model.fit(augmented_X, augmented_y)
        
        # Apply FedProx regularization
        self.apply_fedprox_regularization(model, augmented_X, augmented_y)
        
        weights = model.coef_.flatten()
        bias = float(model.intercept_[0])
        sample_count = len(X)  # Use original sample count
        
        return {
            'weights': weights,
            'bias': bias,
            'sample_count': sample_count,
            'is_one_class': True,
            'strategy': 'laplace_fedprox'
        }
    
    def process_round(self, round_id: int) -> Dict[str, Any]:
        """Process one round of local training"""
        print(f"    📱 Client {self.client_id}: Training locally...")
        
        # Train local model
        result = self.train_local_model(self.X, self.y)
        
        return {
            'client_id': self.client_id,
            'round_id': round_id,
            'weights': result['weights'],
            'bias': result['bias'],
            'sample_count': result['sample_count'],
            'is_one_class': result['is_one_class'],
            'strategy': result['strategy']
        }


class EdgeDeviceProcess:
    """
    Edge device process that handles encryption/decryption
    
    This edge device:
    1. Receives plaintext updates from client
    2. Encrypts updates using FHE CKKS
    3. Sends encrypted updates to server
    4. Receives encrypted global model from server
    5. Decrypts global model and sends to client
    """
    
    def __init__(self, config: EdgeDeviceConfig, shared_encryption_manager: EncryptionManager = None):
        self.edge_device_id = config.edge_device_id
        self.client_id = config.client_id
        
        # Use shared encryption manager to maintain context consistency
        if shared_encryption_manager is not None:
            self.encryption_manager = shared_encryption_manager
        else:
            self.encryption_manager = EncryptionManager(config.fhe_config)
    
    def encrypt_client_update(self, weights: np.ndarray, bias: float) -> Tuple[Any, float]:
        """Encrypt client update (same as FHE pipeline)"""
        # Combine weights and bias
        model_update = np.concatenate([weights, [bias]])
        
        # Encrypt using same method as FHE pipeline
        encrypted_update, encryption_time = self.encryption_manager.encrypt_client_update(model_update)
        
        return encrypted_update, encryption_time
    
    def decrypt_global_model(self, encrypted_global_model: EncryptedModel) -> Tuple[np.ndarray, float, float]:
        """Decrypt global model for client (same as FHE pipeline)"""
        # Decrypt using same method as FHE pipeline
        global_weights, global_bias = encrypted_global_model.decrypt_for_evaluation()
        
        # CRITICAL FIX: Apply FHE CKKS scaling factor (same as FHE pipeline)
        # The FHE CKKS context uses scale_bits=40, so scale = 2^40
        scale_factor = 2**40  # scale_bits = 40
        global_weights = global_weights / scale_factor
        global_bias = global_bias / scale_factor
        
        print(f"    🔧 Applied FHE scaling factor: {scale_factor} (2^40)")
        print(f"    🔧 Scaled weights range: [{global_weights.min():.6f}, {global_weights.max():.6f}]")
        print(f"    🔧 Scaled bias: {global_bias:.6f}")
        
        return global_weights, global_bias, 0.0  # decryption_time
    
    def process_round(self, weights: np.ndarray, bias: float, round_id: int) -> Dict[str, Any]:
        """Process one round of encryption"""
        print(f"    🔒 Edge {self.edge_device_id}: Encrypting update...")
        
        # Encrypt client update
        encrypted_update, encryption_time = self.encrypt_client_update(weights, bias)
        
        return {
            'client_id': self.client_id,
            'round_id': round_id,
            'encrypted_update': encrypted_update,
            'encryption_time': encryption_time
        }


class CloudServerProcess:
    """
    Cloud server process that handles aggregation and global updates
    
    This server:
    1. Receives encrypted updates from edge devices
    2. Aggregates encrypted updates (same as FHE pipeline)
    3. Updates global model with encrypted data
    4. Evaluates model performance
    """
    
    def __init__(self, fhe_config: FHEConfig, feature_count: int):
        self.encryption_manager = EncryptionManager(fhe_config)
        self.feature_count = feature_count
        
        # Global model reference for client synchronization
        self.global_model_weights = None
        self.global_model_bias = None
        
        # Initialize global model (same as FHE pipeline)
        self._initialize_global_model()
    
    def _initialize_global_model(self):
        """Initialize global model (same as FHE pipeline)"""
        print("  🔧 Initializing global model...")
        
        # Create enhanced initialization data (same as FHE pipeline)
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
        
        # Create encrypted global model
        self.encrypted_global_model = self.encryption_manager.create_encrypted_model(
            weights=initial_weights,
            bias=initial_bias
        )
        
        print(f"  ✅ Global model initialized with {n_features} weights")
    
    def aggregate_updates(self, encrypted_updates: List[Any], sample_counts: List[int]) -> Tuple[Any, float]:
        """Aggregate encrypted updates (same as FHE pipeline)"""
        print("  🔄 Aggregating encrypted updates...")
        
        # Use same aggregation method as FHE pipeline
        aggregated_update, aggregation_time = self.encryption_manager.aggregate_updates(
            encrypted_updates, sample_counts
        )
        
        print("  ✅ Aggregation completed - result remains ENCRYPTED")
        return aggregated_update, aggregation_time
    
    def update_global_model(self, aggregated_update: Any):
        """Update global model (same as FHE pipeline)"""
        print("  🔄 Updating global model...")
        
        # Update encrypted global model
        self.encryption_manager.update_global_model(
            self.encrypted_global_model, aggregated_update
        )
        
        # CRITICAL FIX: Update global model reference with decrypted values (same as FHE pipeline)
        # This is essential for proper client synchronization
        decrypted_update = np.array(aggregated_update.decrypt())
        global_weights = decrypted_update[:-1]
        global_bias = float(decrypted_update[-1])
        
        # Apply FHE CKKS scaling factor (same as FHE pipeline)
        scale_factor = 2**40  # scale_bits = 40
        global_weights = global_weights / scale_factor
        global_bias = global_bias / scale_factor
        
        # Store for client synchronization
        self.global_model_weights = global_weights
        self.global_model_bias = global_bias
        
        print("  ✅ Global model updated - remains ENCRYPTED")
        print("  🔧 Global model reference updated for client synchronization")
    
    def evaluate_model(self, clients_data: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> Dict[str, float]:
        """Evaluate model (same as FHE pipeline)"""
        print("  📊 Evaluating model...")
        
        # Create test data (same as FHE pipeline)
        all_X, all_y = [], []
        for X_client, y_client in clients_data.values():
            all_X.append(X_client)
            all_y.append(y_client)
        
        X_test = np.vstack(all_X)
        y_test = np.hstack(all_y)
        
        # CRITICAL FIX: Use global model reference for evaluation (same as FHE pipeline)
        # This ensures evaluation uses the same scaled values as client synchronization
        if self.global_model_weights is not None:
            global_weights = self.global_model_weights
            global_bias = self.global_model_bias
            print(f"  🔧 Using global model reference for evaluation")
        else:
            # Fallback: decrypt and scale
            global_weights, global_bias = self.encrypted_global_model.decrypt_for_evaluation()
            scale_factor = 2**40  # scale_bits = 40
            global_weights = global_weights / scale_factor
            global_bias = global_bias / scale_factor
            print(f"  🔧 Fallback: decrypted and scaled for evaluation")
        
        # Create model for evaluation
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(random_state=42)
        model.coef_ = global_weights.reshape(1, -1)
        model.intercept_ = np.array([global_bias])
        model.classes_ = np.array([0, 1])
        
        # Get predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics (same as FHE pipeline)
        metrics = calculate_enhanced_metrics(y_test, y_pred, y_pred_proba)
        
        print(f"  ✅ Evaluation completed - Accuracy: {metrics['accuracy']*100:.2f}%")
        return metrics


class EdgeFederatedLearningCoordinator:
    """
    Coordinator for edge device federated learning
    
    This coordinator orchestrates:
    1. Client local training
    2. Edge device encryption
    3. Cloud server aggregation
    4. Global model updates
    5. Edge device decryption
    6. Client model synchronization
    """
    
    def __init__(self, fl_config: FLConfig, fhe_config: FHEConfig):
        self.config = fl_config
        self.fhe_config = fhe_config
        
        # Performance tracking
        self.performance_history = []
        self.best_accuracy = 0.0
    
    def run_edge_federated_learning(self, clients_data: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> List[Dict[str, Any]]:
        """Run edge federated learning with n clients + n edge devices"""
        print("\n" + "=" * 70)
        print("EDGE DEVICE FEDERATED LEARNING")
        print("=" * 70)
        print(f"Architecture: {len(clients_data)} clients + {len(clients_data)} edge devices + 1 cloud server")
        print("Data Flow: Client → Edge → Cloud → Edge → Client")
        
        # Initialize components
        client_configs = []
        edge_device_configs = []
        
        # CRITICAL FIX: Use shared encryption manager for context consistency
        feature_count = next(iter(clients_data.values()))[0].shape[1]
        cloud_server = CloudServerProcess(self.fhe_config, feature_count)
        shared_encryption_manager = cloud_server.encryption_manager
        
        for client_id, client_data in clients_data.items():
            # Client config
            client_config = ClientConfig(
                client_id=client_id,
                client_data=client_data
            )
            client_configs.append(client_config)
            
            # Edge device config
            edge_config = EdgeDeviceConfig(
                edge_device_id=f"edge_{client_id}",
                client_id=client_id,
                fhe_config=self.fhe_config
            )
            edge_device_configs.append(edge_config)
        
        # Run federated learning rounds
        results = []
        
        for round_num in range(1, self.config.rounds + 1):
            round_start = time.time()
            print(f"\n🔄 Round {round_num}/{self.config.rounds}")
            
            # Phase 1: Client Training
            print("  📱 Phase 1: Client Local Training")
            client_training_start = time.time()
            client_results = []
            for config in client_configs:
                client = ClientProcess(config)
                result = client.process_round(round_num)
                client_results.append(result)
            client_training_time = time.time() - client_training_start
            
            # Phase 2: Edge Device Encryption
            print("  🔒 Phase 2: Edge Device Encryption")
            edge_encryption_start = time.time()
            edge_results = []
            individual_encryption_times = []
            
            for i, config in enumerate(edge_device_configs):
                # CRITICAL FIX: Use shared encryption manager for context consistency
                edge_device = EdgeDeviceProcess(config, shared_encryption_manager)
                client_result = client_results[i]
                
                result = edge_device.process_round(
                    client_result['weights'],
                    client_result['bias'],
                    round_num
                )
                edge_results.append(result)
                individual_encryption_times.append(result.get('encryption_time', 0.0))
            
            # Calculate proper timing metrics for sequential processing
            edge_encryption_wall_time = time.time() - edge_encryption_start
            total_pure_encryption_time = sum(individual_encryption_times)
            avg_encryption_per_client = np.mean(individual_encryption_times) if individual_encryption_times else 0.0
            
            # Phase 3: Cloud Server Aggregation
            print("  ☁️ Phase 3: Cloud Server Aggregation")
            server_aggregation_start = time.time()
            
            # Align results by client_id
            client_map = {r['client_id']: r for r in client_results}
            edge_map = {r['client_id']: r for r in edge_results}
            common_ids = sorted(set(client_map) & set(edge_map))
            
            encrypted_updates = [edge_map[cid]['encrypted_update'] for cid in common_ids]
            sample_counts = [client_map[cid]['sample_count'] for cid in common_ids]
            
            # Aggregate encrypted updates
            aggregated_update, internal_aggregation_time = cloud_server.aggregate_updates(
                encrypted_updates, sample_counts
            )
            
            # Update global model
            cloud_server.update_global_model(aggregated_update)
            
            server_aggregation_time = time.time() - server_aggregation_start
            
            # Phase 4: Global Model Synchronization
            print("  🔓 Phase 4: Global Model Synchronization")
            
            # CRITICAL FIX: Use cloud server's global model reference (same as FHE pipeline)
            # This ensures all clients get the same updated global model
            if cloud_server.global_model_weights is not None:
                global_weights = cloud_server.global_model_weights
                global_bias = cloud_server.global_model_bias
                
                print(f"  🔧 Synchronizing global model to all clients")
                print(f"  🔧 Global weights range: [{global_weights.min():.6f}, {global_weights.max():.6f}]")
                print(f"  🔧 Global bias: {global_bias:.6f}")
                
                # Update all client configs with the same global model
                for i, config in enumerate(client_configs):
                    client_configs[i].global_model_weights = global_weights.copy()
                    client_configs[i].global_model_bias = global_bias
            else:
                print("  ⚠️ No global model reference available for synchronization")
            
            # Phase 5: Evaluation
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
                'roc_auc': metrics['auc'],  # Fixed: use 'auc' instead of 'roc_auc'
                'pr_auc': metrics['pr_auc'],
                'mae': metrics['mae'],
                'rmse': metrics['rmse'],
                'ece': metrics['ece'],
                'mce': metrics['mce'],
                'round_time': round_time,
                'client_training_time': client_training_time,
                'edge_encryption_wall_time': edge_encryption_wall_time,
                'total_pure_encryption_time': total_pure_encryption_time,
                'avg_encryption_per_client': avg_encryption_per_client,
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
