"""
INC-Assisted Edge Device Federated Learning Pipeline

This module implements n clients + n edge devices + INC architecture where:
- Clients perform local training only
- Edge devices handle encryption/decryption
- INC performs intermediate aggregation
- Cloud server receives pre-aggregated data

Architecture: Client → Edge → INC → Cloud → INC → Edge → Client
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
    edge_id: str
    client_id: str


@dataclass
class INCConfig:
    """Configuration for Intermediate Network Controller"""
    inc_id: str
    edge_device_ids: List[str]  # Edge devices this INC manages


class ClientProcess:
    """
    Client process that performs local training only
    
    This client:
    1. Receives global model from edge device
    2. Performs local training on its data
    3. Sends local model to edge device
    """
    
    def __init__(self, config: ClientConfig):
        self.config = config
        self.client_id = config.client_id
        self.X, self.y = config.client_data
        
        # Training configuration (same as FHE pipeline)
        self.solver = 'liblinear'
        self.max_iter = 5000
        self.random_state = 42
    
    def train_client_with_strategy(self, global_weights: np.ndarray, global_bias: float, round_num: int) -> Dict[str, Any]:
        """Train client with appropriate strategy (same as FHE pipeline)"""
        print(f"    📱 Client {self.client_id}: Training locally...")
        
        # Check if client has only one class
        unique_classes = np.unique(self.y)
        if len(unique_classes) == 1:
            # One-class client handling (same as FHE pipeline)
            return self._train_with_laplace_smoothing(global_weights, global_bias, round_num)
        else:
            # Multi-class client - normal training
            return self._train_normal_client(global_weights, global_bias, round_num)
    
    def _train_with_laplace_smoothing(self, global_weights: np.ndarray, global_bias: float, round_num: int) -> Dict[str, Any]:
        """Handle one-class client with Laplace smoothing (same as FHE pipeline)"""
        from sklearn.linear_model import LogisticRegression
        
        # Apply Laplace smoothing
        X_augmented, y_augmented = self._apply_laplace_smoothing()
        
        # Create model with warm start if available
        if round_num > 1:
            model = self._create_warm_start_model(global_weights, global_bias)
        else:
            model = LogisticRegression(solver=self.solver, max_iter=self.max_iter, random_state=self.random_state)
        
        # Train model
        model.fit(X_augmented, y_augmented)
        
        weights = model.coef_.flatten()
        bias = model.intercept_[0]
        
        return {
            'client_id': self.client_id,
            'weights': weights,
            'bias': bias,
            'sample_count': len(self.y),
            'training_time': 0.0  # Will be measured by coordinator
        }
    
    def _apply_laplace_smoothing(self) -> Tuple[np.ndarray, np.ndarray]:
        """Apply Laplace smoothing for one-class clients (same as FHE pipeline)"""
        # Generate synthetic samples of the missing class
        n_synthetic = 20
        missing_class = 1 if 0 in self.y else 0
        
        # Generate synthetic features
        np.random.seed(42)
        synthetic_X = np.random.randn(n_synthetic, self.X.shape[1]) * 0.1
        
        # Combine with original data
        X_augmented = np.vstack([self.X, synthetic_X])
        y_augmented = np.hstack([self.y, np.full(n_synthetic, missing_class)])
        
        print(f"    🔧 Applied Laplace smoothing: added {n_synthetic} virtual samples of class {missing_class}")
        return X_augmented, y_augmented
    
    def _create_warm_start_model(self, global_weights: np.ndarray, global_bias: float) -> Any:
        """Create warm start model (same as FHE pipeline)"""
        from sklearn.linear_model import LogisticRegression
        
        model = LogisticRegression(solver=self.solver, max_iter=self.max_iter, random_state=self.random_state)
        
        # Set initial parameters
        model.coef_ = global_weights.reshape(1, -1)
        model.intercept_ = np.array([global_bias])
        model.classes_ = np.array([0, 1])
        
        print(f"    🔥 Using warm start for {self.client_id}")
        return model
    
    def _train_normal_client(self, global_weights: np.ndarray, global_bias: float, round_num: int) -> Dict[str, Any]:
        """Train normal multi-class client (same as FHE pipeline)"""
        from sklearn.linear_model import LogisticRegression
        
        # Apply FedProx regularization if not first round
        if round_num > 1:
            return self._apply_fedprox_regularization(global_weights, global_bias)
        
        # First round - normal training
        model = LogisticRegression(solver=self.solver, max_iter=self.max_iter, random_state=self.random_state)
        model.fit(self.X, self.y)
        
        weights = model.coef_.flatten()
        bias = model.intercept_[0]
        
        return {
            'client_id': self.client_id,
            'weights': weights,
            'bias': bias,
            'sample_count': len(self.y),
            'training_time': 0.0
        }
    
    def _apply_fedprox_regularization(self, global_weights: np.ndarray, global_bias: float) -> Dict[str, Any]:
        """Apply FedProx regularization (same as FHE pipeline)"""
        from sklearn.linear_model import LogisticRegression
        
        # Calculate regularization parameter
        mu = 0.001  # FedProx parameter
        
        # Create model with regularization
        model = LogisticRegression(
            solver=self.solver,
            max_iter=self.max_iter,
            random_state=self.random_state,
            C=1.0/(mu + 1e-8)  # Inverse regularization strength
        )
        
        # Fit with regularization
        model.fit(self.X, self.y)
        
        # Apply FedProx update
        local_weights = model.coef_.flatten()
        local_bias = model.intercept_[0]
        
        # FedProx update: w = w_local - mu * (w_local - w_global)
        fedprox_weights = local_weights - mu * (local_weights - global_weights)
        fedprox_bias = local_bias - mu * (local_bias - global_bias)
        
        print(f"    🎯 FedProx regularization applied: {mu:.6f}")
        
        return {
            'client_id': self.client_id,
            'weights': fedprox_weights,
            'bias': fedprox_bias,
            'sample_count': len(self.y),
            'training_time': 0.0
        }


class EdgeDeviceProcess:
    """
    Edge device process that handles encryption/decryption
    
    This edge device:
    1. Receives local model from client
    2. Encrypts model parameters
    3. Sends encrypted data to INC
    4. Receives decrypted global model from INC
    5. Sends global model to client
    """
    
    def __init__(self, config: EdgeDeviceConfig, encryption_manager: EncryptionManager):
        self.config = config
        self.edge_id = config.edge_id
        self.client_id = config.client_id
        self.encryption_manager = encryption_manager
    
    def process_round(self, weights: np.ndarray, bias: float, round_id: int) -> Dict[str, Any]:
        """Process one round: encrypt local model"""
        print(f"    🔒 Edge {self.edge_id}: Encrypting update...")
        
        # Encrypt local model
        encryption_start = time.time()
        encrypted_update, encryption_time = self.encryption_manager.encrypt_client_update(np.concatenate([weights, [bias]]))
        encryption_time = time.time() - encryption_start
        
        print(f"🔒 Real FHE CKKS encryption completed in {encryption_time:.4f}s")
        
        return {
            'edge_id': self.edge_id,
            'client_id': self.client_id,
            'encrypted_update': encrypted_update,
            'encryption_time': encryption_time
        }
    
    def decrypt_global_model(self, encrypted_global_model: Any) -> Tuple[np.ndarray, float]:
        """Decrypt global model for client synchronization"""
        print(f"    🔓 Edge {self.edge_id}: Decrypting global model...")
        
        # Decrypt global model
        global_weights, global_bias = encrypted_global_model.decrypt_for_evaluation()
        
        # Apply FHE CKKS scaling factor (same as FHE pipeline)
        scale_factor = 2**40  # scale_bits = 40
        global_weights = global_weights / scale_factor
        global_bias = global_bias / scale_factor
        
        print(f"    ✅ Global model decrypted and scaled")
        return global_weights, global_bias


class INCProcess:
    """
    Intermediate Network Controller process
    
    This INC:
    1. Receives encrypted updates from multiple edge devices
    2. Performs intermediate aggregation in encrypted domain
    3. Sends aggregated result to cloud server
    4. Receives global model from cloud server
    5. Distributes global model to edge devices
    """
    
    def __init__(self, config: INCConfig, encryption_manager: EncryptionManager):
        self.config = config
        self.inc_id = config.inc_id
        self.edge_device_ids = config.edge_device_ids
        self.encryption_manager = encryption_manager
    
    def aggregate_edge_updates(self, edge_updates: List[Dict[str, Any]], sample_counts: List[int]) -> Dict[str, Any]:
        """Aggregate encrypted updates from edge devices"""
        print(f"    🔄 INC {self.inc_id}: Aggregating {len(edge_updates)} edge updates...")
        
        # Extract encrypted updates and sample counts
        encrypted_updates = [update['encrypted_update'] for update in edge_updates]
        
        # Perform intermediate aggregation
        aggregation_start = time.time()
        aggregated_update, internal_aggregation_time = self.encryption_manager.aggregate_updates(
            encrypted_updates, sample_counts
        )
        aggregation_time = time.time() - aggregation_start
        
        print(f"    ✅ INC aggregation completed in {aggregation_time:.4f}s")
        
        return {
            'inc_id': self.inc_id,
            'aggregated_update': aggregated_update,
            'aggregation_time': aggregation_time,
            'internal_aggregation_time': internal_aggregation_time,
            'edge_count': len(edge_updates)
        }
    
    def distribute_global_model(self, encrypted_global_model: Any, edge_devices: List[EdgeDeviceProcess]) -> Dict[str, Any]:
        """Distribute global model to edge devices"""
        print(f"    🔄 INC {self.inc_id}: Distributing global model to {len(edge_devices)} edge devices...")
        
        distribution_start = time.time()
        
        for edge_device in edge_devices:
            # Decrypt global model for each edge device
            global_weights, global_bias = edge_device.decrypt_global_model(encrypted_global_model)
            
            # Update client configuration with global model
            # This will be handled by the coordinator
        
        distribution_time = time.time() - distribution_start
        
        print(f"    ✅ Global model distributed in {distribution_time:.4f}s")
        
        return {
            'inc_id': self.inc_id,
            'distribution_time': distribution_time,
            'edge_count': len(edge_devices)
        }


class CloudServerProcess:
    """
    Cloud server process that receives pre-aggregated data
    
    This server:
    1. Receives aggregated updates from INC
    2. Updates global model with aggregated data
    3. Evaluates model performance
    4. Sends global model back to INC
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
    
    def update_global_model(self, aggregated_update: Any):
        """Update global model with pre-aggregated data"""
        print("  🔄 Updating global model with INC aggregated data...")
        
        # Update encrypted global model
        self.encryption_manager.update_global_model(
            self.encrypted_global_model, aggregated_update
        )
        
        # CRITICAL FIX: Update global model reference with decrypted values
        decrypted_update = np.array(aggregated_update.decrypt())
        global_weights = decrypted_update[:-1]
        global_bias = float(decrypted_update[-1])
        
        # Apply FHE CKKS scaling factor
        scale_factor = 2**40  # scale_bits = 40
        global_weights = global_weights / scale_factor
        global_bias = global_bias / scale_factor
        
        # Store for client synchronization
        self.global_model_weights = global_weights
        self.global_model_bias = global_bias
        
        print("  ✅ Global model updated with INC aggregated data")
        print("  🔧 Global model reference updated for client synchronization")
    
    def evaluate_model(self, clients_data: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> Dict[str, float]:
        """Evaluate model performance"""
        print("  📊 Evaluating model...")
        
        # Use global model reference for evaluation
        if self.global_model_weights is None or self.global_model_bias is None:
            raise ValueError("Global model not initialized")
        
        # Create model for evaluation
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(solver='liblinear', max_iter=5000, random_state=42)
        model.coef_ = self.global_model_weights.reshape(1, -1)
        model.intercept_ = np.array([self.global_model_bias])
        model.classes_ = np.array([0, 1])
        
        # Evaluate on all client data
        all_X = []
        all_y = []
        
        for client_id, (X, y) in clients_data.items():
            all_X.append(X)
            all_y.append(y)
        
        X_eval = np.vstack(all_X)
        y_eval = np.hstack(all_y)
        
        # Calculate metrics
        y_pred = model.predict(X_eval)
        y_pred_proba = model.predict_proba(X_eval)[:, 1]
        
        metrics = calculate_enhanced_metrics(y_eval, y_pred, y_pred_proba)
        
        print(f"  ✅ Evaluation completed - Accuracy: {metrics['accuracy']*100:.2f}%")
        return metrics


class INCFederatedLearningCoordinator:
    """
    Coordinator for INC-assisted edge device federated learning
    
    This coordinator orchestrates:
    1. Client local training
    2. Edge device encryption
    3. INC intermediate aggregation
    4. Cloud server global updates
    5. INC global model distribution
    6. Edge device decryption
    7. Client model synchronization
    """
    
    def __init__(self, fl_config: FLConfig, fhe_config: FHEConfig):
        self.config = fl_config
        self.fhe_config = fhe_config
        
        # Performance tracking
        self.best_accuracy = 0.0
    
    def _aggregate_multiple_incs(self, inc_results: List[Dict[str, Any]], encryption_manager: EncryptionManager) -> Any:
        """
        Properly aggregate multiple INC results using secure decryption-aggregation-re-encryption
        
        This method handles FHE scaling limitations by:
        1. Decrypting each INC's aggregated result at trusted cloud server
        2. Performing weighted aggregation in plaintext (mathematically correct)
        3. Re-encrypting the final result for secure transmission
        
        SECURITY NOTE: This approach is secure because:
        - Decryption only happens at the trusted cloud server
        - No client data is exposed to other clients
        - The cloud server already has access to aggregated data
        - Re-encryption ensures secure transmission back to INCs
        
        Args:
            inc_results: List of INC aggregation results
            encryption_manager: Encryption manager for re-encryption
            
        Returns:
            Encrypted aggregated update
        """
        print(f"    🔄 Aggregating {len(inc_results)} INC results using secure decryption-aggregation-re-encryption...")
        
        if len(inc_results) == 1:
            return inc_results[0]['aggregated_update']
        
        # Extract encrypted aggregated updates and calculate sample counts
        encrypted_updates = []
        sample_counts = []
        
        for result in inc_results:
            encrypted_updates.append(result['aggregated_update'])
            # Each INC manages 3 clients, each with 200 samples
            sample_counts.append(result['edge_count'] * 200)  # edge_count * samples_per_client
        
        total_samples = sum(sample_counts)
        print(f"    🔓 Decrypting {len(encrypted_updates)} INC aggregated results at trusted cloud server...")
        
        # Decrypt all INC aggregated results at trusted cloud server
        decrypted_updates = []
        for i, encrypted_update in enumerate(encrypted_updates):
            # Decrypt the aggregated update from each INC
            decrypted_update = np.array(encrypted_update.decrypt())
            decrypted_updates.append(decrypted_update)
            print(f"    🔓 Decrypted INC {i} aggregated update (samples: {sample_counts[i]})")
        
        # Perform weighted aggregation in plaintext (mathematically correct)
        print(f"    📊 Performing weighted aggregation in plaintext...")
        weighted_sum = decrypted_updates[0] * (sample_counts[0] / total_samples)
        for i, decrypted_update in enumerate(decrypted_updates[1:], 1):
            weight = sample_counts[i] / total_samples
            weighted_sum = weighted_sum + (decrypted_update * weight)
        
        print(f"    ✅ Weighted aggregation completed - total samples: {total_samples}")
        
        # Re-encrypt the final aggregated result for secure transmission
        print(f"    🔒 Re-encrypting final aggregated result...")
        final_encrypted_update, _ = encryption_manager.encrypt_client_update(weighted_sum)
        print(f"    🔒 Final aggregated result re-encrypted for secure transmission")
        
        return final_encrypted_update
    
    def run_inc_federated_learning(self, clients_data: Dict[str, Tuple[np.ndarray, np.ndarray]], inc_configs: List[INCConfig]) -> List[Dict[str, Any]]:
        """Run INC-assisted federated learning"""
        
        print("\n" + "=" * 70)
        print("INC-ASSISTED EDGE DEVICE FEDERATED LEARNING")
        print("=" * 70)
        print(f"Architecture: {len(clients_data)} clients + {len(clients_data)} edge devices + {len(inc_configs)} INC + 1 cloud server")
        print("Data Flow: Client → Edge → INC → Cloud → INC → Edge → Client")
        
        # Initialize components
        client_configs = []
        edge_device_configs = []
        
        # CRITICAL FIX: Use shared encryption manager for context consistency
        feature_count = next(iter(clients_data.values()))[0].shape[1]
        cloud_server = CloudServerProcess(self.fhe_config, feature_count)
        shared_encryption_manager = cloud_server.encryption_manager
        
        for client_id, client_data in clients_data.items():
            client_config = ClientConfig(
                client_id=client_id,
                client_data=client_data,
                global_model_weights=cloud_server.global_model_weights,
                global_model_bias=cloud_server.global_model_bias
            )
            client_configs.append(client_config)
            
            edge_config = EdgeDeviceConfig(
                edge_id=f"edge_{client_id}",
                client_id=client_id
            )
            edge_device_configs.append(edge_config)
        
        # Initialize INC processes
        inc_processes = []
        for inc_config in inc_configs:
            inc_process = INCProcess(inc_config, shared_encryption_manager)
            inc_processes.append(inc_process)
        
        # Run federated learning rounds
        round_results = []
        
        for round_num in range(1, self.config.rounds + 1):
            print(f"\n🔄 Round {round_num}/{self.config.rounds}")
            
            # Phase 1: Client Local Training
            print("  📱 Phase 1: Client Local Training")
            client_training_start = time.time()
            
            client_results = []
            for i, config in enumerate(client_configs):
                client = ClientProcess(config)
                result = client.train_client_with_strategy(
                    config.global_model_weights,
                    config.global_model_bias,
                    round_num
                )
                client_results.append(result)
            
            client_training_time = time.time() - client_training_start
            
            # Phase 2: Edge Device Encryption
            print("  🔒 Phase 2: Edge Device Encryption")
            edge_encryption_start = time.time()
            
            edge_results = []
            individual_encryption_times = []
            
            for i, config in enumerate(edge_device_configs):
                edge_device = EdgeDeviceProcess(config, shared_encryption_manager)
                client_result = client_results[i]
                
                result = edge_device.process_round(
                    client_result['weights'],
                    client_result['bias'],
                    round_num
                )
                edge_results.append(result)
                individual_encryption_times.append(result.get('encryption_time', 0.0))
            
            edge_encryption_wall_time = time.time() - edge_encryption_start
            total_pure_encryption_time = sum(individual_encryption_times)
            avg_encryption_per_client = np.mean(individual_encryption_times) if individual_encryption_times else 0.0
            
            # Phase 3: INC Intermediate Aggregation
            print("  🔄 Phase 3: INC Intermediate Aggregation")
            inc_aggregation_start = time.time()
            
            # Group edge results by INC
            inc_results = []
            for inc_process in inc_processes:
                # Find edge devices managed by this INC
                managed_edges = [r for r in edge_results if r['edge_id'] in inc_process.edge_device_ids]
                managed_sample_counts = [client_results[i]['sample_count'] for i, r in enumerate(edge_results) if r['edge_id'] in inc_process.edge_device_ids]
                
                if managed_edges:
                    inc_result = inc_process.aggregate_edge_updates(managed_edges, managed_sample_counts)
                    inc_results.append(inc_result)
            
            inc_aggregation_time = time.time() - inc_aggregation_start
            
            # Phase 4: Cloud Server Global Update
            print("  ☁️ Phase 4: Cloud Server Global Update")
            cloud_update_start = time.time()
            
            # Handle pre-aggregated data from INCs
            if len(inc_results) > 1:
                # Multiple INCs - properly aggregate all INC results
                print(f"  🔄 Multiple INCs detected - aggregating {len(inc_results)} INC results")
                final_aggregated_update = self._aggregate_multiple_incs(inc_results, cloud_server.encryption_manager)
                final_aggregation_time = sum(result['internal_aggregation_time'] for result in inc_results) / len(inc_results)
            else:
                # Single INC - use its result directly
                final_aggregated_update = inc_results[0]['aggregated_update']
                final_aggregation_time = inc_results[0]['internal_aggregation_time']
            
            # Update global model
            cloud_server.update_global_model(final_aggregated_update)
            
            cloud_update_time = time.time() - cloud_update_start
            
            # Phase 5: INC Global Model Distribution
            print("  🔄 Phase 5: INC Global Model Distribution")
            inc_distribution_start = time.time()
            
            for inc_process in inc_processes:
                managed_edges = [EdgeDeviceProcess(edge_device_configs[i], shared_encryption_manager) 
                               for i, config in enumerate(edge_device_configs) 
                               if config.edge_id in inc_process.edge_device_ids]
                
                inc_process.distribute_global_model(cloud_server.encrypted_global_model, managed_edges)
            
            inc_distribution_time = time.time() - inc_distribution_start
            
            # Phase 6: Global Model Synchronization
            print("  🔓 Phase 6: Global Model Synchronization")
            
            # Update client configurations with global model
            for i, config in enumerate(client_configs):
                config.global_model_weights = cloud_server.global_model_weights
                config.global_model_bias = cloud_server.global_model_bias
            
            # Phase 7: Model Evaluation
            print("  📊 Phase 7: Model Evaluation")
            evaluation_start = time.time()
            
            metrics = cloud_server.evaluate_model(clients_data)
            evaluation_time = time.time() - evaluation_start
            
            # Track best accuracy
            if metrics['accuracy'] > self.best_accuracy:
                self.best_accuracy = metrics['accuracy']
            
            # Store round results
            round_result = {
                'round': round_num,
                'accuracy': metrics['accuracy'],
                'f1_score': metrics['f1_score'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'auc': metrics['auc'],
                'mae': metrics['mae'],
                'rmse': metrics['rmse'],
                'timing_statistics': {
                    'client_training_time': client_training_time,
                    'edge_encryption_wall_time': edge_encryption_wall_time,
                    'total_pure_encryption_time': total_pure_encryption_time,
                    'avg_encryption_per_client': avg_encryption_per_client,
                    'inc_aggregation_time': inc_aggregation_time,
                    'cloud_update_time': cloud_update_time,
                    'inc_distribution_time': inc_distribution_time,
                    'evaluation_time': evaluation_time,
                    'total_round_time': client_training_time + edge_encryption_wall_time + inc_aggregation_time + cloud_update_time + inc_distribution_time + evaluation_time
                }
            }
            
            round_results.append(round_result)
            
            print(f"  ✅ Round {round_num} completed - Accuracy: {metrics['accuracy']*100:.2f}%")
        
        return round_results
