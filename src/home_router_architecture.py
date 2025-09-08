"""
Realistic Edge Device Architecture with Home Router Encryption
Smartwatch → Home Router (Encryption) → Server → Home Router (Decryption) → Smartwatch
"""

import time
import numpy as np
import pandas as pd
import tenseal as ts
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import json

@dataclass
class SmartwatchConfig:
    """Configuration for smartwatch device"""
    device_id: str
    device_type: str = "smartwatch"
    battery_level: float = 100.0
    processing_power: float = 1.0
    storage_capacity: int = 100  # MB
    network_latency: float = 0.05  # Local network latency

@dataclass
class HomeRouterConfig:
    """Configuration for home router with FHE capabilities"""
    router_id: str
    fhe_capability: bool = True
    processing_power: float = 5.0  # Higher than smartwatch
    encryption_speed: float = 1.0
    network_latency: float = 0.1  # Internet latency
    connected_devices: List[str] = None

class Smartwatch:
    """
    Simulates a realistic smartwatch device
    - Collects sensor data
    - Trains local model
    - Sends raw model updates to home router
    - Receives decrypted global model from home router
    """
    
    def __init__(self, config: SmartwatchConfig):
        self.config = config
        self.local_data: List[Dict] = []
        self.local_model = None
        self.resource_usage: Dict[str, float] = {
            'cpu_usage': 0.0,
            'memory_usage': 0.0,
            'battery_drain': 0.0
        }
        
    def load_sensor_data(self, csv_data: pd.DataFrame, participant_id: int) -> None:
        """Load CSV data as smartwatch sensor data"""
        print(f"⌚ {self.config.device_id}: Loading sensor data...")
        
        # Filter data for this participant
        device_data = csv_data[csv_data['participant_id'] == participant_id].copy()
        
        # Simulate sensor data collection
        device_data['timestamp'] = pd.date_range(
            start=datetime.now() - timedelta(days=len(device_data)),
            periods=len(device_data),
            freq='1H'
        )
        
        # Add sensor noise
        device_data['heart_rate_noise'] = np.random.normal(0, 2, len(device_data))
        device_data['steps_noise'] = np.random.normal(0, 50, len(device_data))
        
        self.local_data = device_data.to_dict('records')
        
        print(f"  ✅ Loaded {len(self.local_data)} sensor readings")
        print(f"  🔋 Battery: {self.config.battery_level:.1f}%")
        
    def train_local_model(self, model_params: Dict) -> Dict[str, Any]:
        """Train local model on smartwatch"""
        print(f"⌚ {self.config.device_id}: Training local model...")
        
        # Simulate device resource usage
        self.resource_usage['cpu_usage'] = 80.0
        self.resource_usage['memory_usage'] = 60.0
        self.resource_usage['battery_drain'] = 8.0
        
        # Update battery
        self.config.battery_level -= self.resource_usage['battery_drain']
        
        # Convert to training format
        if not self.local_data:
            raise ValueError("No sensor data available")
        
        df_local = pd.DataFrame(self.local_data)
        features = [
            'age', 'height_cm', 'weight_kg', 'bmi', 'avg_heart_rate',
            'resting_heart_rate', 'hours_sleep', 'stress_level', 'daily_steps',
            'calories_burned', 'hydration_level'
        ]
        
        X = df_local[features].values
        y = df_local['health_status'].values
        
        # Train local model
        from sklearn.linear_model import LogisticRegression
        self.local_model = LogisticRegression(**model_params)
        self.local_model.fit(X, y)
        
        # Simulate training time
        training_time = len(X) * 0.002 / self.config.processing_power
        
        print(f"  ✅ Local model trained on {len(X)} samples")
        print(f"  ⏱️  Training time: {training_time:.3f}s")
        print(f"  🔋 Battery: {self.config.battery_level:.1f}%")
        
        return {
            'model_weights': self.local_model.coef_.flatten(),
            'model_bias': self.local_model.intercept_[0],
            'sample_count': len(X),
            'training_time': training_time,
            'device_id': self.config.device_id,
            'battery_level': self.config.battery_level
        }
    
    def send_model_update_to_router(self, model_update: np.ndarray) -> Dict[str, Any]:
        """Send raw model update to home router (no encryption on device)"""
        print(f"⌚ {self.config.device_id}: Sending model update to home router...")
        
        # Simulate local network communication
        network_start = time.time()
        time.sleep(self.config.network_latency)
        network_time = time.time() - network_start
        
        # Simulate resource usage
        self.resource_usage['battery_drain'] += 1.0
        self.config.battery_level -= 1.0
        
        print(f"  📡 Raw model update sent to home router")
        print(f"  ⏱️  Local network time: {network_time:.3f}s")
        print(f"  📊 Data size: {len(model_update)} parameters")
        print(f"  🔋 Battery: {self.config.battery_level:.1f}%")
        
        return {
            'timestamp': datetime.now().isoformat(),
            'action': 'send_to_router',
            'data_size': len(model_update),
            'network_time': network_time,
            'battery_level': self.config.battery_level
        }
    
    def receive_global_model_from_router(self, decrypted_global_model: Dict) -> Dict[str, Any]:
        """Receive decrypted global model from home router"""
        print(f"⌚ {self.config.device_id}: Receiving global model from home router...")
        
        # Simulate local network communication
        network_start = time.time()
        time.sleep(self.config.network_latency)
        network_time = time.time() - network_start
        
        # Simulate resource usage
        self.resource_usage['battery_drain'] += 1.0
        self.config.battery_level -= 1.0
        
        # Update local model
        if self.local_model is not None:
            self.local_model.coef_ = decrypted_global_model['weights'].reshape(1, -1)
            self.local_model.intercept_ = np.array([decrypted_global_model['bias']])
            self.local_model.classes_ = np.array([0, 1])
        
        print(f"  📡 Decrypted global model received from home router")
        print(f"  ⏱️  Local network time: {network_time:.3f}s")
        print(f"  🔋 Battery: {self.config.battery_level:.1f}%")
        
        return {
            'timestamp': datetime.now().isoformat(),
            'action': 'receive_from_router',
            'network_time': network_time,
            'battery_level': self.config.battery_level
        }

class HomeRouter:
    """
    Simulates a home router with FHE encryption capabilities
    - Receives raw model updates from smartwatches
    - Encrypts model updates using FHE CKKS
    - Sends encrypted updates to server
    - Receives encrypted global model from server
    - Decrypts global model for local devices
    """
    
    def __init__(self, config: HomeRouterConfig):
        self.config = config
        self.connected_devices: List[str] = config.connected_devices or []
        self.fhe_encryption = None
        self.encrypted_updates: List[np.ndarray] = []
        self.communication_log: List[Dict] = []
        self.resource_usage: Dict[str, float] = {
            'cpu_usage': 0.0,
            'memory_usage': 0.0,
            'encryption_load': 0.0
        }
        
    def initialize_fhe_encryption(self, fhe_config) -> None:
        """Initialize FHE encryption on home router"""
        print(f"🏠 {self.config.router_id}: Initializing FHE encryption...")
        
        from src.real_fhe_ckks import RealFHEEncryption, RealFHEConfig, RealEncryptedModel
        self.fhe_encryption = RealFHEEncryption(fhe_config)
        
        print(f"  ✅ Real FHE CKKS encryption initialized")
        print(f"  🔒 Ready to encrypt model updates")
        
    def receive_model_update_from_device(self, device_id: str, model_update: np.ndarray) -> Dict[str, Any]:
        """Receive raw model update from smartwatch"""
        print(f"🏠 {self.config.router_id}: Receiving model update from {device_id}...")
        
        # Simulate local network communication
        network_start = time.time()
        time.sleep(0.01)  # Very fast local network
        network_time = time.time() - network_start
        
        print(f"  📡 Raw model update received from {device_id}")
        print(f"  ⏱️  Local network time: {network_time:.3f}s")
        print(f"  📊 Data size: {len(model_update)} parameters")
        
        return {
            'timestamp': datetime.now().isoformat(),
            'action': 'receive_from_device',
            'device_id': device_id,
            'data_size': len(model_update),
            'network_time': network_time
        }
    
    def encrypt_model_update(self, model_update: np.ndarray) -> Tuple[ts.CKKSVector, float]:
        """Encrypt model update using real FHE CKKS on home router"""
        print(f"🏠 {self.config.router_id}: Encrypting model update with REAL FHE CKKS...")
        
        # Simulate encryption resource usage
        self.resource_usage['cpu_usage'] = 90.0
        self.resource_usage['memory_usage'] = 80.0
        self.resource_usage['encryption_load'] += 1.0
        
        # Perform REAL FHE encryption
        encrypted_update, encryption_time = self.fhe_encryption.encrypt_model_update(model_update)
        
        # Store encrypted update
        self.encrypted_updates.append(encrypted_update)
        
        print(f"  🔒 Model update encrypted using REAL FHE CKKS")
        print(f"  ⏱️  Encryption time: {encryption_time:.3f}s")
        print(f"  🔒 Encryption load: {self.resource_usage['encryption_load']:.1f}")
        print(f"  📊 Ciphertext size: {len(encrypted_update.serialize())} bytes")
        
        return encrypted_update, encryption_time
    
    def send_encrypted_update_to_server(self, encrypted_update: ts.CKKSVector) -> Dict[str, Any]:
        """Send encrypted update to server"""
        print(f"🏠 {self.config.router_id}: Sending encrypted update to server...")
        
        # Simulate internet communication
        network_start = time.time()
        time.sleep(self.config.network_latency)
        network_time = time.time() - network_start
        
        # Get ciphertext size
        ciphertext_size = len(encrypted_update.serialize())
        
        # Log communication
        communication_log = {
            'timestamp': datetime.now().isoformat(),
            'action': 'send_to_server',
            'data_size': ciphertext_size,
            'network_time': network_time,
            'encryption_load': self.resource_usage['encryption_load']
        }
        self.communication_log.append(communication_log)
        
        print(f"  📡 Encrypted update sent to server")
        print(f"  ⏱️  Internet network time: {network_time:.3f}s")
        print(f"  📊 Encrypted data size: {ciphertext_size} bytes")
        
        return communication_log
    
    def receive_encrypted_global_model_from_server(self, encrypted_global_model) -> Dict[str, Any]:
        """Receive encrypted global model from server"""
        print(f"🏠 {self.config.router_id}: Receiving encrypted global model from server...")
        
        # Simulate internet communication
        network_start = time.time()
        time.sleep(self.config.network_latency)
        network_time = time.time() - network_start
        
        # Log communication
        communication_log = {
            'timestamp': datetime.now().isoformat(),
            'action': 'receive_from_server',
            'network_time': network_time,
            'encryption_load': self.resource_usage['encryption_load']
        }
        self.communication_log.append(communication_log)
        
        print(f"  📡 Encrypted global model received from server")
        print(f"  ⏱️  Internet network time: {network_time:.3f}s")
        
        return communication_log
    
    def decrypt_global_model(self, encrypted_global_model) -> Dict[str, Any]:
        """Decrypt global model for local devices"""
        print(f"🏠 {self.config.router_id}: Decrypting global model...")
        
        # Simulate decryption resource usage
        self.resource_usage['cpu_usage'] = 95.0
        self.resource_usage['memory_usage'] = 85.0
        self.resource_usage['encryption_load'] += 1.0
        
        # Decrypt global model
        decryption_start = time.time()
        decrypted_weights, decrypted_bias = encrypted_global_model.decrypt_for_evaluation()
        decryption_time = time.time() - decryption_start
        
        print(f"  🔓 Global model decrypted")
        print(f"  ⏱️  Decryption time: {decryption_time:.3f}s")
        print(f"  🔒 Encryption load: {self.resource_usage['encryption_load']:.1f}")
        
        return {
            'weights': decrypted_weights,
            'bias': decrypted_bias,
            'decryption_time': decryption_time,
            'encryption_load': self.resource_usage['encryption_load']
        }
    
    def send_decrypted_global_model_to_devices(self, decrypted_global_model: Dict) -> Dict[str, Any]:
        """Send decrypted global model to connected devices"""
        print(f"🏠 {self.config.router_id}: Sending decrypted global model to devices...")
        
        # Simulate local network broadcast
        network_start = time.time()
        time.sleep(0.01)  # Very fast local network
        network_time = time.time() - network_start
        
        print(f"  📡 Decrypted global model sent to {len(self.connected_devices)} devices")
        print(f"  ⏱️  Local network time: {network_time:.3f}s")
        
        return {
            'timestamp': datetime.now().isoformat(),
            'action': 'broadcast_to_devices',
            'device_count': len(self.connected_devices),
            'network_time': network_time
        }
    
    def get_router_status(self) -> Dict[str, Any]:
        """Get current router status"""
        return {
            'router_id': self.config.router_id,
            'fhe_capability': self.config.fhe_capability,
            'connected_devices': len(self.connected_devices),
            'encrypted_updates_count': len(self.encrypted_updates),
            'communication_log_count': len(self.communication_log),
            'resource_usage': self.resource_usage.copy(),
            'encryption_load': self.resource_usage['encryption_load']
        }

class HomeRouterManager:
    """
    Manages home routers and their connected smartwatch devices
    """
    
    def __init__(self, router_configs: List[HomeRouterConfig]):
        self.routers = {config.router_id: HomeRouter(config) for config in router_configs}
        self.device_to_router_mapping: Dict[str, str] = {}
        
    def assign_devices_to_routers(self, device_ids: List[str]) -> None:
        """Assign smartwatch devices to home routers"""
        print("🏠 Assigning devices to home routers...")
        
        router_ids = list(self.routers.keys())
        
        for i, device_id in enumerate(device_ids):
            router_id = router_ids[i % len(router_ids)]
            self.device_to_router_mapping[device_id] = router_id
            self.routers[router_id].connected_devices.append(device_id)
            
            print(f"  📱 {device_id} → 🏠 {router_id}")
        
        print(f"✅ {len(device_ids)} devices assigned to {len(router_ids)} routers")
    
    def get_router_for_device(self, device_id: str) -> str:
        """Get the router ID for a specific device"""
        return self.device_to_router_mapping.get(device_id)
    
    def get_all_router_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all routers"""
        return {router_id: router.get_router_status() 
                for router_id, router in self.routers.items()}

def main():
    """Test the home router architecture"""
    print("🏠 Testing Home Router Architecture with FHE Encryption")
    
    # Create router configurations
    router_configs = [
        HomeRouterConfig(
            router_id="home_router_1",
            fhe_capability=True,
            processing_power=5.0,
            connected_devices=[]
        ),
        HomeRouterConfig(
            router_id="home_router_2", 
            fhe_capability=True,
            processing_power=5.0,
            connected_devices=[]
        )
    ]
    
    # Create router manager
    router_manager = HomeRouterManager(router_configs)
    
    # Assign devices to routers
    device_ids = [f"smartwatch_{i}" for i in range(6)]
    router_manager.assign_devices_to_routers(device_ids)
    
    # Show router status
    router_status = router_manager.get_all_router_status()
    for router_id, status in router_status.items():
        print(f"🏠 {router_id}: {status['connected_devices']} devices, "
              f"FHE: {status['fhe_capability']}")

if __name__ == "__main__":
    main()
