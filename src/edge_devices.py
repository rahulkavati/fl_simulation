"""
Edge Device Simulation for FHE CKKS Research
Simulates realistic smartwatch edge devices with local processing capabilities
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import json

@dataclass
class EdgeDeviceConfig:
    """Configuration for edge device simulation"""
    device_id: str
    device_type: str = "smartwatch"  # smartwatch, fitness_tracker, health_monitor
    processing_power: float = 1.0    # Relative processing capability
    battery_level: float = 100.0     # Battery percentage
    storage_capacity: int = 1000     # Local storage capacity (MB)
    network_latency: float = 0.1     # Network latency in seconds
    encryption_capability: bool = True
    fhe_support: bool = True

class EdgeDevice:
    """
    Simulates a realistic edge device (smartwatch) with:
    - Local data processing
    - FHE encryption capabilities
    - Network communication
    - Battery and resource management
    """
    
    def __init__(self, config: EdgeDeviceConfig):
        self.config = config
        self.local_data: List[Dict] = []
        self.local_model = None
        self.encrypted_updates: List[np.ndarray] = []
        self.communication_log: List[Dict] = []
        self.resource_usage: Dict[str, float] = {
            'cpu_usage': 0.0,
            'memory_usage': 0.0,
            'battery_drain': 0.0,
            'network_usage': 0.0
        }
        
    def load_smartwatch_data(self, csv_data: pd.DataFrame, participant_id: int) -> None:
        """
        Load CSV data as if it's coming from smartwatch sensors
        Simulates realistic sensor data collection
        """
        print(f"📱 {self.config.device_id}: Loading smartwatch sensor data...")
        
        # Filter data for this participant (simulating individual device)
        device_data = csv_data[csv_data['participant_id'] == participant_id].copy()
        
        # Simulate sensor data collection with realistic timestamps
        device_data['timestamp'] = pd.date_range(
            start=datetime.now() - timedelta(days=len(device_data)),
            periods=len(device_data),
            freq='1H'
        )
        
        # Add sensor noise and realistic variations
        device_data['heart_rate_noise'] = np.random.normal(0, 2, len(device_data))
        device_data['steps_noise'] = np.random.normal(0, 50, len(device_data))
        
        # Simulate sensor data quality
        device_data['sensor_quality'] = np.random.uniform(0.85, 0.99, len(device_data))
        
        # Convert to local storage format
        self.local_data = device_data.to_dict('records')
        
        print(f"  ✅ Loaded {len(self.local_data)} sensor readings")
        print(f"  📊 Data range: {device_data['date'].min()} to {device_data['date'].max()}")
        print(f"  🔋 Battery level: {self.config.battery_level:.1f}%")
        
    def train_local_model(self, model_params: Dict) -> Dict[str, Any]:
        """
        Train local model on device (simulating on-device ML)
        """
        print(f"📱 {self.config.device_id}: Training local model on device...")
        
        # Simulate device resource usage
        self.resource_usage['cpu_usage'] = 75.0
        self.resource_usage['memory_usage'] = 60.0
        self.resource_usage['battery_drain'] = 5.0
        
        # Update battery level
        self.config.battery_level -= self.resource_usage['battery_drain']
        
        # Convert local data to training format
        if not self.local_data:
            raise ValueError("No local data available for training")
        
        # Create training data from local sensor readings
        df_local = pd.DataFrame(self.local_data)
        
        # Feature engineering (simulating on-device processing)
        features = [
            'age', 'height_cm', 'weight_kg', 'bmi', 'avg_heart_rate',
            'resting_heart_rate', 'hours_sleep', 'stress_level', 'daily_steps',
            'calories_burned', 'hydration_level'
        ]
        
        X = df_local[features].values
        y = df_local['health_status'].values
        
        # Train local model (simulating on-device ML)
        from sklearn.linear_model import LogisticRegression
        self.local_model = LogisticRegression(**model_params)
        self.local_model.fit(X, y)
        
        # Simulate training time based on device capability
        training_time = len(X) * 0.001 / self.config.processing_power
        
        print(f"  ✅ Local model trained on {len(X)} samples")
        print(f"  ⏱️  Training time: {training_time:.3f}s")
        print(f"  🔋 Battery remaining: {self.config.battery_level:.1f}%")
        
        return {
            'model_weights': self.local_model.coef_.flatten(),
            'model_bias': self.local_model.intercept_[0],
            'sample_count': len(X),
            'training_time': training_time,
            'device_id': self.config.device_id,
            'battery_level': self.config.battery_level
        }
    
    def encrypt_model_update(self, model_update: np.ndarray, fhe_encryption) -> Tuple[np.ndarray, float]:
        """
        Encrypt model update using FHE CKKS (simulating on-device encryption)
        """
        print(f"📱 {self.config.device_id}: Encrypting model update...")
        
        # Simulate encryption resource usage
        self.resource_usage['cpu_usage'] = 90.0
        self.resource_usage['memory_usage'] = 80.0
        self.resource_usage['battery_drain'] = 8.0
        
        # Update battery level
        self.config.battery_level -= self.resource_usage['battery_drain']
        
        # Perform FHE encryption
        encryption_start = time.time()
        encrypted_update, encryption_time = fhe_encryption.simulate_fhe_ckks_encryption(model_update)
        encryption_time = time.time() - encryption_start
        
        # Store encrypted update
        self.encrypted_updates.append(encrypted_update)
        
        print(f"  🔒 Model update encrypted")
        print(f"  ⏱️  Encryption time: {encryption_time:.3f}s")
        print(f"  🔋 Battery remaining: {self.config.battery_level:.1f}%")
        
        return encrypted_update, encryption_time
    
    def send_to_server(self, encrypted_update: np.ndarray) -> Dict[str, Any]:
        """
        Send encrypted update to server (simulating network communication)
        """
        print(f"📱 {self.config.device_id}: Sending encrypted update to server...")
        
        # Simulate network communication
        network_start = time.time()
        time.sleep(self.config.network_latency)  # Simulate network delay
        network_time = time.time() - network_start
        
        # Simulate network resource usage
        self.resource_usage['network_usage'] += len(encrypted_update) * 8 / 1024  # KB
        self.resource_usage['battery_drain'] += 2.0
        self.config.battery_level -= 2.0
        
        # Log communication
        communication_log = {
            'timestamp': datetime.now().isoformat(),
            'action': 'send_encrypted_update',
            'data_size': len(encrypted_update),
            'network_time': network_time,
            'battery_level': self.config.battery_level
        }
        self.communication_log.append(communication_log)
        
        print(f"  📡 Encrypted update sent to server")
        print(f"  ⏱️  Network time: {network_time:.3f}s")
        print(f"  📊 Data size: {len(encrypted_update)} bytes")
        print(f"  🔋 Battery remaining: {self.config.battery_level:.1f}%")
        
        return communication_log
    
    def receive_global_model(self, encrypted_global_model) -> Dict[str, Any]:
        """
        Receive encrypted global model from server
        """
        print(f"📱 {self.config.device_id}: Receiving encrypted global model...")
        
        # Simulate network communication
        network_start = time.time()
        time.sleep(self.config.network_latency)
        network_time = time.time() - network_start
        
        # Simulate resource usage
        self.resource_usage['network_usage'] += 1024  # KB for global model
        self.resource_usage['battery_drain'] += 3.0
        self.config.battery_level -= 3.0
        
        # Log communication
        communication_log = {
            'timestamp': datetime.now().isoformat(),
            'action': 'receive_global_model',
            'network_time': network_time,
            'battery_level': self.config.battery_level
        }
        self.communication_log.append(communication_log)
        
        print(f"  📡 Encrypted global model received")
        print(f"  ⏱️  Network time: {network_time:.3f}s")
        print(f"  🔋 Battery remaining: {self.config.battery_level:.1f}%")
        
        return communication_log
    
    def decrypt_and_update(self, encrypted_global_model) -> Dict[str, Any]:
        """
        Decrypt global model and update local model (simulating on-device decryption)
        """
        print(f"📱 {self.config.device_id}: Decrypting global model and updating...")
        
        # Simulate decryption resource usage
        self.resource_usage['cpu_usage'] = 85.0
        self.resource_usage['memory_usage'] = 70.0
        self.resource_usage['battery_drain'] = 6.0
        
        # Update battery level
        self.config.battery_level -= self.resource_usage['battery_drain']
        
        # Decrypt global model
        decryption_start = time.time()
        decrypted_weights, decrypted_bias = encrypted_global_model.decrypt_for_evaluation()
        decryption_time = time.time() - decryption_start
        
        # Update local model
        if self.local_model is not None:
            self.local_model.coef_ = decrypted_weights.reshape(1, -1)
            self.local_model.intercept_ = np.array([decrypted_bias])
            self.local_model.classes_ = np.array([0, 1])
        
        print(f"  🔓 Global model decrypted and local model updated")
        print(f"  ⏱️  Decryption time: {decryption_time:.3f}s")
        print(f"  🔋 Battery remaining: {self.config.battery_level:.1f}%")
        
        return {
            'decryption_time': decryption_time,
            'battery_level': self.config.battery_level,
            'device_id': self.config.device_id
        }
    
    def get_device_status(self) -> Dict[str, Any]:
        """Get current device status for monitoring"""
        return {
            'device_id': self.config.device_id,
            'device_type': self.config.device_type,
            'battery_level': self.config.battery_level,
            'local_data_count': len(self.local_data),
            'encrypted_updates_count': len(self.encrypted_updates),
            'communication_log_count': len(self.communication_log),
            'resource_usage': self.resource_usage.copy(),
            'fhe_support': self.config.fhe_support
        }

class EdgeDeviceManager:
    """
    Manages multiple edge devices for federated learning simulation
    """
    
    def __init__(self, device_configs: List[EdgeDeviceConfig]):
        self.devices = {config.device_id: EdgeDevice(config) for config in device_configs}
        self.network_topology = self._create_network_topology()
        
    def _create_network_topology(self) -> Dict[str, List[str]]:
        """Create realistic network topology"""
        device_ids = list(self.devices.keys())
        return {
            'server': device_ids,
            'devices': {device_id: ['server'] for device_id in device_ids}
        }
    
    def load_data_to_devices(self, csv_data: pd.DataFrame, participant_mapping: Dict[str, int]):
        """Load CSV data to devices as if from smartwatch sensors"""
        print("📱 Loading smartwatch data to edge devices...")
        
        for device_id, participant_id in participant_mapping.items():
            if device_id in self.devices:
                self.devices[device_id].load_smartwatch_data(csv_data, participant_id)
        
        print(f"✅ Data loaded to {len(self.devices)} edge devices")
    
    def get_all_device_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all devices"""
        return {device_id: device.get_device_status() 
                for device_id, device in self.devices.items()}
    
    def simulate_network_communication(self, from_device: str, to_device: str, data: Any) -> float:
        """Simulate network communication between devices"""
        if from_device in self.devices and to_device in self.network_topology.get('devices', {}).get(from_device, []):
            # Simulate network latency
            latency = np.random.uniform(0.05, 0.2)  # 50-200ms
            time.sleep(latency)
            return latency
        return 0.0
