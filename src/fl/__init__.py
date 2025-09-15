"""
Federated Learning Core Module

This module provides the core functionality for federated learning including:
- Data processing and client dataset creation
- Model evaluation and metrics calculation
- Configuration management
- Client statistics logging

Key Components:
1. FLConfig: Configuration for federated learning parameters
2. DataProcessor: Handles data loading, preprocessing, and client creation
3. ModelEvaluator: Provides model evaluation capabilities

Author: AI Assistant
Date: 2025
"""

import os
import json
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from src.utils.client_statistics import create_client_logger


@dataclass
class FLConfig:
    """
    Federated Learning Configuration
    
    This class defines the parameters for federated learning:
    - rounds: Number of federated learning rounds
    - clients: Number of clients to simulate
    - min_samples_per_client: Minimum samples required per client
    - test_size: Fraction of data to use for testing
    - random_state: Random seed for reproducibility
    - model_params: Parameters for the machine learning model
    """
    rounds: int = 5
    clients: int = 10
    min_samples_per_client: int = 50
    test_size: float = 0.2
    random_state: int = 42
    model_params: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize default model parameters if not provided"""
        if self.model_params is None:
            self.model_params = {
                'penalty': 'l2',
                'C': 1.0,
                'fit_intercept': True,
                'solver': 'lbfgs',
                'max_iter': 10000,
                'class_weight': 'balanced'
            }


class DataProcessor:
    """
    Handles data loading, preprocessing, and client dataset creation
    
    This class provides comprehensive data processing capabilities:
    - Load and preprocess health fitness data
    - Create client datasets from participant data
    - Handle one-class clients without exclusion
    - Scale data for federated learning
    - Log client statistics and distributions
    """
    
    def __init__(self, config: FLConfig):
        """
        Initialize data processor with configuration
        
        Args:
            config: Federated learning configuration
        """
        self.config = config
        self.scaler = None
        self.feature_columns = None
        
        # Initialize client statistics logger
        self.client_logger = create_client_logger("logs/client_statistics.log")
    
    def load_health_fitness_data(self, data_path: str = 'data/health_fitness_data.csv') -> Tuple[pd.DataFrame, List[str]]:
        """
        Load and preprocess the health fitness dataset
        
        This method loads the health fitness data and performs preprocessing:
        - Loads data from CSV file
        - Creates binary health status from fitness level
        - Selects and engineers features
        - Handles categorical variables
        - Creates derived features
        
        Args:
            data_path: Path to the health fitness data CSV file
            
        Returns:
            Tuple[pd.DataFrame, List[str]]: Processed dataframe and feature columns
        """
        print("Loading Health Fitness Dataset...")
        
        if not os.path.exists(data_path):
            print(f"Health data not found at {data_path}")
            return None, None
        
        # Load the dataset
        df = pd.read_csv(data_path)
        print(f"Loaded {len(df):,} records from {df['participant_id'].nunique()} participants")
        
        # Clean the data
        print("Preprocessing data...")
        df = df.copy()
        df['health_condition'] = df['health_condition'].fillna('None')
        
        # Convert fitness_level to binary health status using median threshold
        fitness_threshold = df['fitness_level'].median()
        df['health_status'] = (df['fitness_level'] >= fitness_threshold).astype(int)
        
        # Print health status distribution
        print(f"Health Status Distribution:")
        print(f"  Unhealthy (0): {len(df[df['health_status'] == 0]):,} ({len(df[df['health_status'] == 0])/len(df)*100:.1f}%)")
        print(f"  Healthy (1): {len(df[df['health_status'] == 1]):,} ({len(df[df['health_status'] == 1])/len(df)*100:.1f}%)")
        print(f"  Fitness Threshold: {fitness_threshold:.2f}")
        
        # Select basic features for federated learning
        self.feature_columns = [
            'age', 'height_cm', 'weight_kg', 'bmi',
            'avg_heart_rate', 'resting_heart_rate', 
            'blood_pressure_systolic', 'blood_pressure_diastolic',
            'hours_sleep', 'stress_level', 'daily_steps', 
            'calories_burned', 'hydration_level'
        ]
        
        # Add derived features for better model performance
        df['steps_per_calorie'] = df['daily_steps'] / (df['calories_burned'] + 1)
        df['sleep_efficiency'] = df['hours_sleep'] / (df['stress_level'] + 1)
        df['cardio_health'] = (df['resting_heart_rate'] <= 70).astype(int)
        df['high_activity'] = (df['daily_steps'] >= 10000).astype(int)
        df['good_sleep'] = (df['hours_sleep'] >= 7.0).astype(int)
        df['low_stress'] = (df['stress_level'] <= 5.0).astype(int)
        
        # Add derived features to feature list
        self.feature_columns.extend([
            'steps_per_calorie', 'sleep_efficiency', 'cardio_health',
            'high_activity', 'good_sleep', 'low_stress'
        ])
        
        # Handle categorical variables by encoding them
        df['gender_encoded'] = LabelEncoder().fit_transform(df['gender'])
        df['intensity_encoded'] = LabelEncoder().fit_transform(df['intensity'])
        
        self.feature_columns.extend(['gender_encoded', 'intensity_encoded'])
        
        print(f"Using {len(self.feature_columns)} features for federated learning")
        
        return df, self.feature_columns
    
    def create_client_datasets(self, df: pd.DataFrame) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Create client datasets from the health fitness data
        
        This method creates client datasets by:
        - Selecting participants for federated learning
        - Creating individual client datasets
        - Handling one-class clients without exclusion
        - Logging client statistics and distributions
        - Saving client data to CSV files for inspection
        
        Args:
            df: Processed health fitness dataframe
            
        Returns:
            Dict[str, Tuple[np.ndarray, np.ndarray]]: Client datasets with features and labels
        """
        print(f"Creating {self.config.clients} client datasets...")
        
        # Get unique participants
        participants = df['participant_id'].unique()
        
        # Select subset of participants for federated learning
        if self.config.clients < len(participants):
            np.random.seed(self.config.random_state)
            selected_participants = np.random.choice(
                participants, self.config.clients, replace=False
            )
        else:
            selected_participants = participants[:self.config.clients]
        
        clients_data = {}
        
        # Create clients directory if it doesn't exist
        os.makedirs("data/clients", exist_ok=True)
        
        # Clear existing client files
        print("Clearing existing client files...")
        existing_files = [f for f in os.listdir("data/clients") if f.startswith("client_") and f.endswith(".csv")]
        for file in existing_files:
            os.remove(os.path.join("data/clients", file))
        print(f"Removed {len(existing_files)} existing client files")
        
        # Create client datasets
        for i, participant_id in enumerate(selected_participants):
            # Get data for this participant
            participant_data = df[df['participant_id'] == participant_id]
            
            if len(participant_data) >= self.config.min_samples_per_client:
                # Extract features and labels
                X = participant_data[self.feature_columns].values
                y = participant_data['health_status'].values
                
                # Handle any remaining NaN values
                X = np.nan_to_num(X, nan=0.0)
                
                # Include all clients, even those with one class
                unique_classes = np.unique(y)
                clients_data[f"client_{i}"] = (X, y)
                
                # Log client creation with statistics
                class_distribution = {int(cls): int(np.sum(y == cls)) for cls in unique_classes}
                is_one_class = len(unique_classes) < 2
                
                self.client_logger.log_client_creation(
                    client_id=f"client_{i}",
                    sample_count=len(participant_data),
                    class_distribution=class_distribution,
                    is_one_class=is_one_class,
                    strategy_used="combined" if is_one_class else None
                )
                
                # Log one-class clients for monitoring
                if is_one_class:
                    print(f"    ⚠️  One-class client detected: {unique_classes[0]} only")
                
                # Create comprehensive client dataset for visualization
                client_df = participant_data.copy()
                client_df['health_status'] = y  # Ensure we have the binary health status
                
                # Add derived features that were computed
                client_df['steps_per_calorie'] = client_df['daily_steps'] / (client_df['calories_burned'] + 1)
                client_df['sleep_efficiency'] = client_df['hours_sleep'] / (client_df['stress_level'] + 1)
                client_df['cardio_health'] = (client_df['resting_heart_rate'] <= 70).astype(int)
                client_df['high_activity'] = (client_df['daily_steps'] >= 10000).astype(int)
                client_df['good_sleep'] = (client_df['hours_sleep'] >= 7.0).astype(int)
                client_df['low_stress'] = (client_df['stress_level'] <= 5.0).astype(int)
                
                # Add encoded categorical variables
                client_df['gender_encoded'] = LabelEncoder().fit_transform(client_df['gender'])
                client_df['intensity_encoded'] = LabelEncoder().fit_transform(client_df['intensity'])
                
                # Save client dataset to CSV for manual inspection
                client_filename = f"data/clients/client_{i}.csv"
                client_df.to_csv(client_filename, index=False)
                
                # Print detailed client information
                print(f"  Client {i} (Participant {participant_id}):")
                print(f"    - Samples: {len(participant_data)}")
                print(f"    - Classes: {len(unique_classes)} (Health Status: {np.bincount(y)})")
                print(f"    - Age Range: {participant_data['age'].min()}-{participant_data['age'].max()}")
                print(f"    - Gender: {participant_data['gender'].iloc[0]}")
                print(f"    - Fitness Level: {participant_data['fitness_level'].mean():.2f} ± {participant_data['fitness_level'].std():.2f}")
                print(f"    - Health Status Distribution:")
                print(f"      * Unhealthy (0): {np.sum(y == 0)} ({np.sum(y == 0)/len(y)*100:.1f}%)")
                print(f"      * Healthy (1): {np.sum(y == 1)} ({np.sum(y == 1)/len(y)*100:.1f}%)")
                print(f"    - Saved to: {client_filename}")
            else:
                print(f"  Client {i}: Insufficient data ({len(participant_data)} < {self.config.min_samples_per_client})")
                
                # Log client exclusion
                self.client_logger.log_client_exclusion(
                    client_id=f"client_{i}",
                    reason="insufficient_data",
                    sample_count=len(participant_data)
                )
        
        print(f"\nCreated {len(clients_data)} client datasets")
        print(f"Client datasets saved to: data/clients/")
        print(f"You can now manually inspect each client's data by opening the CSV files")
        
        # Print and save client statistics
        self.client_logger.print_summary()
        self.client_logger.save_statistics("logs/client_creation_statistics.json")
        
        return clients_data
    
    def scale_client_data(self, clients_data: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Scale client data using global statistics
        
        This method scales all client data using a global scaler fitted on all data.
        This ensures consistent scaling across all clients while maintaining privacy.
        
        Args:
            clients_data: Dictionary of client datasets
            
        Returns:
            Dict[str, Tuple[np.ndarray, np.ndarray]]: Scaled client datasets
        """
        print("Scaling client data...")
        
        # Collect all data for scaling
        all_X = []
        for X, y in clients_data.values():
            all_X.append(X)
        
        # Fit scaler on all data
        self.scaler = StandardScaler()
        self.scaler.fit(np.vstack(all_X))
        
        # Apply scaling to each client's data
        scaled_clients = {}
        for client_id, (X, y) in clients_data.items():
            X_scaled = self.scaler.transform(X)
            scaled_clients[client_id] = (X_scaled, y)
        
        print(f"Scaled data for {len(scaled_clients)} clients")
        return scaled_clients


class ModelEvaluator:
    """
    Handles model evaluation and metrics calculation
    
    This class provides evaluation capabilities for federated learning models:
    - Encrypted model evaluation (with decryption only for evaluation)
    - Comprehensive metrics calculation
    - Temporary model creation for evaluation
    """
    
    @staticmethod
    def evaluate_encrypted_model(global_model, clients_data: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> Dict[str, float]:
        """
        Evaluate encrypted model - decrypt ONLY for evaluation
        
        This method evaluates an encrypted model by:
        - Decrypting the model only for evaluation purposes
        - Creating a temporary model for prediction
        - Evaluating on all client data
        - Returning comprehensive metrics
        
        Args:
            global_model: Encrypted global model
            clients_data: Dictionary of client datasets for evaluation
            
        Returns:
            Dict[str, float]: Evaluation metrics including accuracy, F1-score, precision, recall
        """
        print("  Evaluating encrypted model (decrypting ONLY for evaluation)...")
        
        # Decrypt ONLY for evaluation (in production, this would happen on client side)
        decrypted_weights, decrypted_bias = global_model.decrypt_for_evaluation()
        
        # Create temporary model for evaluation
        temp_model = LogisticRegression()
        temp_model.coef_ = decrypted_weights.reshape(1, -1)
        temp_model.intercept_ = np.array([decrypted_bias])
        temp_model.classes_ = np.array([0, 1])  # Set classes for prediction
        
        # Evaluate on all client data
        all_x_test = []
        all_y_test = []
        for X, y in clients_data.values():
            all_x_test.append(X)
            all_y_test.append(y)
        
        x_test = np.vstack(all_x_test)
        y_test = np.concatenate(all_y_test)
        
        # Get predictions
        y_pred = temp_model.predict(x_test)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred)
        }
        
        print(f"  Evaluation completed - model re-encrypted")
        
        return metrics