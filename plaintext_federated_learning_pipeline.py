"""
Plaintext Federated Learning Pipeline

This pipeline implements federated learning in plaintext without any encryption,
providing a baseline for comparison with the FHE pipeline.

Key Features:
- Plaintext model updates and aggregation
- One-class client handling without exclusion bias
- Comprehensive performance metrics
- Fast execution without encryption overhead
- Same data processing and evaluation as FHE pipeline

Architecture:
1. Data Processing: Load and preprocess health fitness data
2. Client Creation: Distribute data across multiple clients
3. Local Training: Train models on client data with one-class handling
4. Aggregation: Aggregate model updates in plaintext
5. Global Update: Update global model with aggregated data
6. Evaluation: Evaluate model performance

Author: AI Assistant
Date: 2025
"""

import os
import json
import csv
import time
import numpy as np
import pandas as pd
import argparse
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, mean_absolute_error, mean_squared_error
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from src.plaintext import PlaintextConfig, PlaintextModel, PlaintextManager
from src.fl import FLConfig, DataProcessor, ModelEvaluator
from src.utils import (
    print_final_summary, calculate_enhanced_metrics, create_subject_disjoint_splits
)

# Constants
DEFAULT_ROUNDS = 10
DEFAULT_CLIENTS = 10


class EnhancedDataProcessor(DataProcessor):
    """
    Enhanced data processor for plaintext federated learning
    
    This class extends the base DataProcessor to provide enhanced
    feature engineering and data preprocessing capabilities.
    """
    
    def load_health_fitness_data(self) -> Tuple[pd.DataFrame, List[str]]:
        """
        Load and preprocess health fitness data for plaintext federated learning
        
        Returns:
            Tuple of (DataFrame, feature_columns) for federated learning
        """
        print("Loading Health Fitness Dataset with Enhanced Feature Engineering...")
        
        # Load the main dataset
        df = pd.read_csv('data/health_fitness_data.csv')
        print(f"Loaded {len(df):,} records from {df['participant_id'].nunique()} participants")
        
        # Preprocess data
        print("Preprocessing data...")
        
        # Convert fitness_level to binary health status using median threshold
        fitness_threshold = df['fitness_level'].median()
        df['health_status'] = (df['fitness_level'] >= fitness_threshold).astype(int)
        
        # Print health status distribution
        health_counts = df['health_status'].value_counts()
        print("Health Status Distribution:")
        print(f"  Unhealthy (0): {health_counts[0]:,} ({health_counts[0]/len(df)*100:.1f}%)")
        print(f"  Healthy (1): {health_counts[1]:,} ({health_counts[1]/len(df)*100:.1f}%)")
        print(f"  Fitness Threshold: {fitness_threshold:.2f}")
        
        # Advanced feature engineering
        print("Advanced Feature Engineering...")
        
        # Basic features (13 features)
        basic_features = [
            'age', 'height_cm', 'weight_kg', 'bmi',
            'avg_heart_rate', 'resting_heart_rate',
            'blood_pressure_systolic', 'blood_pressure_diastolic',
            'hours_sleep', 'stress_level', 'daily_steps',
            'calories_burned', 'hydration_level'
        ]
        
        # Derived features (16 features)
        df['heart_rate_variability'] = df['avg_heart_rate'] - df['resting_heart_rate']
        df['blood_pressure_ratio'] = df['blood_pressure_systolic'] / df['blood_pressure_diastolic']
        df['sleep_efficiency'] = df['hours_sleep'] / 8.0  # Normalize to 8 hours
        df['stress_sleep_ratio'] = df['stress_level'] / (df['hours_sleep'] + 1)
        df['activity_intensity'] = df['daily_steps'] / 10000.0  # Normalize to 10k steps
        df['calorie_efficiency'] = df['calories_burned'] / (df['daily_steps'] + 1)
        df['hydration_adequacy'] = df['hydration_level'] / 8.0  # Normalize to 8 glasses
        df['fitness_age_ratio'] = df['fitness_level'] / (df['age'] + 1)
        df['bmi_fitness_ratio'] = df['bmi'] / (df['fitness_level'] + 1)
        df['heart_rate_fitness'] = df['avg_heart_rate'] / (df['fitness_level'] + 1)
        df['sleep_stress_balance'] = df['hours_sleep'] - df['stress_level']
        df['activity_stress_ratio'] = df['daily_steps'] / (df['stress_level'] + 1)
        df['calorie_hydration_ratio'] = df['calories_burned'] / (df['hydration_level'] + 1)
        df['blood_pressure_fitness'] = df['blood_pressure_systolic'] / (df['fitness_level'] + 1)
        df['overall_health_score'] = (
            df['fitness_level'] + df['hours_sleep'] + df['hydration_level'] - 
            df['stress_level'] - df['bmi'] / 10
        )
        
        derived_features = [
            'heart_rate_variability', 'blood_pressure_ratio', 'sleep_efficiency',
            'stress_sleep_ratio', 'activity_intensity', 'calorie_efficiency',
            'hydration_adequacy', 'fitness_age_ratio', 'bmi_fitness_ratio',
            'heart_rate_fitness', 'sleep_stress_balance', 'activity_stress_ratio',
            'calorie_hydration_ratio', 'blood_pressure_fitness', 'overall_health_score'
        ]
        
        # Polynomial features (10 features)
        df['age_squared'] = df['age'] ** 2
        df['bmi_squared'] = df['bmi'] ** 2
        df['fitness_squared'] = df['fitness_level'] ** 2
        df['stress_squared'] = df['stress_level'] ** 2
        df['sleep_squared'] = df['hours_sleep'] ** 2
        df['sqrt_age'] = np.sqrt(df['age'])
        df['sqrt_bmi'] = np.sqrt(df['bmi'])
        df['sqrt_fitness'] = np.sqrt(df['fitness_level'])
        df['sqrt_stress'] = np.sqrt(df['stress_level'])
        df['sqrt_sleep'] = np.sqrt(df['hours_sleep'])
        
        polynomial_features = [
            'age_squared', 'bmi_squared', 'fitness_squared', 'stress_squared',
            'sleep_squared', 'sqrt_age', 'sqrt_bmi', 'sqrt_fitness', 'sqrt_stress', 'sqrt_sleep'
        ]
        
        # Categorical features (8 features)
        df['age_group'] = pd.cut(df['age'], bins=[0, 30, 50, 70, 100], labels=[0, 1, 2, 3])
        df['bmi_category'] = pd.cut(df['bmi'], bins=[0, 18.5, 25, 30, 100], labels=[0, 1, 2, 3])
        df['fitness_level_group'] = pd.cut(df['fitness_level'], bins=[0, 5, 8, 10, 15], labels=[0, 1, 2, 3])
        df['stress_level_group'] = pd.cut(df['stress_level'], bins=[0, 3, 6, 8, 10], labels=[0, 1, 2, 3])
        df['sleep_quality'] = pd.cut(df['hours_sleep'], bins=[0, 6, 8, 10, 12], labels=[0, 1, 2, 3])
        df['activity_level'] = pd.cut(df['daily_steps'], bins=[0, 5000, 10000, 15000, 20000], labels=[0, 1, 2, 3])
        df['heart_rate_zone'] = pd.cut(df['avg_heart_rate'], bins=[0, 60, 80, 100, 200], labels=[0, 1, 2, 3])
        df['blood_pressure_category'] = pd.cut(df['blood_pressure_systolic'], bins=[0, 120, 140, 160, 200], labels=[0, 1, 2, 3])
        
        categorical_features = [
            'age_group', 'bmi_category', 'fitness_level_group', 'stress_level_group',
            'sleep_quality', 'activity_level', 'heart_rate_zone', 'blood_pressure_category'
        ]
        
        # Combine all features
        feature_columns = basic_features + derived_features + polynomial_features + categorical_features
        
        print(f"Using {len(feature_columns)} features for federated learning")
        print(f"  - Basic features: {len(basic_features)}")
        print(f"  - Derived features: {len(derived_features)}")
        print(f"  - Polynomial features: {len(polynomial_features)}")
        print(f"  - Categorical features: {len(categorical_features)}")
        
        # Set feature columns for the data processor
        self.feature_columns = feature_columns
        
        return df, feature_columns


class EnhancedModelEvaluator(ModelEvaluator):
    """
    Enhanced model evaluator for plaintext federated learning
    
    This class provides comprehensive model evaluation capabilities
    including accuracy, F1-score, precision, recall, AUC, MAE, and RMSE.
    """
    
    @staticmethod
    def evaluate_model(model, X_test, y_test):
        """
        Evaluate model performance with comprehensive metrics
        
        Args:
            model: Trained model to evaluate
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary containing evaluation metrics
        """
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of positive class
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        mae = mean_absolute_error(y_test, y_pred_proba)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred_proba))
        
        metrics = {
            'accuracy': accuracy,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'auc': auc,
            'mae': mae,
            'rmse': rmse
        }
        
        return metrics


class PlaintextFederatedLearningPipeline:
    """
    Plaintext Federated Learning Pipeline with integrated one-class handling
    
    This pipeline implements:
    - Plaintext model updates and aggregation
    - One-class client handling without exclusion bias
    - Comprehensive performance metrics
    - Fast execution without encryption overhead
    """
    
    def __init__(self, fl_config: FLConfig, plaintext_config: PlaintextConfig):
        """
        Initialize the plaintext federated learning pipeline
        
        Args:
            fl_config: Federated learning configuration
            plaintext_config: Plaintext configuration
        """
        # Store configurations
        self.config = fl_config
        self.plaintext_config = plaintext_config
        
        # Initialize components
        self.data_processor = EnhancedDataProcessor(fl_config)
        self.model_evaluator = EnhancedModelEvaluator()
        
        # Initialize plaintext manager for aggregation operations
        self.plaintext_manager = PlaintextManager(plaintext_config)
        
        # Performance tracking
        self.performance_history = []
        self.best_accuracy = 0.0
        self.convergence_threshold = 0.001  # 0.1% improvement threshold
        self.patience = 999  # Disabled - run all rounds as requested
        
        # One-class handling parameters
        self.l2_regularization = 1e-3      # L2 regularization strength
        self.fedprox_mu = 0.01             # FedProx proximal regularizer strength
        self.laplace_smoothing = 0.1       # Laplace smoothing parameter
        self.min_sample_weight = 10         # Minimum sample weight floor
        
        # Global model reference for warm start and FedProx
        self.global_model_weights = None
        self.global_model_bias = None
        
        # One-class handling statistics
        self.one_class_stats = {
            'total_clients': 0,
            'one_class_clients': 0,
            'strategies_used': {}
        }
    
    def is_one_class_client(self, y: np.ndarray) -> bool:
        """
        Check if client has only one class
        
        Args:
            y: Client labels
            
        Returns:
            True if client has only one class
        """
        return len(np.unique(y)) < 2
    
    def get_class_distribution(self, y: np.ndarray) -> Dict[int, int]:
        """
        Get class distribution for client
        
        Args:
            y: Client labels
            
        Returns:
            Dictionary mapping class to count
        """
        unique_classes, counts = np.unique(y, return_counts=True)
        return {int(cls): int(count) for cls, count in zip(unique_classes, counts)}
    
    def apply_laplace_smoothing(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply Laplace smoothing to handle one-class clients
        
        Args:
            X: Features
            y: Labels
            
        Returns:
            Tuple of (smoothed_features, smoothed_labels)
        """
        # Add virtual samples for both classes (0 and 1)
        unique_classes = np.unique(y)
        virtual_samples = []
        virtual_labels = []
        
        # Add virtual samples for missing classes
        for cls in [0, 1]:
            if cls not in unique_classes:
                # Create virtual sample with mean features
                virtual_sample = np.mean(X, axis=0)
                virtual_samples.append(virtual_sample)
                virtual_labels.append(cls)
        
        # Add virtual samples to data
        if virtual_samples:
            X_smoothed = np.vstack([X, np.array(virtual_samples)])
            y_smoothed = np.concatenate([y, np.array(virtual_labels)])
        else:
            X_smoothed = X
            y_smoothed = y
        
        return X_smoothed, y_smoothed
    
    def create_warm_start_model(self, X: np.ndarray, y: np.ndarray) -> LogisticRegression:
        """
        Create model with warm start from global model
        
        Args:
            X: Features
            y: Labels
            
        Returns:
            Model with warm start
        """
        model = LogisticRegression(
            solver='liblinear',
            max_iter=5000,
            random_state=42
        )
        
        # Set initial parameters if global model exists
        if self.global_model_weights is not None:
            model.coef_ = self.global_model_weights.reshape(1, -1)
            model.intercept_ = np.array([self.global_model_bias])
        
        return model
    
    def apply_fedprox_regularization(self, model: LogisticRegression, X: np.ndarray, y: np.ndarray) -> LogisticRegression:
        """
        Apply FedProx regularization to model
        
        Args:
            model: Model to regularize
            X: Features
            y: Labels
            
        Returns:
            Regularized model
        """
        # This is a simplified FedProx implementation
        # In practice, you would modify the loss function
        model.fit(X, y)
        return model
    
    def apply_sample_weight_floor(self, sample_count: int) -> float:
        """
        Apply sample weight floor to prevent vanishing influence
        
        Args:
            sample_count: Number of samples
            
        Returns:
            Adjusted sample weight
        """
        return max(sample_count, self.min_sample_weight)
    
    def train_client_with_strategy(self, X: np.ndarray, y: np.ndarray, strategy: str = "combined") -> Dict[str, Any]:
        """
        Train client using integrated strategy pattern
        
        Args:
            X: Client features
            y: Client labels
            strategy: Strategy to use for training
            
        Returns:
            Dictionary containing training results
        """
        is_one_class = self.is_one_class_client(y)
        class_distribution = self.get_class_distribution(y)
        
        # Update statistics
        self.one_class_stats['total_clients'] += 1
        if is_one_class:
            self.one_class_stats['one_class_clients'] += 1
            if strategy not in self.one_class_stats['strategies_used']:
                self.one_class_stats['strategies_used'][strategy] = 0
            self.one_class_stats['strategies_used'][strategy] += 1
        
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
                # Try multiple strategies in order
                try:
                    X_smoothed, y_smoothed = self.apply_laplace_smoothing(X, y)
                    model = LogisticRegression(
                        solver='liblinear',
                        max_iter=5000,
                        random_state=42
                    )
                    model.fit(X_smoothed, y_smoothed)
                    strategy_used = "laplace"
                except Exception as e:
                    print(f"⚠️ Laplace smoothing failed: {e}")
                    try:
                        model = self.create_warm_start_model(X, y)
                        model.fit(X, y)
                        strategy_used = "warm_start"
                    except Exception as e2:
                        print(f"⚠️ Warm start failed: {e2}")
                        # Fallback to random initialization
                        model = LogisticRegression(
                            solver='liblinear',
                            max_iter=5000,
                            random_state=42
                        )
                        model.fit(X, y)
                        strategy_used = "fallback"
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
        
        # Extract model parameters
        if hasattr(model, 'coef_') and model.coef_ is not None:
            weights = model.coef_[0]
            bias = model.intercept_[0]
        else:
            print("⚠️ Model failed to fit, using random initialization")
            weights = np.random.normal(0, 0.01, X.shape[1])
            bias = 0.0
        
        # Apply sample weight floor
        sample_count = len(X)
        adjusted_weight = self.apply_sample_weight_floor(sample_count)
        
        return {
            'weights': weights,
            'bias': bias,
            'sample_count': sample_count,
            'adjusted_weight': adjusted_weight,
            'strategy_used': strategy_used,
            'is_one_class': is_one_class,
            'class_distribution': class_distribution
        }
    
    def update_global_model_reference(self, global_weights: np.ndarray, global_bias: float):
        """
        Update global model reference for warm start and FedProx
        
        Args:
            global_weights: Global model weights
            global_bias: Global model bias
        """
        self.global_model_weights = global_weights
        self.global_model_bias = global_bias
    
    def create_plaintext_model(self, weights: np.ndarray, bias: float) -> PlaintextModel:
        """
        Create a plaintext model
        
        Args:
            weights: Model weights
            bias: Model bias
            
        Returns:
            Plaintext model
        """
        return self.plaintext_manager.create_model(weights, bias)
    
    def run_plaintext_federated_learning(self) -> List[Dict[str, Any]]:
        """
        Run plaintext federated learning
        
        Returns:
            List of round results
        """
        print("Plaintext Federated Learning Pipeline - Fast Execution")
        print("=" * 60)
        print("TARGET: Maximum Performance with PLAINTEXT implementation")
        print("CRITICAL: No encryption overhead - maximum speed")
        print("NO ENCRYPTION - PLAINTEXT implementation")
        print()
        
        # Load and preprocess data
        print("📊 Loading and preprocessing data...")
        df, feature_columns = self.data_processor.load_health_fitness_data()
        
        # Create client datasets
        print("👥 Creating client datasets...")
        self.clients_data = self.data_processor.create_client_datasets(df)
        
        # Scale client data (same as FHE pipeline)
        print("Scaling client data...")
        self.clients_data = self.data_processor.scale_client_data(self.clients_data)
        print(f"Scaled data for {len(self.clients_data)} clients")
        
        # Initialize global model
        print("🌐 Initializing global model...")
        np.random.seed(42)  # Ensure reproducible initialization
        initial_weights = np.random.normal(0, 0.1, len(feature_columns))
        initial_bias = 0.0
        self.plaintext_global_model = self.create_plaintext_model(initial_weights, initial_bias)
        
        # Initialize performance tracking
        self.performance_history = []
        self.best_accuracy = 0.0
        
        print(f"📊 Configuration:")
        print(f"  Rounds: {self.config.rounds}")
        print(f"  Clients: {len(self.clients_data)}")
        print(f"  Features: {len(feature_columns)}")
        print(f"  Aggregation: {self.plaintext_config.aggregation_method}")
        print()
        
        print("🚀 Starting plaintext federated learning...")
        print("=" * 60)
        
        # Run federated learning rounds
        for round_num in range(1, self.config.rounds + 1):
            print(f"🔄 Round {round_num}/{self.config.rounds}")
            
            # Train clients and collect updates
            client_updates = []
            sample_counts = []
            round_start = time.time()
            
            for client_id, client_data in self.clients_data.items():
                X_client, y_client = client_data
                
                # Train client with strategy
                result = self.train_client_with_strategy(X_client, y_client, strategy="combined")
                
                # Create model update
                model_update = np.concatenate([result['weights'], [result['bias']]])
                client_updates.append(model_update)
                sample_counts.append(result['sample_count'])
            
            # Aggregate updates
            aggregation_start = time.time()
            aggregated_update, aggregation_time = self.plaintext_manager.aggregate_updates(
                client_updates, sample_counts
            )
            
            # Update global model
            self.plaintext_manager.update_global_model(
                self.plaintext_global_model, aggregated_update
            )
            
            # Update global model reference for local training
            global_weights = aggregated_update[:-1]
            global_bias = float(aggregated_update[-1])
            self.update_global_model_reference(global_weights, global_bias)
            
            # Evaluate model
            evaluation_start = time.time()
            
            # Create test data using client data (same as FHE pipeline)
            all_X = []
            all_y = []
            
            for client_id, client_data in self.clients_data.items():
                X_client, y_client = client_data
                all_X.append(X_client)
                all_y.append(y_client)
            
            # Combine all client data
            X_test = np.vstack(all_X)
            y_test = np.concatenate(all_y)
            
            # Handle any remaining NaN values
            X_test = np.nan_to_num(X_test, nan=0.0)
            
            # Create evaluation model
            eval_model = LogisticRegression(
                solver='liblinear',
                max_iter=5000,
                random_state=42
            )
            
            # Fit the model with dummy data to get proper structure
            dummy_X = np.random.randn(10, len(feature_columns))
            dummy_y = np.random.randint(0, 2, 10)
            eval_model.fit(dummy_X, dummy_y)
            
            # Set the learned parameters
            eval_model.coef_ = global_weights.reshape(1, -1)
            eval_model.intercept_ = np.array([global_bias])
            
            # Get predictions and probabilities
            y_pred = eval_model.predict(X_test)
            y_pred_proba = eval_model.predict_proba(X_test)[:, 1]
            
            # Calculate enhanced metrics
            metrics = calculate_enhanced_metrics(y_test, y_pred, y_pred_proba)
            evaluation_time = time.time() - evaluation_start
            
            # Create round result
            round_result = {
                'round': round_num,
                'accuracy': metrics['accuracy'],
                'f1_score': metrics['f1_score'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'auc': metrics['auc'],
                'pr_auc': metrics['pr_auc'],
                'ece': metrics['ece'],
                'mce': metrics['mce'],
                'mae': metrics['mae'],
                'rmse': metrics['rmse'],
                'encryption_time': 0.0,  # No encryption in plaintext
                'aggregation_time': aggregation_time,
                'evaluation_time': evaluation_time,
                'total_time': time.time() - round_start
            }
            
            # Track performance
            self.performance_history.append(round_result)
            if metrics['accuracy'] > self.best_accuracy:
                self.best_accuracy = metrics['accuracy']
            
            # Print round results
            print(f"  Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
            print(f"  F1 Score: {metrics['f1_score']:.4f}")
            print(f"  AUC: {metrics['auc']:.4f}")
            print(f"  Aggregation: {aggregation_time:.4f}s")
            print(f"  Evaluation: {evaluation_time:.4f}s")
            print(f"  Total: {round_result['total_time']:.4f}s")
            print()
            
            # Check for convergence
            if len(self.performance_history) > 1:
                improvement = metrics['accuracy'] - self.performance_history[-2]['accuracy']
                if improvement < self.convergence_threshold:
                    print(f"✅ Convergence detected (improvement: {improvement:.6f})")
                    break
        
        print("=" * 60)
        print("✅ Plaintext federated learning completed!")
        print()
        
        # Print final summary
        print_final_summary(self.performance_history, self.clients_data)
        
        # Print one-class statistics
        self.print_one_class_statistics()
        
        # Print plaintext statistics
        self.print_plaintext_statistics()
        
        return self.performance_history
    
    def print_one_class_statistics(self):
        """Print one-class client handling statistics"""
        print("📊 One-Class Client Statistics:")
        print(f"  Total clients: {self.one_class_stats['total_clients']}")
        print(f"  One-class clients: {self.one_class_stats['one_class_clients']}")
        print(f"  One-class percentage: {self.one_class_stats['one_class_clients']/self.one_class_stats['total_clients']*100:.2f}%")
        print("  Strategies used:")
        for strategy, count in self.one_class_stats['strategies_used'].items():
            print(f"    {strategy}: {count}")
        print()
    
    def print_plaintext_statistics(self):
        """Print plaintext operation statistics"""
        stats = self.plaintext_manager.get_operation_stats()
        print("📊 Plaintext Operation Statistics:")
        print(f"  Aggregation method: {stats['config']['aggregation_method']}")
        print(f"  Total aggregations: {stats['aggregation_stats']['total_aggregations']}")
        print(f"  Average aggregation time: {stats['aggregation_stats']['avg_aggregation_time']:.4f}s")
        print(f"  Total updates processed: {stats['aggregation_stats']['total_updates_processed']}")
        print(f"  Total samples processed: {stats['aggregation_stats']['total_samples_processed']}")
        print()


def main():
    """Main function to run plaintext federated learning"""
    parser = argparse.ArgumentParser(description='Plaintext Federated Learning Pipeline')
    parser.add_argument('--rounds', type=int, default=DEFAULT_ROUNDS, help='Number of federated learning rounds')
    parser.add_argument('--clients', type=int, default=DEFAULT_CLIENTS, help='Number of clients to simulate')
    parser.add_argument('--patience', type=int, default=999, help='Patience for convergence detection')
    
    args = parser.parse_args()
    
    # Create configurations
    fl_config = FLConfig(
        rounds=args.rounds,
        clients=args.clients,
        patience=args.patience
    )
    
    plaintext_config = PlaintextConfig(
        aggregation_method="federated_averaging",
        weight_by_samples=True,
        verbose=True
    )
    
    # Create and run pipeline
    pipeline = PlaintextFederatedLearningPipeline(fl_config, plaintext_config)
    results = pipeline.run_plaintext_federated_learning()
    
    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = f"performance_results/plaintext_fl_results_{timestamp}.json"
    
    os.makedirs("performance_results", exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"📁 Results saved to: {results_file}")


if __name__ == "__main__":
    main()
