"""
Customizable FHE CKKS Data Flow - Configurable Clients, Rounds, and Routers
Research-Paper Ready Implementation with Real FHE and 47 Features
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime
import argparse
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION PARAMETERS - CUSTOMIZE THESE!
# =============================================================================

class CustomConfig:
    """Configuration class for customizable FHE data flow"""
    
    def __init__(self, 
                 num_clients=6,           # Number of smartwatch clients
                 num_rounds=3,            # Number of federated learning rounds
                 num_routers=2,           # Number of home routers
                 use_real_data=True,      # Use real CSV data or synthetic
                 feature_engineering=True, # Apply 47-feature engineering
                 verbose=True):           # Show detailed output
        self.num_clients = num_clients
        self.num_rounds = num_rounds
        self.num_routers = num_routers
        self.use_real_data = use_real_data
        self.feature_engineering = feature_engineering
        self.verbose = verbose
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration parameters"""
        if self.num_clients < 1:
            raise ValueError("Number of clients must be at least 1")
        if self.num_rounds < 1:
            raise ValueError("Number of rounds must be at least 1")
        if self.num_routers < 1:
            raise ValueError("Number of routers must be at least 1")
        if self.num_clients > 50:
            print("⚠️  Warning: Large number of clients may slow down execution")
        if self.num_rounds > 20:
            print("⚠️  Warning: Large number of rounds may take significant time")
    
    def print_config(self):
        """Print current configuration"""
        print("🔧 CUSTOM CONFIGURATION:")
        print(f"  👥 Clients: {self.num_clients}")
        print(f"  🔄 Rounds: {self.num_rounds}")
        print(f"  🏠 Routers: {self.num_routers}")
        print(f"  📊 Real Data: {self.use_real_data}")
        print(f"  🛠️  Feature Engineering: {self.feature_engineering}")
        print(f"  📝 Verbose: {self.verbose}")

def load_and_enhance_csv_data(config):
    """
    Load real health fitness data with EXACT same feature engineering as main pipeline
    """
    if config.verbose:
        print("📊 Loading Real Health Fitness Data with EXACT Feature Engineering...")
    
    csv_path = "data/health_fitness_data.csv"
    if not os.path.exists(csv_path):
        if config.verbose:
            print(f"❌ Health data not found at {csv_path}")
        return create_enhanced_synthetic_data(config)
    
    # Load the real CSV data
    df = pd.read_csv(csv_path)
    if config.verbose:
        print(f"✅ Loaded {len(df):,} records from {df['participant_id'].nunique()} participants")
    
    # Preprocessing
    if config.verbose:
        print("Preprocessing data...")
    
    # Convert fitness_level to binary health status
    fitness_threshold = df['fitness_level'].median()
    df['health_status'] = (df['fitness_level'] >= fitness_threshold).astype(int)
    
    if config.verbose:
        print(f"Health Status Distribution:")
        print(f"  Unhealthy (0): {len(df[df['health_status'] == 0]):,} ({len(df[df['health_status'] == 0])/len(df)*100:.1f}%)")
        print(f"  Healthy (1): {len(df[df['health_status'] == 1]):,} ({len(df[df['health_status'] == 1])/len(df)*100:.1f}%)")
        print(f"  Fitness Threshold: {fitness_threshold:.2f}")
    
    if config.feature_engineering:
        # Advanced feature engineering - EXACT SAME AS MAIN PIPELINE
        if config.verbose:
            print("Advanced Feature Engineering...")
        
        # 1. Basic features
        basic_features = [
            'age', 'height_cm', 'weight_kg', 'bmi',
            'avg_heart_rate', 'resting_heart_rate', 
            'blood_pressure_systolic', 'blood_pressure_diastolic',
            'hours_sleep', 'stress_level', 'daily_steps', 
            'calories_burned', 'hydration_level'
        ]
        
        # 2. Derived features
        df['steps_per_calorie'] = df['daily_steps'] / (df['calories_burned'] + 1)
        df['sleep_efficiency'] = df['hours_sleep'] / (df['stress_level'] + 1)
        df['cardio_health'] = (df['resting_heart_rate'] <= 70).astype(int)
        df['high_activity'] = (df['daily_steps'] >= 10000).astype(int)
        df['good_sleep'] = (df['hours_sleep'] >= 7.0).astype(int)
        df['low_stress'] = (df['stress_level'] <= 5.0).astype(int)
        
        # 3. Advanced derived features
        df['heart_rate_variability'] = df['avg_heart_rate'] - df['resting_heart_rate']
        df['blood_pressure_ratio'] = df['blood_pressure_systolic'] / (df['blood_pressure_diastolic'] + 1)
        df['metabolic_efficiency'] = df['calories_burned'] / (df['weight_kg'] + 1)
        df['sleep_quality_score'] = df['hours_sleep'] * (10 - df['stress_level']) / 10
        df['activity_intensity'] = df['daily_steps'] * df['calories_burned'] / 1000
        df['health_score'] = (df['fitness_level'] + df['sleep_quality_score']) / 2
        
        # 4. Interaction features
        df['age_fitness_interaction'] = df['age'] * df['fitness_level']
        df['sleep_stress_interaction'] = df['hours_sleep'] * (10 - df['stress_level'])
        df['activity_sleep_interaction'] = df['daily_steps'] * df['hours_sleep']
        df['heart_rate_sleep_interaction'] = df['avg_heart_rate'] * df['hours_sleep']
        
        # 5. Polynomial features (degree 2)
        poly_features = ['age', 'fitness_level', 'avg_heart_rate', 'hours_sleep', 'daily_steps']
        for feature in poly_features:
            df[f'{feature}_squared'] = df[feature] ** 2
            df[f'{feature}_sqrt'] = np.sqrt(np.abs(df[feature]))
        
        # 6. Categorical encoding
        df['gender_encoded'] = LabelEncoder().fit_transform(df['gender'])
        df['intensity_encoded'] = LabelEncoder().fit_transform(df['intensity'])
        df['activity_type_encoded'] = LabelEncoder().fit_transform(df['activity_type'])
        
        # 7. Temporal features
        df['date'] = pd.to_datetime(df['date'])
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # 8. Health condition encoding
        df['has_health_condition'] = (df['health_condition'] != 'None').astype(int)
        df['smoking_status_encoded'] = LabelEncoder().fit_transform(df['smoking_status'])
        
        # Collect all features - EXACT SAME AS MAIN PIPELINE
        derived_features = [
            'steps_per_calorie', 'sleep_efficiency', 'cardio_health',
            'high_activity', 'good_sleep', 'low_stress',
            'heart_rate_variability', 'blood_pressure_ratio', 'metabolic_efficiency',
            'sleep_quality_score', 'activity_intensity', 'health_score',
            'age_fitness_interaction', 'sleep_stress_interaction', 
            'activity_sleep_interaction', 'heart_rate_sleep_interaction'
        ]
        
        poly_feature_names = []
        for feature in poly_features:
            poly_feature_names.extend([f'{feature}_squared', f'{feature}_sqrt'])
        
        categorical_features = [
            'gender_encoded', 'intensity_encoded', 'activity_type_encoded',
            'day_of_week', 'month', 'is_weekend', 'has_health_condition', 'smoking_status_encoded'
        ]
        
        feature_columns = basic_features + derived_features + poly_feature_names + categorical_features
        
        if config.verbose:
            print(f"Using {len(feature_columns)} features for federated learning")
            print(f"Feature breakdown:")
            print(f"  Basic features: {len(basic_features)}")
            print(f"  Derived features: {len(derived_features)}")
            print(f"  Polynomial features: {len(poly_feature_names)}")
            print(f"  Categorical features: {len(categorical_features)}")
            print(f"  TOTAL FEATURES: {len(feature_columns)}")
    else:
        # Use only basic features
        feature_columns = [
            'age', 'height_cm', 'weight_kg', 'bmi',
            'avg_heart_rate', 'resting_heart_rate', 
            'blood_pressure_systolic', 'blood_pressure_diastolic',
            'hours_sleep', 'stress_level', 'daily_steps', 
            'calories_burned', 'hydration_level', 'fitness_level'
        ]
        if config.verbose:
            print(f"Using {len(feature_columns)} basic features for federated learning")
    
    return df, feature_columns

def create_enhanced_synthetic_data(config):
    """
    Create enhanced synthetic data with EXACT same features as main pipeline
    """
    if config.verbose:
        print("📊 Creating Enhanced Synthetic Health Data...")
    
    np.random.seed(42)
    n_samples = config.num_clients * 200  # 200 samples per client
    n_participants = config.num_clients
    
    data = []
    for participant_id in range(n_participants):
        participant_samples = n_samples // n_participants
        
        # Generate realistic health data
        age = np.random.normal(35, 10, participant_samples)
        height_cm = np.random.normal(170, 10, participant_samples)
        weight_kg = np.random.normal(70, 15, participant_samples)
        bmi = weight_kg / ((height_cm / 100) ** 2)
        
        # Generate correlated features
        fitness_level = np.random.normal(5, 2, participant_samples)
        daily_steps = np.random.normal(8000, 3000, participant_samples)
        calories_burned = daily_steps * 0.04 + np.random.normal(0, 100, participant_samples)
        hours_sleep = np.random.normal(7.5, 1.5, participant_samples)
        stress_level = np.random.normal(5, 2, participant_samples)
        avg_heart_rate = np.random.normal(75, 10, participant_samples)
        resting_heart_rate = np.random.normal(70, 8, participant_samples)
        blood_pressure_systolic = np.random.normal(120, 15, participant_samples)
        blood_pressure_diastolic = np.random.normal(80, 10, participant_samples)
        hydration_level = np.random.normal(60, 10, participant_samples)
        
        # Create health status based on multiple factors
        health_score = (
            fitness_level * 0.3 +
            (daily_steps / 10000) * 0.2 +
            (hours_sleep / 8) * 0.2 +
            (10 - stress_level) / 10 * 0.2 +
            (80 - resting_heart_rate) / 80 * 0.1
        )
        health_status = (health_score > 0.5).astype(int)
        
        # Ensure balanced classes
        if np.sum(health_status) == 0:
            health_status[np.random.choice(len(health_status), size=len(health_status)//2)] = 1
        elif np.sum(health_status) == len(health_status):
            health_status[np.random.choice(len(health_status), size=len(health_status)//2)] = 0
        
        # Create DataFrame
        participant_df = pd.DataFrame({
            'participant_id': participant_id,
            'age': age,
            'height_cm': height_cm,
            'weight_kg': weight_kg,
            'bmi': bmi,
            'avg_heart_rate': avg_heart_rate,
            'resting_heart_rate': resting_heart_rate,
            'blood_pressure_systolic': blood_pressure_systolic,
            'blood_pressure_diastolic': blood_pressure_diastolic,
            'hours_sleep': hours_sleep,
            'stress_level': stress_level,
            'daily_steps': daily_steps,
            'calories_burned': calories_burned,
            'hydration_level': hydration_level,
            'fitness_level': fitness_level,
            'health_status': health_status,
            'gender': np.random.choice(['Male', 'Female'], participant_samples),
            'intensity': np.random.choice(['Low', 'Medium', 'High'], participant_samples),
            'activity_type': np.random.choice(['Walking', 'Running', 'Cycling'], participant_samples),
            'health_condition': np.random.choice(['None', 'Diabetes', 'Hypertension'], participant_samples),
            'smoking_status': np.random.choice(['Non-smoker', 'Former smoker', 'Current smoker'], participant_samples),
            'date': pd.date_range('2023-01-01', periods=participant_samples, freq='D')
        })
        
        data.append(participant_df)
    
    df = pd.concat(data, ignore_index=True)
    
    if config.feature_engineering:
        # Apply EXACT same feature engineering as main pipeline
        # 2. Derived features
        df['steps_per_calorie'] = df['daily_steps'] / (df['calories_burned'] + 1)
        df['sleep_efficiency'] = df['hours_sleep'] / (df['stress_level'] + 1)
        df['cardio_health'] = (df['resting_heart_rate'] <= 70).astype(int)
        df['high_activity'] = (df['daily_steps'] >= 10000).astype(int)
        df['good_sleep'] = (df['hours_sleep'] >= 7.0).astype(int)
        df['low_stress'] = (df['stress_level'] <= 5.0).astype(int)
        
        # 3. Advanced derived features
        df['heart_rate_variability'] = df['avg_heart_rate'] - df['resting_heart_rate']
        df['blood_pressure_ratio'] = df['blood_pressure_systolic'] / (df['blood_pressure_diastolic'] + 1)
        df['metabolic_efficiency'] = df['calories_burned'] / (df['weight_kg'] + 1)
        df['sleep_quality_score'] = df['hours_sleep'] * (10 - df['stress_level']) / 10
        df['activity_intensity'] = df['daily_steps'] * df['calories_burned'] / 1000
        df['health_score'] = (df['fitness_level'] + df['sleep_quality_score']) / 2
        
        # 4. Interaction features
        df['age_fitness_interaction'] = df['age'] * df['fitness_level']
        df['sleep_stress_interaction'] = df['hours_sleep'] * (10 - df['stress_level'])
        df['activity_sleep_interaction'] = df['daily_steps'] * df['hours_sleep']
        df['heart_rate_sleep_interaction'] = df['avg_heart_rate'] * df['hours_sleep']
        
        # 5. Polynomial features (degree 2)
        poly_features = ['age', 'fitness_level', 'avg_heart_rate', 'hours_sleep', 'daily_steps']
        for feature in poly_features:
            df[f'{feature}_squared'] = df[feature] ** 2
            df[f'{feature}_sqrt'] = np.sqrt(np.abs(df[feature]))
        
        # 6. Categorical encoding
        df['gender_encoded'] = LabelEncoder().fit_transform(df['gender'])
        df['intensity_encoded'] = LabelEncoder().fit_transform(df['intensity'])
        df['activity_type_encoded'] = LabelEncoder().fit_transform(df['activity_type'])
        
        # 7. Temporal features
        df['date'] = pd.to_datetime(df['date'])
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # 8. Health condition encoding
        df['has_health_condition'] = (df['health_condition'] != 'None').astype(int)
        df['smoking_status_encoded'] = LabelEncoder().fit_transform(df['smoking_status'])
        
        # Collect all features - EXACT SAME AS MAIN PIPELINE
        basic_features = [
            'age', 'height_cm', 'weight_kg', 'bmi',
            'avg_heart_rate', 'resting_heart_rate', 
            'blood_pressure_systolic', 'blood_pressure_diastolic',
            'hours_sleep', 'stress_level', 'daily_steps', 
            'calories_burned', 'hydration_level'
        ]
        
        derived_features = [
            'steps_per_calorie', 'sleep_efficiency', 'cardio_health',
            'high_activity', 'good_sleep', 'low_stress',
            'heart_rate_variability', 'blood_pressure_ratio', 'metabolic_efficiency',
            'sleep_quality_score', 'activity_intensity', 'health_score',
            'age_fitness_interaction', 'sleep_stress_interaction', 
            'activity_sleep_interaction', 'heart_rate_sleep_interaction'
        ]
        
        poly_feature_names = []
        for feature in poly_features:
            poly_feature_names.extend([f'{feature}_squared', f'{feature}_sqrt'])
        
        categorical_features = [
            'gender_encoded', 'intensity_encoded', 'activity_type_encoded',
            'day_of_week', 'month', 'is_weekend', 'has_health_condition', 'smoking_status_encoded'
        ]
        
        feature_columns = basic_features + derived_features + poly_feature_names + categorical_features
    else:
        # Use only basic features
        feature_columns = [
            'age', 'height_cm', 'weight_kg', 'bmi',
            'avg_heart_rate', 'resting_heart_rate', 
            'blood_pressure_systolic', 'blood_pressure_diastolic',
            'hours_sleep', 'stress_level', 'daily_steps', 
            'calories_burned', 'hydration_level', 'fitness_level'
        ]
    
    if config.verbose:
        print(f"✅ Enhanced synthetic data created: {len(df):,} records from {df['participant_id'].nunique()} participants")
        print(f"📊 Health Status Distribution:")
        print(f"  Unhealthy (0): {len(df[df['health_status'] == 0]):,} ({len(df[df['health_status'] == 0])/len(df)*100:.1f}%)")
        print(f"  Healthy (1): {len(df[df['health_status'] == 1]):,} ({len(df[df['health_status'] == 1])/len(df)*100:.1f}%)")
        print(f"📊 Features: {len(feature_columns)}")
    
    return df, feature_columns

def run_customizable_fhe_data_flow(config):
    """
    Run customizable FHE CKKS data flow with configurable parameters
    """
    print("🚀 RUNNING CUSTOMIZABLE FHE CKKS DATA FLOW")
    print("="*80)
    
    # Print configuration
    config.print_config()
    
    # Step 1: Load enhanced CSV data
    if config.verbose:
        print("\n📊 STEP 1: Loading Enhanced CSV Data")
        print("-" * 40)
    
    if config.use_real_data:
        df, feature_columns = load_and_enhance_csv_data(config)
    else:
        df, feature_columns = create_enhanced_synthetic_data(config)
    
    # Step 2: Import required modules
    if config.verbose:
        print("\n🔧 STEP 2: Importing FHE Modules")
        print("-" * 40)
    
    try:
        from src.real_fhe_ckks import RealFHEConfig, RealFHEEncryption, RealEncryptedModel
        from src.home_router_architecture import Smartwatch, SmartwatchConfig, HomeRouter, HomeRouterConfig, HomeRouterManager
        from src.fl import FLConfig
        if config.verbose:
            print("✅ All modules imported successfully")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    # Step 3: Initialize FHE system
    if config.verbose:
        print("\n🔐 STEP 3: Initializing Enhanced FHE System")
        print("-" * 40)
    
    # Create configurations
    fl_config = FLConfig(rounds=config.num_rounds, clients=config.num_clients)
    fhe_config = RealFHEConfig()
    
    # Initialize FHE encryption
    fhe_encryption = RealFHEEncryption(fhe_config)
    if config.verbose:
        print("✅ Enhanced FHE system initialized")
    
    # Step 4: Create smartwatches and home routers
    if config.verbose:
        print(f"\n⌚ STEP 4: Creating {config.num_clients} Smartwatches and {config.num_routers} Home Routers")
        print("-" * 40)
    
    # Create smartwatches
    smartwatches = {}
    for i in range(config.num_clients):
        config_sw = SmartwatchConfig(
            device_id=f"smartwatch_{i}",
            battery_level=np.random.uniform(85, 100),
            processing_power=np.random.uniform(0.8, 1.2)
        )
        smartwatches[config_sw.device_id] = Smartwatch(config_sw)
    
    # Create home routers
    router_configs = []
    for i in range(config.num_routers):
        router_config = HomeRouterConfig(
            router_id=f"home_router_{i}", 
            fhe_capability=True
        )
        router_configs.append(router_config)
    
    router_manager = HomeRouterManager(router_configs)
    
    if config.verbose:
        print(f"✅ Created {len(smartwatches)} smartwatches and {len(router_configs)} home routers")
    
    # Step 5: Load data to smartwatches
    if config.verbose:
        print(f"\n📊 STEP 5: Loading Data to {config.num_clients} Smartwatches")
        print("-" * 40)
    
    participants = df['participant_id'].unique()
    for i, device_id in enumerate(smartwatches.keys()):
        if i < len(participants):
            participant_id = participants[i]
            participant_data = df[df['participant_id'] == participant_id]
            
            # Ensure balanced classes
            if len(participant_data['health_status'].unique()) >= 2:
                smartwatches[device_id].load_sensor_data(participant_data, participant_id)
            else:
                # Create balanced dataset by sampling from multiple participants
                balanced_data = []
                balanced_data.append(participant_data)
                
                # Add data from other participants to ensure both classes
                other_participants = df[df['participant_id'] != participant_id]
                
                # Sample from healthy and unhealthy groups
                healthy_data = other_participants[other_participants['health_status'] == 1].sample(n=min(50, len(other_participants[other_participants['health_status'] == 1])))
                unhealthy_data = other_participants[other_participants['health_status'] == 0].sample(n=min(50, len(other_participants[other_participants['health_status'] == 0])))
                
                balanced_data.extend([healthy_data, unhealthy_data])
                
                # Combine all data
                final_balanced_data = pd.concat(balanced_data, ignore_index=True)
                
                # Ensure we have both classes
                if len(final_balanced_data['health_status'].unique()) >= 2:
                    smartwatches[device_id].load_sensor_data(final_balanced_data, participant_id)
                else:
                    # Fallback: use synthetic data
                    if config.verbose:
                        print(f"  ⚠️  {device_id}: Using synthetic data as fallback")
                    synthetic_df, _ = create_enhanced_synthetic_data(config)
                    smartwatches[device_id].load_sensor_data(synthetic_df, participant_id)
    
    if config.verbose:
        print("✅ Data loaded to all smartwatches")
    
    # Step 6: Initialize encrypted global model
    if config.verbose:
        print("\n🔐 STEP 6: Initializing Enhanced Encrypted Global Model")
        print("-" * 40)
    
    if config.verbose:
        print(f"📊 Selected {len(feature_columns)} features for training")
    
    # Create initial model
    lr_model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
    
    # Get balanced sample data for initialization
    healthy_samples = df[df['health_status'] == 1].sample(n=50, random_state=42)
    unhealthy_samples = df[df['health_status'] == 0].sample(n=50, random_state=42)
    sample_df = pd.concat([healthy_samples, unhealthy_samples]).sample(frac=1, random_state=42)
    
    sample_data = sample_df[feature_columns].fillna(0).astype(float)
    sample_target = sample_df['health_status']
    
    if config.verbose:
        print(f"📊 Sample data shape: {sample_data.shape}")
        print(f"📊 Sample target distribution: {np.bincount(sample_target)}")
    
    # Fit initial model
    lr_model.fit(sample_data, sample_target)
    
    # Extract weights
    lr_weights = lr_model.coef_[0]
    lr_bias = lr_model.intercept_[0]
    
    # Initialize encrypted model
    encrypted_global_model = RealEncryptedModel(lr_weights, lr_bias, fhe_encryption.context)
    
    if config.verbose:
        print(f"✅ Enhanced encrypted global model initialized with {len(feature_columns)} features")
    
    # Step 7: Run federated learning rounds
    if config.verbose:
        print(f"\n🔄 STEP 7: Running {config.num_rounds} Federated Learning Rounds")
        print("-" * 40)
    
    round_results = []
    test_data = None
    
    for round_num in range(config.num_rounds):
        if config.verbose:
            print(f"\n🔄 Round {round_num + 1}/{config.num_rounds}")
            print("-" * 30)
        
        round_start_time = time.time()
        
        # Phase 1: Local training
        if config.verbose:
            print("  📊 Phase 1: Local Training")
        local_updates = []
        sample_counts = []
        
        for device_id, smartwatch in smartwatches.items():
            if len(smartwatch.local_data) > 0:
                # Train simple logistic regression locally
                df_local = pd.DataFrame(smartwatch.local_data)
                
                # Get feature columns that exist in the data
                available_features = [col for col in feature_columns if col in df_local.columns]
                if not available_features:
                    if config.verbose:
                        print(f"    ⚠️  {device_id}: No matching features found, skipping")
                    continue
                
                X_train = df_local[available_features].fillna(0).astype(float)
                y_train = df_local['health_status'].values
                
                # Ensure we have both classes
                if len(np.unique(y_train)) < 2:
                    if config.verbose:
                        print(f"    ⚠️  {device_id}: Only one class in data, skipping")
                    continue
                
                # Create and train simple model
                local_model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
                local_model.fit(X_train, y_train)
                
                # Create model update including bias as last element for accurate aggregation
                weights_only = local_model.coef_[0]
                bias_value = float(local_model.intercept_[0])
                model_update = np.concatenate([weights_only, np.array([bias_value], dtype=float)])
                local_updates.append(model_update)
                sample_counts.append(len(X_train))
                
                if config.verbose:
                    print(f"    ✅ {device_id}: Trained model ({len(X_train)} samples)")
        
        if not local_updates:
            if config.verbose:
                print("    ⚠️  No valid updates available, skipping round")
            continue
        
        # Phase 2: Encrypt updates
        if config.verbose:
            print("  🔐 Phase 2: Encrypting Updates")
        encrypted_updates = []
        encryption_times = []
        
        for update in local_updates:
            start_time = time.time()
            encrypted_update, _ = fhe_encryption.encrypt_model_update(update)
            encryption_time = time.time() - start_time
            
            encrypted_updates.append(encrypted_update)
            encryption_times.append(encryption_time)
        
        avg_encryption_time = np.mean(encryption_times)
        if config.verbose:
            print(f"    ✅ Encrypted {len(encrypted_updates)} updates (avg: {avg_encryption_time:.4f}s)")
        
        # Phase 3: Server aggregates encrypted updates
        if config.verbose:
            print("  ☁️  Phase 3: Server Aggregates Encrypted Updates")
        start_time = time.time()
        
        aggregated_weights, aggregated_bias = fhe_encryption.aggregate_encrypted_updates(
            encrypted_updates, sample_counts
        )
        
        aggregation_time = time.time() - start_time
        if config.verbose:
            print(f"    ✅ Aggregated encrypted updates ({aggregation_time:.4f}s)")
        
        # Phase 4: Update global model
        if config.verbose:
            print("  🔄 Phase 4: Updating Global Model")
        # Decrypt aggregated vector and robustly split weights/bias if present
        aggregated_vector = np.array(aggregated_weights.decrypt())
        if len(aggregated_vector) == len(feature_columns) + 1:
            updated_weights = aggregated_vector[:-1]
            updated_bias = float(aggregated_vector[-1])
        else:
            updated_weights = aggregated_vector
            updated_bias = float(aggregated_bias) if aggregated_bias is not None else 0.0
        encrypted_global_model = RealEncryptedModel(updated_weights, updated_bias, fhe_encryption.context)
        
        # Phase 5: Decrypt and evaluate
        if config.verbose:
            print("  📊 Phase 5: Evaluating Global Model")
        
        # Create test data from all clients
        if test_data is None:
            all_X_test = []
            all_y_test = []
            
            for device_id, smartwatch in smartwatches.items():
                if len(smartwatch.local_data) > 0:
                    df_local = pd.DataFrame(smartwatch.local_data)
                    
                    # Get feature columns that exist in the data
                    available_features = [col for col in feature_columns if col in df_local.columns]
                    if available_features:
                        X_test = df_local[available_features].fillna(0).astype(float)
                        y_test = df_local['health_status'].values
                        
                        # Ensure we have both classes
                        if len(np.unique(y_test)) >= 2:
                            all_X_test.append(X_test)
                            all_y_test.append(y_test)
            
            if all_X_test:
                test_data = (np.vstack(all_X_test), np.hstack(all_y_test))
        
        if test_data is not None:
            X_test, y_test = test_data
            
            # Decrypt global model for evaluation
            global_weights = np.array(encrypted_global_model.encrypted_weights.decrypt())
            global_bias = float(encrypted_global_model.encrypted_bias.decrypt()[0])
            
            # Ensure weights match the number of features
            if len(global_weights) != X_test.shape[1]:
                if config.verbose:
                    print(f"    ⚠️  Weight dimension mismatch: {len(global_weights)} vs {X_test.shape[1]}")
                # Use only the matching features
                min_features = min(len(global_weights), X_test.shape[1])
                global_weights = global_weights[:min_features]
                X_test = X_test[:, :min_features]
            
            # Create global model
            global_model = LogisticRegression(random_state=42, max_iter=1000)
            global_model.coef_ = global_weights.reshape(1, -1)
            global_model.intercept_ = np.array([global_bias])
            global_model.classes_ = np.array([0, 1])
            
            # Evaluate
            y_pred = global_model.predict(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            
            if config.verbose:
                print(f"    📊 Global Model Performance:")
                print(f"      Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
                print(f"      F1-Score: {f1:.4f}")
                print(f"      Precision: {precision:.4f}")
                print(f"      Recall: {recall:.4f}")
        
        # Record round results
        round_time = time.time() - round_start_time
        round_result = {
            'round': round_num + 1,
            'accuracy': accuracy if test_data is not None else 0.0,
            'f1_score': f1 if test_data is not None else 0.0,
            'precision': precision if test_data is not None else 0.0,
            'recall': recall if test_data is not None else 0.0,
            'encryption_time': avg_encryption_time,
            'aggregation_time': aggregation_time,
            'total_time': round_time,
            'clients_participated': len(local_updates)
        }
        
        round_results.append(round_result)
        
        if config.verbose:
            print(f"  ⏱️  Round {round_num + 1} completed in {round_time:.4f}s")
    
    # Step 8: Final evaluation and analysis
    if config.verbose:
        print("\n📊 STEP 8: Final Evaluation and Analysis")
        print("-" * 40)
    
    if round_results:
        # Calculate final metrics
        final_accuracy = round_results[-1]['accuracy']
        final_f1 = round_results[-1]['f1_score']
        final_precision = round_results[-1]['precision']
        final_recall = round_results[-1]['recall']
        
        total_encryption_time = sum([r['encryption_time'] for r in round_results])
        total_aggregation_time = sum([r['aggregation_time'] for r in round_results])
        total_time = sum([r['total_time'] for r in round_results])
        
        print(f"🎯 FINAL RESULTS:")
        print(f"  📊 Accuracy: {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")
        print(f"  📊 F1-Score: {final_f1:.4f}")
        print(f"  📊 Precision: {final_precision:.4f}")
        print(f"  📊 Recall: {final_recall:.4f}")
        print(f"  ⏱️  Total Encryption Time: {total_encryption_time:.4f}s")
        print(f"  ⏱️  Total Aggregation Time: {total_aggregation_time:.4f}s")
        print(f"  ⏱️  Total Time: {total_time:.4f}s")
        print(f"  🔄 Rounds Completed: {len(round_results)}")
        print(f"  👥 Clients Participated: {round_results[-1]['clients_participated']}")
        print(f"  🏠 Routers Used: {config.num_routers}")
        print(f"  📊 Features Used: {len(feature_columns)}")
        
        # Step 9: Save results
        if config.verbose:
            print("\n💾 STEP 9: Saving Custom Results")
            print("-" * 40)
        
        # Create results directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = f"custom_fhe_results_{config.num_clients}c_{config.num_rounds}r_{config.num_routers}rt_{timestamp}"
        os.makedirs(results_dir, exist_ok=True)
        
        # Save detailed results
        results_summary = {
            'experiment_type': f'Customizable FHE CKKS - {config.num_clients} Clients, {config.num_rounds} Rounds, {config.num_routers} Routers',
            'timestamp': timestamp,
            'configuration': {
                'clients': config.num_clients,
                'rounds': config.num_rounds,
                'routers': config.num_routers,
                'use_real_data': config.use_real_data,
                'feature_engineering': config.feature_engineering,
                'features': len(feature_columns),
                'fhe_config': {
                    'poly_modulus_degree': 8192,
                    'coeff_mod_bit_sizes': [40, 40, 40, 40],
                    'scale_bits': 40
                }
            },
            'final_performance': {
                'accuracy': final_accuracy,
                'f1_score': final_f1,
                'precision': final_precision,
                'recall': final_recall
            },
            'performance_metrics': {
                'total_encryption_time': total_encryption_time,
                'total_aggregation_time': total_aggregation_time,
                'total_time': total_time,
                'avg_encryption_time_per_round': total_encryption_time / len(round_results),
                'avg_aggregation_time_per_round': total_aggregation_time / len(round_results)
            },
            'round_results': round_results,
            'feature_columns': feature_columns
        }
        
        # Save results
        results_file = os.path.join(results_dir, "custom_fhe_results.json")
        with open(results_file, 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        print(f"✅ Custom results saved to: {results_file}")
    
    print(f"\n🎉 CUSTOMIZABLE FHE CKKS DATA FLOW COMPLETED SUCCESSFULLY!")
    print("="*80)
    
    return True

# =============================================================================
# EXAMPLE USAGE - CUSTOMIZE THESE PARAMETERS!
# =============================================================================

def main():
    """
    Main entry point with CLI flags to customize clients, rounds, and routers.
    """
    parser = argparse.ArgumentParser(description="Customizable FHE CKKS Data Flow")
    parser.add_argument("--clients", type=int, default=4, help="Number of smartwatch clients")
    parser.add_argument("--rounds", type=int, default=3, help="Number of federated learning rounds")
    parser.add_argument("--routers", type=int, default=1, help="Number of home routers")
    parser.add_argument("--real-data", action="store_true", help="Use real CSV data (default: synthetic)")
    parser.add_argument("--synthetic", action="store_true", help="Force synthetic data (overrides --real-data)")
    parser.add_argument("--features-47", dest="features_47", action="store_true", help="Use 47 engineered features")
    parser.add_argument("--features-basic", dest="features_basic", action="store_true", help="Use basic features only")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--quiet", action="store_true", help="Minimal output")

    args = parser.parse_args()

    use_real_data = True if args.real_data and not args.synthetic else False
    feature_engineering = True if args.features_47 and not args.features_basic else False
    verbose = True if args.verbose and not args.quiet else False

    print("🔧 CUSTOMIZABLE FHE CKKS DATA FLOW")
    print("="*50)

    config = CustomConfig(
        num_clients=args.clients,
        num_rounds=args.rounds,
        num_routers=args.routers,
        use_real_data=use_real_data,
        feature_engineering=feature_engineering,
        verbose=verbose
    )

    print(f"\n🚀 Running experiment with:")
    print(f"  👥 {config.num_clients} clients")
    print(f"  🔄 {config.num_rounds} rounds")
    print(f"  🏠 {config.num_routers} routers")
    print(f"  📊 Real data: {config.use_real_data}")
    print(f"  🛠️  Feature engineering: {config.feature_engineering}")

    success = run_customizable_fhe_data_flow(config)

    if success:
        print("\n✅ Customizable FHE data flow completed successfully!")
        print("📊 Check the results directory for detailed analysis")
    else:
        print("\n❌ Customizable FHE data flow failed!")
        print("🔍 Check the error messages above for troubleshooting")

if __name__ == "__main__":
    main()
