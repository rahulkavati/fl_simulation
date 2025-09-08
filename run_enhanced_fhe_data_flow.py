"""
Enhanced FHE CKKS Data Flow with Advanced Feature Engineering
Research-Paper Ready Implementation with Real FHE and Advanced ML
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def load_and_enhance_csv_data():
    """
    Load real health fitness data with advanced feature engineering
    """
    print("📊 Loading Real Health Fitness Data with Advanced Feature Engineering...")
    
    csv_path = "data/health_fitness_data.csv"
    if not os.path.exists(csv_path):
        print(f"❌ Health data not found at {csv_path}")
        print("Creating synthetic data as fallback...")
        return create_enhanced_synthetic_data()
    
    # Load the real CSV data
    df = pd.read_csv(csv_path)
    print(f"✅ Loaded {len(df):,} records from {df['participant_id'].nunique()} participants")
    
    # Advanced preprocessing
    df = df.copy()
    df['health_condition'] = df['health_condition'].fillna('None')
    
    # Handle string columns that might contain non-numeric values
    string_columns = ['gender', 'intensity', 'activity_type', 'health_condition']
    for col in string_columns:
        if col in df.columns:
            df[col] = df[col].astype(str)
    
    # Convert fitness_level to binary health status
    fitness_threshold = df['fitness_level'].median()
    df['health_status'] = (df['fitness_level'] >= fitness_threshold).astype(int)
    
    print(f"📊 Original Health Status Distribution:")
    print(f"  Unhealthy (0): {len(df[df['health_status'] == 0]):,} ({len(df[df['health_status'] == 0])/len(df)*100:.1f}%)")
    print(f"  Healthy (1): {len(df[df['health_status'] == 1]):,} ({len(df[df['health_status'] == 1])/len(df)*100:.1f}%)")
    print(f"  Fitness Threshold: {fitness_threshold:.2f}")
    
    # Advanced feature engineering
    print("🔧 Applying Advanced Feature Engineering...")
    
    # 1. Derived features
    df['steps_per_calorie'] = df['daily_steps'] / (df['calories_burned'] + 1)
    df['sleep_efficiency'] = df['hours_sleep'] / (df['stress_level'] + 1)
    df['cardio_health'] = (df['resting_heart_rate'] <= 70).astype(int)
    df['high_activity'] = (df['daily_steps'] >= 10000).astype(int)
    df['good_sleep'] = (df['hours_sleep'] >= 7.0).astype(int)
    df['low_stress'] = (df['stress_level'] <= 5.0).astype(int)
    
    # 2. Categorical encoding
    le_gender = LabelEncoder()
    le_intensity = LabelEncoder()
    le_activity = LabelEncoder()
    
    df['gender_encoded'] = le_gender.fit_transform(df['gender'])
    df['intensity_encoded'] = le_intensity.fit_transform(df['intensity'])
    df['activity_type_encoded'] = le_activity.fit_transform(df['activity_type'])
    
    # 3. Polynomial features for key variables
    poly_features = ['age', 'fitness_level', 'daily_steps', 'calories_burned']
    for feature in poly_features:
        df[f'{feature}_squared'] = df[feature] ** 2
        df[f'{feature}_log'] = np.log(df[feature] + 1)
    
    # 4. Interaction features
    df['age_fitness_interaction'] = df['age'] * df['fitness_level']
    df['sleep_stress_interaction'] = df['hours_sleep'] * df['stress_level']
    df['steps_calories_interaction'] = df['daily_steps'] * df['calories_burned']
    
    # 5. Temporal features
    df['day_of_week'] = pd.to_datetime(df['date']).dt.dayofweek
    df['month'] = pd.to_datetime(df['date']).dt.month
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # 6. Health risk indicators
    df['high_bmi'] = (df['bmi'] > 25).astype(int)
    df['high_heart_rate'] = (df['resting_heart_rate'] > 80).astype(int)
    df['low_sleep'] = (df['hours_sleep'] < 6).astype(int)
    df['high_stress'] = (df['stress_level'] > 7).astype(int)
    
    print(f"✅ Feature engineering completed. Total features: {len(df.columns)}")
    
    # Balance the data by participant
    balanced_data = []
    participants = df['participant_id'].unique()
    
    print(f"📊 Balancing data for {len(participants)} participants...")
    
    for participant_id in participants:
        participant_data = df[df['participant_id'] == participant_id]
        
        # Check if participant has both classes
        unique_classes = participant_data['health_status'].unique()
        
        if len(unique_classes) >= 2:
            # Participant has both classes - use their data
            balanced_data.append(participant_data)
        else:
            # Participant has only one class - sample from other participants
            other_data = df[df['participant_id'] != participant_id]
            
            # Get some samples from the missing class
            missing_class = 1 if unique_classes[0] == 0 else 0
            missing_class_data = other_data[other_data['health_status'] == missing_class]
            
            if len(missing_class_data) > 0:
                # Sample some data from the missing class
                sample_size = min(50, len(missing_class_data))
                sampled_data = missing_class_data.sample(n=sample_size, random_state=42)
                
                # Combine participant data with sampled data
                combined_data = pd.concat([participant_data, sampled_data], ignore_index=True)
                balanced_data.append(combined_data)
            else:
                # Fallback: use participant data as is
                balanced_data.append(participant_data)
    
    # Combine all balanced data
    final_df = pd.concat(balanced_data, ignore_index=True)
    
    print(f"📊 Balanced Health Status Distribution:")
    print(f"  Unhealthy (0): {len(final_df[final_df['health_status'] == 0]):,} ({len(final_df[final_df['health_status'] == 0])/len(final_df)*100:.1f}%)")
    print(f"  Healthy (1): {len(final_df[final_df['health_status'] == 1]):,} ({len(final_df[final_df['health_status'] == 1])/len(final_df)*100:.1f}%)")
    
    return final_df

def create_enhanced_synthetic_data():
    """
    Create enhanced synthetic data with advanced features
    """
    print("📊 Creating Enhanced Synthetic Health Data...")
    
    np.random.seed(42)
    n_samples = 2000
    n_participants = 15
    
    data = []
    for participant_id in range(n_participants):
        participant_samples = n_samples // n_participants
        
        # Generate realistic health data
        age = np.random.normal(35, 10, participant_samples)
        height = np.random.normal(170, 10, participant_samples)
        weight = np.random.normal(70, 15, participant_samples)
        bmi = weight / ((height / 100) ** 2)
        
        # Generate correlated features
        fitness_level = np.random.normal(5, 2, participant_samples)
        daily_steps = np.random.normal(8000, 3000, participant_samples)
        calories_burned = daily_steps * 0.04 + np.random.normal(0, 100, participant_samples)
        hours_sleep = np.random.normal(7.5, 1.5, participant_samples)
        stress_level = np.random.normal(5, 2, participant_samples)
        resting_heart_rate = np.random.normal(70, 10, participant_samples)
        
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
            'height': height,
            'weight': weight,
            'bmi': bmi,
            'fitness_level': fitness_level,
            'daily_steps': daily_steps,
            'calories_burned': calories_burned,
            'hours_sleep': hours_sleep,
            'stress_level': stress_level,
            'resting_heart_rate': resting_heart_rate,
            'health_status': health_status,
            'gender': np.random.choice(['Male', 'Female'], participant_samples),
            'intensity': np.random.choice(['Low', 'Medium', 'High'], participant_samples),
            'activity_type': np.random.choice(['Walking', 'Running', 'Cycling'], participant_samples),
            'health_condition': np.random.choice(['None', 'Diabetes', 'Hypertension'], participant_samples),
            'date': pd.date_range('2023-01-01', periods=participant_samples, freq='D')
        })
        
        data.append(participant_df)
    
    df = pd.concat(data, ignore_index=True)
    
    # Apply same feature engineering as real data
    df['steps_per_calorie'] = df['daily_steps'] / (df['calories_burned'] + 1)
    df['sleep_efficiency'] = df['hours_sleep'] / (df['stress_level'] + 1)
    df['cardio_health'] = (df['resting_heart_rate'] <= 70).astype(int)
    df['high_activity'] = (df['daily_steps'] >= 10000).astype(int)
    df['good_sleep'] = (df['hours_sleep'] >= 7.0).astype(int)
    df['low_stress'] = (df['stress_level'] <= 5.0).astype(int)
    
    # Categorical encoding
    le_gender = LabelEncoder()
    le_intensity = LabelEncoder()
    le_activity = LabelEncoder()
    
    df['gender_encoded'] = le_gender.fit_transform(df['gender'])
    df['intensity_encoded'] = le_intensity.fit_transform(df['intensity'])
    df['activity_type_encoded'] = le_activity.fit_transform(df['activity_type'])
    
    # Polynomial features
    poly_features = ['age', 'fitness_level', 'daily_steps', 'calories_burned']
    for feature in poly_features:
        df[f'{feature}_squared'] = df[feature] ** 2
        df[f'{feature}_log'] = np.log(df[feature] + 1)
    
    # Interaction features
    df['age_fitness_interaction'] = df['age'] * df['fitness_level']
    df['sleep_stress_interaction'] = df['hours_sleep'] * df['stress_level']
    df['steps_calories_interaction'] = df['daily_steps'] * df['calories_burned']
    
    # Temporal features
    df['day_of_week'] = pd.to_datetime(df['date']).dt.dayofweek
    df['month'] = pd.to_datetime(df['date']).dt.month
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # Health risk indicators
    df['high_bmi'] = (df['bmi'] > 25).astype(int)
    df['high_heart_rate'] = (df['resting_heart_rate'] > 80).astype(int)
    df['low_sleep'] = (df['hours_sleep'] < 6).astype(int)
    df['high_stress'] = (df['stress_level'] > 7).astype(int)
    
    print(f"✅ Enhanced synthetic data created: {len(df):,} records from {df['participant_id'].nunique()} participants")
    print(f"📊 Health Status Distribution:")
    print(f"  Unhealthy (0): {len(df[df['health_status'] == 0]):,} ({len(df[df['health_status'] == 0])/len(df)*100:.1f}%)")
    print(f"  Healthy (1): {len(df[df['health_status'] == 1]):,} ({len(df[df['health_status'] == 1])/len(df)*100:.1f}%)")
    
    return df

def create_ensemble_model():
    """
    Create ensemble model for better performance
    """
    # Individual models
    lr_model = LogisticRegression(
        random_state=42,
        max_iter=1000,
        class_weight='balanced',
        solver='lbfgs'
    )
    
    rf_model = RandomForestClassifier(
        n_estimators=50,
        random_state=42,
        class_weight='balanced',
        max_depth=10
    )
    
    # Ensemble model
    ensemble_model = VotingClassifier(
        estimators=[
            ('lr', lr_model),
            ('rf', rf_model)
        ],
        voting='soft'
    )
    
    return ensemble_model

def run_federated_learning_rounds_simple(fl_config, smartwatches, fhe_encryption, encrypted_global_model, feature_columns):
    """
    Run simple federated learning rounds (fallback)
    """
    print("\n🔄 Running Simple Federated Learning Rounds")
    print("-" * 40)
    
    round_results = []
    test_data = None
    
    for round_num in range(fl_config.rounds):
        print(f"\n🔄 Round {round_num + 1}/{fl_config.rounds}")
        print("-" * 30)
        
        round_start_time = time.time()
        
        # Phase 1: Local training
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
                    print(f"    ⚠️  {device_id}: No matching features found, skipping")
                    continue
                
                X_train = df_local[available_features].fillna(0).astype(float)
                y_train = df_local['health_status'].values
                
                # Ensure we have both classes
                if len(np.unique(y_train)) < 2:
                    print(f"    ⚠️  {device_id}: Only one class in data, skipping")
                    continue
                
                # Create and train simple model
                local_model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
                local_model.fit(X_train, y_train)
                
                # Create model update
                model_update = np.append(local_model.coef_[0], local_model.intercept_[0])
                local_updates.append(model_update)
                sample_counts.append(len(X_train))
                
                print(f"    ✅ {device_id}: Trained simple model ({len(X_train)} samples)")
        
        # Phase 2: Encrypt updates
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
        print(f"    ✅ Encrypted {len(encrypted_updates)} updates (avg: {avg_encryption_time:.4f}s)")
        
        # Phase 3: Server aggregates encrypted updates
        print("  ☁️  Phase 3: Server Aggregates Encrypted Updates")
        start_time = time.time()
        
        aggregated_weights, aggregated_bias = fhe_encryption.aggregate_encrypted_updates(
            encrypted_updates, sample_counts
        )
        
        aggregation_time = time.time() - start_time
        print(f"    ✅ Aggregated encrypted updates ({aggregation_time:.4f}s)")
        
        # Phase 4: Update global model
        print("  🔄 Phase 4: Updating Global Model")
        encrypted_global_model = RealEncryptedModel(
            np.array(aggregated_weights.decrypt()), 
            aggregated_bias, 
            fhe_encryption.context
        )
        
        # Phase 5: Decrypt and evaluate
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
            global_bias = encrypted_global_model.encrypted_bias.decrypt()[0]
            
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
        
        print(f"  ⏱️  Round {round_num + 1} completed in {round_time:.4f}s")
    
    # Final results
    print("\n🎯 SIMPLE FHE RESULTS:")
    print(f"  📊 Final Accuracy: {round_results[-1]['accuracy']:.4f} ({round_results[-1]['accuracy']*100:.2f}%)")
    print(f"  📊 Final F1-Score: {round_results[-1]['f1_score']:.4f}")
    
    return True

def run_enhanced_fhe_data_flow():
    """
    Run enhanced FHE CKKS data flow with advanced ML techniques
    """
    print("🚀 RUNNING ENHANCED FHE CKKS DATA FLOW")
    print("="*80)
    
    # Step 1: Load enhanced CSV data
    print("\n📊 STEP 1: Loading Enhanced CSV Data")
    print("-" * 40)
    df = load_and_enhance_csv_data()
    
    # Step 2: Import required modules
    print("\n🔧 STEP 2: Importing FHE Modules")
    print("-" * 40)
    try:
        from src.real_fhe_ckks import RealFHEConfig, RealFHEEncryption, RealEncryptedModel
        from src.home_router_architecture import Smartwatch, SmartwatchConfig, HomeRouter, HomeRouterConfig, HomeRouterManager
        from src.fl import FLConfig
        print("✅ All modules imported successfully")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    # Step 3: Initialize FHE system
    print("\n🔐 STEP 3: Initializing Enhanced FHE System")
    print("-" * 40)
    
    # Create configurations
    fl_config = FLConfig(rounds=5, clients=8)
    fhe_config = RealFHEConfig()
    
    # Initialize FHE encryption
    fhe_encryption = RealFHEEncryption(fhe_config)
    print("✅ Enhanced FHE system initialized")
    
    # Step 4: Create smartwatches and home routers
    print("\n⌚ STEP 4: Creating Smartwatches and Home Routers")
    print("-" * 40)
    
    # Create smartwatches
    smartwatches = {}
    for i in range(fl_config.clients):
        config = SmartwatchConfig(
            device_id=f"smartwatch_{i}",
            battery_level=np.random.uniform(85, 100),
            processing_power=np.random.uniform(0.8, 1.2)
        )
        smartwatches[config.device_id] = Smartwatch(config)
    
    # Create home routers
    router_configs = [
        HomeRouterConfig(router_id="home_router_0", fhe_capability=True),
        HomeRouterConfig(router_id="home_router_1", fhe_capability=True)
    ]
    router_manager = HomeRouterManager(router_configs)
    
    print(f"✅ Created {len(smartwatches)} smartwatches and {len(router_configs)} home routers")
    
    # Step 5: Load data to smartwatches
    print("\n📊 STEP 5: Loading Enhanced Data to Smartwatches")
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
                    print(f"  ⚠️  {device_id}: Using synthetic data as fallback")
                    synthetic_df = create_enhanced_synthetic_data()
                    smartwatches[device_id].load_sensor_data(synthetic_df, participant_id)
    
    print("✅ Enhanced data loaded to all smartwatches")
    
    # Step 6: Initialize encrypted global model
    print("\n🔐 STEP 6: Initializing Enhanced Encrypted Global Model")
    print("-" * 40)
    
    # Get feature columns (exclude non-feature columns and ensure numeric)
    exclude_columns = [
        'participant_id', 'health_status', 'date', 'gender', 'intensity', 
        'activity_type', 'health_condition'
    ]
    
    # Only include numeric columns
    feature_columns = []
    for col in df.columns:
        if col not in exclude_columns and pd.api.types.is_numeric_dtype(df[col]):
            feature_columns.append(col)
    
    print(f"📊 Selected {len(feature_columns)} numeric features for training")
    
    # Create initial model with ensemble
    ensemble_model = create_ensemble_model()
    
    # Get sample data for initialization
    # Ensure balanced sample data
    healthy_samples = df[df['health_status'] == 1].sample(n=50, random_state=42)
    unhealthy_samples = df[df['health_status'] == 0].sample(n=50, random_state=42)
    sample_df = pd.concat([healthy_samples, unhealthy_samples]).sample(frac=1, random_state=42)
    
    sample_data = sample_df[feature_columns]
    sample_target = sample_df['health_status']
    
    # Ensure sample data is numeric and has no NaN values
    sample_data = sample_data.fillna(0)
    sample_data = sample_data.astype(float)
    
    print(f"📊 Sample data shape: {sample_data.shape}")
    print(f"📊 Sample target distribution: {np.bincount(sample_target)}")
    
    # Fit ensemble model
    try:
        ensemble_model.fit(sample_data, sample_target)
        print("✅ Ensemble model fitted successfully")
    except Exception as e:
        print(f"⚠️  Ensemble model fitting failed: {e}")
        print("📊 Falling back to simple logistic regression...")
        
        # Fallback to simple logistic regression
        lr_model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
        lr_model.fit(sample_data, sample_target)
        
        # Extract weights
        lr_weights = lr_model.coef_[0]
        lr_bias = lr_model.intercept_[0]
        
        # Initialize encrypted model
        encrypted_global_model = RealEncryptedModel(lr_weights, lr_bias, fhe_encryption.context)
        
        print(f"✅ Simple encrypted global model initialized with {len(feature_columns)} features")
        return run_federated_learning_rounds_simple(fl_config, smartwatches, fhe_encryption, encrypted_global_model, feature_columns)
    
    # Extract weights from logistic regression component
    lr_weights = ensemble_model.named_estimators_['lr'].coef_[0]
    lr_bias = ensemble_model.named_estimators_['lr'].intercept_[0]
    
    # Initialize encrypted model
    encrypted_global_model = RealEncryptedModel(lr_weights, lr_bias, fhe_encryption.context)
    
    print(f"✅ Enhanced encrypted global model initialized with {len(feature_columns)} features")
    
    # Step 7: Run federated learning rounds
    print("\n🔄 STEP 7: Running Enhanced Federated Learning Rounds")
    print("-" * 40)
    
    round_results = []
    test_data = None
    
    for round_num in range(fl_config.rounds):
        print(f"\n🔄 Round {round_num + 1}/{fl_config.rounds}")
        print("-" * 30)
        
        round_start_time = time.time()
        
        # Phase 1: Local training
        print("  📊 Phase 1: Local Training with Ensemble Models")
        local_updates = []
        sample_counts = []
        
        for device_id, smartwatch in smartwatches.items():
            if len(smartwatch.local_data) > 0:
                # Train ensemble model locally
                df_local = pd.DataFrame(smartwatch.local_data)
                
                # Get feature columns that exist in the data
                available_features = [col for col in feature_columns if col in df_local.columns]
                if not available_features:
                    print(f"    ⚠️  {device_id}: No matching features found, skipping")
                    continue
                
                X_train = df_local[available_features].fillna(0).astype(float)
                y_train = df_local['health_status'].values
                
                # Ensure we have both classes
                if len(np.unique(y_train)) < 2:
                    print(f"    ⚠️  {device_id}: Only one class in data, skipping")
                    continue
                
                # Create and train ensemble model
                local_ensemble = create_ensemble_model()
                local_ensemble.fit(X_train, y_train)
                
                # Extract logistic regression weights
                lr_weights = local_ensemble.named_estimators_['lr'].coef_[0]
                lr_bias = local_ensemble.named_estimators_['lr'].intercept_[0]
                
                # Create model update
                model_update = np.append(lr_weights, lr_bias)
                local_updates.append(model_update)
                sample_counts.append(len(X_train))
                
                print(f"    ✅ {device_id}: Trained ensemble model ({len(X_train)} samples)")
        
        # Phase 2: Encrypt updates
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
        print(f"    ✅ Encrypted {len(encrypted_updates)} updates (avg: {avg_encryption_time:.4f}s)")
        
        # Phase 3: Server aggregates encrypted updates
        print("  ☁️  Phase 3: Server Aggregates Encrypted Updates")
        start_time = time.time()
        
        aggregated_weights, aggregated_bias = fhe_encryption.aggregate_encrypted_updates(
            encrypted_updates, sample_counts
        )
        
        aggregation_time = time.time() - start_time
        print(f"    ✅ Aggregated encrypted updates ({aggregation_time:.4f}s)")
        
        # Phase 4: Update global model
        print("  🔄 Phase 4: Updating Global Model")
        encrypted_global_model = RealEncryptedModel(
            np.array(aggregated_weights.decrypt()), 
            aggregated_bias, 
            fhe_encryption.context
        )
        
        # Phase 5: Decrypt and evaluate
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
            global_bias = encrypted_global_model.encrypted_bias.decrypt()[0]
            
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
        
        print(f"  ⏱️  Round {round_num + 1} completed in {round_time:.4f}s")
    
    # Step 8: Final evaluation and analysis
    print("\n📊 STEP 8: Final Evaluation and Analysis")
    print("-" * 40)
    
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
    
    # Step 9: Save results
    print("\n💾 STEP 9: Saving Enhanced Results")
    print("-" * 40)
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"enhanced_fhe_results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Save detailed results
    results_summary = {
        'experiment_type': 'Enhanced FHE CKKS with Advanced ML',
        'timestamp': timestamp,
        'configuration': {
            'rounds': fl_config.rounds,
            'clients': fl_config.clients,
            'features': len(feature_columns),
            'fhe_config': {
                'poly_modulus_degree': fhe_config.poly_modulus_degree,
                'coeff_mod_bit_sizes': fhe_config.coeff_mod_bit_sizes,
                'scale_bits': fhe_config.scale_bits
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
    results_file = os.path.join(results_dir, "enhanced_fhe_results.json")
    with open(results_file, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"✅ Enhanced results saved to: {results_file}")
    
    # Step 10: Statistical analysis
    print("\n📊 STEP 10: Statistical Analysis")
    print("-" * 40)
    
    # Calculate improvement over rounds
    if len(round_results) > 1:
        initial_accuracy = round_results[0]['accuracy']
        final_accuracy = round_results[-1]['accuracy']
        improvement = final_accuracy - initial_accuracy
        
        print(f"📈 Performance Improvement:")
        print(f"  Initial Accuracy: {initial_accuracy:.4f} ({initial_accuracy*100:.2f}%)")
        print(f"  Final Accuracy: {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")
        print(f"  Improvement: {improvement:.4f} ({improvement*100:.2f}%)")
        
        # Calculate convergence
        accuracy_values = [r['accuracy'] for r in round_results]
        if len(accuracy_values) > 2:
            # Calculate standard deviation of last 3 rounds
            last_3_rounds = accuracy_values[-3:]
            convergence_std = np.std(last_3_rounds)
            print(f"  Convergence Stability: {convergence_std:.6f}")
            
            if convergence_std < 0.01:
                print("  ✅ Model has converged (stable performance)")
            else:
                print("  ⚠️  Model may need more rounds for convergence")
    
    print("\n🎉 ENHANCED FHE CKKS DATA FLOW COMPLETED SUCCESSFULLY!")
    print("="*80)
    
    return True

if __name__ == "__main__":
    success = run_enhanced_fhe_data_flow()
    if success:
        print("\n✅ Enhanced FHE data flow completed successfully!")
        print("📊 Check the results directory for detailed analysis")
    else:
        print("\n❌ Enhanced FHE data flow failed!")
        print("🔍 Check the error messages above for troubleshooting")
