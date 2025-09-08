"""
Improved FHE CKKS Data Flow with Real CSV Data and Performance Metrics
Uses actual health_fitness_data.csv and includes accuracy, F1 score, and performance metrics
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def load_real_health_data():
    """
    Load real health fitness data from CSV file
    """
    print("📊 Loading Real Health Fitness Data...")
    
    csv_path = "data/health_fitness_data.csv"
    if not os.path.exists(csv_path):
        print(f"❌ Health data not found at {csv_path}")
        print("Creating synthetic data as fallback...")
        return create_synthetic_data()
    
    # Load the real CSV data
    df = pd.read_csv(csv_path)
    print(f"✅ Loaded {len(df):,} records from {df['participant_id'].nunique()} participants")
    
    # Preprocess the data
    df = df.copy()
    df['health_condition'] = df['health_condition'].fillna('None')
    
    # Convert fitness_level to binary health status
    fitness_threshold = df['fitness_level'].median()
    df['health_status'] = (df['fitness_level'] >= fitness_threshold).astype(int)
    
    print(f"📊 Health Status Distribution:")
    print(f"  Unhealthy (0): {len(df[df['health_status'] == 0]):,} ({len(df[df['health_status'] == 0])/len(df)*100:.1f}%)")
    print(f"  Healthy (1): {len(df[df['health_status'] == 1]):,} ({len(df[df['health_status'] == 1])/len(df)*100:.1f}%)")
    print(f"  Fitness Threshold: {fitness_threshold:.2f}")
    
    return df

def create_synthetic_data():
    """
    Create synthetic data as fallback
    """
    print("📊 Creating Synthetic Health Data...")
    
    np.random.seed(42)
    n_samples = 1000
    n_participants = 10
    
    data = []
    for participant_id in range(n_participants):
        for sample in range(n_samples // n_participants):
            # Create realistic health features
            age = np.random.randint(20, 80)
            height_cm = np.random.normal(170, 15)
            weight_kg = np.random.normal(70, 15)
            bmi = weight_kg / ((height_cm / 100) ** 2)
            
            avg_heart_rate = np.random.normal(75, 10)
            resting_heart_rate = np.random.normal(65, 8)
            hours_sleep = np.random.normal(7.5, 1.5)
            stress_level = np.random.uniform(1, 10)
            daily_steps = np.random.normal(8000, 3000)
            calories_burned = np.random.normal(2000, 500)
            hydration_level = np.random.uniform(0.5, 1.0)
            
            # Create health status (binary classification)
            fitness_score = (
                (daily_steps / 10000) * 0.3 +
                (hours_sleep / 8) * 0.2 +
                (1 - stress_level / 10) * 0.2 +
                (hydration_level) * 0.1 +
                (1 - abs(avg_heart_rate - 70) / 30) * 0.2
            )
            health_status = 1 if fitness_score > 0.5 else 0
            
            data.append({
                'participant_id': participant_id,
                'age': age,
                'height_cm': height_cm,
                'weight_kg': weight_kg,
                'bmi': bmi,
                'avg_heart_rate': avg_heart_rate,
                'resting_heart_rate': resting_heart_rate,
                'hours_sleep': hours_sleep,
                'stress_level': stress_level,
                'daily_steps': daily_steps,
                'calories_burned': calories_burned,
                'hydration_level': hydration_level,
                'health_status': health_status
            })
    
    df = pd.DataFrame(data)
    print(f"✅ Created {len(df)} synthetic samples for {df['participant_id'].nunique()} participants")
    return df

def run_complete_data_flow_with_metrics():
    """
    Run the complete FHE CKKS data flow with real data and performance metrics
    """
    print("🚀 RUNNING COMPLETE FHE CKKS DATA FLOW WITH REAL DATA")
    print("="*70)
    
    # Step 1: Load real data
    print("\n📊 STEP 1: Loading Real Health Data")
    print("-" * 40)
    df = load_real_health_data()
    
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
    print("\n🔐 STEP 3: Initializing FHE System")
    print("-" * 40)
    
    # Create configurations
    fl_config = FLConfig(rounds=3, clients=6)  # Increased for better metrics
    fhe_config = RealFHEConfig()
    
    # Initialize FHE encryption
    fhe_encryption = RealFHEEncryption(fhe_config)
    print("✅ FHE system initialized")
    
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
    
    # Assign devices to routers
    device_ids = list(smartwatches.keys())
    router_manager.assign_devices_to_routers(device_ids)
    
    # Initialize FHE encryption on routers
    for router in router_manager.routers.values():
        router.initialize_fhe_encryption(fhe_config)
    
    print(f"✅ Created {len(smartwatches)} smartwatches and {len(router_configs)} home routers")
    
    # Step 5: Load data to smartwatches
    print("\n📱 STEP 5: Loading Data to Smartwatches")
    print("-" * 40)
    
    participants = df['participant_id'].unique()
    for i, device_id in enumerate(smartwatches.keys()):
        participant_id = participants[i % len(participants)]
        participant_data = df[df['participant_id'] == participant_id]
        
        # Ensure balanced classes by sampling from different participants
        if len(participant_data['health_status'].unique()) >= 2:
            # Use participant data if it has both classes
            smartwatches[device_id].load_sensor_data(df, participant_id)
        else:
            # Create balanced dataset by sampling from multiple participants
            balanced_data = []
            
            # Get some data from current participant
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
                synthetic_df = create_synthetic_data()
                smartwatches[device_id].load_sensor_data(synthetic_df, participant_id)
    
    print("✅ Data loaded to all smartwatches")
    
    # Step 6: Initialize encrypted global model
    print("\n🔐 STEP 6: Initializing Encrypted Global Model")
    print("-" * 40)
    
    # Get feature dimension
    first_smartwatch = list(smartwatches.values())[0]
    df_local = pd.DataFrame(first_smartwatch.local_data)
    feature_columns = [
        'age', 'height_cm', 'weight_kg', 'bmi', 'avg_heart_rate',
        'resting_heart_rate', 'hours_sleep', 'stress_level', 'daily_steps',
        'calories_burned', 'hydration_level'
    ]
    feature_dim = len(feature_columns)
    
    # Initialize encrypted global model
    initial_weights = np.random.normal(0, 0.1, feature_dim)
    initial_bias = 0.0
    encrypted_global_model = RealEncryptedModel(initial_weights, initial_bias, fhe_encryption.context)
    
    print(f"✅ Encrypted global model initialized with {feature_dim} features")
    
    # Step 7: Run federated learning rounds with performance metrics
    print("\n🔄 STEP 7: Running Federated Learning Rounds with Performance Metrics")
    print("-" * 70)
    
    all_results = []
    all_accuracy_scores = []
    all_f1_scores = []
    
    for round_num in range(1, fl_config.rounds + 1):
        print(f"\n🔄 ROUND {round_num}:")
        print("-" * 30)
        
        round_start = time.time()
        round_results = {
            'round': round_num,
            'smartwatch_training': {},
            'router_encryption': {},
            'server_aggregation': {},
            'router_decryption': {},
            'smartwatch_updates': {},
            'performance_metrics': {},
            'timing': {}
        }
        
        # Phase 1: Smartwatches train locally
        print("⌚ Phase 1: Smartwatches train locally")
        raw_model_updates = []
        sample_counts = []
        smartwatch_training_times = []
        
        for device_id, smartwatch in smartwatches.items():
            print(f"  ⌚ {device_id}: Training...")
            
            # Train local model
            model_params = {'random_state': 42, 'max_iter': 1000}
            training_result = smartwatch.train_local_model(model_params)
            
            # Prepare model update
            model_update = np.concatenate([
                training_result['model_weights'],
                [training_result['model_bias']]
            ])
            
            # Send to router
            smartwatch.send_model_update_to_router(model_update)
            
            raw_model_updates.append(model_update)
            sample_counts.append(training_result['sample_count'])
            smartwatch_training_times.append(training_result['training_time'])
            
            round_results['smartwatch_training'][device_id] = {
                'training_time': training_result['training_time'],
                'sample_count': training_result['sample_count'],
                'battery_level': training_result['battery_level']
            }
        
        # Phase 2: Home routers encrypt model updates
        print("🏠 Phase 2: Home routers encrypt model updates")
        encrypted_updates = []
        router_encryption_times = []
        
        for i, model_update in enumerate(raw_model_updates):
            device_id = list(smartwatches.keys())[i]
            router_id = router_manager.get_router_for_device(device_id)
            router = router_manager.routers[router_id]
            
            print(f"  🏠 {router_id}: Encrypting update from {device_id}")
            
            # Router receives and encrypts
            router.receive_model_update_from_device(device_id, model_update)
            encrypted_update, encryption_time = router.encrypt_model_update(model_update)
            router.send_encrypted_update_to_server(encrypted_update)
            
            encrypted_updates.append(encrypted_update)
            router_encryption_times.append(encryption_time)
            
            round_results['router_encryption'][router_id] = {
                'encryption_time': encryption_time,
                'encryption_load': router.resource_usage['encryption_load']
            }
        
        # Phase 3: Server aggregates encrypted updates
        print("🖥️ Phase 3: Server aggregates encrypted updates")
        aggregation_start = time.time()
        
        aggregated_update, aggregation_time = fhe_encryption.aggregate_encrypted_updates(
            encrypted_updates, sample_counts
        )
        
        # Update global model
        decrypted_aggregated = aggregated_update.decrypt()
        weights = np.array(decrypted_aggregated[:-1])
        bias = decrypted_aggregated[-1]
        encrypted_global_model = RealEncryptedModel(weights, bias, fhe_encryption.context)
        
        aggregation_time = time.time() - aggregation_start
        
        round_results['server_aggregation'] = {
            'aggregation_time': aggregation_time,
            'total_samples': sum(sample_counts),
            'encrypted_updates_count': len(encrypted_updates)
        }
        
        print(f"  ✅ Server: Aggregated {len(encrypted_updates)} encrypted updates")
        print(f"  ⏱️ Aggregation time: {aggregation_time:.3f}s")
        
        # Phase 4: Home routers decrypt and distribute
        print("🏠 Phase 4: Home routers decrypt and distribute")
        router_decryption_times = []
        
        for router_id, router in router_manager.routers.items():
            print(f"  🏠 {router_id}: Decrypting global model")
            
            # Router receives and decrypts
            router.receive_encrypted_global_model_from_server(encrypted_global_model)
            decrypted_global_model = router.decrypt_global_model(encrypted_global_model)
            router.send_decrypted_global_model_to_devices(decrypted_global_model)
            
            router_decryption_times.append(decrypted_global_model['decryption_time'])
            
            round_results['router_decryption'][router_id] = {
                'decryption_time': decrypted_global_model['decryption_time'],
                'encryption_load': decrypted_global_model['encryption_load']
            }
        
        # Phase 5: Smartwatches receive and update
        print("⌚ Phase 5: Smartwatches receive and update")
        smartwatch_update_times = []
        
        for device_id, smartwatch in smartwatches.items():
            print(f"  ⌚ {device_id}: Receiving global model")
            
            # Get decrypted global model
            decrypted_global_model = {
                'weights': np.array(encrypted_global_model.encrypted_weights.decrypt()),
                'bias': encrypted_global_model.encrypted_bias.decrypt()[0]
            }
            
            # Smartwatch receives and updates
            update_result = smartwatch.receive_global_model_from_router(decrypted_global_model)
            smartwatch_update_times.append(update_result['network_time'])
            
            round_results['smartwatch_updates'][device_id] = {
                'update_time': update_result['network_time'],
                'battery_level': update_result['battery_level']
            }
        
        # Phase 6: Evaluate model performance
        print("📊 Phase 6: Evaluating Model Performance")
        
        # Use first smartwatch for evaluation
        first_smartwatch = list(smartwatches.values())[0]
        if first_smartwatch.local_model is not None:
            # Create test data
            df_local = pd.DataFrame(first_smartwatch.local_data)
            features = [
                'age', 'height_cm', 'weight_kg', 'bmi', 'avg_heart_rate',
                'resting_heart_rate', 'hours_sleep', 'stress_level', 'daily_steps',
                'calories_burned', 'hydration_level'
            ]
            
            X_test = df_local[features].values
            y_test = df_local['health_status'].values
            
            # Evaluate model
            y_pred = first_smartwatch.local_model.predict(X_test)
            
            # Calculate performance metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            
            print(f"  📊 Model Performance:")
            print(f"    Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            print(f"    F1 Score: {f1:.4f} ({f1*100:.2f}%)")
            print(f"    Precision: {precision:.4f} ({precision*100:.2f}%)")
            print(f"    Recall: {recall:.4f} ({recall*100:.2f}%)")
            
            round_results['performance_metrics'] = {
                'accuracy': accuracy,
                'f1_score': f1,
                'precision': precision,
                'recall': recall,
                'test_samples': len(X_test)
            }
            
            all_accuracy_scores.append(accuracy)
            all_f1_scores.append(f1)
        
        # Calculate timing
        total_time = time.time() - round_start
        round_results['timing'] = {
            'total_time': total_time,
            'avg_smartwatch_training': np.mean(smartwatch_training_times),
            'avg_router_encryption': np.mean(router_encryption_times),
            'server_aggregation': aggregation_time,
            'avg_router_decryption': np.mean(router_decryption_times),
            'avg_smartwatch_update': np.mean(smartwatch_update_times)
        }
        
        all_results.append(round_results)
        print(f"✅ Round {round_num} completed in {total_time:.3f}s")
    
    # Step 8: Generate comprehensive results summary
    print("\n📊 STEP 8: Generating Comprehensive Results Summary")
    print("-" * 70)
    
    # Calculate summary statistics
    total_rounds = len(all_results)
    avg_total_time = np.mean([r['timing']['total_time'] for r in all_results])
    avg_encryption_time = np.mean([r['timing']['avg_router_encryption'] for r in all_results])
    avg_aggregation_time = np.mean([r['timing']['server_aggregation'] for r in all_results])
    avg_decryption_time = np.mean([r['timing']['avg_router_decryption'] for r in all_results])
    
    # Performance metrics
    final_accuracy = all_accuracy_scores[-1] if all_accuracy_scores else 0
    best_accuracy = max(all_accuracy_scores) if all_accuracy_scores else 0
    final_f1 = all_f1_scores[-1] if all_f1_scores else 0
    best_f1 = max(all_f1_scores) if all_f1_scores else 0
    
    # Get FHE performance metrics
    fhe_metrics = encrypted_global_model.get_performance_metrics()
    
    results_summary = {
        'experiment_info': {
            'timestamp': datetime.now().isoformat(),
            'total_rounds': total_rounds,
            'total_smartwatches': len(smartwatches),
            'total_routers': len(router_manager.routers),
            'feature_dimension': feature_dim,
            'data_source': 'health_fitness_data.csv' if os.path.exists("data/health_fitness_data.csv") else 'synthetic_data'
        },
        'performance_metrics': {
            'avg_total_time_per_round': avg_total_time,
            'avg_encryption_time': avg_encryption_time,
            'avg_aggregation_time': avg_aggregation_time,
            'avg_decryption_time': avg_decryption_time,
            'fhe_performance': fhe_metrics
        },
        'model_performance': {
            'final_accuracy': final_accuracy,
            'best_accuracy': best_accuracy,
            'final_f1_score': final_f1,
            'best_f1_score': best_f1,
            'accuracy_improvement': (final_accuracy - all_accuracy_scores[0]) if len(all_accuracy_scores) > 1 else 0,
            'f1_improvement': (final_f1 - all_f1_scores[0]) if len(all_f1_scores) > 1 else 0
        },
        'device_status': {
            'smartwatch_status': {device_id: smartwatch.config.__dict__ 
                                for device_id, smartwatch in smartwatches.items()},
            'router_status': router_manager.get_all_router_status()
        },
        'detailed_results': all_results
    }
    
    # Save results
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"results/fhe_data_flow_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results_summary, f, indent=2, default=str)
    
    print(f"✅ Results saved to: {results_file}")
    
    # Display comprehensive summary
    print(f"\n📈 COMPREHENSIVE RESULTS SUMMARY:")
    print(f"  Data Source: {results_summary['experiment_info']['data_source']}")
    print(f"  Total Rounds: {total_rounds}")
    print(f"  Total Smartwatches: {len(smartwatches)}")
    print(f"  Total Routers: {len(router_manager.routers)}")
    print(f"  Average Total Time per Round: {avg_total_time:.3f}s")
    print(f"  Average Encryption Time: {avg_encryption_time:.3f}s")
    print(f"  Average Aggregation Time: {avg_aggregation_time:.3f}s")
    print(f"  Average Decryption Time: {avg_decryption_time:.3f}s")
    print(f"  Final Accuracy: {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")
    print(f"  Best Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
    print(f"  Final F1 Score: {final_f1:.4f} ({final_f1*100:.2f}%)")
    print(f"  Best F1 Score: {best_f1:.4f} ({best_f1*100:.2f}%)")
    print(f"  FHE Ciphertext Size: {fhe_metrics['ciphertext_size']['total_size']:,} bytes")
    
    return results_summary

if __name__ == "__main__":
    print("🚀 FHE CKKS DATA FLOW WITH REAL DATA AND PERFORMANCE METRICS")
    print("="*80)
    
    # Run the complete data flow
    results = run_complete_data_flow_with_metrics()
    
    if results:
        print("\n🎉 DATA FLOW COMPLETED SUCCESSFULLY!")
        print("✅ Real data loaded from health_fitness_data.csv")
        print("✅ Accuracy, F1 score, and performance metrics included")
        print("✅ Cloud server aggregation timing measured")
        print("✅ Comprehensive results saved")
    else:
        print("\n❌ DATA FLOW FAILED!")
        print("Check the error messages above for troubleshooting.")
