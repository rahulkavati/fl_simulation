"""
Complete Guide: How to Run the FHE CKKS Data Flow and Check Results
Step-by-step instructions for running the realistic FHE implementation
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime

def create_simple_test_data():
    """
    Create simple test data to avoid data loading issues
    """
    print("📊 Creating Simple Test Data...")
    
    # Create synthetic health data
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
    
    # Save to CSV
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/simple_health_data.csv", index=False)
    
    print(f"✅ Created {len(df)} samples for {df['participant_id'].nunique()} participants")
    print(f"📁 Saved to: data/simple_health_data.csv")
    
    return df

def run_complete_data_flow():
    """
    Run the complete FHE CKKS data flow with simple test data
    """
    print("🚀 RUNNING COMPLETE FHE CKKS DATA FLOW")
    print("="*60)
    
    # Step 1: Create test data
    print("\n📊 STEP 1: Creating Test Data")
    print("-" * 30)
    df = create_simple_test_data()
    
    # Step 2: Import required modules
    print("\n🔧 STEP 2: Importing FHE Modules")
    print("-" * 30)
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
    print("-" * 30)
    
    # Create configurations
    fl_config = FLConfig(rounds=2, clients=4)  # Reduced for testing
    fhe_config = RealFHEConfig()
    
    # Initialize FHE encryption
    fhe_encryption = RealFHEEncryption(fhe_config)
    print("✅ FHE system initialized")
    
    # Step 4: Create smartwatches and home routers
    print("\n⌚ STEP 4: Creating Smartwatches and Home Routers")
    print("-" * 30)
    
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
    print("-" * 30)
    
    participants = df['participant_id'].unique()
    for i, device_id in enumerate(smartwatches.keys()):
        participant_id = participants[i % len(participants)]
        participant_data = df[df['participant_id'] == participant_id]
        
        # Ensure balanced classes
        if len(participant_data['health_status'].unique()) >= 2:
            smartwatches[device_id].load_sensor_data(df, participant_id)
        else:
            # Add some data from other participants to balance
            other_data = df[df['participant_id'] != participant_id]
            balanced_data = pd.concat([participant_data, other_data.sample(n=20)])
            smartwatches[device_id].load_sensor_data(balanced_data, participant_id)
    
    print("✅ Data loaded to all smartwatches")
    
    # Step 6: Initialize encrypted global model
    print("\n🔐 STEP 6: Initializing Encrypted Global Model")
    print("-" * 30)
    
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
    
    # Step 7: Run federated learning rounds
    print("\n🔄 STEP 7: Running Federated Learning Rounds")
    print("-" * 30)
    
    all_results = []
    
    for round_num in range(1, fl_config.rounds + 1):
        print(f"\n🔄 ROUND {round_num}:")
        print("-" * 20)
        
        round_start = time.time()
        round_results = {
            'round': round_num,
            'smartwatch_training': {},
            'router_encryption': {},
            'server_aggregation': {},
            'router_decryption': {},
            'smartwatch_updates': {},
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
    
    # Step 8: Generate results summary
    print("\n📊 STEP 8: Generating Results Summary")
    print("-" * 30)
    
    # Calculate summary statistics
    total_rounds = len(all_results)
    avg_total_time = np.mean([r['timing']['total_time'] for r in all_results])
    avg_encryption_time = np.mean([r['timing']['avg_router_encryption'] for r in all_results])
    avg_aggregation_time = np.mean([r['timing']['server_aggregation'] for r in all_results])
    avg_decryption_time = np.mean([r['timing']['avg_router_decryption'] for r in all_results])
    
    # Get FHE performance metrics
    fhe_metrics = encrypted_global_model.get_performance_metrics()
    
    results_summary = {
        'experiment_info': {
            'timestamp': datetime.now().isoformat(),
            'total_rounds': total_rounds,
            'total_smartwatches': len(smartwatches),
            'total_routers': len(router_manager.routers),
            'feature_dimension': feature_dim
        },
        'performance_metrics': {
            'avg_total_time_per_round': avg_total_time,
            'avg_encryption_time': avg_encryption_time,
            'avg_aggregation_time': avg_aggregation_time,
            'avg_decryption_time': avg_decryption_time,
            'fhe_performance': fhe_metrics
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
    
    # Display summary
    print(f"\n📈 RESULTS SUMMARY:")
    print(f"  Total Rounds: {total_rounds}")
    print(f"  Total Smartwatches: {len(smartwatches)}")
    print(f"  Total Routers: {len(router_manager.routers)}")
    print(f"  Average Total Time per Round: {avg_total_time:.3f}s")
    print(f"  Average Encryption Time: {avg_encryption_time:.3f}s")
    print(f"  Average Aggregation Time: {avg_aggregation_time:.3f}s")
    print(f"  Average Decryption Time: {avg_decryption_time:.3f}s")
    print(f"  FHE Ciphertext Size: {fhe_metrics['ciphertext_size']['total_size']} bytes")
    
    return results_summary

def show_how_to_check_results():
    """
    Show how to check and analyze the results
    """
    print("\n📊 HOW TO CHECK RESULTS:")
    print("="*60)
    
    print("\n1. 📁 Results Files:")
    print("   - Results are saved in: results/fhe_data_flow_results_TIMESTAMP.json")
    print("   - Contains complete experiment data and performance metrics")
    
    print("\n2. 📈 Key Metrics to Check:")
    print("   - Encryption/Decryption timing (real FHE measurements)")
    print("   - Aggregation performance (real encrypted operations)")
    print("   - Ciphertext sizes (actual encrypted data sizes)")
    print("   - Device resource usage (battery, CPU, memory)")
    print("   - Network communication timing")
    
    print("\n3. 🔍 Analysis Commands:")
    print("   ```python")
    print("   import json")
    print("   ")
    print("   # Load results")
    print("   with open('results/fhe_data_flow_results_TIMESTAMP.json', 'r') as f:")
    print("       results = json.load(f)")
    print("   ")
    print("   # Check performance metrics")
    print("   print('Encryption time:', results['performance_metrics']['avg_encryption_time'])")
    print("   print('Aggregation time:', results['performance_metrics']['avg_aggregation_time'])")
    print("   print('Ciphertext size:', results['performance_metrics']['fhe_performance']['ciphertext_size'])")
    print("   ```")
    
    print("\n4. 📊 Visualization:")
    print("   - Use the detailed_results section for round-by-round analysis")
    print("   - Plot timing trends across rounds")
    print("   - Compare encryption vs decryption performance")
    print("   - Analyze device resource usage patterns")

if __name__ == "__main__":
    print("🚀 FHE CKKS DATA FLOW RUNNER")
    print("Complete guide to run the realistic FHE implementation")
    print("="*80)
    
    # Run the complete data flow
    results = run_complete_data_flow()
    
    if results:
        show_how_to_check_results()
        print("\n🎉 DATA FLOW COMPLETED SUCCESSFULLY!")
        print("Check the results file for detailed performance metrics.")
    else:
        print("\n❌ DATA FLOW FAILED!")
        print("Check the error messages above for troubleshooting.")
