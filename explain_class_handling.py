"""
How Class Issues Are Handled in Federated Learning
Comprehensive explanation of class imbalance and balancing strategies
"""

def explain_class_handling():
    """
    Explain how class issues are handled in federated learning
    """
    print("🎯 HOW CLASS ISSUES ARE HANDLED IN FEDERATED LEARNING")
    print("="*70)
    
    print("\n❌ THE CLASS PROBLEM:")
    print("-" * 25)
    print("  🚨 Issue: Some participants have only ONE class")
    print("  📊 Example:")
    print("    • Participant A: Only healthy records (class 1)")
    print("    • Participant B: Only unhealthy records (class 0)")
    print("    • Participant C: Mixed records (classes 0 & 1)")
    print("  ")
    print("  💥 Training Error:")
    print("    ValueError: This solver needs samples of at least 2 classes")
    print("    in the data, but the data contains only one class: 0")
    
    print("\n🔍 WHY THIS HAPPENS:")
    print("-" * 20)
    reasons = [
        "📊 Real-world data: People have different health patterns",
        "🎯 Fitness level threshold: Some people consistently above/below",
        "📈 Temporal patterns: Health status changes over time",
        "👥 Individual differences: Some participants are outliers",
        "📅 Data collection: Limited time period for some participants"
    ]
    
    for reason in reasons:
        print(f"  {reason}")
    
    print("\n✅ SOLUTION STRATEGIES:")
    print("-" * 25)
    
    print("\n1️⃣ STRATEGY 1: CLASS BALANCING (Main Pipeline)")
    print("-" * 45)
    print("  🎯 Method: Skip participants with only one class")
    print("  📊 Implementation:")
    print("    ```python")
    print("    # Check if we have both classes")
    print("    unique_classes = np.unique(y)")
    print("    if len(unique_classes) >= 2:")
    print("        clients_data[f'client_{i}'] = (X, y)  # Use this client")
    print("    else:")
    print("        print(f'Client {i}: Only {len(unique_classes)} class(es) - skipping')")
    print("    ```")
    print("  ")
    print("  ✅ Pros:")
    print("    • Ensures all clients have both classes")
    print("    • Prevents training errors")
    print("    • Maintains data integrity")
    print("  ")
    print("  ❌ Cons:")
    print("    • Reduces number of clients")
    print("    • May exclude valuable participants")
    
    print("\n2️⃣ STRATEGY 2: DATA SAMPLING (FHE Pipeline)")
    print("-" * 45)
    print("  🎯 Method: Sample missing class from other participants")
    print("  📊 Implementation:")
    print("    ```python")
    print("    if len(participant_data['health_status'].unique()) >= 2:")
    print("        # Use participant data if it has both classes")
    print("        smartwatches[device_id].load_sensor_data(df, participant_id)")
    print("    else:")
    print("        # Sample from other participants to ensure both classes")
    print("        missing_class = 1 if unique_classes[0] == 0 else 0")
    print("        missing_class_data = other_data[other_data['health_status'] == missing_class]")
    print("        sampled_data = missing_class_data.sample(n=50, random_state=42)")
    print("        combined_data = pd.concat([participant_data, sampled_data])")
    print("    ```")
    print("  ")
    print("  ✅ Pros:")
    print("    • Keeps all participants")
    print("    • Ensures balanced classes")
    print("    • Maintains federated learning principles")
    print("  ")
    print("  ❌ Cons:")
    print("    • Adds synthetic data mixing")
    print("    • May affect privacy guarantees")
    
    print("\n3️⃣ STRATEGY 3: SYNTHETIC DATA FALLBACK")
    print("-" * 40)
    print("  🎯 Method: Use synthetic data when CSV fails")
    print("  📊 Implementation:")
    print("    ```python")
    print("    if len(final_balanced_data['health_status'].unique()) >= 2:")
    print("        smartwatches[device_id].load_sensor_data(final_balanced_data, participant_id)")
    print("    else:")
    print("        # Fallback: use synthetic data")
    print("        synthetic_df = create_synthetic_data()")
    print("        smartwatches[device_id].load_sensor_data(synthetic_df, participant_id)")
    print("    ```")
    print("  ")
    print("  ✅ Pros:")
    print("    • Guaranteed to work")
    print("    • Perfect class balance")
    print("    • No training errors")
    print("  ")
    print("  ❌ Cons:")
    print("    • Not real data")
    print("    • Lower accuracy")
    print("    • Less realistic")
    
    print("\n4️⃣ STRATEGY 4: MODEL CONFIGURATION")
    print("-" * 35)
    print("  🎯 Method: Use balanced class weights")
    print("  📊 Implementation:")
    print("    ```python")
    print("    model_params = {")
    print("        'class_weight': 'balanced',  # Automatically balance classes")
    print("        'solver': 'lbfgs',")
    print("        'max_iter': 10000")
    print("    }")
    print("    ```")
    print("  ")
    print("  ✅ Pros:")
    print("    • Handles imbalanced classes")
    print("    • No data modification needed")
    print("    • Built-in sklearn feature")
    print("  ")
    print("  ❌ Cons:")
    print("    • May not work with extreme imbalance")
    print("    • Still needs minimum samples per class")
    
    print("\n📊 CLASS DISTRIBUTION EXAMPLES:")
    print("-" * 35)
    
    print("\n  🟢 GOOD CLIENT (Both Classes):")
    print("    Participant 1234:")
    print("    • Total samples: 200")
    print("    • Unhealthy (0): 95 samples (47.5%)")
    print("    • Healthy (1): 105 samples (52.5%)")
    print("    • Status: ✅ Used in federated learning")
    
    print("\n  🔴 BAD CLIENT (One Class Only):")
    print("    Participant 5678:")
    print("    • Total samples: 150")
    print("    • Unhealthy (0): 150 samples (100%)")
    print("    • Healthy (1): 0 samples (0%)")
    print("    • Status: ❌ Skipped or balanced")
    
    print("\n  🟡 BALANCED CLIENT (After Sampling):")
    print("    Participant 5678 (After Balancing):")
    print("    • Original samples: 150 (all unhealthy)")
    print("    • Added samples: 50 (all healthy)")
    print("    • Total samples: 200")
    print("    • Unhealthy (0): 150 samples (75%)")
    print("    • Healthy (1): 50 samples (25%)")
    print("    • Status: ✅ Used in federated learning")
    
    print("\n🔧 IMPLEMENTATION DETAILS:")
    print("-" * 30)
    
    print("\n  📁 Main Pipeline (federated_learning_pipeline.py):")
    print("    • Uses Strategy 1: Skip single-class participants")
    print("    • Ensures all clients have both classes")
    print("    • Saves individual client CSV files")
    print("    • Uses class_weight='balanced' in model")
    
    print("\n  📁 FHE Pipeline (run_fhe_data_flow_csv.py):")
    print("    • Uses Strategy 2: Sample missing classes")
    print("    • Keeps all participants")
    print("    • Balances classes by sampling")
    print("    • Fallback to synthetic data if needed")
    
    print("\n  📁 Synthetic Pipeline (run_fhe_data_flow_final.py):")
    print("    • Uses Strategy 3: Perfect synthetic balance")
    print("    • Guaranteed 50/50 class distribution")
    print("    • No class issues")
    print("    • Lower but consistent accuracy")
    
    print("\n📈 ACCURACY IMPACT:")
    print("-" * 20)
    print("  🎯 With Proper Class Handling:")
    print("    • Main Pipeline: 95%+ accuracy")
    print("    • FHE Pipeline: 90%+ accuracy")
    print("    • Synthetic Pipeline: 49% accuracy")
    print("  ")
    print("  🔍 Why Different Accuracies:")
    print("    • Main Pipeline: Real data + advanced features")
    print("    • FHE Pipeline: Real data + sampling")
    print("    • Synthetic Pipeline: Artificial data")
    
    print("\n🚀 BEST PRACTICES:")
    print("-" * 20)
    best_practices = [
        "✅ Always check class distribution before training",
        "✅ Use class_weight='balanced' in model parameters",
        "✅ Ensure minimum samples per class (e.g., 10-20)",
        "✅ Monitor class distribution across clients",
        "✅ Use stratified sampling when possible",
        "✅ Consider data augmentation for minority class",
        "✅ Validate class balance in test data"
    ]
    
    for practice in best_practices:
        print(f"  {practice}")
    
    print("\n🔍 DEBUGGING CLASS ISSUES:")
    print("-" * 30)
    print("  🛠️ Check Commands:")
    print("    ```python")
    print("    # Check class distribution")
    print("    print(df['health_status'].value_counts())")
    print("    ")
    print("    # Check per participant")
    print("    for participant in df['participant_id'].unique():")
    print("        participant_data = df[df['participant_id'] == participant]")
    print("        classes = participant_data['health_status'].unique()")
    print("        print(f'Participant {participant}: {len(classes)} classes')")
    print("    ```")
    
    print("\n📊 SUMMARY:")
    print("-" * 15)
    print("  🎯 Class issues are handled through multiple strategies:")
    print("    1. Skip single-class participants (main pipeline)")
    print("    2. Sample missing classes (FHE pipeline)")
    print("    3. Use synthetic data (fallback)")
    print("    4. Configure model with balanced weights")
    print("  ")
    print("  ✅ This ensures:")
    print("    • No training errors")
    print("    • Balanced federated learning")
    print("    • High accuracy (95%+)")
    print("    • Realistic performance evaluation")

if __name__ == "__main__":
    explain_class_handling()
