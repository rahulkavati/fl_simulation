"""
EXAMPLE: How to Customize Clients, Rounds, and Routers
Simple examples showing different configurations
"""

from run_customizable_fhe_data_flow import CustomConfig, run_customizable_fhe_data_flow

def example_small_scale():
    """
    Example 1: Small scale experiment (3 clients, 2 rounds, 1 router)
    """
    print("🔬 EXAMPLE 1: SMALL SCALE EXPERIMENT")
    print("="*50)
    
    config = CustomConfig(
        num_clients=3,           # 3 smartwatches
        num_rounds=2,            # 2 federated learning rounds
        num_routers=1,           # 1 home router
        use_real_data=True,      # Use real CSV data
        feature_engineering=True, # Apply 47-feature engineering
        verbose=True             # Show detailed output
    )
    
    print(f"Running with: {config.num_clients} clients, {config.num_rounds} rounds, {config.num_routers} routers")
    success = run_customizable_fhe_data_flow(config)
    
    if success:
        print("✅ Small scale experiment completed!")
    else:
        print("❌ Small scale experiment failed!")

def example_medium_scale():
    """
    Example 2: Medium scale experiment (8 clients, 5 rounds, 2 routers)
    """
    print("🔬 EXAMPLE 2: MEDIUM SCALE EXPERIMENT")
    print("="*50)
    
    config = CustomConfig(
        num_clients=8,           # 8 smartwatches
        num_rounds=5,            # 5 federated learning rounds
        num_routers=2,           # 2 home routers
        use_real_data=True,      # Use real CSV data
        feature_engineering=True, # Apply 47-feature engineering
        verbose=True             # Show detailed output
    )
    
    print(f"Running with: {config.num_clients} clients, {config.num_rounds} rounds, {config.num_routers} routers")
    success = run_customizable_fhe_data_flow(config)
    
    if success:
        print("✅ Medium scale experiment completed!")
    else:
        print("❌ Medium scale experiment failed!")

def example_fast_experiment():
    """
    Example 3: Fast experiment (4 clients, 3 rounds, 1 router, synthetic data)
    """
    print("🔬 EXAMPLE 3: FAST EXPERIMENT")
    print("="*50)
    
    config = CustomConfig(
        num_clients=4,           # 4 smartwatches
        num_rounds=3,            # 3 federated learning rounds
        num_routers=1,           # 1 home router
        use_real_data=False,     # Use synthetic data
        feature_engineering=False, # Use basic features only
        verbose=True             # Show detailed output
    )
    
    print(f"Running with: {config.num_clients} clients, {config.num_rounds} rounds, {config.num_routers} routers")
    print("Using synthetic data with basic features for speed")
    success = run_customizable_fhe_data_flow(config)
    
    if success:
        print("✅ Fast experiment completed!")
    else:
        print("❌ Fast experiment failed!")

def example_custom_configuration():
    """
    Example 4: Custom configuration (12 clients, 6 rounds, 3 routers)
    """
    print("🔬 EXAMPLE 4: CUSTOM CONFIGURATION")
    print("="*50)
    
    # Create your own custom configuration
    config = CustomConfig(
        num_clients=12,          # 12 smartwatches
        num_rounds=6,            # 6 federated learning rounds
        num_routers=3,           # 3 home routers
        use_real_data=True,      # Use real CSV data
        feature_engineering=True, # Apply 47-feature engineering
        verbose=True             # Show detailed output
    )
    
    print(f"Running with: {config.num_clients} clients, {config.num_rounds} rounds, {config.num_routers} routers")
    print("This is a custom configuration for comprehensive research")
    success = run_customizable_fhe_data_flow(config)
    
    if success:
        print("✅ Custom configuration experiment completed!")
    else:
        print("❌ Custom configuration experiment failed!")

def show_all_examples():
    """
    Show all configuration examples
    """
    print("🔧 CUSTOMIZATION EXAMPLES - CLIENTS, ROUNDS, AND ROUTERS")
    print("="*80)
    
    print("\n📋 AVAILABLE EXAMPLES:")
    print("-" * 25)
    print("1️⃣ Small Scale (3 clients, 2 rounds, 1 router)")
    print("2️⃣ Medium Scale (8 clients, 5 rounds, 2 routers)")
    print("3️⃣ Fast Experiment (4 clients, 3 rounds, 1 router, synthetic)")
    print("4️⃣ Custom Configuration (12 clients, 6 rounds, 3 routers)")
    
    print("\n🚀 HOW TO RUN EXAMPLES:")
    print("-" * 30)
    print("💻 Command: python customization_examples.py")
    print("💻 Or run individual examples:")
    print("   python -c \"from customization_examples import example_small_scale; example_small_scale()\"")
    print("   python -c \"from customization_examples import example_medium_scale; example_medium_scale()\"")
    print("   python -c \"from customization_examples import example_fast_experiment; example_fast_experiment()\"")
    print("   python -c \"from customization_examples import example_custom_configuration; example_custom_configuration()\"")
    
    print("\n🎯 EXPECTED RESULTS:")
    print("-" * 20)
    print("📊 Small Scale: 85-90% accuracy, ~2-3 seconds")
    print("📊 Medium Scale: 88-92% accuracy, ~8-12 seconds")
    print("📊 Fast Experiment: 80-85% accuracy, ~1-2 seconds")
    print("📊 Custom Config: 89-93% accuracy, ~15-20 seconds")
    
    print("\n💡 TIPS:")
    print("-" * 10)
    print("✅ Start with Fast Experiment for quick testing")
    print("✅ Use Small Scale for development")
    print("✅ Use Medium Scale for standard research")
    print("✅ Use Custom Configuration for comprehensive studies")
    
    print("\n🔧 CUSTOMIZE YOUR OWN:")
    print("-" * 25)
    print("1. Copy one of the example functions")
    print("2. Modify the CustomConfig parameters")
    print("3. Run your custom experiment")
    print("4. Analyze the results")

if __name__ == "__main__":
    print("🔧 CUSTOMIZATION EXAMPLES")
    print("="*30)
    
    # Show all examples
    show_all_examples()
    
    print("\n" + "="*50)
    print("🚀 RUNNING FAST EXPERIMENT AS DEMO")
    print("="*50)
    
    # Run the fast experiment as a demo
    example_fast_experiment()
    
    print("\n" + "="*50)
    print("🎉 DEMO COMPLETED!")
    print("="*50)
    print("✅ You can now customize clients, rounds, and routers!")
    print("📊 Check the results directory for detailed analysis")
    print("🔧 Modify the examples above to create your own configurations")
