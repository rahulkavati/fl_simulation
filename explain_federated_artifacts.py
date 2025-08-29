#!/usr/bin/env python3
"""
Explanation and demonstration of Federated Artifacts in FL Simulation
"""

import os
import sys
import numpy as np
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def explain_federated_artifacts():
    """Explain what federated artifacts are"""
    
    print("🔍 Understanding Federated Artifacts")
    print("=" * 50)
    
    print("\n📚 What are Federated Artifacts?")
    print("   Federated Artifacts are files and data structures that capture")
    print("   the complete history and state of the federated learning process.")
    print("   They serve as the 'memory' of the FL system.")
    
    print("\n🎯 Purpose of Federated Artifacts:")
    print("   1. 📊 Model History: Track how the global model evolves")
    print("   2. 🔄 Round Tracking: Record each training round")
    print("   3. 📈 Performance Monitoring: Store accuracy and metrics")
    print("   4. 🚀 Reproducibility: Enable experiments to be recreated")
    print("   5. 📋 Audit Trail: Maintain compliance and transparency")
    
    print("\n📁 Types of Artifacts in Your System:")
    print("   1. 🔐 Client Updates: Individual client model updates")
    print("   2. 🔒 Encrypted Updates: Secure client contributions")
    print("   3. 📦 Aggregated Results: Combined client updates")
    print("   4. 🌐 Global Model Snapshots: Global model state per round")
    print("   5. 📊 Metrics & Analytics: Performance measurements")

def show_current_artifacts():
    """Show what artifacts currently exist"""
    
    print("\n📂 Current Federated Artifacts in Your System:")
    print("=" * 50)
    
    # 1. Client Updates (JSON)
    print("\n1️⃣ Client Updates (Plaintext):")
    json_dir = "updates/json"
    if os.path.exists(json_dir):
        json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
        rounds = set()
        clients = set()
        for f in json_files:
            if 'client_' in f and 'round_' in f:
                client_num = f.split('client_')[1].split('_')[0]
                round_num = f.split('round_')[1].split('.')[0]
                rounds.add(int(round_num))
                clients.add(int(client_num))
        
        print(f"   📁 Directory: {json_dir}")
        print(f"   📄 Total files: {len(json_files)}")
        print(f"   🔄 Rounds: {sorted(rounds)}")
        print(f"   👥 Clients: {sorted(clients)}")
        print(f"   📊 Structure: {len(clients)} clients × {len(rounds)} rounds")
    
    # 2. Encrypted Updates
    print("\n2️⃣ Encrypted Updates:")
    encrypted_dir = "updates/encrypted"
    if os.path.exists(encrypted_dir):
        encrypted_files = [f for f in os.listdir(encrypted_dir) if f.endswith('.json')]
        print(f"   📁 Directory: {encrypted_dir}")
        print(f"   📄 Files: {encrypted_files}")
        print(f"   🔐 Status: {'✅ Encrypted' if encrypted_files else '❌ No encrypted files'}")
    
    # 3. Aggregated Results
    print("\n3️⃣ Aggregated Results:")
    outbox_dir = "Sriven/outbox"
    if os.path.exists(outbox_dir):
        agg_files = [f for f in os.listdir(outbox_dir) if f.endswith('.json')]
        print(f"   📁 Directory: {outbox_dir}")
        print(f"   📄 Files: {agg_files}")
        print(f"   📦 Status: {'✅ Aggregated' if agg_files else '❌ No aggregated files'}")
    
    # 4. Global Model Snapshots
    print("\n4️⃣ Global Model Snapshots:")
    global_dir = "federated_artifacts/global"
    if os.path.exists(global_dir):
        global_files = [f for f in os.listdir(global_dir) if f.endswith('.npz')]
        rounds = []
        for f in global_files:
            if 'round_' in f:
                round_num = int(f.split('round_')[1].split('.')[0])
                rounds.append(round_num)
        
        print(f"   📁 Directory: {global_dir}")
        print(f"   📄 Files: {global_files}")
        print(f"   🔄 Rounds captured: {sorted(rounds)}")
        print(f"   📊 Status: {'✅ Global snapshots saved' if global_files else '❌ No global snapshots'}")
    
    # 5. Metrics and Analytics
    print("\n5️⃣ Metrics & Analytics:")
    metrics_dir = "metrics"
    if os.path.exists(metrics_dir):
        metric_files = [f for f in os.listdir(metrics_dir) if f.endswith(('.json', '.csv', '.png'))]
        print(f"   📁 Directory: {metrics_dir}")
        print(f"   📄 Files: {metric_files}")
        print(f"   📊 Status: {'✅ Metrics available' if metric_files else '❌ No metrics'}")
    
    # 6. Encryption Keys
    print("\n6️⃣ Encryption Keys:")
    keys_dir = "Huzaif/keys"
    if os.path.exists(keys_dir):
        key_files = [f for f in os.listdir(keys_dir) if f.endswith(('.ctx', '.b64'))]
        print(f"   📁 Directory: {keys_dir}")
        print(f"   🔑 Files: {key_files}")
        print(f"   🔐 Status: {'✅ Encryption keys available' if key_files else '❌ No encryption keys'}")

def demonstrate_artifact_usage():
    """Demonstrate how artifacts are used"""
    
    print("\n🔧 How Federated Artifacts Are Used:")
    print("=" * 50)
    
    try:
        from cloud.global_update import CloudServer, load_aggregated_update
        
        print("\n1️⃣ Loading Global Model Snapshots:")
        global_dir = "federated_artifacts/global"
        if os.path.exists(global_dir):
            # Find the latest round
            global_files = [f for f in os.listdir(global_dir) if f.endswith('.npz')]
            if global_files:
                latest_file = sorted(global_files)[-1]
                latest_path = os.path.join(global_dir, latest_file)
                
                print(f"   📁 Latest snapshot: {latest_file}")
                data = np.load(latest_path)
                print(f"   📊 Contains: {list(data.files)}")
                print(f"   🔄 Round: {latest_file.split('round_')[1].split('.')[0]}")
        
        print("\n2️⃣ Loading Aggregated Updates:")
        decrypted_file = "Sriven/outbox/agg_round_0.decrypted.json"
        if os.path.exists(decrypted_file):
            agg_update = load_aggregated_update(decrypted_file)
            print(f"   📁 File: {decrypted_file}")
            print(f"   📊 Weight delta length: {len(agg_update['weight_delta'])}")
            print(f"   📊 Bias delta: {agg_update['bias_delta']:.6f}")
            print(f"   🔄 Round ID: {agg_update.get('round_id', 'N/A')}")
        
        print("\n3️⃣ Using Artifacts in Global Update:")
        cloud = CloudServer(input_dim=4)
        print(f"   ✅ Cloud server created")
        print(f"   📊 Initial round: {cloud.round}")
        
        # Apply a sample update to show artifact creation
        sample_update = {
            "weight_delta": [0.1, -0.2, 0.3, -0.1],
            "bias_delta": 0.05
        }
        
        cloud.update_global_model(sample_update)
        print(f"   🔄 Applied update, new round: {cloud.round}")
        print(f"   💾 Artifact saved to: federated_artifacts/global/global_round_{cloud.round}.npz")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error demonstrating artifacts: {e}")
        return False

def show_artifact_benefits():
    """Show the benefits of federated artifacts"""
    
    print("\n💡 Benefits of Federated Artifacts:")
    print("=" * 40)
    
    benefits = [
        ("🔍 Transparency", "Complete audit trail of model evolution"),
        ("📈 Reproducibility", "Recreate experiments exactly"),
        ("🚀 Debugging", "Identify issues in specific rounds"),
        ("📊 Analysis", "Analyze learning patterns and convergence"),
        ("🔒 Compliance", "Meet regulatory requirements"),
        ("📋 Documentation", "Document the FL process"),
        ("🔄 Rollback", "Revert to previous model states"),
        ("📱 Deployment", "Track which models are deployed")
    ]
    
    for benefit, description in benefits:
        print(f"   {benefit}: {description}")

def main():
    """Main function to explain federated artifacts"""
    
    print("🚀 Federated Artifacts in FL Simulation")
    print("=" * 60)
    
    # Explain what they are
    explain_federated_artifacts()
    
    # Show current artifacts
    show_current_artifacts()
    
    # Demonstrate usage
    demonstrate_artifact_usage()
    
    # Show benefits
    show_artifact_benefits()
    
    print("\n🎯 Summary:")
    print("   Federated Artifacts are the backbone of your FL system.")
    print("   They provide transparency, reproducibility, and auditability.")
    print("   Your system is already creating and using them effectively!")
    
    print("\n💡 Next Steps:")
    print("   1. Review the artifacts in each directory")
    print("   2. Use them to analyze your FL training process")
    print("   3. Leverage them for debugging and optimization")

if __name__ == "__main__":
    main()
