"""
Integration Example: Routing Protocol with INC Federated Learning

This example shows how to integrate routing protocols into our existing INC implementation
to provide load balancing, fault tolerance, and traffic optimization.
"""

from src.routing_protocol import RoutingProtocol, RoutingAlgorithm, RoutingEnhancedINCCoordinator
from src.inc_edge_fl import INCFederatedLearningCoordinator
import time


def demonstrate_routing_benefits():
    """Demonstrate the benefits of routing protocols in INC FL"""
    
    print("=" * 70)
    print("ROUTING PROTOCOL BENEFITS IN INC FEDERATED LEARNING")
    print("=" * 70)
    
    # Scenario 1: Load Balancing
    print("\n1. LOAD BALANCING BENEFITS:")
    print("   Without Routing:")
    print("   - Client1-3 → INC1 (Fixed assignment)")
    print("   - Client4-6 → INC2 (Fixed assignment)")
    print("   - Problem: INC1 overloaded, INC2 underutilized")
    
    print("\n   With Routing Protocol:")
    print("   - Dynamic assignment based on INC load")
    print("   - Client1 → INC1 (load: 30%)")
    print("   - Client2 → INC2 (load: 20%)")
    print("   - Client3 → INC1 (load: 40%)")
    print("   - Client4 → INC2 (load: 35%)")
    print("   - Benefit: Balanced load across INCs")
    
    # Scenario 2: Fault Tolerance
    print("\n2. FAULT TOLERANCE BENEFITS:")
    print("   Without Routing:")
    print("   - INC1 fails → Client1-3 cannot participate")
    print("   - System loses 50% of clients")
    
    print("\n   With Routing Protocol:")
    print("   - INC1 fails → Automatic failover to INC2")
    print("   - Client1-3 → INC2 (automatic rerouting)")
    print("   - System maintains 100% client participation")
    print("   - Benefit: Higher system reliability")
    
    # Scenario 3: Traffic Optimization
    print("\n3. TRAFFIC OPTIMIZATION BENEFITS:")
    print("   Without Routing:")
    print("   - Fixed paths regardless of network conditions")
    print("   - High latency links used even when better paths available")
    
    print("\n   With Routing Protocol:")
    print("   - Dynamic path selection based on latency")
    print("   - Automatic rerouting when better paths available")
    print("   - Benefit: Reduced communication delays")
    
    # Scenario 4: Scalability
    print("\n4. SCALABILITY BENEFITS:")
    print("   Without Routing:")
    print("   - Manual configuration for new INCs")
    print("   - Static topology updates required")
    
    print("\n   With Routing Protocol:")
    print("   - Automatic discovery of new INCs")
    print("   - Dynamic topology updates")
    print("   - Benefit: Easier system expansion")


def compare_performance_with_without_routing():
    """Compare performance with and without routing protocols"""
    
    print("\n" + "=" * 70)
    print("PERFORMANCE COMPARISON")
    print("=" * 70)
    
    # Simulate different scenarios
    scenarios = [
        {
            'name': 'Normal Operation',
            'inc_failures': 0,
            'network_congestion': 'low',
            'client_distribution': 'balanced'
        },
        {
            'name': 'INC Failure',
            'inc_failures': 1,
            'network_congestion': 'low',
            'client_distribution': 'unbalanced'
        },
        {
            'name': 'Network Congestion',
            'inc_failures': 0,
            'network_congestion': 'high',
            'client_distribution': 'balanced'
        },
        {
            'name': 'Mixed Issues',
            'inc_failures': 1,
            'network_congestion': 'high',
            'client_distribution': 'unbalanced'
        }
    ]
    
    for scenario in scenarios:
        print(f"\nScenario: {scenario['name']}")
        print("-" * 40)
        
        # Simulate without routing
        without_routing = simulate_without_routing(scenario)
        
        # Simulate with routing
        with_routing = simulate_with_routing(scenario)
        
        # Calculate improvements
        accuracy_improvement = with_routing['accuracy'] - without_routing['accuracy']
        latency_reduction = without_routing['avg_latency'] - with_routing['avg_latency']
        reliability_improvement = with_routing['reliability'] - without_routing['reliability']
        
        print(f"  Accuracy: {without_routing['accuracy']:.2f}% → {with_routing['accuracy']:.2f}% ({accuracy_improvement:+.2f}%)")
        print(f"  Latency: {without_routing['avg_latency']:.2f}ms → {with_routing['avg_latency']:.2f}ms ({latency_reduction:+.2f}ms)")
        print(f"  Reliability: {without_routing['reliability']:.2f}% → {with_routing['reliability']:.2f}% ({reliability_improvement:+.2f}%)")


def simulate_without_routing(scenario):
    """Simulate performance without routing protocol"""
    base_accuracy = 85.0
    base_latency = 50.0
    base_reliability = 95.0
    
    # Adjust based on scenario
    if scenario['inc_failures'] > 0:
        base_accuracy -= 15.0  # Significant drop due to client loss
        base_reliability -= 20.0  # Poor fault tolerance
    
    if scenario['network_congestion'] == 'high':
        base_latency += 30.0  # Increased latency
    
    if scenario['client_distribution'] == 'unbalanced':
        base_accuracy -= 5.0  # Slight drop due to load imbalance
    
    return {
        'accuracy': max(0, base_accuracy),
        'avg_latency': base_latency,
        'reliability': max(0, base_reliability)
    }


def simulate_with_routing(scenario):
    """Simulate performance with routing protocol"""
    base_accuracy = 85.0
    base_latency = 50.0
    base_reliability = 95.0
    
    # Routing protocol benefits
    if scenario['inc_failures'] > 0:
        base_accuracy -= 2.0  # Minimal drop due to failover
        base_reliability -= 2.0  # Good fault tolerance
    
    if scenario['network_congestion'] == 'high':
        base_latency += 10.0  # Reduced latency increase due to path optimization
    
    if scenario['client_distribution'] == 'unbalanced':
        base_accuracy += 3.0  # Improvement due to load balancing
    
    # Additional routing benefits
    base_accuracy += 2.0  # General improvement from optimization
    base_latency -= 5.0  # General improvement from path selection
    base_reliability += 3.0  # General improvement from fault tolerance
    
    return {
        'accuracy': min(100, base_accuracy),
        'avg_latency': max(1, base_latency),
        'reliability': min(100, base_reliability)
    }


def routing_protocol_implementation_example():
    """Show how to implement routing in our INC system"""
    
    print("\n" + "=" * 70)
    print("IMPLEMENTATION EXAMPLE")
    print("=" * 70)
    
    print("\n1. Enhanced INC Coordinator with Routing:")
    print("""
    class RoutingEnhancedINCCoordinator:
        def __init__(self, fl_config, fhe_config):
            self.routing_protocol = RoutingProtocol(RoutingAlgorithm.ADAPTIVE)
            self.inc_coordinator = INCFederatedLearningCoordinator(fl_config, fhe_config)
            
        def run_federated_learning_with_routing(self, clients_data, inc_configs):
            # Initialize routing topology
            self.routing_protocol.initialize_topology(clients_data.keys(), inc_configs)
            
            # Run federated learning with routing
            for round_num in range(self.fl_config.rounds):
                # Phase 1: Client Training (unchanged)
                client_results = self.train_clients(clients_data)
                
                # Phase 2: Edge Encryption with Routing
                edge_results = self.encrypt_with_routing(client_results)
                
                # Phase 3: INC Aggregation with Load Balancing
                inc_results = self.aggregate_with_load_balancing(edge_results)
                
                # Phase 4: Cloud Update (unchanged)
                self.update_global_model(inc_results)
                
                # Phase 5: Distribution with Routing
                self.distribute_with_routing()
    """)
    
    print("\n2. Key Routing Features:")
    print("   - Automatic INC selection based on load")
    print("   - Fault detection and failover")
    print("   - Path optimization for reduced latency")
    print("   - Dynamic load balancing")
    print("   - Network topology management")
    
    print("\n3. Routing Algorithms Available:")
    print("   - Round Robin: Simple load distribution")
    print("   - Least Connections: Load-based selection")
    print("   - Least Latency: Latency-based selection")
    print("   - Adaptive: Multi-factor optimization")


if __name__ == "__main__":
    demonstrate_routing_benefits()
    compare_performance_with_without_routing()
    routing_protocol_implementation_example()
