"""
Routing Protocol Implementation for INC-Assisted Federated Learning

This module implements routing protocols to enhance the INC architecture by:
1. Load balancing across multiple INCs
2. Fault tolerance and failover
3. Dynamic path selection
4. Traffic optimization
"""

import time
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from enum import Enum
import heapq
import threading
from collections import defaultdict


class RoutingAlgorithm(Enum):
    """Types of routing algorithms"""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_LATENCY = "least_latency"
    ADAPTIVE = "adaptive"


@dataclass
class NetworkNode:
    """Represents a network node (INC, Edge, Client)"""
    node_id: str
    node_type: str  # 'client', 'edge', 'inc', 'cloud'
    capacity: int = 1000  # Maximum concurrent connections
    current_load: int = 0
    latency: float = 0.0  # Average latency in ms
    bandwidth: float = 1000.0  # Bandwidth in Mbps
    is_active: bool = True
    last_heartbeat: float = 0.0


@dataclass
class NetworkLink:
    """Represents a network link between nodes"""
    source_id: str
    target_id: str
    bandwidth: float = 1000.0
    latency: float = 1.0
    reliability: float = 0.99
    is_active: bool = True


@dataclass
class RoutingTable:
    """Routing table for a node"""
    node_id: str
    routes: Dict[str, List[str]]  # destination -> path
    metrics: Dict[str, Dict[str, float]]  # destination -> metric -> value


class NetworkTopology:
    """Manages the network topology and routing information"""
    
    def __init__(self):
        self.nodes: Dict[str, NetworkNode] = {}
        self.links: Dict[Tuple[str, str], NetworkLink] = {}
        self.routing_tables: Dict[str, RoutingTable] = {}
        self.update_lock = threading.Lock()
    
    def add_node(self, node: NetworkNode):
        """Add a node to the topology"""
        with self.update_lock:
            self.nodes[node.node_id] = node
            self.routing_tables[node.node_id] = RoutingTable(
                node_id=node.node_id,
                routes={},
                metrics={}
            )
    
    def add_link(self, link: NetworkLink):
        """Add a link to the topology"""
        with self.update_lock:
            self.links[(link.source_id, link.target_id)] = link
            self.links[(link.target_id, link.source_id)] = link  # Bidirectional
    
    def update_node_status(self, node_id: str, is_active: bool, latency: float = None):
        """Update node status and metrics"""
        with self.update_lock:
            if node_id in self.nodes:
                self.nodes[node_id].is_active = is_active
                self.nodes[node_id].last_heartbeat = time.time()
                if latency is not None:
                    self.nodes[node_id].latency = latency
    
    def get_active_nodes(self, node_type: str = None) -> List[NetworkNode]:
        """Get active nodes, optionally filtered by type"""
        with self.update_lock:
            active_nodes = [node for node in self.nodes.values() if node.is_active]
            if node_type:
                active_nodes = [node for node in active_nodes if node.node_type == node_type]
            return active_nodes
    
    def calculate_shortest_path(self, source: str, destination: str) -> List[str]:
        """Calculate shortest path using Dijkstra's algorithm"""
        if source not in self.nodes or destination not in self.nodes:
            return []
        
        # Dijkstra's algorithm
        distances = {node_id: float('inf') for node_id in self.nodes}
        distances[source] = 0
        previous = {}
        pq = [(0, source)]
        
        while pq:
            current_dist, current_node = heapq.heappop(pq)
            
            if current_node == destination:
                break
            
            if current_dist > distances[current_node]:
                continue
            
            # Check all neighbors
            for (src, dst), link in self.links.items():
                if src == current_node and link.is_active:
                    neighbor = dst
                    if neighbor in self.nodes and self.nodes[neighbor].is_active:
                        # Calculate cost based on latency and reliability
                        cost = link.latency / link.reliability
                        new_dist = current_dist + cost
                        
                        if new_dist < distances[neighbor]:
                            distances[neighbor] = new_dist
                            previous[neighbor] = current_node
                            heapq.heappush(pq, (new_dist, neighbor))
        
        # Reconstruct path
        path = []
        current = destination
        while current is not None:
            path.append(current)
            current = previous.get(current)
        
        return path[::-1] if path and path[0] == source else []


class RoutingProtocol:
    """Implements routing protocol for INC federated learning"""
    
    def __init__(self, algorithm: RoutingAlgorithm = RoutingAlgorithm.ADAPTIVE):
        self.algorithm = algorithm
        self.topology = NetworkTopology()
        self.load_balancer = LoadBalancer()
        self.fault_detector = FaultDetector()
        
        # Statistics
        self.routing_stats = {
            'total_requests': 0,
            'successful_routes': 0,
            'failed_routes': 0,
            'load_balanced_requests': 0,
            'failover_events': 0
        }
    
    def initialize_topology(self, clients: List[str], edges: List[str], incs: List[str], cloud: str):
        """Initialize the network topology"""
        # Add nodes
        for client_id in clients:
            self.topology.add_node(NetworkNode(
                node_id=client_id,
                node_type='client',
                capacity=100,
                latency=5.0
            ))
        
        for edge_id in edges:
            self.topology.add_node(NetworkNode(
                node_id=edge_id,
                node_type='edge',
                capacity=500,
                latency=2.0
            ))
        
        for inc_id in incs:
            self.topology.add_node(NetworkNode(
                node_id=inc_id,
                node_type='inc',
                capacity=1000,
                latency=1.0
            ))
        
        self.topology.add_node(NetworkNode(
            node_id=cloud,
            node_type='cloud',
            capacity=10000,
            latency=0.5
        ))
        
        # Add links (simplified topology)
        for client_id in clients:
            edge_id = f"edge_{client_id}"
            if edge_id in edges:
                self.topology.add_link(NetworkLink(
                    source_id=client_id,
                    target_id=edge_id,
                    bandwidth=100.0,
                    latency=1.0
                ))
        
        for edge_id in edges:
            for inc_id in incs:
                self.topology.add_link(NetworkLink(
                    source_id=edge_id,
                    target_id=inc_id,
                    bandwidth=1000.0,
                    latency=2.0
                ))
        
        for inc_id in incs:
            self.topology.add_link(NetworkLink(
                source_id=inc_id,
                target_id=cloud,
                bandwidth=10000.0,
                latency=5.0
            ))
    
    def select_inc_for_client(self, client_id: str, edge_id: str) -> str:
        """Select the best INC for a client based on routing algorithm"""
        self.routing_stats['total_requests'] += 1
        
        active_incs = self.topology.get_active_nodes('inc')
        if not active_incs:
            self.routing_stats['failed_routes'] += 1
            return None
        
        if self.algorithm == RoutingAlgorithm.ROUND_ROBIN:
            selected_inc = self.load_balancer.round_robin(active_incs)
        
        elif self.algorithm == RoutingAlgorithm.LEAST_CONNECTIONS:
            selected_inc = self.load_balancer.least_connections(active_incs)
        
        elif self.algorithm == RoutingAlgorithm.LEAST_LATENCY:
            selected_inc = self.load_balancer.least_latency(active_incs, edge_id)
        
        elif self.algorithm == RoutingAlgorithm.ADAPTIVE:
            selected_inc = self.load_balancer.adaptive_selection(active_incs, edge_id)
        
        else:
            selected_inc = active_incs[0]  # Default to first available
        
        if selected_inc:
            self.routing_stats['successful_routes'] += 1
            self.load_balancer.update_load(selected_inc.node_id, 1)
            return selected_inc.node_id
        
        return None
    
    def get_optimal_path(self, source: str, destination: str) -> List[str]:
        """Get optimal path between source and destination"""
        path = self.topology.calculate_shortest_path(source, destination)
        
        # Check for failures and implement failover
        if not path or not self._validate_path(path):
            path = self._find_alternative_path(source, destination)
            if path:
                self.routing_stats['failover_events'] += 1
        
        return path
    
    def _validate_path(self, path: List[str]) -> bool:
        """Validate that all nodes and links in path are active"""
        for i in range(len(path) - 1):
            current_node = path[i]
            next_node = path[i + 1]
            
            if not self.topology.nodes[current_node].is_active:
                return False
            
            link_key = (current_node, next_node)
            if link_key not in self.topology.links or not self.topology.links[link_key].is_active:
                return False
        
        return True
    
    def _find_alternative_path(self, source: str, destination: str) -> List[str]:
        """Find alternative path when primary path fails"""
        # Temporarily mark failed nodes as inactive and recalculate
        failed_nodes = [node_id for node_id, node in self.topology.nodes.items() if not node.is_active]
        
        # Try to find path excluding failed nodes
        for failed_node in failed_nodes:
            self.topology.nodes[failed_node].is_active = False
        
        path = self.topology.calculate_shortest_path(source, destination)
        
        # Restore node status
        for failed_node in failed_nodes:
            self.topology.nodes[failed_node].is_active = True
        
        return path
    
    def update_network_metrics(self, node_id: str, latency: float, load: int):
        """Update network metrics for a node"""
        self.topology.update_node_status(node_id, True, latency)
        if node_id in self.topology.nodes:
            self.topology.nodes[node_id].current_load = load
    
    def get_routing_statistics(self) -> Dict[str, Any]:
        """Get routing protocol statistics"""
        total_requests = self.routing_stats['total_requests']
        success_rate = (self.routing_stats['successful_routes'] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'total_requests': total_requests,
            'successful_routes': self.routing_stats['successful_routes'],
            'failed_routes': self.routing_stats['failed_routes'],
            'success_rate': success_rate,
            'load_balanced_requests': self.routing_stats['load_balanced_requests'],
            'failover_events': self.routing_stats['failover_events'],
            'active_nodes': len(self.topology.get_active_nodes()),
            'total_nodes': len(self.topology.nodes)
        }


class LoadBalancer:
    """Implements various load balancing algorithms"""
    
    def __init__(self):
        self.round_robin_index = 0
        self.node_loads = defaultdict(int)
        self.node_latencies = defaultdict(float)
    
    def round_robin(self, nodes: List[NetworkNode]) -> NetworkNode:
        """Round robin load balancing"""
        if not nodes:
            return None
        
        selected_node = nodes[self.round_robin_index % len(nodes)]
        self.round_robin_index += 1
        return selected_node
    
    def least_connections(self, nodes: List[NetworkNode]) -> NetworkNode:
        """Select node with least current connections"""
        if not nodes:
            return None
        
        return min(nodes, key=lambda node: node.current_load)
    
    def least_latency(self, nodes: List[NetworkNode], source_edge: str) -> NetworkNode:
        """Select node with least latency"""
        if not nodes:
            return None
        
        # Calculate latency from source edge to each INC
        best_node = None
        best_latency = float('inf')
        
        for node in nodes:
            # Simplified latency calculation
            total_latency = node.latency + self.node_latencies.get(node.node_id, 0)
            if total_latency < best_latency:
                best_latency = total_latency
                best_node = node
        
        return best_node
    
    def adaptive_selection(self, nodes: List[NetworkNode], source_edge: str) -> NetworkNode:
        """Adaptive selection based on multiple factors"""
        if not nodes:
            return None
        
        best_node = None
        best_score = float('inf')
        
        for node in nodes:
            # Calculate composite score based on load, latency, and capacity
            load_factor = node.current_load / node.capacity
            latency_factor = node.latency / 100.0  # Normalize latency
            capacity_factor = 1.0 - (node.capacity / 1000.0)  # Prefer higher capacity
            
            score = load_factor + latency_factor + capacity_factor
            
            if score < best_score:
                best_score = score
                best_node = node
        
        return best_node
    
    def update_load(self, node_id: str, load_change: int):
        """Update load for a node"""
        self.node_loads[node_id] += load_change


class FaultDetector:
    """Detects and handles network faults"""
    
    def __init__(self):
        self.fault_threshold = 5.0  # seconds
        self.heartbeat_interval = 1.0  # seconds
    
    def detect_faults(self, topology: NetworkTopology) -> List[str]:
        """Detect faulty nodes"""
        current_time = time.time()
        faulty_nodes = []
        
        for node_id, node in topology.nodes.items():
            if current_time - node.last_heartbeat > self.fault_threshold:
                faulty_nodes.append(node_id)
                node.is_active = False
        
        return faulty_nodes
    
    def recover_node(self, node_id: str, topology: NetworkTopology):
        """Mark node as recovered"""
        if node_id in topology.nodes:
            topology.nodes[node_id].is_active = True
            topology.nodes[node_id].last_heartbeat = time.time()


# Example usage in INC Federated Learning
class RoutingEnhancedINCCoordinator:
    """INC Coordinator enhanced with routing protocol"""
    
    def __init__(self, fl_config, fhe_config, routing_algorithm: RoutingAlgorithm = RoutingAlgorithm.ADAPTIVE):
        self.fl_config = fl_config
        self.fhe_config = fhe_config
        self.routing_protocol = RoutingProtocol(routing_algorithm)
        
        # Initialize routing for INC architecture
        self._initialize_routing()
    
    def _initialize_routing(self):
        """Initialize routing protocol for INC architecture"""
        # This would be called when setting up the federated learning system
        clients = [f"client_{i}" for i in range(self.fl_config.clients)]
        edges = [f"edge_client_{i}" for i in range(self.fl_config.clients)]
        incs = [f"inc_{i}" for i in range(2)]  # Example: 2 INCs
        cloud = "cloud_server"
        
        self.routing_protocol.initialize_topology(clients, edges, incs, cloud)
    
    def select_inc_for_edge(self, edge_id: str) -> str:
        """Select optimal INC for an edge device using routing protocol"""
        client_id = edge_id.replace("edge_", "")
        return self.routing_protocol.select_inc_for_client(client_id, edge_id)
    
    def get_routing_metrics(self) -> Dict[str, Any]:
        """Get routing protocol metrics"""
        return self.routing_protocol.get_routing_statistics()
