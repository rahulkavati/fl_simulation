"""
Client Statistics and Logging Module

This module provides comprehensive logging and statistics tracking for
one-class clients and client exclusions in federated learning.

Key Features:
- Track one-class client statistics
- Log client creation and exclusion events
- Monitor class distributions
- Record strategy usage
- Generate comprehensive summaries

Author: AI Assistant
Date: 2025
"""

import logging
import json
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
from collections import defaultdict, Counter
import os

logger = logging.getLogger(__name__)


class ClientStatisticsLogger:
    """
    Comprehensive logging and statistics tracking for federated learning clients.
    
    This class provides detailed tracking of:
    - Client creation and exclusion events
    - One-class client detection and handling
    - Class distribution analysis
    - Strategy usage statistics
    - Round-by-round performance tracking
    """
    
    def __init__(self, log_file: Optional[str] = None):
        """
        Initialize the client statistics logger.
        
        Args:
            log_file: Optional file path to save detailed logs
        """
        self.log_file = log_file
        self.stats = {
            'total_clients': 0,
            'one_class_clients': 0,
            'excluded_clients': 0,
            'insufficient_data_clients': 0,
            'class_distributions': defaultdict(int),
            'one_class_strategies': defaultdict(int),
            'round_statistics': [],
            'client_details': {}
        }
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging configuration for detailed client tracking."""
        if self.log_file:
            # Create logs directory if it doesn't exist
            os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
            
            # Configure file handler
            file_handler = logging.FileHandler(self.log_file)
            file_handler.setLevel(logging.INFO)
            
            # Configure formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            
            # Add handler to logger
            logger.addHandler(file_handler)
            logger.setLevel(logging.INFO)
    
    def log_client_creation(self, client_id: str, sample_count: int, 
                          class_distribution: Dict[int, int], 
                          is_one_class: bool = False, 
                          strategy_used: Optional[str] = None):
        """
        Log client creation with detailed statistics.
        
        Args:
            client_id: Unique identifier for the client
            sample_count: Number of samples in the client dataset
            class_distribution: Distribution of classes in the client data
            is_one_class: Whether the client has only one class
            strategy_used: Strategy used for one-class handling (if applicable)
        """
        # Update statistics
        self.stats['total_clients'] += 1
        
        if is_one_class:
            self.stats['one_class_clients'] += 1
            if strategy_used:
                self.stats['one_class_strategies'][strategy_used] += 1
        
        # Update class distributions
        for class_label, count in class_distribution.items():
            self.stats['class_distributions'][f'class_{class_label}'] += count
        
        # Store client details
        self.stats['client_details'][client_id] = {
            'sample_count': sample_count,
            'class_distribution': class_distribution,
            'is_one_class': is_one_class,
            'strategy_used': strategy_used,
            'created_at': datetime.now().isoformat()
        }
        
        # Log to file if configured
        if self.log_file:
            logger.info(f"Client {client_id} created: {sample_count} samples, "
                       f"classes: {class_distribution}, one_class: {is_one_class}")
    
    def log_client_exclusion(self, client_id: str, reason: str, sample_count: int):
        """
        Log client exclusion with reason and statistics.
        
        Args:
            client_id: Unique identifier for the client
            reason: Reason for exclusion (e.g., 'insufficient_data')
            sample_count: Number of samples in the excluded client
        """
        # Update statistics
        self.stats['excluded_clients'] += 1
        
        if reason == 'insufficient_data':
            self.stats['insufficient_data_clients'] += 1
        
        # Store exclusion details
        if client_id not in self.stats['client_details']:
            self.stats['client_details'][client_id] = {}
        
        self.stats['client_details'][client_id].update({
            'excluded': True,
            'exclusion_reason': reason,
            'sample_count': sample_count,
            'excluded_at': datetime.now().isoformat()
        })
        
        # Log to file if configured
        if self.log_file:
            logger.warning(f"Client {client_id} excluded: {reason}, "
                          f"samples: {sample_count}")
    
    def log_round_statistics(self, round_id: int, total_clients: int, 
                           one_class_clients: int, strategies_used: Dict[str, int]):
        """
        Log round-specific statistics.
        
        Args:
            round_id: Current round number
            total_clients: Total number of clients in this round
            one_class_clients: Number of one-class clients in this round
            strategies_used: Dictionary of strategies and their usage counts
        """
        round_stats = {
            'round_id': round_id,
            'total_clients': total_clients,
            'one_class_clients': one_class_clients,
            'strategies_used': strategies_used,
            'timestamp': datetime.now().isoformat()
        }
        
        self.stats['round_statistics'].append(round_stats)
        
        # Log to file if configured
        if self.log_file:
            logger.info(f"Round {round_id}: {total_clients} clients, "
                       f"{one_class_clients} one-class, strategies: {strategies_used}")
    
    def print_summary(self):
        """Print comprehensive summary of client statistics."""
        print("\n" + "=" * 60)
        print("CLIENT STATISTICS SUMMARY")
        print("=" * 60)
        
        # Basic statistics
        print(f"Total clients: {self.stats['total_clients']}")
        print(f"One-class clients: {self.stats['one_class_clients']} "
              f"({self.stats['one_class_clients']/max(1, self.stats['total_clients'])*100:.1f}%)")
        print(f"Multi-class clients: {self.stats['total_clients'] - self.stats['one_class_clients']}")
        print(f"Excluded clients: {self.stats['excluded_clients']} "
              f"({self.stats['excluded_clients']/max(1, self.stats['total_clients'])*100:.1f}%)")
        print(f"Insufficient data clients: {self.stats['insufficient_data_clients']}")
        print(f"Total rounds: {len(self.stats['round_statistics'])}")
        
        # One-class handling strategies
        if self.stats['one_class_strategies']:
            print(f"\nOne-class handling strategies:")
            for strategy, count in self.stats['one_class_strategies'].items():
                print(f"  - {strategy}: {count}")
        
        # Class distributions
        if self.stats['class_distributions']:
            print(f"\nClass distributions:")
            for class_name, count in self.stats['class_distributions'].items():
                print(f"  - {class_name}: {count}")
        
        print("=" * 60)
    
    def save_statistics(self, output_file: str):
        """
        Save comprehensive statistics to JSON file.
        
        Args:
            output_file: Path to save the statistics JSON file
        """
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Convert defaultdicts to regular dicts for JSON serialization
        stats_to_save = {
            'total_clients': self.stats['total_clients'],
            'one_class_clients': self.stats['one_class_clients'],
            'excluded_clients': self.stats['excluded_clients'],
            'insufficient_data_clients': self.stats['insufficient_data_clients'],
            'class_distributions': dict(self.stats['class_distributions']),
            'one_class_strategies': dict(self.stats['one_class_strategies']),
            'round_statistics': self.stats['round_statistics'],
            'client_details': self.stats['client_details'],
            'summary_generated_at': datetime.now().isoformat()
        }
        
        with open(output_file, 'w') as f:
            json.dump(stats_to_save, f, indent=2)
        
        print(f"Client statistics saved to: {output_file}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get current statistics dictionary.
        
        Returns:
            Dict[str, Any]: Current statistics
        """
        return self.stats.copy()
    
    def reset_statistics(self):
        """Reset all statistics to initial state."""
        self.stats = {
            'total_clients': 0,
            'one_class_clients': 0,
            'excluded_clients': 0,
            'insufficient_data_clients': 0,
            'class_distributions': defaultdict(int),
            'one_class_strategies': defaultdict(int),
            'round_statistics': [],
            'client_details': {}
        }


def create_client_logger(log_file: Optional[str] = None) -> ClientStatisticsLogger:
    """
    Create a new client statistics logger instance.
    
    Args:
        log_file: Optional file path to save detailed logs
        
    Returns:
        ClientStatisticsLogger: New logger instance
    """
    return ClientStatisticsLogger(log_file)