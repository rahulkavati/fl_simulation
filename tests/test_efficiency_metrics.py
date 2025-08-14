"""
Unit tests for FL efficiency metrics

Tests cover:
- Metric calculation accuracy
- Data type validation
- Edge cases and error handling
- Performance benchmarks
"""

import unittest
import numpy as np
import tempfile
import os
import json
from unittest.mock import Mock, patch
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.efficiency_metrics import FLEfficiencyMetrics, FLEfficiencyCalculator
from sklearn.linear_model import LogisticRegression


class TestFLEfficiencyMetrics(unittest.TestCase):
    """Test the FLEfficiencyMetrics dataclass"""
    
    def setUp(self):
        """Set up test data"""
        self.sample_metrics = FLEfficiencyMetrics(
            timestamp="2025-01-01T00:00:00",
            num_clients=5,
            num_rounds=3,
            total_samples=1000,
            total_communication_rounds=15,
            bytes_transferred=15360.0,
            communication_overhead=1.46,
            total_training_time=0.25,
            avg_training_time_per_round=0.083,
            convergence_rounds=2,
            initial_accuracy=0.5,
            final_accuracy=0.798,
            accuracy_improvement=0.298,
            final_weight_norm=1.031,
            final_bias=-19.223,
            memory_usage=0.00003,
            cpu_utilization=80.0,
            weight_change_magnitude=[0.5, 0.2, 0.1],
            loss_reduction=[0.5, 0.8, 0.9]
        )
    
    def test_metrics_creation(self):
        """Test metrics object creation"""
        self.assertEqual(self.sample_metrics.num_clients, 5)
        self.assertEqual(self.sample_metrics.num_rounds, 3)
        self.assertEqual(self.sample_metrics.final_accuracy, 0.798)
    
    def test_to_dict_conversion(self):
        """Test conversion to dictionary"""
        metrics_dict = self.sample_metrics.to_dict()
        self.assertIsInstance(metrics_dict, dict)
        self.assertEqual(metrics_dict['num_clients'], 5)
        self.assertEqual(metrics_dict['final_accuracy'], 0.798)
    
    def test_to_json_conversion(self):
        """Test conversion to JSON string"""
        json_str = self.sample_metrics.to_json()
        self.assertIsInstance(json_str, str)
        
        # Verify it can be parsed back
        parsed = json.loads(json_str)
        self.assertEqual(parsed['num_clients'], 5)
    
    def test_save_to_file(self):
        """Test saving metrics to file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            self.sample_metrics.save_to_file(temp_path)
            
            # Verify file was created and contains correct data
            with open(temp_path, 'r') as f:
                saved_data = json.load(f)
            
            self.assertEqual(saved_data['num_clients'], 5)
            self.assertEqual(saved_data['final_accuracy'], 0.798)
            
        finally:
            os.unlink(temp_path)


class TestFLEfficiencyCalculator(unittest.TestCase):
    """Test the FLEfficiencyCalculator class"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.calculator = FLEfficiencyCalculator(
            data_dir=self.temp_dir,
            updates_dir=self.temp_dir
        )
        
        # Create mock model
        self.mock_model = Mock(spec=LogisticRegression)
        self.mock_model.coef_ = np.array([[0.1, 0.2, 0.3, 0.4]])
        self.mock_model.intercept_ = np.array([-0.5])
        
        # Create mock client data
        self.mock_clients_data = {
            'client_1': (np.random.rand(100, 4), np.random.randint(0, 2, 100)),
            'client_2': (np.random.rand(100, 4), np.random.randint(0, 2, 100))
        }
    
    def tearDown(self):
        """Clean up test environment"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test calculator initialization"""
        self.assertEqual(self.calculator.data_dir, self.temp_dir)
        self.assertEqual(self.calculator.updates_dir, self.temp_dir)
        self.assertEqual(len(self.calculator.metrics_history), 0)
    
    def test_communication_efficiency_calculation(self):
        """Test communication efficiency metrics calculation"""
        rounds, bytes_trans, overhead = self.calculator.calculate_communication_efficiency(
            num_clients=5, num_rounds=3
        )
        
        self.assertEqual(rounds, 15)  # 5 clients * 3 rounds
        self.assertEqual(bytes_trans, 15360.0)  # 15 * 1024
        self.assertIsInstance(overhead, float)
        self.assertGreater(overhead, 0)
    
    def test_model_performance_calculation(self):
        """Test model performance metrics calculation"""
        # Mock the model to return predictable predictions
        mock_predictions = np.array([0, 1, 0, 1, 0])
        mock_true_labels = np.array([0, 1, 0, 1, 0])
        
        with patch.object(self.mock_model, 'predict', return_value=mock_predictions):
            init_acc, final_acc, improvement = self.calculator.calculate_model_performance(
                self.mock_model, self.mock_clients_data
            )
        
        self.assertEqual(init_acc, 0.5)  # Default baseline
        self.assertEqual(final_acc, 1.0)  # Perfect predictions
        self.assertEqual(improvement, 0.5)
    
    def test_convergence_metrics_calculation(self):
        """Test convergence metrics calculation"""
        # Create mock update files
        updates_dir = os.path.join(self.temp_dir, "numpy")
        os.makedirs(updates_dir, exist_ok=True)
        
        # Create mock update files
        for round_id in range(3):
            for client_id in range(2):
                update_data = np.array([0.1, 0.2, 0.3, 0.4, 0.5])  # weights + bias
                np.save(os.path.join(updates_dir, f"client_{client_id}_round_{round_id}.npy"), update_data)
        
        weight_changes, loss_reductions = self.calculator.calculate_convergence_metrics(
            updates_dir, num_rounds=3
        )
        
        self.assertEqual(len(weight_changes), 3)
        self.assertEqual(len(loss_reductions), 3)
        self.assertIsInstance(weight_changes[0], float)
        self.assertIsInstance(loss_reductions[0], float)
    
    def test_efficiency_metrics_calculation(self):
        """Test complete efficiency metrics calculation"""
        metrics = self.calculator.calculate_efficiency_metrics(
            clients_data=self.mock_clients_data,
            global_model=self.mock_model,
            num_rounds=3,
            training_time=0.25
        )
        
        # Verify all required fields are present
        self.assertIsInstance(metrics, FLEfficiencyMetrics)
        self.assertEqual(metrics.num_clients, 2)
        self.assertEqual(metrics.num_rounds, 3)
        self.assertEqual(metrics.total_training_time, 0.25)
        self.assertIsInstance(metrics.timestamp, str)
        self.assertIsInstance(metrics.final_weight_norm, float)
    
    def test_save_metrics(self):
        """Test metrics saving functionality"""
        metrics = self.calculator.calculate_efficiency_metrics(
            clients_data=self.mock_clients_data,
            global_model=self.mock_model,
            num_rounds=3
        )
        
        # Test saving with custom name
        experiment_name = "test_experiment"
        self.calculator.save_metrics(metrics, experiment_name)
        
        # Verify metrics were added to history
        self.assertEqual(len(self.calculator.metrics_history), 1)
        self.assertEqual(self.calculator.metrics_history[0], metrics)
        
        # Verify files were created
        metrics_dir = os.path.join(self.temp_dir, "metrics")
        self.assertTrue(os.path.exists(metrics_dir))
        
        # Check for individual experiment file
        experiment_file = os.path.join(metrics_dir, f"{experiment_name}.json")
        self.assertTrue(os.path.exists(experiment_file))
        
        # Check for summary files
        summary_file = os.path.join(metrics_dir, "metrics_summary.json")
        csv_file = os.path.join(metrics_dir, "metrics_history.csv")
        self.assertTrue(os.path.exists(summary_file))
        self.assertTrue(os.path.exists(csv_file))


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions"""
    
    def test_empty_clients_data(self):
        """Test behavior with empty clients data"""
        calculator = FLEfficiencyCalculator("data", "updates")
        mock_model = Mock(spec=LogisticRegression)
        mock_model.coef_ = np.array([[0.1, 0.2]])
        mock_model.intercept_ = np.array([0.0])
        
        # Empty clients data
        empty_clients = {}
        
        with self.assertRaises(StopIteration):
            # This should fail when trying to get next(iter(empty_clients.values()))
            calculator.calculate_efficiency_metrics(
                clients_data=empty_clients,
                global_model=mock_model,
                num_rounds=1
            )
    
    def test_zero_rounds(self):
        """Test behavior with zero training rounds"""
        calculator = FLEfficiencyCalculator("data", "updates")
        rounds, bytes_trans, overhead = calculator.calculate_communication_efficiency(
            num_clients=5, num_rounds=0
        )
        
        self.assertEqual(rounds, 0)
        self.assertEqual(bytes_trans, 0.0)
        self.assertEqual(overhead, 0.0)
    
    def test_large_numbers(self):
        """Test behavior with very large numbers"""
        calculator = FLEfficiencyCalculator("data", "updates")
        
        # Test with very large client/round numbers
        rounds, bytes_trans, overhead = calculator.calculate_communication_efficiency(
            num_clients=1000, num_rounds=100
        )
        
        self.assertEqual(rounds, 100000)
        self.assertGreater(bytes_trans, 0)
        self.assertIsInstance(overhead, float)


class TestPerformanceBenchmarks(unittest.TestCase):
    """Test performance characteristics"""
    
    def test_metrics_calculation_speed(self):
        """Test that metrics calculation is reasonably fast"""
        import time
        
        calculator = FLEfficiencyCalculator("data", "updates")
        mock_model = Mock(spec=LogisticRegression)
        mock_model.coef_ = np.array([[0.1, 0.2, 0.3, 0.4]])
        mock_model.intercept_ = np.array([-0.5])
        
        # Create larger mock data
        large_clients_data = {
            f'client_{i}': (np.random.rand(1000, 4), np.random.randint(0, 2, 1000))
            for i in range(10)
        }
        
        start_time = time.time()
        metrics = calculator.calculate_efficiency_metrics(
            clients_data=large_clients_data,
            global_model=mock_model,
            num_rounds=5
        )
        end_time = time.time()
        
        calculation_time = end_time - start_time
        
        # Should complete in reasonable time (less than 1 second)
        self.assertLess(calculation_time, 1.0)
        self.assertIsInstance(metrics, FLEfficiencyMetrics)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
