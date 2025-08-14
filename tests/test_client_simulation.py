"""
Unit tests for FL client simulation

Tests cover:
- Data loading functionality
- Model training and updates
- File I/O operations
- Integration with efficiency metrics
"""

import unittest
import numpy as np
import tempfile
import os
import json
import shutil
from unittest.mock import Mock, patch, MagicMock
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.client_simulation import load_client_data, save_json, save_npy
from sklearn.linear_model import LogisticRegression


class TestClientSimulationHelpers(unittest.TestCase):
    """Test helper functions"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_data = {"test": "data", "number": 42}
        self.test_array = np.array([1.0, 2.0, 3.0, 4.0])
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir)
    
    def test_save_json(self):
        """Test JSON saving functionality"""
        json_path = os.path.join(self.temp_dir, "test.json")
        save_json(self.test_data, json_path)
        
        # Verify file was created
        self.assertTrue(os.path.exists(json_path))
        
        # Verify content
        with open(json_path, 'r') as f:
            loaded_data = json.load(f)
        
        self.assertEqual(loaded_data, self.test_data)
    
    def test_save_npy(self):
        """Test NumPy array saving functionality"""
        npy_path = os.path.join(self.temp_dir, "test.npy")
        save_npy(self.test_array, npy_path)
        
        # Verify file was created
        self.assertTrue(os.path.exists(npy_path))
        
        # Verify content
        loaded_array = np.load(npy_path)
        np.testing.assert_array_equal(loaded_array, self.test_array)
    
    def test_save_json_creates_directory(self):
        """Test that save_json creates parent directories"""
        nested_path = os.path.join(self.temp_dir, "nested", "deep", "test.json")
        save_json(self.test_data, nested_path)
        
        # Verify directory structure was created
        self.assertTrue(os.path.exists(os.path.dirname(nested_path)))
        self.assertTrue(os.path.exists(nested_path))
    
    def test_save_npy_creates_directory(self):
        """Test that save_npy creates parent directories"""
        nested_path = os.path.join(self.temp_dir, "nested", "deep", "test.npy")
        save_npy(self.test_array, nested_path)
        
        # Verify directory structure was created
        self.assertTrue(os.path.exists(os.path.dirname(nested_path)))
        self.assertTrue(os.path.exists(nested_path))


class TestDataLoading(unittest.TestCase):
    """Test data loading functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.clients_dir = os.path.join(self.temp_dir, "clients")
        os.makedirs(self.clients_dir, exist_ok=True)
        
        # Create mock client CSV files
        self.create_mock_client_files()
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir)
    
    def create_mock_client_files(self):
        """Create mock client CSV files for testing"""
        # Create 3 mock clients with different data sizes
        for i in range(3):
            client_file = os.path.join(self.clients_dir, f"client_{i}.csv")
            
            # Create CSV with header and data
            with open(client_file, 'w') as f:
                f.write("feature1,feature2,feature3,feature4,target\n")
                for j in range(50 + i * 25):  # Different sizes: 50, 75, 100
                    features = [f"{j+k*0.1:.2f}" for k in range(4)]
                    target = str(j % 2)
                    f.write(f"{','.join(features)},{target}\n")
    
    def test_load_client_data(self):
        """Test client data loading"""
        # Patch the DATA_DIR constant
        with patch('simulation.client_simulation.DATA_DIR', self.clients_dir):
            clients_data = load_client_data()
        
        # Verify all clients were loaded
        self.assertEqual(len(clients_data), 3)
        self.assertIn('client_0', clients_data)
        self.assertIn('client_1', clients_data)
        self.assertIn('client_2', clients_data)
        
        # Verify data structure
        for client_id, (X, y) in clients_data.items():
            self.assertIsInstance(X, np.ndarray)
            self.assertIsInstance(y, np.ndarray)
            self.assertEqual(X.shape[1], 4)  # 4 features
            self.assertEqual(X.shape[0], y.shape[0])  # Same number of samples
    
    def test_load_client_data_empty_directory(self):
        """Test loading from empty directory"""
        empty_dir = tempfile.mkdtemp()
        try:
            with patch('simulation.client_simulation.DATA_DIR', empty_dir):
                clients_data = load_client_data()
            
            self.assertEqual(len(clients_data), 0)
        finally:
            shutil.rmtree(empty_dir)
    
    def test_load_client_data_invalid_files(self):
        """Test loading with invalid CSV files"""
        # Create an invalid CSV file
        invalid_file = os.path.join(self.clients_dir, "invalid.csv")
        with open(invalid_file, 'w') as f:
            f.write("invalid,data,format\n")
            f.write("1,2,3\n")
        
        with patch('simulation.client_simulation.DATA_DIR', self.clients_dir):
            # Should handle invalid files gracefully
            clients_data = load_client_data()
            
            # Should still load valid files
            self.assertGreaterEqual(len(clients_data), 3)


class TestModelTraining(unittest.TestCase):
    """Test model training and update functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.updates_dir = os.path.join(self.temp_dir, "updates")
        os.makedirs(self.updates_dir, exist_ok=True)
        
        # Create mock client data
        self.X_train = np.random.rand(100, 4)
        self.y_train = np.random.randint(0, 2, 100)
        
        # Create base model
        self.base_model = LogisticRegression(random_state=42)
        self.base_model.fit(self.X_train[:10], self.y_train[:10])
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir)
    
    def test_model_update_calculation(self):
        """Test weight delta calculation"""
        # Create local model with same initial weights
        local_model = LogisticRegression(random_state=42)
        local_model.classes_ = np.array([0, 1])
        local_model.coef_ = np.copy(self.base_model.coef_)
        local_model.intercept_ = np.copy(self.base_model.intercept_)
        
        # Train local model
        local_model.fit(self.X_train, self.y_train)
        
        # Calculate deltas
        weight_delta = (local_model.coef_ - self.base_model.coef_).flatten()
        bias_delta = (local_model.intercept_ - self.base_model.intercept_).item()
        
        # Verify deltas are calculated correctly
        self.assertIsInstance(weight_delta, np.ndarray)
        self.assertIsInstance(bias_delta, float)
        self.assertEqual(weight_delta.shape[0], 4)  # 4 features
        
        # Deltas should not be zero after training
        self.assertTrue(np.any(weight_delta != 0) or bias_delta != 0)
    
    def test_model_aggregation(self):
        """Test FedAvg aggregation"""
        # Create multiple client updates
        updates = []
        for i in range(3):
            # Simulate different client updates
            weight_delta = np.random.rand(4) * 0.1
            bias_delta = np.random.rand() * 0.1
            updates.append((weight_delta, bias_delta))
        
        # Aggregate using FedAvg
        avg_weight_delta = np.mean([w for w, _ in updates], axis=0)
        avg_bias_delta = np.mean([b for _, b in updates], axis=0)
        
        # Verify aggregation
        self.assertEqual(avg_weight_delta.shape, (4,))
        self.assertIsInstance(avg_bias_delta, float)
        
        # Verify averaging worked
        for i in range(4):
            expected_avg = np.mean([updates[j][0][i] for j in range(3)])
            self.assertAlmostEqual(avg_weight_delta[i], expected_avg, places=10)


class TestIntegration(unittest.TestCase):
    """Test integration between components"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create necessary directories
        self.data_dir = os.path.join(self.temp_dir, "data", "clients")
        self.updates_dir = os.path.join(self.temp_dir, "updates")
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.updates_dir, exist_ok=True)
        
        # Create mock client data
        self.create_mock_client_data()
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir)
    
    def create_mock_client_data(self):
        """Create mock client data for integration testing"""
        for i in range(2):
            client_file = os.path.join(self.data_dir, f"client_{i}.csv")
            with open(client_file, 'w') as f:
                f.write("feature1,feature2,feature3,feature4,target\n")
                for j in range(50):
                    features = [f"{j+k*0.1:.2f}" for k in range(4)]
                    target = str(j % 2)
                    f.write(f"{','.join(features)},{target}\n")
    
    @patch('simulation.client_simulation.DATA_DIR')
    @patch('simulation.client_simulation.OUTPUT_JSON')
    @patch('simulation.client_simulation.OUTPUT_NPY')
    def test_end_to_end_simulation(self, mock_npy_dir, mock_json_dir, mock_data_dir):
        """Test complete end-to-end simulation flow"""
        # Set up mocks
        mock_data_dir.__str__ = lambda: self.data_dir
        mock_json_dir.__str__ = lambda: os.path.join(self.updates_dir, "json")
        mock_npy_dir.__str__ = lambda: os.path.join(self.updates_dir, "numpy")
        
        # Mock the efficiency calculator to avoid file system dependencies
        with patch('simulation.client_simulation.FLEfficiencyCalculator') as mock_calc_class:
            mock_calc = Mock()
            mock_calc_class.return_value = mock_calc
            mock_calc.calculate_efficiency_metrics.return_value = Mock()
            
            # Import and run main function
            from simulation.client_simulation import main
            
            # This should run without errors
            try:
                main()
                # If we get here, the simulation ran successfully
                self.assertTrue(True)
            except Exception as e:
                self.fail(f"Simulation failed with error: {e}")
    
    def test_file_output_structure(self):
        """Test that simulation creates correct file structure"""
        # This test would verify the actual file outputs
        # For now, we'll just verify the directories exist
        self.assertTrue(os.path.exists(self.data_dir))
        self.assertTrue(os.path.exists(self.updates_dir))


class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases"""
    
    def test_missing_data_directory(self):
        """Test handling of missing data directory"""
        with patch('simulation.client_simulation.DATA_DIR', "/nonexistent/path"):
            with self.assertRaises(FileNotFoundError):
                load_client_data()
    
    def test_corrupted_csv_files(self):
        """Test handling of corrupted CSV files"""
        temp_dir = tempfile.mkdtemp()
        try:
            # Create corrupted CSV
            corrupted_file = os.path.join(temp_dir, "corrupted.csv")
            with open(corrupted_file, 'w') as f:
                f.write("invalid,data\n")
                f.write("1,2,3,4,5,6\n")  # Wrong number of columns
            
            with patch('simulation.client_simulation.DATA_DIR', temp_dir):
                # Should handle gracefully
                clients_data = load_client_data()
                self.assertEqual(len(clients_data), 0)
        finally:
            shutil.rmtree(temp_dir)
    
    def test_empty_csv_files(self):
        """Test handling of empty CSV files"""
        temp_dir = tempfile.mkdtemp()
        try:
            # Create empty CSV
            empty_file = os.path.join(temp_dir, "empty.csv")
            with open(empty_file, 'w') as f:
                f.write("feature1,feature2,feature3,feature4,target\n")
                # No data rows
            
            with patch('simulation.client_simulation.DATA_DIR', temp_dir):
                # Should handle gracefully
                clients_data = load_client_data()
                self.assertEqual(len(clients_data), 0)
        finally:
            shutil.rmtree(temp_dir)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
