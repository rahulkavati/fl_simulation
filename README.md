# 🚀 Federated Learning Simulation Framework

A comprehensive, production-ready federated learning simulation framework with advanced efficiency metrics, testing, and visualization capabilities.

**Author**: Rahul Kavati  
**Status**: 🟢 Production Ready with Comprehensive Testing

## 🌟 Key Features

- **🔬 FL Simulation**: Complete federated learning workflow with synthetic health data
- **📊 Efficiency Metrics**: Comprehensive performance analysis and benchmarking
- **🧪 Testing Framework**: 80%+ code coverage with unit, integration, and performance tests
- **📈 Visualization**: Advanced metrics analysis and plotting capabilities
- **🔒 Security Ready**: Designed for future integration with CKKS encryption
- **🚀 CI/CD**: Automated testing and quality assurance via GitHub Actions
- **📚 Documentation**: Comprehensive guides and API documentation

## 🏗️ Architecture Overview

```
fl_simulation/
├── 📁 common/                 # Core utilities and schemas
│   ├── schemas.py            # Data structures and validation
│   └── efficiency_metrics.py # FL performance analysis
├── 📁 simulation/            # FL simulation engine
│   └── client_simulation.py  # Main simulation logic
├── 📁 data/                  # Data generation and storage
│   ├── clients/              # Client datasets
│   └── simulate_health_data.py
├── 📁 updates/               # Model updates storage
│   ├── json/                 # Human-readable updates
│   └── numpy/                # Binary updates for processing
├── 📁 visualize/             # Analysis and visualization
│   └── metrics_analysis.py   # Metrics plotting and analysis
├── 📁 tests/                 # Comprehensive testing suite
│   ├── test_efficiency_metrics.py
│   ├── test_client_simulation.py
│   └── run_tests.py
├── 📁 scripts/               # Automation and utilities
│   └── run_experiments.py    # Multi-experiment runner
└── 📁 .github/workflows/     # CI/CD automation
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd fl_simulation

# Install dependencies
pip install -r requirements.txt

# Verify installation
python3 tests/run_tests.py --check-deps
```

### 2. Run Basic Simulation

```bash
# Run FL simulation with 5 clients, 3 rounds
python3 simulation/client_simulation.py

# Expected output:
# Starting Federated Learning Simulation...
# Loaded data for 5 clients
# Initialized global model with 4 features
# --- Round 1/3 ---
#   Training client client_0 with 200 samples...
#   ...
# Simulation completed successfully!
```

### 3. Analyze Results

```bash
# Generate efficiency metrics and visualizations
python3 visualize/metrics_analysis.py

# View generated plots in metrics/ directory
open metrics/accuracy_analysis.png
open metrics/efficiency_metrics.png
```

## 📊 Efficiency Metrics

Our framework provides comprehensive FL efficiency analysis:

### **Communication Efficiency**
- Total communication rounds
- Bytes transferred
- Communication overhead percentage

### **Training Efficiency**
- Training time per round
- Convergence analysis
- Resource utilization

### **Model Performance**
- Accuracy improvement tracking
- Weight convergence analysis
- Loss reduction metrics

### **Resource Metrics**
- Memory usage optimization
- CPU utilization tracking
- Scalability analysis

## 🧪 Testing Framework

### **Test Coverage: 80%+ Target**

```bash
# Run all tests
python3 tests/run_tests.py

# Run with coverage reporting
python3 tests/run_tests.py --coverage

# Run specific test modules
python3 tests/run_tests.py --module test_efficiency_metrics

# Performance testing
python3 tests/run_tests.py --performance
```

### **Test Categories**

- **Unit Tests**: Individual component validation
- **Integration Tests**: Component interaction testing
- **Performance Tests**: Speed and efficiency benchmarks
- **Edge Case Tests**: Error handling and boundary conditions

### **Automated Quality Assurance**

- **GitHub Actions**: Automated testing on every push/PR
- **Code Coverage**: Track test coverage over time
- **Linting**: Code quality and style enforcement
- **Security**: Vulnerability scanning and dependency checks

## 🔬 Advanced Usage

### **Multi-Experiment Runner**

```bash
# Run multiple experiments with different configurations
python3 scripts/run_experiments.py

# This will run simulations with:
# - 2 rounds, 3 rounds, 4 rounds, 5 rounds
# - Generate comprehensive metrics for comparison
# - Save results for analysis
```

### **Custom Configurations**

```python
# Modify simulation parameters in simulation/client_simulation.py
NUM_ROUNDS = 5          # Number of FL rounds
DATA_DIR = "data/clients"  # Client data directory
OUTPUT_JSON = "updates/json"  # JSON output location
OUTPUT_NPY = "updates/numpy"  # NumPy output location
```

### **Future Integration Ready**

The framework is designed to be easily extended for future collaborative work:

```python
# Updates are saved in standardized formats:
# JSON: Human-readable for debugging
# NumPy: Binary format for encryption processing

# Example update structure:
{
    "client_id": "client_0",
    "round_id": 0,
    "weight_delta": [0.1, -0.2, 0.3, -0.1],
    "bias_delta": 0.05,
    "num_samples": 200
}
```

## 📈 Results and Analysis

### **Generated Files**

After running simulations, you'll find:

```
metrics/
├── fl_simulation_3rounds_5clients.json  # Individual experiment
├── metrics_summary.json                  # Aggregated results
├── metrics_history.csv                   # CSV for analysis
├── accuracy_analysis.png                 # Accuracy plots
├── efficiency_metrics.png                # Efficiency plots
├── convergence_trends.png                # Convergence analysis
└── analysis_report.json                  # Detailed report
```

### **Key Metrics Output**

```
==================================================
FL EFFICIENCY METRICS SUMMARY
==================================================
Communication Rounds: 15
Bytes Transferred: 15.00 KB
Final Accuracy: 0.7980
Accuracy Improvement: 0.2980
Convergence Round: Not reached
Memory Usage: 0.0000 MB
==================================================
```

## 🛠️ Development

### **Adding New Features**

1. **Create Tests First**: Follow TDD principles
2. **Update Documentation**: Keep README current
3. **Run Quality Checks**: Ensure all tests pass
4. **Update Requirements**: Add new dependencies

### **Code Quality Standards**

- **Test Coverage**: Maintain 80%+ coverage
- **Linting**: Follow PEP 8 standards
- **Type Hints**: Use type annotations
- **Documentation**: Comprehensive docstrings

### **Testing Best Practices**

```bash
# Before committing:
python3 tests/run_tests.py --coverage
python3 -m flake8 common simulation visualize
python3 -m black --check common simulation visualize
```

## 🔧 Configuration

### **Environment Variables**

```bash
# Optional: Set custom paths
export FL_DATA_DIR="/path/to/data"
export FL_OUTPUT_DIR="/path/to/output"
export FL_LOG_LEVEL="INFO"
```

### **Performance Tuning**

```python
# In simulation/client_simulation.py
global_model = LogisticRegression(
    penalty=None,
    fit_intercept=True,
    solver="lbfgs",
    max_iter=1000,        # Increase for better convergence
    warm_start=True,
    random_state=42       # For reproducibility
)
```

## 📚 API Reference

### **Core Classes**

#### `FLEfficiencyMetrics`
```python
@dataclass
class FLEfficiencyMetrics:
    timestamp: str
    num_clients: int
    num_rounds: int
    total_samples: int
    # ... and many more metrics
```

#### `FLEfficiencyCalculator`
```python
class FLEfficiencyCalculator:
    def calculate_efficiency_metrics(self, clients_data, global_model, num_rounds, training_time=None)
    def save_metrics(self, metrics, experiment_name=None)
    def calculate_communication_efficiency(self, num_clients, num_rounds)
```

### **Main Simulation Functions**

```python
def main():
    """Main FL simulation workflow"""
    # Load client data
    # Initialize global model
    # Run FL rounds
    # Calculate efficiency metrics
    # Save results
```

## 🚨 Troubleshooting

### **Common Issues**

1. **Import Errors**: Ensure Python path includes project root
2. **Missing Dependencies**: Run `pip install -r requirements.txt`
3. **Test Failures**: Check test output for specific error details
4. **Performance Issues**: Verify system resources and data sizes

### **Debug Mode**

```bash
# Run with maximum verbosity
python3 tests/run_tests.py --verbosity 2

# Single test debugging
python3 -m pytest tests/test_efficiency_metrics.py::TestFLEfficiencyMetrics::test_metrics_creation -s
```

### **Getting Help**

1. Check this README
2. Review test output carefully
3. Check dependency versions
4. Verify test environment
5. Create issue with detailed error information

## 🔮 Future Enhancements

### **Planned Features**

- **Distributed Training**: Multi-node FL simulation
- **Advanced Aggregation**: FedProx, FedNova algorithms
- **Real-time Monitoring**: Live metrics dashboard
- **Benchmarking Suite**: Compare different FL approaches
- **Export Formats**: TensorFlow, PyTorch model export

### **Research Integration**

- **Paper Reproduction**: Standard FL algorithm implementations
- **Custom Algorithms**: Easy integration of new FL methods
- **Performance Analysis**: Comprehensive benchmarking tools
- **Publication Ready**: Generate publication-quality plots

### **Collaborative Extensions**

The framework is designed to be easily extended for collaborative research:

- **Encryption Layer**: Integration with CKKS homomorphic encryption
- **Advanced Aggregation**: Custom aggregation algorithms
- **Multi-Party Computation**: Secure multi-party FL protocols
- **Blockchain Integration**: Decentralized FL coordination

## 🤝 Contributing

We welcome contributions! Please follow our development workflow:

1. **Fork** the repository
2. **Create** a feature branch
3. **Add** comprehensive tests
4. **Ensure** all tests pass
5. **Update** documentation
6. **Submit** a pull request

### **Development Setup**

```bash
# Install development dependencies
pip install -r requirements.txt

# Run quality checks
python3 tests/run_tests.py --coverage
python3 -m flake8 common simulation visualize
python3 -m black common simulation visualize
python3 -m isort common simulation visualize
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Research Community**: FL algorithm implementations and research papers
- **Open Source**: Testing and development tools
- **Academic Institutions**: Supporting federated learning research
- **Industry Partners**: Real-world FL applications and use cases

For detailed author information and research background, see [AUTHOR.md](AUTHOR.md).

---

**Ready to revolutionize your federated learning research? 🚀**

Start with `python3 simulation/client_simulation.py` and explore the comprehensive testing framework with `python3 tests/run_tests.py --coverage`!

