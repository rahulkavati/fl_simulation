# 🚀 Federated Learning Simulation Framework

A comprehensive, production-ready federated learning simulation framework with **advanced encryption**, **secure aggregation**, and **encrypted global model updates**.

**Author**: Rahul Kavati  
**Status**: 🟢 Production Ready with Advanced Security Features

## 🌟 Key Features

- **🔬 FL Simulation**: Complete federated learning workflow with synthetic health data
- **🔐 Advanced Security**: CKKS homomorphic encryption and secure aggregation
- **🌐 Encrypted Global Updates**: Cloud-based model updates using encrypted aggregation
- **📊 Efficiency Metrics**: Comprehensive performance analysis and benchmarking
- **🧪 Testing Framework**: Comprehensive testing with secure approach
- **📈 Visualization**: Advanced metrics analysis and plotting capabilities
- **📁 Federated Artifacts**: Complete audit trail and model history tracking
- **🚀 CI/CD**: Automated testing and quality assurance via GitHub Actions
- **📚 Documentation**: Comprehensive guides and API documentation

## 🔐 Security-First Architecture

This framework implements a **secure-by-design** approach where:
- **Client updates are encrypted** before transmission
- **Aggregation happens in encrypted domain** using homomorphic encryption
- **Global updates use encrypted aggregation directly** - no decryption step
- **Data never exists in plaintext** outside of individual clients

## 🏗️ Architecture Overview

```
fl_simulation/
├── 📁 cloud/                  # Global model management
│   └── global_update.py      # Secure cloud server with encrypted updates
├── 📁 Huzaif/                # Encryption system
│   ├── encrypt_update.py     # Client update encryption
│   └── keys/                 # Encryption keys and parameters
├── 📁 Sriven/                # Secure aggregation system
│   ├── smart_switch_tenseal.py # Aggregation with TenSEAL
│   └── outbox/               # Encrypted aggregated results
├── 📁 common/                # Core utilities and schemas
│   ├── schemas.py            # Data structures and validation
│   └── efficiency_metrics.py # FL performance analysis
├── 📁 simulation/            # FL simulation engine
│   └── client_simulation.py  # Main simulation logic
├── 📁 data/                  # Data generation and storage
│   ├── clients/              # Client datasets
│   └── simulate_health_data.py
├── 📁 updates/               # Model updates storage
│   ├── json/                 # Human-readable updates
│   ├── numpy/                # Binary updates for processing
│   └── encrypted/            # Encrypted client updates
├── 📁 federated_artifacts/   # FL process artifacts
│   └── global/               # Global model snapshots per round
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

## 🔐 Security Architecture

### **Encryption Pipeline (Huzaif)**
- **Client Update Encryption**: Secure client contributions using CKKS
- **Key Management**: Public/private key infrastructure
- **Format Support**: JSON and binary encryption

### **Secure Aggregation (Sriven)**
- **TenSEAL Integration**: Homomorphic encryption for secure aggregation
- **Multi-Round Support**: Process multiple FL rounds
- **Output Management**: Encrypted aggregation results

### **Secure Global Model Updates (Cloud)**
- **Model Management**: PyTorch-based global model
- **Encrypted Updates**: Direct application of encrypted aggregation
- **Round Tracking**: Complete training round history
- **No Decryption**: Aggregated data never decrypted to plaintext

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd fl_simulation

# Install dependencies
pip install -r requirements.txt

# Install additional security packages
pip install tenseal torch scikit-learn

# Verify installation
python tests/run_tests.py --check-deps
```

### 2. Run Basic Simulation

```bash
# Run FL simulation with 5 clients, 5 rounds
python simulation/client_simulation.py

# Expected output:
# Starting Federated Learning Simulation...
# Loaded data for 5 clients
# Initialized global model with 4 features
# --- Round 1/5 ---
#   Training client client_0 with 200 samples...
#   ...
# Simulation completed successfully!
```

### 3. Test Global Update System

```bash
# Test global update functionality
python test_global_update_comprehensive.py

# Test multiple rounds simulation
python test_multiple_rounds.py

# Test step-by-step global updates
python test_global_update_simple.py
```

### 4. Run Complete Pipeline

```bash
# Run the complete FL pipeline (encryption → aggregation → decryption → global update)
python test_integration.py

# This will test:
# 1. Client simulation
# 2. Update encryption
# 3. Secure aggregation
# 4. Result decryption
# 5. Global model updates
```

## 🔍 Understanding Federated Artifacts

### **What Are Federated Artifacts?**
Federated Artifacts are files and data structures that capture the complete history and state of the federated learning process. They serve as the 'memory' of the FL system.

### **Types of Artifacts in Your System:**

1. **🔐 Client Updates** (`updates/json/`): Individual client model updates for each round
2. **🔒 Encrypted Updates** (`updates/encrypted/`): Secure, encrypted client contributions
3. **📦 Aggregated Results** (`Sriven/outbox/`): Combined client updates after aggregation
4. **🌐 Global Model Snapshots** (`federated_artifacts/global/`): Global model state after each update
5. **📊 Metrics & Analytics** (`metrics/`): Performance measurements and analysis
6. **🔑 Encryption Keys** (`Huzaif/keys/`): Public/private keys for secure communication

### **Benefits of Federated Artifacts:**
- **🔍 Transparency**: Complete audit trail of model evolution
- **📈 Reproducibility**: Recreate experiments exactly
- **🚀 Debugging**: Identify issues in specific rounds
- **📊 Analysis**: Analyze learning patterns and convergence
- **🔒 Compliance**: Meet regulatory requirements

## 🧪 Testing Framework

### **Test Suites Available**

```bash
# Secure global update testing (RECOMMENDED)
python test_global_update.py

# Traditional unit and integration tests
python tests/run_tests.py --coverage
```

### **Test Categories**

- **Secure Global Update Tests**: Cloud server functionality with encrypted aggregation
- **Encryption Tests**: Client update encryption
- **Aggregation Tests**: Secure aggregation with TenSEAL
- **Pipeline Tests**: End-to-end FL workflow validation
- **Unit Tests**: Individual component validation
- **Integration Tests**: Component interaction testing

## 🔬 Advanced Usage

### **Secure Global Update System**

```python
from cloud.global_update import CloudServer, load_encrypted_aggregation

# Load encrypted aggregation directly
encrypted_file = "Sriven/outbox/agg_round_0.json"
encrypted_agg = load_encrypted_aggregation(encrypted_file)

# Initialize cloud server
input_dim = encrypted_agg['layout']['weights']
cloud = CloudServer(input_dim=input_dim)

# Update global model securely (NO DECRYPTION NEEDED)
accuracy = cloud.update_global_model_encrypted(encrypted_agg, X_test, y_test)
print(f"Round {cloud.round} accuracy: {accuracy:.4f}")
```

### **Secure Encryption Pipeline**

```bash
# Encrypt client update
python Huzaif/encrypt_update.py --in updates/json/client_0_round_0.json --out updates/encrypted/enc_client_0_round_0.json --ctx Huzaif/keys/secret.ctx

# Aggregate encrypted updates
python Sriven/smart_switch_tenseal.py --fedl_dir updates/encrypted --ctx_b64 Huzaif/keys/params.ctx.b64 --out_dir Sriven/outbox

# Update global model securely (NO DECRYPTION NEEDED)
python test_global_update.py
```

### **Multi-Experiment Runner**

```bash
# Run multiple experiments with different configurations
python scripts/run_experiments.py

# This will run simulations with:
# - Multiple rounds (2, 3, 4, 5)
# - Generate comprehensive metrics for comparison
# - Save results for analysis
```

## 📊 Results and Analysis

### **Generated Files**

After running simulations, you'll find:

```
federated_artifacts/
├── global/                    # Global model snapshots
│   ├── global_round_1.npz    # Round 1 model state
│   ├── global_round_2.npz    # Round 2 model state
│   └── ...
updates/
├── json/                      # Plaintext client updates
├── encrypted/                 # Encrypted client updates
└── numpy/                     # Binary format updates
Sriven/outbox/                 # Encrypted aggregated results
├── agg_round_0.json          # Encrypted aggregation
└── ...
metrics/                       # Performance metrics and plots
```

### **Key Metrics Output**

```
==================================================
FL EFFICIENCY METRICS SUMMARY
==================================================
Communication Rounds: 25
Bytes Transferred: 25.00 KB
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

- **Test Coverage**: Maintain comprehensive testing
- **Linting**: Follow PEP 8 standards
- **Type Hints**: Use type annotations
- **Documentation**: Comprehensive docstrings

### **Testing Best Practices**

```bash
# Before committing:
python test_global_update.py
python tests/run_tests.py --coverage
python -m flake8 cloud Huzaif Sriven common simulation
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

#### `CloudServer`
```python
class CloudServer:
    def __init__(self, input_dim, save_dir="federated_artifacts/global")
    def update_global_model_encrypted(self, encrypted_aggregation, X_test=None, y_test=None)
    def update_global_model(self, aggregated_update, X_test=None, y_test=None)  # Legacy
    def evaluate(self, X_test, y_test)
    def save_snapshot(self, aggregated_update)
```

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
5. **Encryption Errors**: Check encryption keys in `Huzaif/keys/`

### **Debug Mode**

```bash
# Run with maximum verbosity
python test_global_update_comprehensive.py

# Test specific components
python test_global_update_simple.py
python test_multiple_rounds.py

# Check pipeline status
python explain_federated_artifacts.py
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

### **Security Enhancements**

- **Advanced Encryption**: Additional homomorphic encryption schemes
- **Secure Multi-Party Computation**: Multi-party FL protocols
- **Blockchain Integration**: Decentralized FL coordination
- **Privacy Preserving**: Differential privacy integration

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
python test_global_update_comprehensive.py
python test_integration.py
python tests/run_tests.py --coverage
python -m flake8 cloud Huzaif Sriven common simulation
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

**Ready to revolutionize your federated learning research with advanced security? 🚀🔐**

Start with `python simulation/client_simulation.py` and explore the comprehensive testing framework with `python test_global_update_comprehensive.py`!

