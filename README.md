# Edge-Assisted Homomorphic Federated Learning (EAH-FL)

A comprehensive implementation of federated learning with homomorphic encryption and edge computing architecture.

## Project Structure

```
fl_simulation/
├── src/                          # Core source code
│   ├── edge_fl/                  # FHE Edge FL implementation
│   ├── plaintext_edge_fl/        # Plaintext Edge FL implementation
│   ├── encryption/               # Homomorphic encryption utilities
│   ├── fl/                       # Core FL configuration
│   ├── plaintext/                # Plaintext FL implementation
│   └── utils/                    # Utility functions
├── scripts/                      # Execution scripts
│   ├── run_edge_fl.py           # Run FHE Edge FL
│   ├── run_plaintext_edge_fl.py # Run Plaintext Edge FL
│   ├── run_fhe_fl.py            # Run original FHE FL
│   ├── run_plaintext_fl.py      # Run original Plaintext FL
│   ├── timing_analysis_report.py
│   └── research_metrics_summary.py
├── docs/                         # Documentation
│   ├── EAH_FL_DataFlow_Algorithm.md
│   ├── EAH_FL_IEEE_Algorithm.md
│   └── EAH_FL_IEEE_Single_Paragraph_Algorithm.md
├── data/                         # Dataset and client data
│   ├── health_fitness_data.csv
│   └── clients/                  # Individual client datasets
├── results/                      # Original FL results
│   ├── fhe/                      # FHE FL results
│   └── plaintext/                # Plaintext FL results
├── fhe_edge_results/             # FHE Edge FL results
├── plaintext_edge_results/        # Plaintext Edge FL results
├── federated_learning_pipeline.py # Main FHE pipeline
├── plaintext_federated_learning_pipeline.py # Main plaintext pipeline
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Key Features

- **Edge-Assisted Architecture**: n clients + n edge devices for encryption/decryption
- **Homomorphic Encryption**: CKKS scheme using Microsoft TenSEAL
- **Federated Learning**: Weighted aggregation with privacy preservation
- **One-Class Client Handling**: Laplace smoothing, FedProx, and warm-start strategies
- **Comprehensive Metrics**: Accuracy, F1, AUC, MAE, RMSE, and timing statistics

## Quick Start

1. Install dependencies:
```bash
   pip install -r requirements.txt
```

2. Run FHE Edge FL:
```bash
   python scripts/run_edge_fl.py --clients 10 --rounds 10
```

3. Run Plaintext Edge FL:
```bash
   python scripts/run_plaintext_edge_fl.py --clients 10 --rounds 10
   ```

## Algorithms

The project implements **Algorithm 1: Edge-Assisted Homomorphic Federated Learning (EAH-FL)** with four main phases:

1. **Client Local Training**: Local model training with one-class handling
2. **Edge Device Encryption**: Homomorphic encryption of model parameters
3. **Cloud Server Aggregation**: Weighted aggregation in encrypted domain
4. **Edge Device Decryption**: Decryption and client synchronization

## Results

Results are automatically saved to:
- `fhe_edge_results/` - FHE Edge FL results
- `plaintext_edge_results/` - Plaintext Edge FL results
- `results/` - Original FL results

## Documentation

See `docs/` directory for detailed algorithm descriptions and IEEE-style documentation.