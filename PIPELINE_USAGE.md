# 🚀 Complete Pipeline Usage Guide

## Overview

The `run_complete_pipeline.py` script executes the entire federated learning pipeline automatically from data generation to global model update.

## Quick Start

```bash
# Run complete pipeline with default settings (3 rounds, 5 clients)
python run_complete_pipeline.py

# Run with custom settings
python run_complete_pipeline.py --rounds 5 --clients 10

# Clean up previous outputs before running
python run_complete_pipeline.py --cleanup

# Skip prerequisite checks (if you're sure everything is set up)
python run_complete_pipeline.py --skip-checks
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--rounds N` | Number of federated learning rounds | 3 |
| `--clients N` | Number of clients/smartwatches | 5 |
| `--cleanup` | Clean up previous outputs before running | False |
| `--skip-checks` | Skip prerequisite checks | False |

## Pipeline Steps

The script executes these 6 steps automatically:

1. **Generate Synthetic Health Data** - Creates smartwatch health data for each client
2. **Prepare Encryption Context** - Sets up TenSEAL CKKS encryption keys
3. **Run Client Simulation** - Simulates federated learning training rounds
4. **Encrypt Client Updates** - Encrypts all client model updates
5. **Aggregate Encrypted Updates** - Combines encrypted updates by round
6. **Update Global Model** - Applies aggregated updates to global model

## Outputs

After successful execution, you'll have:

- **Health Data**: `data/clients/` - Synthetic smartwatch data
- **Client Updates**: `updates/json/` - Plaintext model updates
- **Encrypted Updates**: `updates/encrypted/` - Encrypted model updates
- **Aggregated Results**: `Sriven/outbox/` - Encrypted aggregated updates
- **Global Model**: `federated_artifacts/global/` - Updated global model snapshots
- **Metrics**: `metrics/` - Performance and efficiency metrics

## Example Output

```
🔐 Secure Federated Learning Pipeline
============================================================
Configuration:
  • Rounds: 3
  • Clients: 5
  • Cleanup: True

============================================================
STEP 1/6: Generate Synthetic Health Data
============================================================
✅ Generate synthetic health data for federated learning completed successfully

============================================================
STEP 2/6: Prepare Encryption Context
============================================================
✅ Generate TenSEAL CKKS encryption context completed successfully

============================================================
STEP 3/6: Run Client Simulation
============================================================
✅ Run federated learning simulation for 3 rounds completed successfully

============================================================
STEP 4/6: Encrypt Client Updates
============================================================
✅ Encrypted 15/15 update files

============================================================
STEP 5/6: Aggregate Encrypted Updates
============================================================
✅ Aggregated 3/3 rounds

============================================================
STEP 6/6: Update Global Model
============================================================
✅ Updated global model for 3/3 rounds

⏱️  Pipeline Execution Time: 11.32 seconds
✅ Successful Steps: 6/6

🎉 PIPELINE EXECUTION COMPLETE
```

## Troubleshooting

### Common Issues

1. **Missing Dependencies**: Install required packages:
   ```bash
   pip install tenseal torch numpy pandas sklearn
   ```

2. **Permission Errors**: Ensure write permissions in the project directory

3. **Memory Issues**: Reduce number of clients or rounds for large datasets

4. **Encoding Errors**: The script handles Unicode properly, but ensure your terminal supports UTF-8

### Error Recovery

- If a step fails, the script stops and shows which step failed
- Use `--cleanup` to start fresh
- Check the error messages for specific issues
- Ensure all required files exist in the project structure

## Security Features

- ✅ **Data Privacy**: Health data never leaves individual clients
- ✅ **Encrypted Transmission**: All updates encrypted with CKKS
- ✅ **Homomorphic Aggregation**: Operations performed on encrypted data
- ✅ **Secure Global Updates**: Decryption only happens in trusted cloud server
- ✅ **No Intermediate Decryption**: Aggregated data stays encrypted throughout pipeline

## Performance

- **Typical Execution Time**: 10-15 seconds for 3 rounds, 5 clients
- **Memory Usage**: ~50-100 MB depending on dataset size
- **Scalability**: Supports up to 100+ clients (may require more memory)

## Next Steps

After running the pipeline:

1. **Test Global Model**: `python test_global_update.py`
2. **Review Metrics**: Check `metrics/` directory for performance data
3. **Analyze Results**: Examine `federated_artifacts/global/` for model snapshots
4. **Customize**: Modify individual components as needed

This single script provides a complete, secure federated learning pipeline that can be easily customized and extended for different use cases.
