#!/usr/bin/env bash
set -e  

# FL Simulation Framework - Complete Workflow
# Author: Rahul Kavati

echo "🚀 Starting FL Simulation Framework..."

# 1) Generate synthetic client datasets
echo "📊 Generating synthetic health data..."
python3 data/simulate_health_data.py --clients 5 --outdir data/clients

# 2) Run federated learning simulation (produces plaintext updates + server aggregation)
echo "🔬 Running FL simulation..."
python3 simulation/client_simulation.py

# 3) Analyze efficiency metrics and generate visualizations
echo "📈 Analyzing efficiency metrics..."
python3 visualize/metrics_analysis.py

# 4) Display results summary
echo "✅ FL Simulation completed successfully!"
echo "📁 Check the following directories for results:"
echo "   - updates/: Model updates and aggregations"
echo "   - metrics/: Efficiency metrics and visualizations"
echo "   - data/clients/: Client datasets"

echo ""
echo "🎯 Next steps:"
echo "   - View metrics: open metrics/accuracy_analysis.png"
echo "   - Run experiments: python3 scripts/run_experiments.py"
echo "   - Run tests: python3 tests/run_tests.py --coverage"
