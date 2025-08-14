# Testing Framework for FL Simulation

This directory contains comprehensive tests for the Federated Learning simulation project.

## 🧪 Test Structure

```
tests/
├── __init__.py                 # Package initialization
├── test_efficiency_metrics.py  # Tests for efficiency metrics calculation
├── test_client_simulation.py   # Tests for client simulation logic
├── run_tests.py               # Test runner script
└── README.md                  # This file
```

## 🚀 Quick Start

### Run All Tests
```bash
# Using the test runner
python tests/run_tests.py

# Using pytest directly
pytest tests/

# Using unittest
python -m unittest discover tests/
```

### Run Specific Tests
```bash
# Run specific test module
python tests/run_tests.py --module test_efficiency_metrics

# Run with coverage
python tests/run_tests.py --coverage

# Run performance tests only
python tests/run_tests.py --performance

# Check dependencies
python tests/run_tests.py --check-deps
```

### Run with pytest
```bash
# Run with coverage and HTML report
pytest tests/ --cov=common --cov=simulation --cov=visualize --cov-report=html

# Run specific test class
pytest tests/test_efficiency_metrics.py::TestFLEfficiencyMetrics

# Run tests matching pattern
pytest tests/ -k "test_accuracy"
```

## 📊 Test Coverage

The testing framework aims for **80%+ code coverage** across all modules:

- **common/**: Efficiency metrics, schemas, utilities
- **simulation/**: Client simulation, training logic
- **visualize/**: Metrics analysis, plotting functions

## 🧩 Test Categories

### 1. Unit Tests
- **TestFLEfficiencyMetrics**: Test the metrics dataclass
- **TestFLEfficiencyCalculator**: Test metrics calculation logic
- **TestClientSimulationHelpers**: Test utility functions

### 2. Integration Tests
- **TestIntegration**: Test component interactions
- **TestDataLoading**: Test data pipeline
- **TestModelTraining**: Test FL training process

### 3. Performance Tests
- **TestPerformanceBenchmarks**: Test calculation speed
- **TestLargeData**: Test with large datasets
- **TestMemoryUsage**: Test memory efficiency

### 4. Edge Case Tests
- **TestErrorHandling**: Test error conditions
- **TestEdgeCases**: Test boundary conditions
- **TestInvalidInput**: Test malformed data

## 🔧 Test Configuration

### pytest.ini
- Coverage reporting (HTML, XML, terminal)
- Test discovery patterns
- Warning filters
- Performance markers

### GitHub Actions
- Automated testing on push/PR
- Multiple Python versions (3.8-3.11)
- Code quality checks (linting, security)
- Coverage reporting

## 📈 Coverage Reports

After running tests with coverage:

```bash
# Generate HTML report
pytest --cov=common --cov=simulation --cov=visualize --cov-report=html

# View in browser
open htmlcov/index.html
```

## 🚨 Test Failures

### Common Issues
1. **Import Errors**: Check Python path and dependencies
2. **File Not Found**: Ensure test data directories exist
3. **Permission Errors**: Check file permissions for temp directories

### Debug Mode
```bash
# Run with maximum verbosity
python tests/run_tests.py --verbosity 2

# Run single test with debugger
python -m pytest tests/test_efficiency_metrics.py::TestFLEfficiencyMetrics::test_metrics_creation -s
```

## 🧹 Test Maintenance

### Adding New Tests
1. Create test file: `test_<module_name>.py`
2. Follow naming convention: `Test<ClassName>`
3. Test methods: `test_<functionality>`
4. Add to appropriate test category

### Test Data
- Use temporary directories (`tempfile.mkdtemp()`)
- Clean up in `tearDown()` methods
- Mock external dependencies
- Create realistic test scenarios

### Best Practices
- **Arrange-Act-Assert**: Structure tests clearly
- **Descriptive Names**: Use clear test method names
- **Isolation**: Each test should be independent
- **Coverage**: Test both success and failure paths

## 🔍 Test Dependencies

### Required Packages
```bash
pip install pytest pytest-cov coverage
pip install flake8 black isort mypy
pip install bandit safety
```

### Optional Packages
```bash
# For advanced testing features
pip install pytest-mock pytest-xdist
pip install hypothesis  # Property-based testing
```

## 📋 Test Checklist

Before committing code:

- [ ] All tests pass locally
- [ ] New functionality has tests
- [ ] Test coverage > 80%
- [ ] No linting errors
- [ ] Security checks pass
- [ ] Performance tests pass

## 🚀 Continuous Integration

The project uses GitHub Actions for:

- **Automated Testing**: Run on every push/PR
- **Code Quality**: Linting, formatting, type checking
- **Security**: Vulnerability scanning
- **Coverage**: Track test coverage over time

## 📚 Additional Resources

- [pytest Documentation](https://docs.pytest.org/)
- [unittest Documentation](https://docs.python.org/3/library/unittest.html)
- [Coverage.py Documentation](https://coverage.readthedocs.io/)
- [Testing Best Practices](https://realpython.com/python-testing/)

## 🤝 Contributing

When adding tests:

1. Follow existing patterns
2. Add appropriate docstrings
3. Include edge cases
4. Test error conditions
5. Update this documentation

## 📞 Support

For testing issues:

1. Check this documentation
2. Review test output carefully
3. Check dependency versions
4. Verify test environment
5. Create issue with test details
