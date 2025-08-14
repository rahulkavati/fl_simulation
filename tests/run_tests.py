#!/usr/bin/env python3
"""
Test runner for FL simulation project

This script provides a convenient way to run all tests with different options:
- Run all tests
- Run specific test modules
- Run with coverage reporting
- Run with different verbosity levels
"""

import os
import sys
import argparse
import subprocess
import unittest
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def discover_and_run_tests(pattern="test_*.py", verbosity=2, coverage=False):
    """Discover and run all tests"""
    # Discover tests
    loader = unittest.TestLoader()
    start_dir = Path(__file__).parent
    suite = loader.discover(start_dir, pattern=pattern)
    
    if coverage:
        return run_with_coverage(suite, verbosity)
    else:
        return run_tests_directly(suite, verbosity)

def run_tests_directly(suite, verbosity):
    """Run tests directly without coverage"""
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    return result.wasSuccessful()

def run_with_coverage(suite, verbosity):
    """Run tests with coverage reporting"""
    try:
        import coverage
    except ImportError:
        print("Coverage not available. Install with: pip install coverage")
        return run_tests_directly(suite, verbosity)
    
    # Start coverage measurement
    cov = coverage.Coverage()
    cov.start()
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    # Stop coverage and generate report
    cov.stop()
    cov.save()
    
    print("\n" + "="*60)
    print("COVERAGE REPORT")
    print("="*60)
    
    # Generate console report
    cov.report()
    
    # Generate HTML report
    html_dir = project_root / "htmlcov"
    cov.html_report(directory=str(html_dir))
    print(f"\nHTML coverage report generated in: {html_dir}")
    
    return result.wasSuccessful()

def run_specific_module(module_name, verbosity=2, coverage=False):
    """Run tests from a specific module"""
    module_path = Path(__file__).parent / f"{module_name}.py"
    
    if not module_path.exists():
        print(f"Test module {module_name} not found!")
        return False
    
    # Import and run the specific module
    spec = unittest.TestLoader().loadTestsFromName(module_name)
    
    if coverage:
        return run_with_coverage(spec, verbosity)
    else:
        runner = unittest.TextTestRunner(verbosity=verbosity)
        result = runner.run(spec)
        return result.wasSuccessful()

def run_performance_tests():
    """Run performance benchmarks"""
    print("Running performance tests...")
    
    # Import and run performance tests
    try:
        from tests.test_efficiency_metrics import TestPerformanceBenchmarks
        suite = unittest.TestLoader().loadTestsFromTestCase(TestPerformanceBenchmarks)
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        return result.wasSuccessful()
    except ImportError as e:
        print(f"Could not import performance tests: {e}")
        return False

def check_test_dependencies():
    """Check if all test dependencies are available"""
    required_packages = [
        'numpy',
        'pandas', 
        'sklearn',  # scikit-learn is imported as sklearn
        'matplotlib',
        'seaborn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("Missing test dependencies:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\nInstall with: pip install " + " ".join(missing_packages))
        return False
    
    return True

def main():
    """Main test runner function"""
    parser = argparse.ArgumentParser(description="Run FL simulation tests")
    parser.add_argument(
        "--module", "-m",
        help="Run tests from specific module (e.g., test_efficiency_metrics)"
    )
    parser.add_argument(
        "--pattern",
        help="Test file pattern (default: test_*.py)"
    )
    parser.add_argument(
        "--verbosity", "-v",
        type=int,
        default=2,
        choices=[0, 1, 2],
        help="Test output verbosity (0=quiet, 1=normal, 2=verbose)"
    )
    parser.add_argument(
        "--coverage", "-c",
        action="store_true",
        help="Run tests with coverage reporting"
    )
    parser.add_argument(
        "--performance",
        action="store_true",
        help="Run only performance tests"
    )
    parser.add_argument(
        "--check-deps",
        action="store_true",
        help="Check test dependencies"
    )
    
    args = parser.parse_args()
    
    print("FL Simulation Test Runner")
    print("="*40)
    
    # Check dependencies if requested
    if args.check_deps:
        if check_test_dependencies():
            print("✅ All test dependencies are available")
        else:
            print("❌ Some test dependencies are missing")
        return
    
    # Check dependencies before running tests
    if not check_test_dependencies():
        print("Cannot run tests due to missing dependencies")
        return
    
    # Run specific type of tests
    if args.performance:
        success = run_performance_tests()
    elif args.module:
        success = run_specific_module(args.module, args.verbosity, args.coverage)
    else:
        success = discover_and_run_tests(args.pattern, args.verbosity, args.coverage)
    
    # Report results
    print("\n" + "="*40)
    if success:
        print("✅ All tests passed!")
        sys.exit(0)
    else:
        print("❌ Some tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
