#!/usr/bin/env python3
"""
Test runner script for pyradtran.

This script provides an easy way to run different test suites:
- Unit tests (fast, no external dependencies) 
- Integration tests (require libradtran installation)
- All tests
- Specific test modules

Usage:
    python run_tests.py               # Run all unit tests
    python run_tests.py --all         # Run all tests including integration
    python run_tests.py --integration # Run only integration tests
    python run_tests.py --module io   # Run only IO tests
    python run_tests.py --coverage    # Run with coverage report
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_pytest(args_list):
    """Run pytest with the given arguments."""
    cmd = [sys.executable, "-m", "pytest"] + args_list
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Run pyradtran tests")
    parser.add_argument(
        "--all", 
        action="store_true", 
        help="Run all tests including integration tests"
    )
    parser.add_argument(
        "--integration", 
        action="store_true", 
        help="Run only integration tests"
    )
    parser.add_argument(
        "--unit", 
        action="store_true", 
        help="Run only unit tests (default)"
    )
    parser.add_argument(
        "--module", 
        choices=["io", "interface", "config", "core"], 
        help="Run tests for specific module"
    )
    parser.add_argument(
        "--coverage", 
        action="store_true", 
        help="Run with coverage report"
    )
    parser.add_argument(
        "--verbose", "-v", 
        action="store_true", 
        help="Verbose output"
    )
    parser.add_argument(
        "--fast", 
        action="store_true", 
        help="Run tests in parallel (faster)"
    )
    
    args = parser.parse_args()
    
    # Build pytest arguments
    pytest_args = []
    
    # Add coverage if requested
    if args.coverage:
        pytest_args.extend([
            "--cov=pyradtran", 
            "--cov-report=html", 
            "--cov-report=term-missing"
        ])
    
    # Add parallelization if requested
    if args.fast:
        pytest_args.extend(["-n", "auto"])
    
    # Determine which tests to run
    if args.integration:
        pytest_args.extend(["-m", "integration"])
    elif args.all:
        # Run all tests
        pass
    elif args.module:
        pytest_args.append(f"tests/test_{args.module}.py")
    else:
        # Default: run unit tests only
        pytest_args.extend(["-m", "not integration"])
    
    # Add verbose if requested
    if args.verbose:
        pytest_args.append("-v")
    
    print("🧪 PyRadtran Test Runner")
    print("=" * 50)
    
    if args.integration:
        print("Running integration tests (requires libradtran installation)...")
    elif args.all:
        print("Running all tests...")
    elif args.module:
        print(f"Running {args.module} tests...")
    else:
        print("Running unit tests...")
    
    return_code = run_pytest(pytest_args)
    
    if return_code == 0:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")
        
    return return_code


if __name__ == "__main__":
    sys.exit(main())
