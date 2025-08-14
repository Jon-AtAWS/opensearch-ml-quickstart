#!/usr/bin/env python3
# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

"""
Test runner for mapping module tests.

This script provides a convenient way to run all mapping-related tests
with proper configuration and reporting.
"""

import sys
import os
import subprocess
import logging

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def run_mapping_tests():
    """Run all mapping unit tests"""
    logger = logging.getLogger(__name__)
    
    logger.info("Starting mapping module tests...")
    
    # Change to project root directory
    os.chdir(project_root)
    
    # Run pytest with mapping tests
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/unit/mapping/",
        "-v",
        "--tb=short",
        "--color=yes",
        "--durations=10"
    ]
    
    logger.info(f"Running command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        logger.info("All mapping tests passed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Tests failed with exit code: {e.returncode}")
        return False
    except Exception as e:
        logger.error(f"Error running tests: {e}")
        return False

def run_mapping_tests_with_coverage():
    """Run mapping tests with coverage reporting"""
    logger = logging.getLogger(__name__)
    
    logger.info("Starting mapping module tests with coverage...")
    
    # Change to project root directory
    os.chdir(project_root)
    
    # Run pytest with coverage
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/unit/mapping/",
        "--cov=mapping",
        "--cov-report=html:htmlcov/mapping",
        "--cov-report=term-missing",
        "-v",
        "--tb=short"
    ]
    
    logger.info(f"Running command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        logger.info("All mapping tests passed with coverage report generated!")
        logger.info("Coverage report available at: htmlcov/mapping/index.html")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Tests failed with exit code: {e.returncode}")
        return False
    except Exception as e:
        logger.error(f"Error running tests: {e}")
        return False

def main():
    """Main function"""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Check if coverage is requested
    run_coverage = "--coverage" in sys.argv or "-c" in sys.argv
    
    if run_coverage:
        success = run_mapping_tests_with_coverage()
    else:
        success = run_mapping_tests()
    
    if success:
        logger.info("✅ All mapping tests completed successfully!")
        sys.exit(0)
    else:
        logger.error("❌ Some mapping tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
