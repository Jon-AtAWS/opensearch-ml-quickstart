#!/usr/bin/env python3
# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

"""
Test runner for configs module tests.

This script runs the unit tests for the configs directory.
Use pytest for more advanced testing features.

Usage:
    python tests/run_configs_tests.py [--integration]
    
Options:
    --integration: Also run integration tests (requires actual config files)
"""

import sys
import os
import logging
import argparse

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def run_unit_tests():
    """Run unit tests that don't require external dependencies"""
    print("=" * 60)
    print("RUNNING CONFIGS UNIT TESTS")
    print("=" * 60)
    
    try:
        # Import and run unit tests
        from tests.unit.configs.test_configuration_manager import (
            TestEnums, TestDataClasses, TestConfigurationManager, TestModuleFunctions
        )
        from tests.unit.configs.test_configs_init import (
            TestConfigsInit, TestConfigsModuleIntegration
        )
        from tests.unit.configs.test_config_validation import (
            TestConfigurationValidation, TestConfigurationEdgeCases,
            TestConfigurationFileHandling, TestConfigurationConstants
        )
        
        # Run tests manually (basic test runner)
        test_classes = [
            TestEnums, TestDataClasses, TestConfigurationManager, TestModuleFunctions,
            TestConfigsInit, TestConfigsModuleIntegration,
            TestConfigurationValidation, TestConfigurationEdgeCases,
            TestConfigurationFileHandling, TestConfigurationConstants
        ]
        
        total_tests = 0
        passed_tests = 0
        failed_tests = 0
        
        for test_class in test_classes:
            print(f"\nRunning {test_class.__name__}...")
            
            # Get all test methods
            test_methods = [method for method in dir(test_class) if method.startswith('test_')]
            
            for test_method in test_methods:
                total_tests += 1
                try:
                    # Create instance and run setup if it exists
                    instance = test_class()
                    if hasattr(instance, 'setup_method'):
                        instance.setup_method()
                    
                    # Run the test method
                    getattr(instance, test_method)()
                    print(f"  ✓ {test_method}")
                    passed_tests += 1
                    
                except Exception as e:
                    print(f"  ✗ {test_method}: {e}")
                    failed_tests += 1
        
        print(f"\n" + "=" * 60)
        print(f"UNIT TEST RESULTS")
        print(f"=" * 60)
        print(f"Total tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Success rate: {(passed_tests/total_tests)*100:.1f}%")
        
        return failed_tests == 0
        
    except ImportError as e:
        print(f"Error importing test modules: {e}")
        print("Make sure all dependencies are installed and PYTHONPATH is set correctly")
        return False
    except Exception as e:
        print(f"Error running unit tests: {e}")
        return False


def run_integration_tests():
    """Run integration tests that require actual configuration files"""
    print("\n" + "=" * 60)
    print("RUNNING CONFIGS INTEGRATION TESTS")
    print("=" * 60)
    print("Note: These tests require actual configuration files and dynaconf")
    
    try:
        from tests.unit.configs.test_integration import (
            test_config_loading, test_model_configs
        )
        
        print("\nTesting configuration loading...")
        try:
            test_config_loading()
            print("  ✓ Configuration loading test passed")
        except Exception as e:
            print(f"  ✗ Configuration loading test failed: {e}")
        
        print("\nTesting model configurations...")
        try:
            test_model_configs()
            print("  ✓ Model configuration test passed")
        except Exception as e:
            print(f"  ✗ Model configuration test failed: {e}")
            
    except Exception as e:
        print(f"Error running integration tests: {e}")


def main():
    """Main test runner"""
    parser = argparse.ArgumentParser(description='Run configs module tests')
    parser.add_argument('--integration', action='store_true', 
                       help='Also run integration tests (requires actual config files)')
    parser.add_argument('--unit-only', action='store_true',
                       help='Run only unit tests (default)')
    
    args = parser.parse_args()
    
    # Run unit tests
    success = run_unit_tests()
    
    # Run integration tests if requested
    if args.integration:
        run_integration_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
