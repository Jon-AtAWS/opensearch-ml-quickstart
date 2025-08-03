# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

"""
Pytest configuration file to set up the test environment.

This file configures the Python path so that tests can import
project modules correctly.
"""

import sys
import os
import pytest
from pathlib import Path
from unittest.mock import patch

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

@pytest.fixture(scope="function", autouse=True)
def reset_config_manager():
    """Reset the configuration manager for each test."""
    # Clear any existing config manager
    import configs.configuration_manager as config_module
    if hasattr(config_module, 'config_manager'):
        delattr(config_module, 'config_manager')
    
    # Set up test configuration path
    test_config_path = os.path.join(os.path.dirname(__file__), "test_config.yaml")
    
    # Create a new config manager with test config
    config_module.config_manager = config_module.ConfigurationManager(test_config_path)
    
    yield
    
    # Clean up after test
    if hasattr(config_module, 'config_manager'):
        delattr(config_module, 'config_manager')

@pytest.fixture
def mock_opensearch_client():
    """Mock OpenSearch client for integration tests."""
    from unittest.mock import Mock
    mock_client = Mock()
    mock_client.info.return_value = {"version": {"number": "2.13.0"}}
    return mock_client

@pytest.fixture(autouse=True)
def mock_opensearch_for_integration_tests(request):
    """Automatically mock OpenSearch client for integration tests to avoid authentication issues."""
    if "integration" in request.keywords:
        # Skip the complex mocking that causes circular imports
        # Integration tests should handle their own mocking if needed
        yield
    else:
        yield

# Configure pytest markers
def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (require external dependencies)"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests (no external dependencies)"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow running"
    )
