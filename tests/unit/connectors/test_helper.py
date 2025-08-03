# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for connector helper functions.
"""

import pytest
from unittest.mock import patch, MagicMock
from connectors.helper import get_remote_connector_configs


class TestGetRemoteConnectorConfigs:
    """Test cases for get_remote_connector_configs function."""

    def test_invalid_connector_type(self):
        """Test that invalid connector type raises ValueError."""
        with pytest.raises(ValueError, match="connector_type must be either sagemaker or bedrock"):
            get_remote_connector_configs("invalid", "os")

    def test_invalid_host_type(self):
        """Test that invalid host type raises ValueError."""
        with pytest.raises(ValueError, match="host_type must either be os or aos"):
            get_remote_connector_configs("sagemaker", "invalid")

    @patch('connectors.helper.get_raw_config_value')
    def test_sagemaker_os_configuration(self, mock_get_config):
        """Test SageMaker OS configuration loading."""
        # Mock configuration values
        mock_config_values = {
            "AWS_ACCESS_KEY_ID": "test_access_key",
            "AWS_SECRET_ACCESS_KEY": "test_secret_key",
            "AWS_REGION": "us-west-2",
            "SAGEMAKER_CONNECTOR_VERSION": "1.0",
            "SAGEMAKER_SPARSE_URL": "https://runtime.sagemaker.us-west-2.amazonaws.com/endpoints/sparse-endpoint/invocations",
            "SAGEMAKER_DENSE_URL": "https://runtime.sagemaker.us-west-2.amazonaws.com/endpoints/dense-endpoint/invocations",
            "SAGEMAKER_DENSE_MODEL_DIMENSION": "384"
        }
        
        def mock_get_config_side_effect(key):
            return mock_config_values.get(key)
        
        mock_get_config.side_effect = mock_get_config_side_effect
        
        result = get_remote_connector_configs("sagemaker", "os")
        
        expected = {
            "access_key": "test_access_key",
            "secret_key": "test_secret_key",
            "region": "us-west-2",
            "connector_version": "1.0",
            "sparse_url": "https://runtime.sagemaker.us-west-2.amazonaws.com/endpoints/sparse-endpoint/invocations",
            "dense_url": "https://runtime.sagemaker.us-west-2.amazonaws.com/endpoints/dense-endpoint/invocations",
            "model_dimensions": "384"
        }
        
        assert result == expected

    @patch('connectors.helper.get_raw_config_value')
    def test_sagemaker_aos_configuration(self, mock_get_config):
        """Test SageMaker AOS configuration loading with correct ARN mappings."""
        # Mock configuration values
        mock_config_values = {
            "SAGEMAKER_DENSE_ARN": "arn:aws:sagemaker:us-west-2:123456789012:endpoint/dense-endpoint",
            "SAGEMAKER_SPARSE_ARN": "arn:aws:sagemaker:us-west-2:123456789012:endpoint/sparse-endpoint",
            "SAGEMAKER_CONNECTOR_ROLE_NAME": "sagemaker_connector_role",
            "SAGEMAKER_CREATE_CONNECTOR_ROLE_NAME": "sagemaker_create_connector_role",
            "AWS_REGION": "us-west-2",
            "SAGEMAKER_CONNECTOR_VERSION": "1.0",
            "SAGEMAKER_SPARSE_URL": "https://runtime.sagemaker.us-west-2.amazonaws.com/endpoints/sparse-endpoint/invocations",
            "SAGEMAKER_DENSE_URL": "https://runtime.sagemaker.us-west-2.amazonaws.com/endpoints/dense-endpoint/invocations",
            "SAGEMAKER_DENSE_MODEL_DIMENSION": "384"
        }
        
        def mock_get_config_side_effect(key):
            return mock_config_values.get(key)
        
        mock_get_config.side_effect = mock_get_config_side_effect
        
        result = get_remote_connector_configs("sagemaker", "aos")
        
        expected = {
            "dense_arn": "arn:aws:sagemaker:us-west-2:123456789012:endpoint/dense-endpoint",
            "sparse_arn": "arn:aws:sagemaker:us-west-2:123456789012:endpoint/sparse-endpoint",
            "connector_role_name": "sagemaker_connector_role",
            "create_connector_role_name": "sagemaker_create_connector_role",
            "region": "us-west-2",
            "connector_version": "1.0",
            "sparse_url": "https://runtime.sagemaker.us-west-2.amazonaws.com/endpoints/sparse-endpoint/invocations",
            "dense_url": "https://runtime.sagemaker.us-west-2.amazonaws.com/endpoints/dense-endpoint/invocations",
            "model_dimensions": "384"
        }
        
        assert result == expected
        
        # Verify that the ARN mappings are correct (this was the bug we fixed)
        assert result["dense_arn"] == mock_config_values["SAGEMAKER_DENSE_ARN"]
        assert result["sparse_arn"] == mock_config_values["SAGEMAKER_SPARSE_ARN"]

    @patch('connectors.helper.get_raw_config_value')
    def test_bedrock_os_configuration(self, mock_get_config):
        """Test Bedrock OS configuration loading."""
        # Mock configuration values
        mock_config_values = {
            "AWS_ACCESS_KEY_ID": "test_access_key",
            "AWS_SECRET_ACCESS_KEY": "test_secret_key",
            "AWS_REGION": "us-west-2",
            "BEDROCK_CONNECTOR_VERSION": "1.0",
            "BEDROCK_EMBEDDING_URL": "https://bedrock-runtime.us-west-2.amazonaws.com/model/amazon.titan-embed-text-v1/invoke",
            "BEDROCK_MODEL_DIMENSION": "1536"
        }
        
        def mock_get_config_side_effect(key):
            return mock_config_values.get(key)
        
        mock_get_config.side_effect = mock_get_config_side_effect
        
        result = get_remote_connector_configs("bedrock", "os")
        
        expected = {
            "access_key": "test_access_key",
            "secret_key": "test_secret_key",
            "region": "us-west-2",
            "connector_version": "1.0",
            "dense_url": "https://bedrock-runtime.us-west-2.amazonaws.com/model/amazon.titan-embed-text-v1/invoke",
            "model_dimensions": "1536"
        }
        
        assert result == expected

    @patch('connectors.helper.get_raw_config_value')
    def test_bedrock_aos_configuration(self, mock_get_config):
        """Test Bedrock AOS configuration loading."""
        # Mock configuration values
        mock_config_values = {
            "BEDROCK_ARN": "arn:aws:bedrock:*::foundation-model/*",
            "BEDROCK_CONNECTOR_ROLE_NAME": "bedrock_connector_role",
            "BEDROCK_CREATE_CONNECTOR_ROLE_NAME": "bedrock_create_connector_role",
            "AWS_REGION": "us-west-2",
            "BEDROCK_CONNECTOR_VERSION": "1.0",
            "BEDROCK_MODEL_DIMENSION": "1536",
            "BEDROCK_EMBEDDING_URL": "https://bedrock-runtime.us-west-2.amazonaws.com/model/amazon.titan-embed-text-v1/invoke"
        }
        
        def mock_get_config_side_effect(key):
            return mock_config_values.get(key)
        
        mock_get_config.side_effect = mock_get_config_side_effect
        
        result = get_remote_connector_configs("bedrock", "aos")
        
        expected = {
            "dense_arn": "arn:aws:bedrock:*::foundation-model/*",
            "connector_role_name": "bedrock_connector_role",
            "create_connector_role_name": "bedrock_create_connector_role",
            "region": "us-west-2",
            "connector_version": "1.0",
            "model_dimensions": "1536",
            "dense_url": "https://bedrock-runtime.us-west-2.amazonaws.com/model/amazon.titan-embed-text-v1/invoke"
        }
        
        assert result == expected

    @patch('connectors.helper.get_raw_config_value')
    def test_missing_required_configs_sagemaker_os(self, mock_get_config):
        """Test that missing required configs raise ValueError for SageMaker OS."""
        # Mock missing configuration (return None for some required fields)
        mock_config_values = {
            "AWS_ACCESS_KEY_ID": "test_access_key",
            "AWS_SECRET_ACCESS_KEY": None,  # Missing required field
            "AWS_REGION": "us-west-2",
            "SAGEMAKER_CONNECTOR_VERSION": "1.0",
            "SAGEMAKER_SPARSE_URL": "https://runtime.sagemaker.us-west-2.amazonaws.com/endpoints/sparse-endpoint/invocations",
            "SAGEMAKER_DENSE_URL": "https://runtime.sagemaker.us-west-2.amazonaws.com/endpoints/dense-endpoint/invocations",
            "SAGEMAKER_DENSE_MODEL_DIMENSION": "384"
        }
        
        def mock_get_config_side_effect(key):
            return mock_config_values.get(key)
        
        mock_get_config.side_effect = mock_get_config_side_effect
        
        with pytest.raises(ValueError, match="Missing required OS Sagemaker configurations"):
            get_remote_connector_configs("sagemaker", "os")

    @patch('connectors.helper.get_raw_config_value')
    def test_missing_required_configs_sagemaker_aos(self, mock_get_config):
        """Test that missing required configs raise ValueError for SageMaker AOS."""
        # Mock missing configuration (return empty string for some required fields)
        mock_config_values = {
            "SAGEMAKER_DENSE_ARN": "arn:aws:sagemaker:us-west-2:123456789012:endpoint/dense-endpoint",
            "SAGEMAKER_SPARSE_ARN": "",  # Missing required field (empty string)
            "SAGEMAKER_CONNECTOR_ROLE_NAME": "sagemaker_connector_role",
            "SAGEMAKER_CREATE_CONNECTOR_ROLE_NAME": "sagemaker_create_connector_role",
            "AWS_REGION": "us-west-2",
            "SAGEMAKER_CONNECTOR_VERSION": "1.0",
            "SAGEMAKER_SPARSE_URL": "https://runtime.sagemaker.us-west-2.amazonaws.com/endpoints/sparse-endpoint/invocations",
            "SAGEMAKER_DENSE_URL": "https://runtime.sagemaker.us-west-2.amazonaws.com/endpoints/dense-endpoint/invocations",
            "SAGEMAKER_DENSE_MODEL_DIMENSION": "384"
        }
        
        def mock_get_config_side_effect(key):
            return mock_config_values.get(key)
        
        mock_get_config.side_effect = mock_get_config_side_effect
        
        with pytest.raises(ValueError, match="Missing required AOS Sagemaker configurations"):
            get_remote_connector_configs("sagemaker", "aos")

    @patch('connectors.helper.get_raw_config_value')
    def test_missing_required_configs_bedrock_os(self, mock_get_config):
        """Test that missing required configs raise ValueError for Bedrock OS."""
        # Mock missing configuration
        mock_config_values = {
            "AWS_ACCESS_KEY_ID": "test_access_key",
            "AWS_SECRET_ACCESS_KEY": "test_secret_key",
            "AWS_REGION": None,  # Missing required field
            "BEDROCK_CONNECTOR_VERSION": "1.0",
            "BEDROCK_EMBEDDING_URL": "https://bedrock-runtime.us-west-2.amazonaws.com/model/amazon.titan-embed-text-v1/invoke",
            "BEDROCK_MODEL_DIMENSION": "1536"
        }
        
        def mock_get_config_side_effect(key):
            return mock_config_values.get(key)
        
        mock_get_config.side_effect = mock_get_config_side_effect
        
        with pytest.raises(ValueError, match="Missing required OS Bedrock configurations"):
            get_remote_connector_configs("bedrock", "os")

    @patch('connectors.helper.get_raw_config_value')
    def test_missing_required_configs_bedrock_aos(self, mock_get_config):
        """Test that missing required configs raise ValueError for Bedrock AOS."""
        # Mock missing configuration
        mock_config_values = {
            "BEDROCK_ARN": "arn:aws:bedrock:*::foundation-model/*",
            "BEDROCK_CONNECTOR_ROLE_NAME": None,  # Missing required field
            "BEDROCK_CREATE_CONNECTOR_ROLE_NAME": "bedrock_create_connector_role",
            "AWS_REGION": "us-west-2",
            "BEDROCK_CONNECTOR_VERSION": "1.0",
            "BEDROCK_MODEL_DIMENSION": "1536",
            "BEDROCK_EMBEDDING_URL": "https://bedrock-runtime.us-west-2.amazonaws.com/model/amazon.titan-embed-text-v1/invoke"
        }
        
        def mock_get_config_side_effect(key):
            return mock_config_values.get(key)
        
        mock_get_config.side_effect = mock_get_config_side_effect
        
        with pytest.raises(ValueError, match="Missing required AOS Bedrock configurations"):
            get_remote_connector_configs("bedrock", "aos")

    @patch('connectors.helper.get_raw_config_value')
    def test_arn_mapping_correctness(self, mock_get_config):
        """Test that ARN mappings are correct (regression test for the bug we fixed)."""
        # This test specifically verifies that dense_arn maps to SAGEMAKER_DENSE_ARN
        # and sparse_arn maps to SAGEMAKER_SPARSE_ARN (not swapped)
        
        mock_config_values = {
            "SAGEMAKER_DENSE_ARN": "arn:aws:sagemaker:us-west-2:123456789012:endpoint/DENSE-endpoint",
            "SAGEMAKER_SPARSE_ARN": "arn:aws:sagemaker:us-west-2:123456789012:endpoint/SPARSE-endpoint",
            "SAGEMAKER_CONNECTOR_ROLE_NAME": "sagemaker_connector_role",
            "SAGEMAKER_CREATE_CONNECTOR_ROLE_NAME": "sagemaker_create_connector_role",
            "AWS_REGION": "us-west-2",
            "SAGEMAKER_CONNECTOR_VERSION": "1.0",
            "SAGEMAKER_SPARSE_URL": "https://runtime.sagemaker.us-west-2.amazonaws.com/endpoints/sparse-endpoint/invocations",
            "SAGEMAKER_DENSE_URL": "https://runtime.sagemaker.us-west-2.amazonaws.com/endpoints/dense-endpoint/invocations",
            "SAGEMAKER_DENSE_MODEL_DIMENSION": "384"
        }
        
        def mock_get_config_side_effect(key):
            return mock_config_values.get(key)
        
        mock_get_config.side_effect = mock_get_config_side_effect
        
        result = get_remote_connector_configs("sagemaker", "aos")
        
        # The key assertion: dense_arn should map to SAGEMAKER_DENSE_ARN (not SAGEMAKER_SPARSE_ARN)
        assert result["dense_arn"] == "arn:aws:sagemaker:us-west-2:123456789012:endpoint/DENSE-endpoint"
        assert result["sparse_arn"] == "arn:aws:sagemaker:us-west-2:123456789012:endpoint/SPARSE-endpoint"
        
        # Verify the mapping is not swapped
        assert result["dense_arn"] != mock_config_values["SAGEMAKER_SPARSE_ARN"]
        assert result["sparse_arn"] != mock_config_values["SAGEMAKER_DENSE_ARN"]
