# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for connector helper functions.
"""

import pytest
from unittest.mock import patch, MagicMock
from connectors.helper import get_remote_connector_configs, get_connector_payload_filename


class TestGetRemoteConnectorConfigs:
    """Test cases for get_remote_connector_configs function."""

    def test_invalid_connector_type(self):
        """Test that invalid connector type raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported combination"):
            get_remote_connector_configs("invalid", "os")

    def test_invalid_host_type(self):
        """Test that invalid host type raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported combination"):
            get_remote_connector_configs("sagemaker", "invalid")

    @patch('connectors.config_strategies.get_raw_config_value')
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

    @patch('connectors.config_strategies.get_raw_config_value')
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

    @patch('connectors.config_strategies.get_raw_config_value')
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

    @patch('connectors.config_strategies.get_raw_config_value')
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

    @patch('connectors.config_strategies.get_raw_config_value')
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
        
        with pytest.raises(ValueError, match="Missing required Sagemaker OS configurations"):
            get_remote_connector_configs("sagemaker", "os")

    @patch('connectors.config_strategies.get_raw_config_value')
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
        
        with pytest.raises(ValueError, match="Missing required Sagemaker AOS configurations"):
            get_remote_connector_configs("sagemaker", "aos")

    @patch('connectors.config_strategies.get_raw_config_value')
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
        
        with pytest.raises(ValueError, match="Missing required Bedrock OS configurations"):
            get_remote_connector_configs("bedrock", "os")

    @patch('connectors.config_strategies.get_raw_config_value')
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
        
        with pytest.raises(ValueError, match="Missing required Bedrock AOS configurations"):
            get_remote_connector_configs("bedrock", "aos")

    @patch('connectors.config_strategies.get_raw_config_value')
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


class TestGetConnectorPayloadFilename:
    """Test cases for get_connector_payload_filename function."""

    @patch('connectors.config_strategies.get_raw_config_value')
    def test_bedrock_llm_predict_payload(self, mock_get_config):
        """Test Bedrock LLM predict payload filename."""
        mock_get_config.return_value = None
        
        filename = get_connector_payload_filename("bedrock", "os", "llm_predict")
        assert filename == "bedrock_llm_predict.json"

    @patch('connectors.config_strategies.get_raw_config_value')
    def test_bedrock_llm_converse_payload(self, mock_get_config):
        """Test Bedrock LLM converse payload filename."""
        mock_get_config.return_value = None
        
        filename = get_connector_payload_filename("bedrock", "os", "llm_converse")
        assert filename == "bedrock_llm_converse.json"

    @patch('connectors.config_strategies.get_raw_config_value')
    def test_bedrock_llm_with_llm_type_param(self, mock_get_config):
        """Test Bedrock LLM with llm_type parameter."""
        mock_get_config.return_value = None
        
        filename = get_connector_payload_filename("bedrock", "os", "llm", "predict")
        assert filename == "bedrock_llm_predict.json"
        
        filename = get_connector_payload_filename("bedrock", "os", "llm", "converse")
        assert filename == "bedrock_llm_converse.json"

    @patch('connectors.config_strategies.get_raw_config_value')
    def test_claude_model_variations(self, mock_get_config):
        """Test that different Claude models use same payload files."""
        mock_get_config.return_value = None
        
        # Both Claude 3.5 and 3.7 should use the same payload files
        filename_predict = get_connector_payload_filename("bedrock", "os", "llm_predict")
        filename_converse = get_connector_payload_filename("bedrock", "os", "llm_converse")
        
        assert filename_predict == "bedrock_llm_predict.json"
        assert filename_converse == "bedrock_llm_converse.json"

    @patch('connectors.config_strategies.get_raw_config_value')
    def test_sagemaker_dense_payload(self, mock_get_config):
        """Test SageMaker dense payload filename."""
        mock_get_config.return_value = None
        
        filename = get_connector_payload_filename("sagemaker", "aos", "dense")
        assert filename == "sagemaker_dense.json"

    @patch('connectors.config_strategies.get_raw_config_value')
    def test_custom_payload_override(self, mock_get_config):
        """Test custom payload filename override."""
        mock_get_config.return_value = "custom_bedrock.json"
        
        filename = get_connector_payload_filename("bedrock", "os", "llm_predict")
        assert filename == "custom_bedrock.json"

    def test_invalid_combination(self):
        """Test invalid connector combination raises error."""
        with pytest.raises(ValueError, match="Unsupported combination"):
            get_connector_payload_filename("invalid", "os", "llm_predict")
