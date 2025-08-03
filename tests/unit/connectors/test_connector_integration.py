# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

"""
Integration tests for connector classes and configuration validation.
"""

import pytest
from unittest.mock import patch, MagicMock
from connectors.helper import get_remote_connector_configs
from connectors.embedding_connector import EmbeddingConnector


class TestConnectorIntegration:
    """Integration tests for connector functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_os_client = MagicMock()
        self.mock_os_client.transport = MagicMock()
        self.mock_os_client.transport.perform_request = MagicMock()
        
        self.base_args = {
            "os_client": self.mock_os_client,
            "opensearch_domain_url": "https://test-domain.us-west-2.es.amazonaws.com",
            "opensearch_domain_arn": "arn:aws:es:us-west-2:123456789012:domain/test-domain",
            "opensearch_username": "admin",
            "opensearch_password": "password",
            "aws_user_name": "test_user",
            "region": "us-west-2"
        }

    @patch('connectors.helper.get_raw_config_value')
    def test_sagemaker_dense_sparse_configuration_consistency(self, mock_get_config):
        """Test that SageMaker dense and sparse configurations are consistent and correctly mapped."""
        # Mock configuration values with clear distinction between dense and sparse
        mock_config_values = {
            "SAGEMAKER_DENSE_ARN": "arn:aws:sagemaker:us-west-2:123456789012:endpoint/DENSE-model-endpoint",
            "SAGEMAKER_SPARSE_ARN": "arn:aws:sagemaker:us-west-2:123456789012:endpoint/SPARSE-model-endpoint",
            "SAGEMAKER_CONNECTOR_ROLE_NAME": "sagemaker_connector_role",
            "SAGEMAKER_CREATE_CONNECTOR_ROLE_NAME": "sagemaker_create_connector_role",
            "AWS_REGION": "us-west-2",
            "SAGEMAKER_CONNECTOR_VERSION": "1.0",
            "SAGEMAKER_SPARSE_URL": "https://runtime.sagemaker.us-west-2.amazonaws.com/endpoints/SPARSE-endpoint/invocations",
            "SAGEMAKER_DENSE_URL": "https://runtime.sagemaker.us-west-2.amazonaws.com/endpoints/DENSE-endpoint/invocations",
            "SAGEMAKER_DENSE_MODEL_DIMENSION": "384"
        }
        
        def mock_get_config_side_effect(key):
            return mock_config_values.get(key)
        
        mock_get_config.side_effect = mock_get_config_side_effect
        
        # Test dense configuration
        dense_config = get_remote_connector_configs("sagemaker", "aos")
        assert dense_config["dense_arn"] == "arn:aws:sagemaker:us-west-2:123456789012:endpoint/DENSE-model-endpoint"
        assert dense_config["sparse_arn"] == "arn:aws:sagemaker:us-west-2:123456789012:endpoint/SPARSE-model-endpoint"
        assert dense_config["dense_url"] == "https://runtime.sagemaker.us-west-2.amazonaws.com/endpoints/DENSE-endpoint/invocations"
        assert dense_config["sparse_url"] == "https://runtime.sagemaker.us-west-2.amazonaws.com/endpoints/SPARSE-endpoint/invocations"
        
        # Verify that dense and sparse are not swapped
        assert "DENSE" in dense_config["dense_arn"]
        assert "SPARSE" in dense_config["sparse_arn"]
        assert "DENSE" in dense_config["dense_url"]
        assert "SPARSE" in dense_config["sparse_url"]

    @patch('connectors.helper.get_raw_config_value')
    def test_bedrock_dense_configuration(self, mock_get_config):
        """Test that Bedrock dense configuration is correctly loaded."""
        # Mock configuration values for Bedrock
        mock_config_values = {
            "BEDROCK_ARN": "arn:aws:bedrock:*::foundation-model/amazon.titan-embed-text-v1",
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
        
        # Test Bedrock configuration
        bedrock_config = get_remote_connector_configs("bedrock", "aos")
        assert bedrock_config["dense_arn"] == "arn:aws:bedrock:*::foundation-model/amazon.titan-embed-text-v1"
        assert bedrock_config["dense_url"] == "https://bedrock-runtime.us-west-2.amazonaws.com/model/amazon.titan-embed-text-v1/invoke"
        assert bedrock_config["model_dimensions"] == "1536"
        assert bedrock_config["connector_role_name"] == "bedrock_connector_role"

    @patch('connectors.embedding_connector.get_remote_connector_configs')
    def test_embedding_connector_dense_sparse_switching(self, mock_get_configs):
        """Test that EmbeddingConnector correctly handles dense and sparse embedding types."""
        # Mock configuration that supports both dense and sparse
        mock_config = {
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
        mock_get_configs.return_value = mock_config
        
        with patch.object(EmbeddingConnector, '_get_connector_id', return_value="test_connector_id"):
            # Test dense embedding connector
            dense_connector = EmbeddingConnector(
                provider="sagemaker",
                os_type="aos",
                connector_configs={"embedding_type": "dense"},
                **self.base_args
            )
            
            dense_info = dense_connector.get_provider_model_info()
            assert dense_info["embedding_type"] == "dense"
            assert dense_info["model_arn"] == "arn:aws:sagemaker:us-west-2:123456789012:endpoint/dense-endpoint"
            
            # Test sparse embedding connector
            sparse_connector = EmbeddingConnector(
                provider="sagemaker",
                os_type="aos",
                connector_configs={"embedding_type": "sparse"},
                **self.base_args
            )
            
            sparse_info = sparse_connector.get_provider_model_info()
            assert sparse_info["embedding_type"] == "sparse"
            assert sparse_info["model_arn"] == "arn:aws:sagemaker:us-west-2:123456789012:endpoint/sparse-endpoint"

    @patch('connectors.embedding_connector.get_remote_connector_configs')
    def test_configuration_merging_preserves_essential_fields(self, mock_get_configs):
        """Test that configuration merging preserves essential fields from auto-loaded config."""
        # Mock auto-loaded configuration with all required fields
        auto_config = {
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
        mock_get_configs.return_value = auto_config
        
        # Partial configuration that would be missing essential fields
        partial_config = {
            "model_name": "custom_model",
            "embedding_type": "dense",
            "custom_field": "custom_value"
        }
        
        with patch.object(EmbeddingConnector, '_get_connector_id', return_value="test_connector_id"):
            connector = EmbeddingConnector(
                provider="sagemaker",
                os_type="aos",
                connector_configs=partial_config,
                **self.base_args
            )
            
            # Verify that essential fields from auto-config are preserved
            info = connector.get_provider_model_info()
            assert info["model_arn"] == "arn:aws:sagemaker:us-west-2:123456789012:endpoint/dense-endpoint"
            assert info["connector_role"] == "sagemaker_connector_role"
            assert info["region"] == "us-west-2"
            
            # Verify that provided config values are also present
            assert connector.get_embedding_type() == "dense"

    @patch('connectors.embedding_connector.get_remote_connector_configs')
    def test_provider_capability_validation(self, mock_get_configs):
        """Test that provider capabilities are correctly validated."""
        mock_config = {
            "dense_url": "https://test.com",
            "model_dimensions": "384",
            "region": "us-west-2",
            "connector_version": "1.0"
        }
        mock_get_configs.return_value = mock_config
        
        # Test that SageMaker supports both dense and sparse
        with patch.object(EmbeddingConnector, '_get_connector_id', return_value="test_connector_id"):
            # Dense should work
            sagemaker_dense = EmbeddingConnector(
                provider="sagemaker",
                os_type="aos",
                connector_configs={"embedding_type": "dense"},
                **self.base_args
            )
            assert sagemaker_dense.get_embedding_type() == "dense"
            
            # Sparse should work
            mock_config["sparse_url"] = "https://test-sparse.com"
            mock_get_configs.return_value = mock_config
            sagemaker_sparse = EmbeddingConnector(
                provider="sagemaker",
                os_type="aos",
                connector_configs={"embedding_type": "sparse"},
                **self.base_args
            )
            assert sagemaker_sparse.get_embedding_type() == "sparse"
        
        # Test that Bedrock only supports dense
        with patch.object(EmbeddingConnector, '_get_connector_id', return_value="test_connector_id"):
            # Dense should work
            bedrock_dense = EmbeddingConnector(
                provider="bedrock",
                os_type="aos",
                connector_configs={"embedding_type": "dense"},
                **self.base_args
            )
            assert bedrock_dense.get_embedding_type() == "dense"
            
            # Sparse should fail
            with pytest.raises(ValueError, match="Provider 'bedrock' doesn't support 'sparse' embeddings"):
                EmbeddingConnector(
                    provider="bedrock",
                    os_type="aos",
                    connector_configs={"embedding_type": "sparse"},
                    **self.base_args
                )

    @patch('connectors.helper.get_raw_config_value')
    def test_all_provider_host_combinations(self, mock_get_config):
        """Test all valid provider and host type combinations."""
        # Mock all required configuration values
        mock_config_values = {
            # SageMaker OS
            "AWS_ACCESS_KEY_ID": "test_access_key",
            "AWS_SECRET_ACCESS_KEY": "test_secret_key",
            "AWS_REGION": "us-west-2",
            "SAGEMAKER_CONNECTOR_VERSION": "1.0",
            "SAGEMAKER_SPARSE_URL": "https://runtime.sagemaker.us-west-2.amazonaws.com/endpoints/sparse-endpoint/invocations",
            "SAGEMAKER_DENSE_URL": "https://runtime.sagemaker.us-west-2.amazonaws.com/endpoints/dense-endpoint/invocations",
            "SAGEMAKER_DENSE_MODEL_DIMENSION": "384",
            
            # SageMaker AOS
            "SAGEMAKER_DENSE_ARN": "arn:aws:sagemaker:us-west-2:123456789012:endpoint/dense-endpoint",
            "SAGEMAKER_SPARSE_ARN": "arn:aws:sagemaker:us-west-2:123456789012:endpoint/sparse-endpoint",
            "SAGEMAKER_CONNECTOR_ROLE_NAME": "sagemaker_connector_role",
            "SAGEMAKER_CREATE_CONNECTOR_ROLE_NAME": "sagemaker_create_connector_role",
            
            # Bedrock OS
            "BEDROCK_CONNECTOR_VERSION": "1.0",
            "BEDROCK_EMBEDDING_URL": "https://bedrock-runtime.us-west-2.amazonaws.com/model/amazon.titan-embed-text-v1/invoke",
            "BEDROCK_MODEL_DIMENSION": "1536",
            
            # Bedrock AOS
            "BEDROCK_ARN": "arn:aws:bedrock:*::foundation-model/*",
            "BEDROCK_CONNECTOR_ROLE_NAME": "bedrock_connector_role",
            "BEDROCK_CREATE_CONNECTOR_ROLE_NAME": "bedrock_create_connector_role"
        }
        
        def mock_get_config_side_effect(key):
            return mock_config_values.get(key)
        
        mock_get_config.side_effect = mock_get_config_side_effect
        
        # Test all valid combinations
        valid_combinations = [
            ("sagemaker", "os"),
            ("sagemaker", "aos"),
            ("bedrock", "os"),
            ("bedrock", "aos")
        ]
        
        for provider, host_type in valid_combinations:
            config = get_remote_connector_configs(provider, host_type)
            
            # Verify that each combination returns a valid configuration
            assert "region" in config
            assert "connector_version" in config
            
            if provider == "sagemaker":
                if host_type == "os":
                    assert "access_key" in config
                    assert "secret_key" in config
                    assert "dense_url" in config
                    assert "sparse_url" in config
                else:  # aos
                    assert "dense_arn" in config
                    assert "sparse_arn" in config
                    assert "connector_role_name" in config
            
            elif provider == "bedrock":
                if host_type == "os":
                    assert "access_key" in config
                    assert "secret_key" in config
                    assert "dense_url" in config
                else:  # aos
                    assert "dense_arn" in config
                    assert "connector_role_name" in config
