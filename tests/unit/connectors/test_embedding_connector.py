# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for EmbeddingConnector class.
"""

import pytest
from unittest.mock import patch, MagicMock, Mock
from connectors.embedding_connector import EmbeddingConnector


class TestEmbeddingConnector:
    """Test cases for EmbeddingConnector class."""

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

    @patch('connectors.embedding_connector.get_remote_connector_configs')
    def test_init_sagemaker_aos_auto_config(self, mock_get_configs):
        """Test EmbeddingConnector initialization with SageMaker AOS auto-loaded config."""
        # Mock auto-loaded configuration
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
            connector = EmbeddingConnector(
                provider="sagemaker",
                os_type="aos",
                **self.base_args
            )
        
        assert connector.get_provider() == "sagemaker"
        assert connector.get_os_type() == "aos"
        assert connector.get_embedding_type() == "dense"  # Default
        assert connector.get_model_dimensions() == "384"
        mock_get_configs.assert_called_once_with("sagemaker", "aos")

    @patch('connectors.embedding_connector.get_remote_connector_configs')
    def test_init_bedrock_aos_auto_config(self, mock_get_configs):
        """Test EmbeddingConnector initialization with Bedrock AOS auto-loaded config."""
        # Mock auto-loaded configuration
        mock_config = {
            "dense_arn": "arn:aws:bedrock:*::foundation-model/*",
            "connector_role_name": "bedrock_connector_role",
            "create_connector_role_name": "bedrock_create_connector_role",
            "region": "us-west-2",
            "connector_version": "1.0",
            "model_dimensions": "1536",
            "dense_url": "https://bedrock-runtime.us-west-2.amazonaws.com/model/amazon.titan-embed-text-v1/invoke"
        }
        mock_get_configs.return_value = mock_config
        
        with patch.object(EmbeddingConnector, '_get_connector_id', return_value="test_connector_id"):
            connector = EmbeddingConnector(
                provider="bedrock",
                os_type="aos",
                **self.base_args
            )
        
        assert connector.get_provider() == "bedrock"
        assert connector.get_os_type() == "aos"
        assert connector.get_embedding_type() == "dense"  # Default
        assert connector.get_model_dimensions() == "1536"
        mock_get_configs.assert_called_once_with("bedrock", "aos")

    @patch('connectors.embedding_connector.get_remote_connector_configs')
    def test_init_with_merged_config(self, mock_get_configs):
        """Test EmbeddingConnector initialization with merged configuration (the fix we implemented)."""
        # Mock auto-loaded configuration
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
        
        # Provided configuration (partial, like what dense_exact_search.py provides)
        provided_config = {
            "model_name": "aos_sagemaker",
            "embedding_type": "dense",
            "model_dimensions": "512",  # Override the auto-loaded value
            "connector_id": "existing_connector_id"
        }
        
        with patch.object(EmbeddingConnector, '_get_connector_id', return_value="test_connector_id"):
            connector = EmbeddingConnector(
                provider="sagemaker",
                os_type="aos",
                connector_configs=provided_config,
                **self.base_args
            )
        
        # Verify that the configuration was merged correctly
        assert connector.get_provider() == "sagemaker"
        assert connector.get_os_type() == "aos"
        assert connector.get_embedding_type() == "dense"
        assert connector.get_model_dimensions() == "512"  # Should use provided value
        
        # Verify that auto-loaded config was called and merged
        mock_get_configs.assert_called_once_with("sagemaker", "aos")

    @patch('connectors.embedding_connector.get_remote_connector_configs')
    def test_init_sparse_embedding_type(self, mock_get_configs):
        """Test EmbeddingConnector initialization with sparse embedding type."""
        # Mock auto-loaded configuration
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
        
        # Provided configuration with sparse embedding type
        provided_config = {
            "embedding_type": "sparse"
        }
        
        with patch.object(EmbeddingConnector, '_get_connector_id', return_value="test_connector_id"):
            connector = EmbeddingConnector(
                provider="sagemaker",
                os_type="aos",
                connector_configs=provided_config,
                **self.base_args
            )
        
        assert connector.get_embedding_type() == "sparse"

    @patch('connectors.embedding_connector.get_remote_connector_configs')
    def test_unsupported_embedding_type(self, mock_get_configs):
        """Test that unsupported embedding type raises ValueError."""
        mock_config = {
            "dense_url": "https://test.com",
            "model_dimensions": "384",
            "region": "us-west-2",
            "connector_version": "1.0"
        }
        mock_get_configs.return_value = mock_config
        
        # Bedrock doesn't support sparse embeddings
        provided_config = {
            "embedding_type": "sparse"
        }
        
        with pytest.raises(ValueError, match="Provider 'bedrock' doesn't support 'sparse' embeddings"):
            EmbeddingConnector(
                provider="bedrock",
                os_type="aos",
                connector_configs=provided_config,
                **self.base_args
            )

    @patch('connectors.embedding_connector.get_remote_connector_configs')
    def test_provider_capabilities(self, mock_get_configs):
        """Test provider capabilities are correctly defined."""
        mock_config = {
            "dense_url": "https://test.com",
            "model_dimensions": "384",
            "region": "us-west-2",
            "connector_version": "1.0"
        }
        mock_get_configs.return_value = mock_config
        
        with patch.object(EmbeddingConnector, '_get_connector_id', return_value="test_connector_id"):
            # Test SageMaker capabilities
            sagemaker_connector = EmbeddingConnector(
                provider="sagemaker",
                os_type="aos",
                **self.base_args
            )
            sagemaker_capabilities = sagemaker_connector.get_provider_capabilities()
            assert "dense" in sagemaker_capabilities
            assert "sparse" in sagemaker_capabilities
            
            # Test Bedrock capabilities
            bedrock_connector = EmbeddingConnector(
                provider="bedrock",
                os_type="aos",
                **self.base_args
            )
            bedrock_capabilities = bedrock_connector.get_provider_capabilities()
            assert "dense" in bedrock_capabilities
            assert "sparse" not in bedrock_capabilities

    @patch('connectors.embedding_connector.get_remote_connector_configs')
    def test_get_provider_model_info_sagemaker_dense(self, mock_get_configs):
        """Test get_provider_model_info for SageMaker dense embeddings."""
        mock_config = {
            "dense_arn": "arn:aws:sagemaker:us-west-2:123456789012:endpoint/dense-endpoint",
            "sparse_arn": "arn:aws:sagemaker:us-west-2:123456789012:endpoint/sparse-endpoint",
            "connector_role_name": "sagemaker_connector_role",
            "region": "us-west-2",
            "connector_version": "1.0",
            "dense_url": "https://runtime.sagemaker.us-west-2.amazonaws.com/endpoints/dense-endpoint/invocations",
            "model_dimensions": "384"
        }
        mock_get_configs.return_value = mock_config
        
        with patch.object(EmbeddingConnector, '_get_connector_id', return_value="test_connector_id"):
            connector = EmbeddingConnector(
                provider="sagemaker",
                os_type="aos",
                **self.base_args
            )
            
            info = connector.get_provider_model_info()
            
            assert info["provider"] == "sagemaker"
            assert info["os_type"] == "aos"
            assert info["embedding_type"] == "dense"
            assert info["model_dimensions"] == "384"
            assert info["model_arn"] == "arn:aws:sagemaker:us-west-2:123456789012:endpoint/dense-endpoint"
            assert info["connector_role"] == "sagemaker_connector_role"
            assert info["authentication"] == "iam_role"

    @patch('connectors.embedding_connector.get_remote_connector_configs')
    def test_get_provider_model_info_sagemaker_sparse(self, mock_get_configs):
        """Test get_provider_model_info for SageMaker sparse embeddings."""
        mock_config = {
            "dense_arn": "arn:aws:sagemaker:us-west-2:123456789012:endpoint/dense-endpoint",
            "sparse_arn": "arn:aws:sagemaker:us-west-2:123456789012:endpoint/sparse-endpoint",
            "connector_role_name": "sagemaker_connector_role",
            "region": "us-west-2",
            "connector_version": "1.0",
            "sparse_url": "https://runtime.sagemaker.us-west-2.amazonaws.com/endpoints/sparse-endpoint/invocations",
            "model_dimensions": "384",
            "embedding_type": "sparse"
        }
        mock_get_configs.return_value = mock_config
        
        with patch.object(EmbeddingConnector, '_get_connector_id', return_value="test_connector_id"):
            connector = EmbeddingConnector(
                provider="sagemaker",
                os_type="aos",
                **self.base_args
            )
            
            info = connector.get_provider_model_info()
            
            assert info["provider"] == "sagemaker"
            assert info["os_type"] == "aos"
            assert info["embedding_type"] == "sparse"
            assert info["model_arn"] == "arn:aws:sagemaker:us-west-2:123456789012:endpoint/sparse-endpoint"

    @patch('connectors.embedding_connector.get_remote_connector_configs')
    def test_get_provider_model_info_bedrock_dense(self, mock_get_configs):
        """Test get_provider_model_info for Bedrock dense embeddings."""
        mock_config = {
            "dense_arn": "arn:aws:bedrock:*::foundation-model/*",
            "connector_role_name": "bedrock_connector_role",
            "region": "us-west-2",
            "connector_version": "1.0",
            "dense_url": "https://bedrock-runtime.us-west-2.amazonaws.com/model/amazon.titan-embed-text-v1/invoke",
            "model_dimensions": "1536"
        }
        mock_get_configs.return_value = mock_config
        
        with patch.object(EmbeddingConnector, '_get_connector_id', return_value="test_connector_id"):
            connector = EmbeddingConnector(
                provider="bedrock",
                os_type="aos",
                **self.base_args
            )
            
            info = connector.get_provider_model_info()
            
            assert info["provider"] == "bedrock"
            assert info["os_type"] == "aos"
            assert info["embedding_type"] == "dense"
            assert info["model_dimensions"] == "1536"
            assert info["model_arn"] == "arn:aws:bedrock:*::foundation-model/*"
            assert info["connector_role"] == "bedrock_connector_role"
            assert info["authentication"] == "iam_role"

    @patch('connectors.embedding_connector.get_remote_connector_configs')
    def test_get_provider_model_info_os_deployment(self, mock_get_configs):
        """Test get_provider_model_info for OS (self-managed) deployment."""
        mock_config = {
            "access_key": "test_access_key",
            "secret_key": "test_secret_key",
            "region": "us-west-2",
            "connector_version": "1.0",
            "dense_url": "https://runtime.sagemaker.us-west-2.amazonaws.com/endpoints/dense-endpoint/invocations",
            "model_dimensions": "384"
        }
        mock_get_configs.return_value = mock_config
        
        with patch.object(EmbeddingConnector, '_get_connector_id', return_value="test_connector_id"):
            connector = EmbeddingConnector(
                provider="sagemaker",
                os_type="os",
                **self.base_args
            )
            
            info = connector.get_provider_model_info()
            
            assert info["provider"] == "sagemaker"
            assert info["os_type"] == "os"
            assert info["embedding_type"] == "dense"
            assert info["model_dimensions"] == "384"
            assert info["model_url"] == "https://runtime.sagemaker.us-west-2.amazonaws.com/endpoints/dense-endpoint/invocations"
            assert info["authentication"] == "aws_credentials"
            assert "model_arn" not in info  # OS deployment doesn't use ARNs

    @patch('connectors.embedding_connector.get_remote_connector_configs')
    def test_validation_missing_dense_url(self, mock_get_configs):
        """Test validation fails when dense_url is missing."""
        # Mock configuration missing dense_url
        mock_config = {
            "region": "us-west-2",
            "connector_version": "1.0",
            "model_dimensions": "384"
            # Missing dense_url
        }
        mock_get_configs.return_value = mock_config
        
        with pytest.raises(ValueError, match="dense_url is required"):
            EmbeddingConnector(
                provider="sagemaker",
                os_type="aos",
                **self.base_args
            )

    @patch('connectors.embedding_connector.get_remote_connector_configs')
    def test_validation_missing_sparse_url(self, mock_get_configs):
        """Test validation fails when sparse_url is missing for sparse embeddings."""
        # Mock configuration missing sparse_url
        mock_config = {
            "region": "us-west-2",
            "connector_version": "1.0",
            "model_dimensions": "384",
            "embedding_type": "sparse"
            # Missing sparse_url
        }
        mock_get_configs.return_value = mock_config
        
        with pytest.raises(ValueError, match="sparse_url is required"):
            EmbeddingConnector(
                provider="sagemaker",
                os_type="aos",
                connector_configs={"embedding_type": "sparse"},
                **self.base_args
            )

    @patch('connectors.embedding_connector.get_remote_connector_configs')
    def test_str_representation(self, mock_get_configs):
        """Test string representation of EmbeddingConnector."""
        mock_config = {
            "dense_url": "https://test.com",
            "region": "us-west-2",
            "connector_version": "1.0",
            "model_dimensions": "384"
        }
        mock_get_configs.return_value = mock_config
        
        with patch.object(EmbeddingConnector, '_get_connector_id', return_value="test_connector_id"):
            connector = EmbeddingConnector(
                provider="sagemaker",
                os_type="aos",
                **self.base_args
            )
            
            str_repr = str(connector)
            assert "EmbeddingConnector" in str_repr
            assert "provider=sagemaker" in str_repr  # Updated to match actual format
            assert "os_type=aos" in str_repr
            assert "type=dense" in str_repr
            assert "dimensions=384" in str_repr
