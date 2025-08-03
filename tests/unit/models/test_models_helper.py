# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for models.helper module"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from opensearchpy import OpenSearch
from opensearch_py_ml.ml_commons import MLCommonClient

from models.helper import get_ml_model_group, get_ml_model
from models.ml_model_group import MlModelGroup
from models.local_ml_model import LocalMlModel
from models.remote_ml_model import RemoteMlModel
from connectors import EmbeddingConnector


class TestGetMlModelGroup:
    """Test cases for get_ml_model_group function"""

    def setup_method(self):
        """Setup method run before each test"""
        self.mock_os_client = Mock(spec=OpenSearch)
        self.mock_ml_commons_client = Mock(spec=MLCommonClient)

    @patch('models.helper.MlModelGroup')
    def test_get_ml_model_group(self, mock_ml_model_group_class):
        """Test get_ml_model_group function"""
        mock_group = Mock(spec=MlModelGroup)
        mock_ml_model_group_class.return_value = mock_group
        
        result = get_ml_model_group(self.mock_os_client, self.mock_ml_commons_client)
        
        # Verify MlModelGroup was created with correct parameters
        mock_ml_model_group_class.assert_called_once_with(
            os_client=self.mock_os_client,
            ml_commons_client=self.mock_ml_commons_client,
        )
        
        assert result == mock_group


class TestGetMlModel:
    """Test cases for get_ml_model function"""

    def setup_method(self):
        """Setup method run before each test"""
        self.mock_os_client = Mock(spec=OpenSearch)
        self.mock_ml_commons_client = Mock(spec=MLCommonClient)
        self.model_group_id = "test_group_id"
        
        # Base model config
        self.base_model_config = {
            "model_version": "1.0.1",
            "model_format": "TORCH_SCRIPT"
        }

    @patch('models.helper.LocalMlModel')
    def test_get_ml_model_local(self, mock_local_ml_model_class):
        """Test get_ml_model with local model type"""
        mock_model = Mock(spec=LocalMlModel)
        mock_local_ml_model_class.return_value = mock_model
        
        model_config = self.base_model_config.copy()
        model_config["model_name"] = "test/local/model"
        
        result = get_ml_model(
            host_type="os",
            model_type="local",
            model_group_id=self.model_group_id,
            model_config=model_config,
            os_client=self.mock_os_client,
            ml_commons_client=self.mock_ml_commons_client
        )
        
        # Verify LocalMlModel was created with correct parameters
        mock_local_ml_model_class.assert_called_once_with(
            os_client=self.mock_os_client,
            ml_commons_client=self.mock_ml_commons_client,
            model_group_id=self.model_group_id,
            model_name="test/local/model",
            model_configs=model_config,
        )
        
        assert result == mock_model

    @patch('models.helper.LocalMlModel')
    def test_get_ml_model_local_no_model_name(self, mock_local_ml_model_class):
        """Test get_ml_model with local model type and no model name"""
        mock_model = Mock(spec=LocalMlModel)
        mock_local_ml_model_class.return_value = mock_model
        
        model_config = self.base_model_config.copy()
        # No model_name in config
        
        result = get_ml_model(
            host_type="os",
            model_type="local",
            model_group_id=self.model_group_id,
            model_config=model_config,
            os_client=self.mock_os_client,
            ml_commons_client=self.mock_ml_commons_client
        )
        
        # Verify LocalMlModel was created with None model_name
        mock_local_ml_model_class.assert_called_once_with(
            os_client=self.mock_os_client,
            ml_commons_client=self.mock_ml_commons_client,
            model_group_id=self.model_group_id,
            model_name=None,
            model_configs=model_config,
        )

    @patch('models.helper.EmbeddingConnector')
    @patch('models.helper.RemoteMlModel')
    def test_get_ml_model_os_sagemaker(self, mock_remote_ml_model_class, mock_embedding_connector_class):
        """Test get_ml_model with OS + SageMaker combination"""
        mock_connector = Mock(spec=EmbeddingConnector)
        mock_embedding_connector_class.return_value = mock_connector
        mock_model = Mock(spec=RemoteMlModel)
        mock_remote_ml_model_class.return_value = mock_model
        
        model_config = self.base_model_config.copy()
        model_config["embedding_type"] = "dense"
        
        result = get_ml_model(
            host_type="os",
            model_type="sagemaker",
            model_group_id=self.model_group_id,
            model_config=model_config,
            os_client=self.mock_os_client,
            ml_commons_client=self.mock_ml_commons_client
        )
        
        # Verify EmbeddingConnector was created
        mock_embedding_connector_class.assert_called_once_with(
            os_client=self.mock_os_client,
            provider="sagemaker",
            os_type="os",
            connector_name="os_sagemaker_dense",
            connector_configs=model_config,
        )
        
        # Verify RemoteMlModel was created
        mock_remote_ml_model_class.assert_called_once_with(
            os_client=self.mock_os_client,
            ml_commons_client=self.mock_ml_commons_client,
            ml_connector=mock_connector,
            model_group_id=self.model_group_id,
            model_name="os_sagemaker_dense",
            model_configs=model_config,
        )
        
        assert result == mock_model

    @patch('models.helper.EmbeddingConnector')
    @patch('models.helper.RemoteMlModel')
    def test_get_ml_model_os_bedrock(self, mock_remote_ml_model_class, mock_embedding_connector_class):
        """Test get_ml_model with OS + Bedrock combination"""
        mock_connector = Mock(spec=EmbeddingConnector)
        mock_embedding_connector_class.return_value = mock_connector
        mock_model = Mock(spec=RemoteMlModel)
        mock_remote_ml_model_class.return_value = mock_model
        
        model_config = self.base_model_config.copy()
        model_config["embedding_type"] = "sparse"
        
        result = get_ml_model(
            host_type="os",
            model_type="bedrock",
            model_group_id=self.model_group_id,
            model_config=model_config,
            os_client=self.mock_os_client,
            ml_commons_client=self.mock_ml_commons_client
        )
        
        # Verify EmbeddingConnector was created with sparse embedding
        mock_embedding_connector_class.assert_called_once_with(
            os_client=self.mock_os_client,
            provider="bedrock",
            os_type="os",
            connector_name="os_bedrock_sparse",
            connector_configs=model_config,
        )
        
        # Verify RemoteMlModel was created with sparse name
        mock_remote_ml_model_class.assert_called_once_with(
            os_client=self.mock_os_client,
            ml_commons_client=self.mock_ml_commons_client,
            ml_connector=mock_connector,
            model_group_id=self.model_group_id,
            model_name="os_bedrock_sparse",
            model_configs=model_config,
        )

    @patch('models.helper.get_opensearch_config')
    @patch('models.helper.EmbeddingConnector')
    @patch('models.helper.RemoteMlModel')
    def test_get_ml_model_aos_sagemaker(self, mock_remote_ml_model_class, mock_embedding_connector_class, mock_get_opensearch_config):
        """Test get_ml_model with AOS + SageMaker combination"""
        # Mock AOS configuration
        mock_aos_config = Mock()
        mock_aos_config.host_url = "https://test.aos.amazonaws.com"
        mock_aos_config.region = "us-west-2"
        mock_aos_config.domain_name = "test-domain"
        mock_aos_config.username = "admin"
        mock_aos_config.password = "password"
        mock_aos_config.aws_user_name = "test_user"
        mock_get_opensearch_config.return_value = mock_aos_config
        
        mock_connector = Mock(spec=EmbeddingConnector)
        mock_embedding_connector_class.return_value = mock_connector
        mock_model = Mock(spec=RemoteMlModel)
        mock_remote_ml_model_class.return_value = mock_model
        
        model_config = self.base_model_config.copy()
        
        result = get_ml_model(
            host_type="aos",
            model_type="sagemaker",
            model_group_id=self.model_group_id,
            model_config=model_config,
            os_client=self.mock_os_client,
            ml_commons_client=self.mock_ml_commons_client
        )
        
        # Verify get_opensearch_config was called
        mock_get_opensearch_config.assert_called_once_with("aos")
        
        # Verify EmbeddingConnector was created with AOS parameters
        mock_embedding_connector_class.assert_called_once_with(
            os_client=self.mock_os_client,
            provider="sagemaker",
            os_type="aos",
            opensearch_domain_url="https://test.aos.amazonaws.com",
            opensearch_domain_arn="arn:aws:es:us-west-2:*:domain/test-domain",
            opensearch_username="admin",
            opensearch_password="password",
            aws_user_name="test_user",
            region="us-west-2",
            connector_name="aos_sagemaker_dense",
            connector_configs=model_config,
        )

    @patch('models.helper.get_opensearch_config')
    @patch('models.helper.EmbeddingConnector')
    @patch('models.helper.RemoteMlModel')
    def test_get_ml_model_aos_bedrock(self, mock_remote_ml_model_class, mock_embedding_connector_class, mock_get_opensearch_config):
        """Test get_ml_model with AOS + Bedrock combination"""
        # Mock AOS configuration
        mock_aos_config = Mock()
        mock_aos_config.host_url = "https://test.aos.amazonaws.com"
        mock_aos_config.region = "us-east-1"
        mock_aos_config.domain_name = "bedrock-domain"
        mock_aos_config.username = "bedrock_user"
        mock_aos_config.password = "bedrock_pass"
        mock_aos_config.aws_user_name = "bedrock_aws_user"
        mock_get_opensearch_config.return_value = mock_aos_config
        
        mock_connector = Mock(spec=EmbeddingConnector)
        mock_embedding_connector_class.return_value = mock_connector
        mock_model = Mock(spec=RemoteMlModel)
        mock_remote_ml_model_class.return_value = mock_model
        
        model_config = self.base_model_config.copy()
        model_config["embedding_type"] = "dense"
        
        result = get_ml_model(
            host_type="aos",
            model_type="bedrock",
            model_group_id=self.model_group_id,
            model_config=model_config,
            os_client=self.mock_os_client,
            ml_commons_client=self.mock_ml_commons_client
        )
        
        # Verify EmbeddingConnector was created with correct AOS Bedrock parameters
        mock_embedding_connector_class.assert_called_once_with(
            os_client=self.mock_os_client,
            provider="bedrock",
            os_type="aos",
            opensearch_domain_url="https://test.aos.amazonaws.com",
            opensearch_domain_arn="arn:aws:es:us-east-1:*:domain/bedrock-domain",
            opensearch_username="bedrock_user",
            opensearch_password="bedrock_pass",
            aws_user_name="bedrock_aws_user",
            region="us-east-1",
            connector_name="aos_bedrock_dense",
            connector_configs=model_config,
        )

    def test_get_ml_model_unsupported_combination(self):
        """Test get_ml_model with unsupported combination"""
        model_config = self.base_model_config.copy()
        
        with pytest.raises(ValueError, match="Unsupported combination: host_type='invalid', model_type='bedrock'"):
            get_ml_model(
                host_type="invalid",
                model_type="bedrock",
                model_group_id=self.model_group_id,
                model_config=model_config,
                os_client=self.mock_os_client,
                ml_commons_client=self.mock_ml_commons_client
            )

    def test_get_ml_model_embedding_type_default(self):
        """Test get_ml_model with default embedding type (dense)"""
        model_config = self.base_model_config.copy()
        # No embedding_type specified, should default to dense
        
        with patch('models.helper.EmbeddingConnector') as mock_connector_class:
            with patch('models.helper.RemoteMlModel') as mock_model_class:
                mock_connector = Mock()
                mock_connector_class.return_value = mock_connector
                mock_model = Mock()
                mock_model_class.return_value = mock_model
                
                get_ml_model(
                    host_type="os",
                    model_type="bedrock",
                    model_group_id=self.model_group_id,
                    model_config=model_config,
                    os_client=self.mock_os_client,
                    ml_commons_client=self.mock_ml_commons_client
                )
                
                # Should use "dense" as default
                mock_model_class.assert_called_once()
                call_args = mock_model_class.call_args
                assert call_args[1]["model_name"] == "os_bedrock_dense"

    @patch('models.helper.EmbeddingConnector')
    @patch('models.helper.RemoteMlModel')
    def test_get_ml_model_connector_name_generation(self, mock_remote_ml_model_class, mock_embedding_connector_class):
        """Test connector name generation for different combinations"""
        test_cases = [
            ("os", "bedrock", "dense", "os_bedrock_dense"),
            ("os", "bedrock", "sparse", "os_bedrock_sparse"),
            ("os", "sagemaker", "dense", "os_sagemaker_dense"),
            ("aos", "bedrock", "dense", "aos_bedrock_dense"),
            ("aos", "sagemaker", "sparse", "aos_sagemaker_sparse"),
        ]
        
        for host_type, model_type, embedding_type, expected_name in test_cases:
            mock_embedding_connector_class.reset_mock()
            mock_remote_ml_model_class.reset_mock()
            
            model_config = self.base_model_config.copy()
            model_config["embedding_type"] = embedding_type
            
            # Mock AOS config if needed
            if host_type == "aos":
                with patch('models.helper.get_opensearch_config') as mock_get_config:
                    mock_config = Mock()
                    mock_config.host_url = "https://test.amazonaws.com"
                    mock_config.region = "us-west-2"
                    mock_config.domain_name = "test"
                    mock_config.username = "user"
                    mock_config.password = "pass"
                    mock_config.aws_user_name = "aws_user"
                    mock_get_config.return_value = mock_config
                    
                    get_ml_model(
                        host_type=host_type,
                        model_type=model_type,
                        model_group_id=self.model_group_id,
                        model_config=model_config,
                        os_client=self.mock_os_client,
                        ml_commons_client=self.mock_ml_commons_client
                    )
            else:
                get_ml_model(
                    host_type=host_type,
                    model_type=model_type,
                    model_group_id=self.model_group_id,
                    model_config=model_config,
                    os_client=self.mock_os_client,
                    ml_commons_client=self.mock_ml_commons_client
                )
            
            # Verify connector name
            call_args = mock_embedding_connector_class.call_args
            assert call_args[1]["connector_name"] == expected_name
            
            # Verify model name
            call_args = mock_remote_ml_model_class.call_args
            assert call_args[1]["model_name"] == expected_name

    def test_get_ml_model_model_config_passthrough(self):
        """Test that model_config is passed through correctly"""
        custom_config = {
            "custom_param": "custom_value",
            "embedding_type": "dense",
            "model_dimension": 768
        }
        
        with patch('models.helper.LocalMlModel') as mock_local_model_class:
            mock_model = Mock()
            mock_local_model_class.return_value = mock_model
            
            get_ml_model(
                host_type="os",
                model_type="local",
                model_group_id=self.model_group_id,
                model_config=custom_config,
                os_client=self.mock_os_client,
                ml_commons_client=self.mock_ml_commons_client
            )
            
            # Verify custom config was passed through
            call_args = mock_local_model_class.call_args
            assert call_args[1]["model_configs"] == custom_config

    @patch('models.helper.get_opensearch_config')
    def test_get_ml_model_aos_domain_arn_construction(self, mock_get_opensearch_config):
        """Test AOS domain ARN construction"""
        mock_aos_config = Mock()
        mock_aos_config.host_url = "https://test.aos.amazonaws.com"
        mock_aos_config.region = "eu-west-1"
        mock_aos_config.domain_name = "my-test-domain"
        mock_aos_config.username = "admin"
        mock_aos_config.password = "password"
        mock_aos_config.aws_user_name = "test_user"
        mock_get_opensearch_config.return_value = mock_aos_config
        
        with patch('models.helper.EmbeddingConnector') as mock_connector_class:
            with patch('models.helper.RemoteMlModel'):
                get_ml_model(
                    host_type="aos",
                    model_type="bedrock",
                    model_group_id=self.model_group_id,
                    model_config=self.base_model_config,
                    os_client=self.mock_os_client,
                    ml_commons_client=self.mock_ml_commons_client
                )
                
                # Verify ARN construction
                call_args = mock_connector_class.call_args
                expected_arn = "arn:aws:es:eu-west-1:*:domain/my-test-domain"
                assert call_args[1]["opensearch_domain_arn"] == expected_arn

    def test_get_ml_model_parameter_validation(self):
        """Test parameter validation and types"""
        # Test with minimal valid parameters
        with patch('models.helper.LocalMlModel') as mock_local_model_class:
            mock_model = Mock()
            mock_local_model_class.return_value = mock_model
            
            result = get_ml_model(
                host_type="os",
                model_type="local",
                model_group_id=self.model_group_id,
                model_config={},  # Empty config should work
                os_client=self.mock_os_client,
                ml_commons_client=self.mock_ml_commons_client
            )
            
            assert result == mock_model
            
            # Verify all required parameters were passed
            call_args = mock_local_model_class.call_args
            assert call_args[1]["os_client"] == self.mock_os_client
            assert call_args[1]["ml_commons_client"] == self.mock_ml_commons_client
            assert call_args[1]["model_group_id"] == self.model_group_id
            assert call_args[1]["model_configs"] == {}

    def test_get_ml_model_all_supported_combinations(self):
        """Test all supported host_type and model_type combinations"""
        supported_combinations = [
            ("os", "local"),
            ("os", "bedrock"),
            ("os", "sagemaker"),
            ("aos", "bedrock"),
            ("aos", "sagemaker"),
        ]
        
        for host_type, model_type in supported_combinations:
            with patch('models.helper.LocalMlModel') if model_type == "local" else patch('models.helper.RemoteMlModel'):
                if model_type != "local":
                    with patch('models.helper.EmbeddingConnector'):
                        if host_type == "aos":
                            with patch('models.helper.get_opensearch_config') as mock_get_config:
                                mock_config = Mock()
                                mock_config.host_url = "https://test.amazonaws.com"
                                mock_config.region = "us-west-2"
                                mock_config.domain_name = "test"
                                mock_config.username = "user"
                                mock_config.password = "pass"
                                mock_config.aws_user_name = "aws_user"
                                mock_get_config.return_value = mock_config
                                
                                # Should not raise exception
                                get_ml_model(
                                    host_type=host_type,
                                    model_type=model_type,
                                    model_group_id=self.model_group_id,
                                    model_config=self.base_model_config,
                                    os_client=self.mock_os_client,
                                    ml_commons_client=self.mock_ml_commons_client
                                )
                        else:
                            # Should not raise exception
                            get_ml_model(
                                host_type=host_type,
                                model_type=model_type,
                                model_group_id=self.model_group_id,
                                model_config=self.base_model_config,
                                os_client=self.mock_os_client,
                                ml_commons_client=self.mock_ml_commons_client
                            )
                else:
                    # Local model case
                    get_ml_model(
                        host_type=host_type,
                        model_type=model_type,
                        model_group_id=self.model_group_id,
                        model_config=self.base_model_config,
                        os_client=self.mock_os_client,
                        ml_commons_client=self.mock_ml_commons_client
                    )
