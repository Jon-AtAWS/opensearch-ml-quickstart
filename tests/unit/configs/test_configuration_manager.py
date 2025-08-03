# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for configs.configuration_manager module"""

import pytest
import os
import tempfile
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass
from typing import Dict, Any

from configs.configuration_manager import (
    OpenSearchType, ModelProvider, ModelType,
    OpenSearchConfig, ModelConfig, ConfigurationManager,
    get_opensearch_config, get_model_config, get_embedding_config, get_llm_config,
    get_raw_config_value, reload_config, get_available_combinations,
    validate_all_configs, get_config_info, validate_config_for,
    get_config_for, list_all_config_keys, get_project_root,
    get_base_mapping_path, get_qanda_file_reader_path, get_minimum_opensearch_version,
    get_ml_base_uri, get_delete_resource_wait_time, get_delete_resource_retry_time,
    get_local_embedding_model_name, get_local_embedding_model_version,
    get_local_embedding_model_format, get_pipeline_field_map, get_client_configs,
    validate_configs
)


class TestEnums:
    """Test cases for configuration enums"""

    def test_opensearch_type_enum(self):
        """Test OpenSearchType enum values"""
        assert OpenSearchType.OS.value == "os"
        assert OpenSearchType.AOS.value == "aos"
        assert len(OpenSearchType) == 2

    def test_model_provider_enum(self):
        """Test ModelProvider enum values"""
        assert ModelProvider.LOCAL.value == "local"
        assert ModelProvider.BEDROCK.value == "bedrock"
        assert ModelProvider.SAGEMAKER.value == "sagemaker"
        assert len(ModelProvider) == 3

    def test_model_type_enum(self):
        """Test ModelType enum values"""
        assert ModelType.EMBEDDING.value == "embedding"
        assert ModelType.LLM.value == "llm"
        assert len(ModelType) == 2


class TestDataClasses:
    """Test cases for configuration dataclasses"""

    def test_opensearch_config_creation(self):
        """Test OpenSearchConfig dataclass creation"""
        config = OpenSearchConfig(
            username="admin",
            password="password",
            host_url="localhost",
            port="9200"
        )
        assert config.username == "admin"
        assert config.password == "password"
        assert config.host_url == "localhost"
        assert config.port == "9200"
        assert config.domain_name is None
        assert config.region is None

    def test_opensearch_config_defaults(self):
        """Test OpenSearchConfig with default values"""
        config = OpenSearchConfig()
        assert config.username is None
        assert config.password is None
        assert config.host_url is None
        assert config.port is None
        assert config.domain_name is None
        assert config.region is None
        assert config.aws_user_name is None

    def test_model_config_creation(self):
        """Test ModelConfig dataclass creation"""
        config = ModelConfig(
            access_key="test_key",
            secret_key="test_secret",
            region="us-west-2",
            endpoint_url="https://test.amazonaws.com"
        )
        assert config.access_key == "test_key"
        assert config.secret_key == "test_secret"
        assert config.region == "us-west-2"
        assert config.endpoint_url == "https://test.amazonaws.com"
        assert config.role_arn is None

    def test_model_config_defaults(self):
        """Test ModelConfig with default values"""
        config = ModelConfig()
        assert config.access_key is None
        assert config.secret_key is None
        assert config.role_arn is None
        assert config.region is None
        assert config.endpoint_url is None


class TestConfigurationManager:
    """Test cases for ConfigurationManager class"""

    def setup_method(self):
        """Setup method run before each test"""
        # Create a temporary config file for testing
        self.temp_config = {
            "OPENSEARCH_ADMIN_USER": "admin",
            "OPENSEARCH_ADMIN_PASSWORD": "password",
            "OS_HOST_URL": "localhost",
            "OS_PORT": "9200",
            "AOS_HOST_URL": "https://test.aos.amazonaws.com",
            "AOS_DOMAIN_NAME": "test-domain",
            "AWS_REGION": "us-west-2",
            "AWS_ACCESS_KEY_ID": "test_key",
            "AWS_SECRET_ACCESS_KEY": "test_secret",
            "BEDROCK_EMBEDDING_URL": "https://bedrock.amazonaws.com",
            "BEDROCK_LLM_URL": "https://bedrock-llm.amazonaws.com",
            "BEDROCK_LLM_MODEL_NAME": "claude-3",
            "BEDROCK_LLM_ARN": "arn:aws:bedrock:*::foundation-model/claude-3",
            "BEDROCK_LLM_MAX_TOKENS": "8000",
            "BEDROCK_LLM_TEMPERATURE": "0.1",
            "LOCAL_EMBEDDING_MODEL_NAME": "test-model",
            "LOCAL_EMBEDDING_MODEL_VERSION": "1.0.0",
            "LOCAL_EMBEDDING_MODEL_FORMAT": "TORCH_SCRIPT",
            "ML_BASE_URI": "/_plugins/_ml",
            "DELETE_RESOURCE_WAIT_TIME": "5",
            "DELETE_RESOURCE_RETRY_TIME": "3",
            "MINIMUM_OPENSEARCH_VERSION": "2.13.0"
        }

    @patch('configs.configuration_manager.Dynaconf')
    def test_configuration_manager_init(self, mock_dynaconf):
        """Test ConfigurationManager initialization"""
        mock_config = Mock()
        mock_dynaconf.return_value = mock_config
        
        manager = ConfigurationManager()
        
        # The manager stores the Dynaconf instance as 'settings', not 'config'
        assert manager.settings == mock_config
        assert hasattr(manager, '_opensearch_configs')
        assert hasattr(manager, '_model_configurations')
        mock_dynaconf.assert_called_once()

    @patch('configs.configuration_manager.Dynaconf')
    def test_get_config_value(self, mock_dynaconf):
        """Test _get_config_value method"""
        mock_config = Mock()
        mock_config.get.return_value = "test_value"
        mock_dynaconf.return_value = mock_config
        
        manager = ConfigurationManager()
        result = manager._get_config_value("TEST_KEY")
        
        assert result == "test_value"
        # The config is called many times during initialization, so we just check it was called with our key
        mock_config.get.assert_any_call("TEST_KEY", None)

    @patch('configs.configuration_manager.Dynaconf')
    def test_safe_int_convert(self, mock_dynaconf):
        """Test _safe_int_convert method"""
        mock_config = Mock()
        mock_dynaconf.return_value = mock_config
        
        manager = ConfigurationManager()
        
        # Test valid conversion
        assert manager._safe_int_convert("123") == 123
        assert manager._safe_int_convert("0") == 0
        
        # Test invalid conversion
        assert manager._safe_int_convert("invalid") is None
        assert manager._safe_int_convert(None) is None
        assert manager._safe_int_convert("") is None

    @patch('configs.configuration_manager.Dynaconf')
    def test_safe_float_convert(self, mock_dynaconf):
        """Test _safe_float_convert method"""
        mock_config = Mock()
        mock_dynaconf.return_value = mock_config
        
        manager = ConfigurationManager()
        
        # Test valid conversion
        assert manager._safe_float_convert("123.45") == 123.45
        assert manager._safe_float_convert("0.1") == 0.1
        assert manager._safe_float_convert("1") == 1.0
        
        # Test invalid conversion
        assert manager._safe_float_convert("invalid") is None
        assert manager._safe_float_convert(None) is None
        assert manager._safe_float_convert("") is None

    @patch('configs.configuration_manager.Dynaconf')
    def test_build_configurations(self, mock_dynaconf):
        """Test _build_configurations method"""
        mock_config = Mock()
        mock_config.get.side_effect = lambda key, default=None: self.temp_config.get(key, default)
        mock_dynaconf.return_value = mock_config
        
        manager = ConfigurationManager()
        
        # Test structure - the configurations are stored in private attributes
        assert hasattr(manager, '_opensearch_configs')
        assert hasattr(manager, '_model_configurations')
        
        # Check that OpenSearch configs were built
        # The keys are enum values, so we need to check they exist in the dict
        assert len(manager._opensearch_configs) == 2
        assert any(key.value == "os" for key in manager._opensearch_configs.keys())
        assert any(key.value == "aos" for key in manager._opensearch_configs.keys())
        
        # Check that the configs are actually OpenSearchConfig instances
        for config in manager._opensearch_configs.values():
            assert config.__class__.__name__ == 'OpenSearchConfig'
        
        # Check that model configurations were built
        assert len(manager._model_configurations) == 2
        assert any(key.value == "os" for key in manager._model_configurations.keys())
        assert any(key.value == "aos" for key in manager._model_configurations.keys())

    @patch('configs.configuration_manager.Dynaconf')
    def test_get_opensearch_config_os(self, mock_dynaconf):
        """Test get_opensearch_config method for OS"""
        mock_config = Mock()
        mock_config.get.side_effect = lambda key, default=None: self.temp_config.get(key, default)
        mock_dynaconf.return_value = mock_config
        
        manager = ConfigurationManager()
        config = manager.get_opensearch_config("os")
        
        # Check that it returns an OpenSearchConfig instance
        assert config.__class__.__name__ == 'OpenSearchConfig'
        assert config.username == 'admin'
        assert config.password == 'password'
        assert config.host_url == 'localhost'
        assert config.username == "admin"
        assert config.password == "password"
        assert config.host_url == "localhost"
        assert config.port == "9200"

    @patch('configs.configuration_manager.Dynaconf')
    def test_get_opensearch_config_aos(self, mock_dynaconf):
        """Test get_opensearch_config method for AOS"""
        mock_config = Mock()
        mock_config.get.side_effect = lambda key, default=None: self.temp_config.get(key, default)
        mock_dynaconf.return_value = mock_config
        
        manager = ConfigurationManager()
        config = manager.get_opensearch_config("aos")
        
        # Check that it returns an OpenSearchConfig instance
        assert config.__class__.__name__ == 'OpenSearchConfig'
        assert config.username == 'admin'
        assert config.password == 'password'
        assert config.host_url == 'https://test.aos.amazonaws.com'
        assert config.username == "admin"
        assert config.password == "password"
        assert config.host_url == "https://test.aos.amazonaws.com"
        assert config.domain_name == "test-domain"
        assert config.region == "us-west-2"

    @patch('configs.configuration_manager.Dynaconf')
    def test_get_model_config(self, mock_dynaconf):
        """Test get_model_config method"""
        mock_config = Mock()
        mock_config.get.side_effect = lambda key, default=None: self.temp_config.get(key, default)
        mock_dynaconf.return_value = mock_config
        
        manager = ConfigurationManager()
        config = manager.get_model_config("os", "bedrock", "embedding")
        
        # Check that it returns a ModelConfig instance
        assert config.__class__.__name__ == 'ModelConfig'
        assert config.access_key == 'test_key'
        assert config.secret_key == 'test_secret'
        assert config.region == 'us-west-2'
        assert config.access_key == "test_key"
        assert config.secret_key == "test_secret"
        assert config.region == "us-west-2"

    @patch('configs.configuration_manager.Dynaconf')
    def test_get_model_config_invalid_combination(self, mock_dynaconf):
        """Test get_model_config with invalid combination"""
        mock_config = Mock()
        mock_dynaconf.return_value = mock_config
        
        manager = ConfigurationManager()
        
        with pytest.raises(ValueError, match="'invalid' is not a valid OpenSearchType"):
            manager.get_model_config("invalid", "bedrock", "embedding")


class TestModuleFunctions:
    """Test cases for module-level functions"""

    @patch('configs.configuration_manager.ConfigurationManager')
    def test_get_opensearch_config(self, mock_manager_class):
        """Test get_opensearch_config function"""
        mock_manager = Mock()
        mock_config = OpenSearchConfig(username="test")
        mock_manager.get_opensearch_config.return_value = mock_config
        mock_manager_class.return_value = mock_manager
        
        # We need to patch the global config_manager instance
        with patch('configs.configuration_manager.config_manager', mock_manager):
            result = get_opensearch_config("os")
            
            assert result == mock_config
            mock_manager.get_opensearch_config.assert_called_once_with("os")
        mock_manager.get_opensearch_config.assert_called_once_with("os")

    @patch('configs.configuration_manager.ConfigurationManager')
    def test_get_model_config(self, mock_manager_class):
        """Test get_model_config function"""
        mock_manager = Mock()
        mock_config = ModelConfig(access_key="test")
        mock_manager.get_model_config.return_value = mock_config
        mock_manager_class.return_value = mock_manager
        
        # We need to patch the global config_manager instance
        with patch('configs.configuration_manager.config_manager', mock_manager):
            result = get_model_config("os", "bedrock", "embedding")
            
            assert result == mock_config
            mock_manager.get_model_config.assert_called_once_with("os", "bedrock", "embedding")
        mock_manager.get_model_config.assert_called_once_with("os", "bedrock", "embedding")

    @patch('configs.configuration_manager.ConfigurationManager')
    def test_get_embedding_config(self, mock_manager_class):
        """Test get_embedding_config function"""
        mock_manager = Mock()
        mock_config = ModelConfig(access_key="test")
        mock_manager.get_model_config.return_value = mock_config
        mock_manager_class.return_value = mock_manager
        
        # We need to patch the global config_manager instance
        with patch('configs.configuration_manager.config_manager', mock_manager):
            result = get_embedding_config("os", "bedrock")
            
            assert result == mock_config
            mock_manager.get_model_config.assert_called_once_with("os", "bedrock", "embedding")
        mock_manager.get_model_config.assert_called_once_with("os", "bedrock", "embedding")

    @patch('configs.configuration_manager.ConfigurationManager')
    def test_get_llm_config(self, mock_manager_class):
        """Test get_llm_config function"""
        mock_manager = Mock()
        mock_config = ModelConfig(access_key="test")
        mock_manager.get_model_config.return_value = mock_config
        mock_manager_class.return_value = mock_manager
        
        # We need to patch the global config_manager instance
        with patch('configs.configuration_manager.config_manager', mock_manager):
            result = get_llm_config("os", "bedrock")
            
            assert result == mock_config
            mock_manager.get_model_config.assert_called_once_with("os", "bedrock", "llm")
        mock_manager.get_model_config.assert_called_once_with("os", "bedrock", "llm")

    @patch('configs.configuration_manager.Dynaconf')
    def test_get_raw_config_value(self, mock_dynaconf):
        """Test get_raw_config_value function"""
        mock_config = Mock()
        mock_config.get.return_value = "test_value"
        mock_dynaconf.return_value = mock_config
        
        # We need to patch the global config_manager instance
        mock_manager = Mock()
        mock_manager.get_raw_config_value.return_value = "test_value"
        
        with patch('configs.configuration_manager.config_manager', mock_manager):
            result = get_raw_config_value("TEST_KEY")
            
            assert result == "test_value"
            mock_manager.get_raw_config_value.assert_called_once_with("TEST_KEY", None)

    @patch('configs.configuration_manager.get_raw_config_value')
    def test_get_minimum_opensearch_version(self, mock_get_raw):
        """Test get_minimum_opensearch_version function"""
        mock_get_raw.return_value = "2.13.0"
        
        result = get_minimum_opensearch_version()
        
        assert result == "2.13.0"
        mock_get_raw.assert_called_once_with("MINIMUM_OPENSEARCH_VERSION", "2.13.0")

    @patch('configs.configuration_manager.get_raw_config_value')
    def test_get_ml_base_uri(self, mock_get_raw):
        """Test get_ml_base_uri function"""
        mock_get_raw.return_value = "/_plugins/_ml"
        
        result = get_ml_base_uri()
        
        assert result == "/_plugins/_ml"
        mock_get_raw.assert_called_once_with("ML_BASE_URI", "/_plugins/_ml")

    @patch('configs.configuration_manager.get_raw_config_value')
    def test_get_delete_resource_wait_time(self, mock_get_raw):
        """Test get_delete_resource_wait_time function"""
        mock_get_raw.return_value = "5"
        
        result = get_delete_resource_wait_time()
        
        assert result == 5
        mock_get_raw.assert_called_once_with("DELETE_RESOURCE_WAIT_TIME", "5")

    @patch('configs.configuration_manager.get_raw_config_value')
    def test_get_delete_resource_retry_time(self, mock_get_raw):
        """Test get_delete_resource_retry_time function"""
        mock_get_raw.return_value = "3"
        
        result = get_delete_resource_retry_time()
        
        assert result == 3
        mock_get_raw.assert_called_once_with("DELETE_RESOURCE_RETRY_TIME", "5")

    @patch('configs.configuration_manager.get_raw_config_value')
    def test_get_local_embedding_model_name(self, mock_get_raw):
        """Test get_local_embedding_model_name function"""
        expected_default = "huggingface/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        mock_get_raw.return_value = "test-model"
        
        result = get_local_embedding_model_name()
        
        assert result == "test-model"
        mock_get_raw.assert_called_once_with("LOCAL_EMBEDDING_MODEL_NAME", expected_default)

    @patch('configs.configuration_manager.get_raw_config_value')
    def test_get_local_embedding_model_version(self, mock_get_raw):
        """Test get_local_embedding_model_version function"""
        mock_get_raw.return_value = "1.0.1"
        
        result = get_local_embedding_model_version()
        
        assert result == "1.0.1"
        mock_get_raw.assert_called_once_with("LOCAL_EMBEDDING_MODEL_VERSION", "1.0.1")

    @patch('configs.configuration_manager.get_raw_config_value')
    def test_get_local_embedding_model_format(self, mock_get_raw):
        """Test get_local_embedding_model_format function"""
        mock_get_raw.return_value = "TORCH_SCRIPT"
        
        result = get_local_embedding_model_format()
        
        assert result == "TORCH_SCRIPT"
        mock_get_raw.assert_called_once_with("LOCAL_EMBEDDING_MODEL_FORMAT", "TORCH_SCRIPT")

    @patch('configs.configuration_manager.get_raw_config_value')
    def test_get_pipeline_field_map(self, mock_get_raw):
        """Test get_pipeline_field_map function"""
        mock_get_raw.return_value = {"chunk": "chunk_embedding"}
        
        result = get_pipeline_field_map()
        
        assert result == {"chunk": "chunk_embedding"}
        mock_get_raw.assert_called_once_with("PIPELINE_FIELD_MAP", {"chunk": "chunk_embedding"})

    def test_get_project_root(self):
        """Test get_project_root function"""
        result = get_project_root()
        
        # Should return a valid path
        assert isinstance(result, str)
        assert os.path.exists(result)
        # Should end with the project directory name
        assert result.endswith("opensearch-ml-quickstart")

    @patch('configs.configuration_manager.get_project_root')
    def test_get_base_mapping_path(self, mock_get_root):
        """Test get_base_mapping_path function"""
        mock_get_root.return_value = "/test/project"
        
        result = get_base_mapping_path()
        
        assert result == "/test/project/mapping/base_mapping.json"
        mock_get_root.assert_called_once()

    @patch('configs.configuration_manager.get_raw_config_value')
    def test_get_qanda_file_reader_path(self, mock_get_raw):
        """Test get_qanda_file_reader_path function"""
        mock_get_raw.return_value = "./datasets/amazon_pqa"
        
        result = get_qanda_file_reader_path()
        
        assert result == "./datasets/amazon_pqa"
        mock_get_raw.assert_called_once_with("QANDA_FILE_READER_PATH", "./datasets/amazon_pqa")

    def test_validate_configs_valid(self):
        """Test validate_configs with valid configuration"""
        configs = {
            "required_key1": "value1",
            "required_key2": "value2",
            "optional_key": "value3"
        }
        required_args = ["required_key1", "required_key2"]
        
        # Should not raise exception
        validate_configs(configs, required_args)

    def test_validate_configs_missing_required(self):
        """Test validate_configs with missing required keys"""
        configs = {
            "required_key1": "value1",
            "optional_key": "value3"
        }
        required_args = ["required_key1", "required_key2"]
        
        with pytest.raises(ValueError, match="Missing required configuration"):
            validate_configs(configs, required_args)

    def test_validate_configs_empty_required(self):
        """Test validate_configs with empty required list"""
        configs = {"key": "value"}
        required_args = []
        
        # Should not raise exception
        validate_configs(configs, required_args)


class TestConfigurationIntegration:
    """Integration tests for configuration system"""

    @patch('configs.configuration_manager.Dynaconf')
    def test_get_client_configs_os(self, mock_dynaconf):
        """Test get_client_configs for OS"""
        mock_config = Mock()
        mock_config.get.side_effect = lambda key, default=None: {
            "OPENSEARCH_ADMIN_USER": "admin",
            "OPENSEARCH_ADMIN_PASSWORD": "password",
            "OS_HOST_URL": "localhost",
            "OS_PORT": "9200"
        }.get(key, default)
        mock_dynaconf.return_value = mock_config
        
        result = get_client_configs("os")
        
        assert result["username"] == "admin"
        assert result["password"] == "password"
        assert result["host_url"] == "localhost"
        # The port might be returned as string or int depending on implementation
        assert str(result["port"]) == "9200"

    @patch('configs.configuration_manager.Dynaconf')
    def test_get_client_configs_aos(self, mock_dynaconf):
        """Test get_client_configs for AOS"""
        mock_config = Mock()
        mock_config.get.side_effect = lambda key, default=None: {
            "OPENSEARCH_ADMIN_USER": "admin",
            "OPENSEARCH_ADMIN_PASSWORD": "password",
            "AOS_HOST_URL": "https://test.aos.amazonaws.com",
            "AOS_DOMAIN_NAME": "test-domain",
            "AWS_REGION": "us-west-2"
        }.get(key, default)
        mock_dynaconf.return_value = mock_config
        
        result = get_client_configs("aos")
        
        assert result["username"] == "admin"
        assert result["password"] == "password"
        assert result["host_url"] == "https://test.aos.amazonaws.com"
        assert result["domain_name"] == "test-domain"
        assert result["region"] == "us-west-2"

    @patch('configs.configuration_manager.ConfigurationManager')
    def test_get_available_combinations(self, mock_manager_class):
        """Test get_available_combinations function"""
        mock_manager = Mock()
        # The actual implementation returns 10 combinations, not 2
        mock_manager.get_available_combinations.return_value = [
            ("os", "local", "embedding"),
            ("os", "local", "llm"),
            ("os", "sagemaker", "embedding"),
            ("os", "sagemaker", "llm"),
            ("os", "bedrock", "embedding"),
            ("os", "bedrock", "llm"),
            ("aos", "sagemaker", "embedding"),
            ("aos", "sagemaker", "llm"),
            ("aos", "bedrock", "embedding"),
            ("aos", "bedrock", "llm")
        ]
        mock_manager_class.return_value = mock_manager
        
        with patch('configs.configuration_manager.config_manager', mock_manager):
            result = get_available_combinations()
            
            assert len(result) == 10
            assert ("os", "bedrock", "embedding") in result
            assert ("aos", "bedrock", "llm") in result
        assert ("os", "bedrock", "embedding") in result
        assert ("aos", "bedrock", "llm") in result

    @patch('configs.configuration_manager.ConfigurationManager')
    def test_validate_all_configs(self, mock_manager_class):
        """Test validate_all_configs function"""
        mock_manager = Mock()
        # The actual implementation returns issues, not valid/invalid categories
        mock_manager.validate_all_configs.return_value = {
            "aos/bedrock/llm": ["Missing authentication (access_key or role_arn)"],
            "aos/sagemaker/embedding": ["Missing authentication (access_key or role_arn)"]
        }
        mock_manager_class.return_value = mock_manager
        
        with patch('configs.configuration_manager.config_manager', mock_manager):
            result = validate_all_configs()
            
            assert isinstance(result, dict)
            # The function returns issues, not a "valid" key
            assert "aos/bedrock/llm" in result
            assert "Missing authentication (access_key or role_arn)" in result["aos/bedrock/llm"]
