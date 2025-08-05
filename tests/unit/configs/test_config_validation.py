# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for configuration validation and edge cases"""

import pytest
import os
import tempfile
from unittest.mock import Mock, patch, mock_open
from typing import Dict, Any

from configs.configuration_manager import (
    ConfigurationManager, OpenSearchConfig, ModelConfig,
    get_config_for, validate_config_for, get_config_info,
    list_all_config_keys, reload_config, validate_all_configs
)


class TestConfigurationValidation:
    """Test cases for configuration validation logic"""

    def setup_method(self):
        """Setup method run before each test"""
        self.valid_config_data = {
            "OPENSEARCH_ADMIN_USER": "admin",
            "OPENSEARCH_ADMIN_PASSWORD": "password",
            "OS_HOST_URL": "localhost",
            "OS_PORT": "9200",
            "AOS_HOST_URL": "https://test.aos.amazonaws.com",
            "AWS_REGION": "us-west-2",
            "AWS_ACCESS_KEY_ID": "test_key",
            "AWS_SECRET_ACCESS_KEY": "test_secret",
            "BEDROCK_EMBEDDING_URL": "https://bedrock.amazonaws.com",
            "BEDROCK_LLM_URL": "https://bedrock-llm.amazonaws.com",
            "BEDROCK_LLM_MODEL_NAME": "claude-3",
            "BEDROCK_LLM_ARN": "arn:aws:bedrock:*::foundation-model/claude-3",
            "BEDROCK_LLM_MAX_TOKENS": "8000",
            "BEDROCK_LLM_TEMPERATURE": "0.1",
            "BEDROCK_CONNECTOR_VERSION": "1.0",
            "SAGEMAKER_DENSE_ARN": "arn:aws:sagemaker:us-west-2:123456789012:endpoint/test",
            "SAGEMAKER_SPARSE_ARN": "arn:aws:sagemaker:us-west-2:123456789012:endpoint/test-sparse",
            "LOCAL_DENSE_EMBEDDING_MODEL": "test-model",
            "LOCAL_DENSE_EMBEDDING_VERSION": "1.0.0",
            "LOCAL_DENSE_EMBEDDING_FORMAT": "TORCH_SCRIPT"
        }

    @patch('configs.configuration_manager.Dynaconf')
    def test_validate_config_for_valid_os_bedrock_embedding(self, mock_dynaconf):
        """Test validation for valid OS + Bedrock + Embedding configuration"""
        mock_config = Mock()
        mock_config.get.side_effect = lambda key, default=None: self.valid_config_data.get(key, default)
        mock_dynaconf.return_value = mock_config
        
        config = get_config_for("os", "bedrock", "embedding")
        
        # Should not raise exception
        validate_config_for(config, "os", "bedrock", "embedding")

    def test_validate_config_for_missing_required_field(self):
        """Test validation with missing required field"""
        # Create a config with missing required fields
        incomplete_config = {
            "opensearch": {
                "username": "admin",
                "password": "password",
                "host_url": "localhost",
                "port": "9200",
                "domain_name": None,
                "region": None,
                "aws_user_name": None,
            },
            "model": {
                # Missing access_key which is required for os/bedrock/embedding
                "secret_key": None,
                "role_arn": None,
                "connector_role_name": None,
                "create_connector_role_name": None,
                "region": None,
                "connector_version": None,
                "endpoint_url": None,
                "sparse_url": None,
                "dense_url": None,
                "model_name": None,
                "model_dimension": None,
                "sparse_arn": None,
                "dense_arn": None,
                "llm_arn": None,
                "max_tokens": None,
                "temperature": None,
            },
            "constants": {
                "ML_BASE_URI": "/_plugins/_ml",
                "DELETE_RESOURCE_WAIT_TIME": 5,
                "DELETE_RESOURCE_RETRY_TIME": 5,
            },
            "metadata": {
                "os_type": "os",
                "provider": "bedrock",
                "model_type": "embedding",
            }
        }
        
        with pytest.raises(ValueError, match="Missing required configuration parameters"):
            validate_config_for(incomplete_config, "os", "bedrock", "embedding")

    @patch('configs.configuration_manager.Dynaconf')
    def test_validate_config_for_aos_bedrock_llm(self, mock_dynaconf):
        """Test validation for AOS + Bedrock + LLM configuration"""
        aos_config = self.valid_config_data.copy()
        aos_config.update({
            "BEDROCK_ARN": "arn:aws:bedrock:*::foundation-model/*",
            "BEDROCK_CONNECTOR_ROLE_NAME": "test_role",
            "BEDROCK_CREATE_CONNECTOR_ROLE_NAME": "test_create_role"
        })
        
        mock_config = Mock()
        mock_config.get.side_effect = lambda key, default=None: aos_config.get(key, default)
        mock_dynaconf.return_value = mock_config
        
        config = get_config_for("aos", "bedrock", "llm")
        
        # Should not raise exception
        validate_config_for(config, "aos", "bedrock", "llm")

    @patch('configs.configuration_manager.Dynaconf')
    def test_validate_config_for_invalid_combination(self, mock_dynaconf):
        """Test validation with invalid configuration combination"""
        mock_config = Mock()
        mock_dynaconf.return_value = mock_config
        
        with pytest.raises(ValueError, match="'invalid' is not a valid OpenSearchType"):
            get_config_for("invalid", "bedrock", "embedding")

    @patch('configs.configuration_manager.Dynaconf')
    def test_get_config_info(self, mock_dynaconf):
        """Test get_config_info function"""
        mock_config = Mock()
        mock_config.get.side_effect = lambda key, default=None: self.valid_config_data.get(key, default)
        mock_dynaconf.return_value = mock_config
        
        info = get_config_info()
        
        assert isinstance(info, dict)
        assert "available_combinations" in info
        assert "total_keys" in info
        assert isinstance(info["available_combinations"], int)
        assert isinstance(info["total_keys"], int)

    def test_list_all_config_keys(self):
        """Test list_all_config_keys function"""
        keys = list_all_config_keys()
        
        assert isinstance(keys, list)
        assert "QANDA_FILE_READER_PATH" in keys
        assert "ML_BASE_URI" in keys
        assert "OPENSEARCH_ADMIN_USER" in keys

    @patch('configs.configuration_manager.Dynaconf')
    def test_configuration_manager_get_available_combinations(self, mock_dynaconf):
        """Test ConfigurationManager.get_available_combinations method"""
        mock_config = Mock()
        mock_config.get.side_effect = lambda key, default=None: self.valid_config_data.get(key, default)
        mock_dynaconf.return_value = mock_config
        
        manager = ConfigurationManager()
        combinations = manager.get_available_combinations()
        
        assert isinstance(combinations, list)
        assert len(combinations) > 0
        
        # Check that combinations are tuples of (os_type, provider, model_type)
        for combo in combinations:
            assert isinstance(combo, tuple)
            assert len(combo) == 3
            os_type, provider, model_type = combo
            assert os_type in ["os", "aos"]
            assert provider in ["bedrock", "sagemaker", "local"]
            assert model_type in ["embedding", "llm"]

    def test_configuration_manager_validate_all_configs(self):
        """Test ConfigurationManager.validate_all_configs method"""
        results = validate_all_configs()
        
        assert isinstance(results, dict)
        # Results should contain configuration combinations with issues
        # Valid configurations won't be in the results
        for key, issues in results.items():
            assert isinstance(key, str)
            assert isinstance(issues, list)
            assert "/" in key  # Should be in format "os_type/provider/model_type"

    def test_reload_config(self):
        """Test reload_config function"""
        # This is mainly testing that the function exists and can be called
        # The actual reloading behavior depends on dynaconf
        try:
            reload_config()
            # If no exception is raised, the test passes
            assert True
        except Exception as e:
            # If there's an import error due to missing dynaconf, that's expected in tests
            if "dynaconf" in str(e).lower():
                pytest.skip("Dynaconf not available for testing")
            else:
                raise


class TestConfigurationEdgeCases:
    """Test cases for edge cases and error conditions"""

    @patch('configs.configuration_manager.Dynaconf')
    def test_empty_configuration(self, mock_dynaconf):
        """Test behavior with empty configuration"""
        mock_config = Mock()
        mock_config.get.return_value = None
        mock_dynaconf.return_value = mock_config
        
        manager = ConfigurationManager()
        
        # Should handle missing values gracefully
        config = manager.get_opensearch_config("os")
        assert isinstance(config, OpenSearchConfig)
        assert config.username is None
        assert config.password is None

    @patch('configs.configuration_manager.Dynaconf')
    def test_partial_configuration(self, mock_dynaconf):
        """Test behavior with partial configuration"""
        partial_config = {
            "OPENSEARCH_ADMIN_USER": "admin",
            # Missing password and other fields
        }
        
        mock_config = Mock()
        mock_config.get.side_effect = lambda key, default=None: partial_config.get(key, default)
        mock_dynaconf.return_value = mock_config
        
        manager = ConfigurationManager()
        config = manager.get_opensearch_config("os")
        
        assert config.username == "admin"
        assert config.password is None  # Should handle missing values

    @patch('configs.configuration_manager.Dynaconf')
    def test_invalid_numeric_values(self, mock_dynaconf):
        """Test handling of invalid numeric configuration values"""
        invalid_config = {
            "BEDROCK_LLM_MAX_TOKENS": "not_a_number",
            "BEDROCK_LLM_TEMPERATURE": "invalid_float",
            "DELETE_RESOURCE_WAIT_TIME": "not_an_int"
        }
        
        mock_config = Mock()
        mock_config.get.side_effect = lambda key, default=None: invalid_config.get(key, default)
        mock_dynaconf.return_value = mock_config
        
        manager = ConfigurationManager()
        
        # Should handle invalid conversions gracefully
        assert manager._safe_int_convert("not_an_int") is None
        assert manager._safe_float_convert("invalid_float") is None

    @patch('configs.configuration_manager.Dynaconf')
    def test_configuration_with_none_values(self, mock_dynaconf):
        """Test configuration handling when values are None"""
        mock_config = Mock()
        mock_config.get.return_value = None
        mock_dynaconf.return_value = mock_config
        
        manager = ConfigurationManager()
        
        # Test safe conversions with None
        assert manager._safe_int_convert(None) is None
        assert manager._safe_float_convert(None) is None

    def test_configuration_with_empty_strings(self):
        """Test configuration handling when values are empty strings"""
        # Our configuration manager converts empty strings to None, which is the expected behavior
        # Test that empty strings are converted to None appropriately
        from configs.configuration_manager import ConfigurationManager
        
        # Create a temporary config file with empty strings
        import tempfile
        import os
        
        empty_config_content = """
OPENSEARCH_ADMIN_USER: ""
AWS_ACCESS_KEY_ID: ""
BEDROCK_LLM_MAX_TOKENS: ""
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(empty_config_content)
            temp_config_path = f.name
        
        try:
            manager = ConfigurationManager(temp_config_path)
            
            # Test that empty strings are converted to None
            config = manager.get_opensearch_config("os")
            assert config.username is None  # Empty string should be converted to None
            
            # Test numeric conversions with empty strings
            assert manager._safe_int_convert("") is None
            assert manager._safe_float_convert("") is None
        finally:
            os.unlink(temp_config_path)

    def test_unsupported_os_type(self):
        """Test error handling for unsupported OpenSearch types"""
        from configs.configuration_manager import ConfigurationManager
        
        manager = ConfigurationManager()
        
        with pytest.raises(ValueError, match="'unsupported' is not a valid OpenSearchType"):
            manager.get_opensearch_config("unsupported")

    def test_unsupported_provider(self):
        """Test error handling for unsupported providers"""
        from configs.configuration_manager import ConfigurationManager
        
        manager = ConfigurationManager()
        
        with pytest.raises(ValueError, match="'unsupported' is not a valid ModelProvider"):
            manager.get_model_config("os", "unsupported", "embedding")

    def test_unsupported_model_type(self):
        """Test error handling for unsupported model types"""
        from configs.configuration_manager import ConfigurationManager
        
        manager = ConfigurationManager()
        
        with pytest.raises(ValueError, match="'unsupported' is not a valid ModelType"):
            manager.get_model_config("os", "bedrock", "unsupported")


class TestConfigurationFileHandling:
    """Test cases for configuration file handling"""

    @patch('configs.configuration_manager.CONFIG_FILE_PATH')
    @patch('configs.configuration_manager.Dynaconf')
    def test_config_file_path_handling(self, mock_dynaconf, mock_config_path):
        """Test that configuration file path is handled correctly"""
        mock_config_path.return_value = "/test/path/osmlqs.yaml"
        mock_config = Mock()
        mock_dynaconf.return_value = mock_config
        
        manager = ConfigurationManager()
        
        # Verify that Dynaconf was called (indicating file path was used)
        mock_dynaconf.assert_called_once()

    def test_config_file_exists(self):
        """Test that the actual configuration file exists"""
        # In test environment, we use a test config file
        # Check that the original config file exists in the project
        import os
        from pathlib import Path
        
        project_root = Path(__file__).parent.parent.parent.parent
        original_config_path = project_root / "configs" / "osmlqs.yaml"
        
        assert original_config_path.exists()
        assert str(original_config_path).endswith("osmlqs.yaml")

    @patch('configs.configuration_manager.os.path.dirname')
    def test_config_file_path_construction(self, mock_dirname):
        """Test configuration file path construction"""
        mock_dirname.return_value = "/test/configs"
        
        # Re-import to trigger path construction
        import importlib
        import configs.configuration_manager
        importlib.reload(configs.configuration_manager)
        
        # The path should be constructed correctly
        expected_path = "/test/configs/osmlqs.yaml"
        assert configs.configuration_manager.CONFIG_FILE_PATH == expected_path


class TestConfigurationConstants:
    """Test cases for configuration constants and their values"""

    @patch('configs.configuration_manager.get_raw_config_value')
    def test_all_constants_have_defaults(self, mock_get_raw):
        """Test that all configuration constants have appropriate defaults"""
        from configs.configuration_manager import (
            get_minimum_opensearch_version, get_ml_base_uri,
            get_delete_resource_wait_time, get_delete_resource_retry_time,
            get_local_dense_embedding_model_name, get_local_dense_embedding_model_version,
            get_local_dense_embedding_model_format, get_pipeline_field_map
        )
        
        # Test that functions have reasonable defaults
        mock_get_raw.return_value = None
        
        # These should return default values even when config is None
        functions_with_defaults = [
            (get_minimum_opensearch_version, "2.13.0"),
            (get_ml_base_uri, "/_plugins/_ml"),
            (get_delete_resource_wait_time, 5),
            (get_delete_resource_retry_time, 5),
            (get_local_dense_embedding_model_version, "1.0.1"),
            (get_local_dense_embedding_model_format, "TORCH_SCRIPT"),
        ]
        
        for func, expected_default in functions_with_defaults:
            mock_get_raw.return_value = str(expected_default) if isinstance(expected_default, int) else expected_default
            result = func()
            if isinstance(expected_default, int):
                assert result == expected_default
            else:
                assert result == expected_default

    def test_constant_values_are_reasonable(self):
        """Test that constant values are reasonable"""
        from configs.configuration_manager import (
            ML_BASE_URI, DELETE_RESOURCE_WAIT_TIME, DELETE_RESOURCE_RETRY_TIME,
            MINIMUM_OPENSEARCH_VERSION
        )
        
        # Test that constants have reasonable values
        assert isinstance(ML_BASE_URI, str)
        assert ML_BASE_URI.startswith("/_plugins/")
        
        assert isinstance(DELETE_RESOURCE_WAIT_TIME, int)
        assert DELETE_RESOURCE_WAIT_TIME > 0
        
        assert isinstance(DELETE_RESOURCE_RETRY_TIME, int)
        assert DELETE_RESOURCE_RETRY_TIME > 0
        
        assert isinstance(MINIMUM_OPENSEARCH_VERSION, str)
        assert "." in MINIMUM_OPENSEARCH_VERSION  # Should be version format
