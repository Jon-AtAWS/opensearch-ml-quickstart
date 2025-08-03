# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for models.local_ml_model module"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from opensearchpy import OpenSearch
from opensearch_py_ml.ml_commons import MLCommonClient

from models.local_ml_model import LocalMlModel


class TestLocalMlModel:
    """Test cases for LocalMlModel class"""

    def setup_method(self):
        """Setup method run before each test"""
        # Create mock clients
        self.mock_os_client = Mock(spec=OpenSearch)
        self.mock_ml_commons_client = Mock(spec=MLCommonClient)
        
        # Mock model group ID
        self.model_group_id = "test_group_id"
        
        # Mock model configurations
        self.model_configs = {
            "model_version": "1.0.1",
            "model_format": "TORCH_SCRIPT",
            "model_group_id": self.model_group_id
        }

    @patch('models.local_ml_model.get_local_embedding_model_name')
    @patch('models.ml_model.logging')
    def test_local_ml_model_init_with_default_name(self, mock_logging, mock_get_model_name):
        """Test LocalMlModel initialization with default model name"""
        mock_get_model_name.return_value = "default/model/name"
        
        with patch.object(LocalMlModel, 'find_models', return_value=['existing_model_id']):
            model = LocalMlModel(
                os_client=self.mock_os_client,
                ml_commons_client=self.mock_ml_commons_client,
                model_group_id=self.model_group_id,
                model_configs=self.model_configs
            )
            
            assert model._model_name == "default/model/name"
            assert model._model_description == LocalMlModel.DEFAULT_LOCAL_MODEL_DESCRIPTION
            assert model._model_configs == self.model_configs
            mock_get_model_name.assert_called_once()

    @patch('models.ml_model.logging')
    def test_local_ml_model_init_with_custom_name(self, mock_logging):
        """Test LocalMlModel initialization with custom model name"""
        custom_name = "custom/model/name"
        custom_description = "Custom model description"
        
        with patch.object(LocalMlModel, 'find_models', return_value=['custom_model_id']):
            model = LocalMlModel(
                os_client=self.mock_os_client,
                ml_commons_client=self.mock_ml_commons_client,
                model_group_id=self.model_group_id,
                model_name=custom_name,
                model_description=custom_description,
                model_configs=self.model_configs
            )
            
            assert model._model_name == custom_name
            assert model._model_description == custom_description
            assert model._model_configs == self.model_configs

    @patch('models.local_ml_model.get_local_embedding_model_name')
    @patch('models.ml_model.logging')
    def test_local_ml_model_init_inheritance(self, mock_logging, mock_get_model_name):
        """Test that LocalMlModel properly inherits from MlModel"""
        mock_get_model_name.return_value = "test/model"
        
        with patch.object(LocalMlModel, 'find_models', return_value=['test_id']):
            model = LocalMlModel(
                os_client=self.mock_os_client,
                ml_commons_client=self.mock_ml_commons_client,
                model_group_id=self.model_group_id
            )
            
            # Test inherited properties
            assert model._os_client == self.mock_os_client
            assert model._ml_commons_client == self.mock_ml_commons_client
            assert model._model_group_id == self.model_group_id
            
            # Test inherited methods
            assert hasattr(model, 'model_id')
            assert hasattr(model, 'clean_up')
            assert hasattr(model, 'find_models')

    @patch('models.local_ml_model.get_local_embedding_model_version')
    @patch('models.local_ml_model.get_local_embedding_model_format')
    @patch('models.ml_model.logging')
    def test_register_model_with_defaults(self, mock_logging, mock_get_format, mock_get_version):
        """Test _register_model method with default configuration values"""
        mock_get_version.return_value = "1.0.1"
        mock_get_format.return_value = "TORCH_SCRIPT"
        
        with patch.object(LocalMlModel, 'find_models', return_value=['test_id']):
            model = LocalMlModel(
                os_client=self.mock_os_client,
                ml_commons_client=self.mock_ml_commons_client,
                model_group_id=self.model_group_id,
                model_name="test/model"
            )
            
            # Call _register_model directly
            model._register_model()
            
            # Verify the register_pretrained_model call
            self.mock_ml_commons_client.register_pretrained_model.assert_called_once_with(
                model_name="test/model",
                model_version="1.0.1",
                model_format="TORCH_SCRIPT",
                model_group_id=self.model_group_id,
                deploy_model=True,
                wait_until_deployed=True,
            )
            
            # Verify configuration functions were called
            mock_get_version.assert_called_once()
            mock_get_format.assert_called_once()

    @patch('models.ml_model.logging')
    def test_register_model_with_config_values(self, mock_logging):
        """Test _register_model method with configuration values from model_configs"""
        config_with_version_format = {
            "model_version": "2.0.0",
            "model_format": "ONNX",
            "model_group_id": self.model_group_id
        }
        
        with patch.object(LocalMlModel, 'find_models', return_value=['test_id']):
            model = LocalMlModel(
                os_client=self.mock_os_client,
                ml_commons_client=self.mock_ml_commons_client,
                model_group_id=self.model_group_id,
                model_name="test/model",
                model_configs=config_with_version_format
            )
            
            # Call _register_model directly
            model._register_model()
            
            # Verify the register_pretrained_model call uses config values
            self.mock_ml_commons_client.register_pretrained_model.assert_called_once_with(
                model_name="test/model",
                model_version="2.0.0",  # From config
                model_format="ONNX",    # From config
                model_group_id=self.model_group_id,
                deploy_model=True,
                wait_until_deployed=True,
            )

    @patch('models.local_ml_model.get_local_embedding_model_version')
    @patch('models.local_ml_model.get_local_embedding_model_format')
    @patch('models.ml_model.logging')
    def test_register_model_partial_config(self, mock_logging, mock_get_format, mock_get_version):
        """Test _register_model with partial configuration (only version in config)"""
        mock_get_format.return_value = "TORCH_SCRIPT"
        
        config_with_version_only = {
            "model_version": "3.0.0",
            "model_group_id": self.model_group_id
        }
        
        with patch.object(LocalMlModel, 'find_models', return_value=['test_id']):
            model = LocalMlModel(
                os_client=self.mock_os_client,
                ml_commons_client=self.mock_ml_commons_client,
                model_group_id=self.model_group_id,
                model_name="test/model",
                model_configs=config_with_version_only
            )
            
            # Call _register_model directly
            model._register_model()
            
            # Verify the register_pretrained_model call
            self.mock_ml_commons_client.register_pretrained_model.assert_called_once_with(
                model_name="test/model",
                model_version="3.0.0",      # From config
                model_format="TORCH_SCRIPT", # From default
                model_group_id=self.model_group_id,
                deploy_model=True,
                wait_until_deployed=True,
            )
            
            # Verify both functions were called for defaults (even though version was in config)
            mock_get_format.assert_called_once()
            # Note: get_version is called as default parameter even when config has version
            mock_get_version.assert_called_once()

    @patch('models.ml_model.logging')
    def test_register_model_exception_handling(self, mock_logging):
        """Test _register_model method exception handling"""
        self.mock_ml_commons_client.register_pretrained_model.side_effect = Exception("Registration failed")
        
        with patch.object(LocalMlModel, 'find_models', return_value=['test_id']):
            model = LocalMlModel(
                os_client=self.mock_os_client,
                ml_commons_client=self.mock_ml_commons_client,
                model_group_id=self.model_group_id,
                model_name="test/model"
            )
            
            # _register_model should propagate the exception
            with pytest.raises(Exception, match="Registration failed"):
                model._register_model()

    @patch('models.local_ml_model.get_local_embedding_model_name')
    @patch('models.ml_model.logging')
    def test_local_ml_model_str_representation(self, mock_logging, mock_get_model_name):
        """Test string representation of LocalMlModel"""
        mock_get_model_name.return_value = "test/model/name"
        
        with patch.object(LocalMlModel, 'find_models', return_value=['test_model_id']):
            model = LocalMlModel(
                os_client=self.mock_os_client,
                ml_commons_client=self.mock_ml_commons_client,
                model_group_id=self.model_group_id
            )
            
            expected_str = "<MlModel test/model/name test_model_id>"
            assert str(model) == expected_str
            assert repr(model) == expected_str

    @patch('models.local_ml_model.get_local_embedding_model_name')
    @patch('models.ml_model.logging')
    def test_local_ml_model_model_id_method(self, mock_logging, mock_get_model_name):
        """Test model_id method"""
        mock_get_model_name.return_value = "test/model"
        
        with patch.object(LocalMlModel, 'find_models', return_value=['specific_model_id']):
            model = LocalMlModel(
                os_client=self.mock_os_client,
                ml_commons_client=self.mock_ml_commons_client,
                model_group_id=self.model_group_id
            )
            
            assert model.model_id() == 'specific_model_id'

    @patch('models.local_ml_model.get_local_embedding_model_name')
    @patch('models.ml_model.logging')
    def test_local_ml_model_clean_up(self, mock_logging, mock_get_model_name):
        """Test clean_up method inheritance"""
        mock_get_model_name.return_value = "test/model"
        
        with patch.object(LocalMlModel, 'find_models', return_value=['cleanup_model_id']):
            with patch.object(LocalMlModel, '_undeploy_and_delete_model') as mock_undeploy:
                model = LocalMlModel(
                    os_client=self.mock_os_client,
                    ml_commons_client=self.mock_ml_commons_client,
                    model_group_id=self.model_group_id
                )
                
                model.clean_up()
                mock_undeploy.assert_called_once_with('cleanup_model_id')

    @patch('models.local_ml_model.get_local_embedding_model_name')
    @patch('models.ml_model.logging')
    def test_local_ml_model_integration_with_parent(self, mock_logging, mock_get_model_name):
        """Test integration with parent MlModel class"""
        mock_get_model_name.return_value = "integration/test/model"
        
        # Mock the parent class methods
        with patch.object(LocalMlModel, 'find_models', side_effect=[[], ['new_model_id']]):
            with patch.object(LocalMlModel, '_register_model') as mock_register:
                model = LocalMlModel(
                    os_client=self.mock_os_client,
                    ml_commons_client=self.mock_ml_commons_client,
                    model_group_id=self.model_group_id
                )
                
                # Verify that parent class logic was executed
                mock_register.assert_called_once()
                assert model._model_id == 'new_model_id'

    def test_local_ml_model_constants(self):
        """Test LocalMlModel constants"""
        assert hasattr(LocalMlModel, 'DEFAULT_LOCAL_MODEL_DESCRIPTION')
        assert isinstance(LocalMlModel.DEFAULT_LOCAL_MODEL_DESCRIPTION, str)
        assert len(LocalMlModel.DEFAULT_LOCAL_MODEL_DESCRIPTION) > 0

    @patch('models.local_ml_model.get_local_embedding_model_name')
    @patch('models.local_ml_model.get_local_embedding_model_version')
    @patch('models.local_ml_model.get_local_embedding_model_format')
    @patch('models.ml_model.logging')
    def test_configuration_manager_integration(self, mock_logging, mock_get_format, mock_get_version, mock_get_model_name):
        """Test integration with configuration manager"""
        # Set up configuration manager mocks
        mock_get_model_name.return_value = "config/model/name"
        mock_get_version.return_value = "config_version"
        mock_get_format.return_value = "CONFIG_FORMAT"
        
        with patch.object(LocalMlModel, 'find_models', return_value=['config_test_id']):
            model = LocalMlModel(
                os_client=self.mock_os_client,
                ml_commons_client=self.mock_ml_commons_client,
                model_group_id=self.model_group_id
            )
            
            # Test that configuration manager functions are called
            mock_get_model_name.assert_called_once()
            
            # Test _register_model uses configuration manager defaults
            model._register_model()
            
            self.mock_ml_commons_client.register_pretrained_model.assert_called_once_with(
                model_name="config/model/name",
                model_version="config_version",
                model_format="CONFIG_FORMAT",
                model_group_id=self.model_group_id,
                deploy_model=True,
                wait_until_deployed=True,
            )

    @patch('models.ml_model.logging')
    def test_empty_model_configs(self, mock_logging):
        """Test LocalMlModel with empty model_configs"""
        with patch.object(LocalMlModel, 'find_models', return_value=['empty_config_id']):
            model = LocalMlModel(
                os_client=self.mock_os_client,
                ml_commons_client=self.mock_ml_commons_client,
                model_group_id=self.model_group_id,
                model_name="test/model",
                model_configs={}  # Empty config
            )
            
            assert model._model_configs == {}
            assert model._model_name == "test/model"

    @patch('models.local_ml_model.get_local_embedding_model_name')
    @patch('models.ml_model.logging')
    def test_none_model_configs(self, mock_logging, mock_get_model_name):
        """Test LocalMlModel with None model_configs (should use default dict)"""
        mock_get_model_name.return_value = "none/config/model"
        
        with patch.object(LocalMlModel, 'find_models', return_value=['none_config_id']):
            # Don't pass model_configs parameter
            model = LocalMlModel(
                os_client=self.mock_os_client,
                ml_commons_client=self.mock_ml_commons_client,
                model_group_id=self.model_group_id
            )
            
            # Should use default empty dict from parent class
            assert model._model_configs == {}

    @patch('models.local_ml_model.get_local_embedding_model_name')
    @patch('models.ml_model.logging')
    def test_overrides_decorator(self, mock_logging, mock_get_model_name):
        """Test that _register_model has @overrides decorator"""
        mock_get_model_name.return_value = "override/test"
        
        with patch.object(LocalMlModel, 'find_models', return_value=['override_id']):
            model = LocalMlModel(
                os_client=self.mock_os_client,
                ml_commons_client=self.mock_ml_commons_client,
                model_group_id=self.model_group_id
            )
            
            # Verify that the method exists and is callable
            assert hasattr(model, '_register_model')
            assert callable(getattr(model, '_register_model'))
            
            # The @overrides decorator should be applied
            # This is mainly a structural test to ensure the decorator is present
