# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for models.remote_ml_model module"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from opensearchpy import OpenSearch
from opensearch_py_ml.ml_commons import MLCommonClient

from models.remote_ml_model import RemoteMlModel
from connectors import MlConnector


class TestRemoteMlModel:
    """Test cases for RemoteMlModel class"""

    def setup_method(self):
        """Setup method run before each test"""
        # Create mock clients
        self.mock_os_client = Mock(spec=OpenSearch)
        self.mock_os_client.http = Mock()
        self.mock_os_client.http.post = Mock(return_value={
            "model_id": "test_model_id",
            "task_id": "test_task_id"
        })
        
        self.mock_ml_commons_client = Mock(spec=MLCommonClient)
        self.mock_ml_connector = Mock(spec=MlConnector)
        
        # Mock model group ID
        self.model_group_id = "test_group_id"
        
        # Mock connector ID
        self.mock_ml_connector.connector_id.return_value = "test_connector_id"

    @patch('models.ml_model.logging')
    def test_remote_ml_model_init_dense_default(self, mock_logging):
        """Test RemoteMlModel initialization with dense embedding (default)"""
        model_configs = {"embedding_type": "dense"}
        
        with patch.object(RemoteMlModel, 'find_models', return_value=['dense_model_id']):
            model = RemoteMlModel(
                os_client=self.mock_os_client,
                ml_commons_client=self.mock_ml_commons_client,
                ml_connector=self.mock_ml_connector,
                model_group_id=self.model_group_id,
                model_configs=model_configs
            )
            
            assert model._ml_connector == self.mock_ml_connector
            assert model._model_name == RemoteMlModel.DEFAULT_DENSE_MODEL_NAME
            assert model._model_description == RemoteMlModel.DEFAULT_DENSE_MODEL_DESCRIPTION
            assert model._model_configs == model_configs

    @patch('models.ml_model.logging')
    def test_remote_ml_model_init_sparse(self, mock_logging):
        """Test RemoteMlModel initialization with sparse embedding"""
        model_configs = {"embedding_type": "sparse"}
        
        with patch.object(RemoteMlModel, 'find_models', return_value=['sparse_model_id']):
            model = RemoteMlModel(
                os_client=self.mock_os_client,
                ml_commons_client=self.mock_ml_commons_client,
                ml_connector=self.mock_ml_connector,
                model_group_id=self.model_group_id,
                model_configs=model_configs
            )
            
            assert model._model_name == RemoteMlModel.DEFAULT_SPARSE_MODEL_NAME
            assert model._model_description == RemoteMlModel.DEFAULT_SPARSE_MODEL_DESCRIPTION

    @patch('models.ml_model.logging')
    def test_remote_ml_model_init_no_embedding_type(self, mock_logging):
        """Test RemoteMlModel initialization without embedding_type (defaults to dense)"""
        model_configs = {}
        
        with patch.object(RemoteMlModel, 'find_models', return_value=['default_model_id']):
            model = RemoteMlModel(
                os_client=self.mock_os_client,
                ml_commons_client=self.mock_ml_commons_client,
                ml_connector=self.mock_ml_connector,
                model_group_id=self.model_group_id,
                model_configs=model_configs
            )
            
            # Should default to dense
            assert model._model_name == RemoteMlModel.DEFAULT_DENSE_MODEL_NAME
            assert model._model_description == RemoteMlModel.DEFAULT_DENSE_MODEL_DESCRIPTION

    @patch('models.ml_model.logging')
    def test_remote_ml_model_init_custom_name_description(self, mock_logging):
        """Test RemoteMlModel initialization with custom name and description"""
        custom_name = "Custom Remote Model"
        custom_description = "Custom model description"
        model_configs = {"embedding_type": "dense"}
        
        with patch.object(RemoteMlModel, 'find_models', return_value=['custom_model_id']):
            model = RemoteMlModel(
                os_client=self.mock_os_client,
                ml_commons_client=self.mock_ml_commons_client,
                ml_connector=self.mock_ml_connector,
                model_group_id=self.model_group_id,
                model_name=custom_name,
                model_description=custom_description,
                model_configs=model_configs
            )
            
            assert model._model_name == custom_name
            assert model._model_description == custom_description

    @patch('models.ml_model.logging')
    def test_remote_ml_model_inheritance(self, mock_logging):
        """Test that RemoteMlModel properly inherits from MlModel"""
        with patch.object(RemoteMlModel, 'find_models', return_value=['inherit_test_id']):
            model = RemoteMlModel(
                os_client=self.mock_os_client,
                ml_commons_client=self.mock_ml_commons_client,
                ml_connector=self.mock_ml_connector,
                model_group_id=self.model_group_id
            )
            
            # Test inherited properties
            assert model._os_client == self.mock_os_client
            assert model._ml_commons_client == self.mock_ml_commons_client
            assert model._model_group_id == self.model_group_id
            
            # Test inherited methods
            assert hasattr(model, 'model_id')
            assert hasattr(model, 'find_models')

    @patch('models.remote_ml_model.get_ml_base_uri')
    @patch('models.ml_model.logging')
    def test_register_model_calls_deploy(self, mock_logging, mock_get_ml_base_uri):
        """Test that _register_model calls _deploy_model"""
        mock_get_ml_base_uri.return_value = "/_plugins/_ml"
        
        with patch.object(RemoteMlModel, 'find_models', return_value=['register_test_id']):
            with patch.object(RemoteMlModel, '_deploy_model') as mock_deploy:
                model = RemoteMlModel(
                    os_client=self.mock_os_client,
                    ml_commons_client=self.mock_ml_commons_client,
                    ml_connector=self.mock_ml_connector,
                    model_group_id=self.model_group_id
                )
                
                model._register_model()
                mock_deploy.assert_called_once()

    @patch('models.remote_ml_model.get_ml_base_uri')
    @patch('models.remote_ml_model.time.sleep')
    @patch('models.ml_model.logging')
    def test_deploy_model_success(self, mock_logging, mock_sleep, mock_get_ml_base_uri):
        """Test successful model deployment"""
        mock_get_ml_base_uri.return_value = "/_plugins/_ml"
        
        # Mock HTTP responses
        register_response = {"task_id": "test_task_id"}
        task_response = {"state": "COMPLETED"}
        
        self.mock_os_client.http.post.return_value = register_response
        self.mock_os_client.http.get.return_value = task_response
        
        with patch.object(RemoteMlModel, 'find_models', return_value=['deploy_test_id']):
            model = RemoteMlModel(
                os_client=self.mock_os_client,
                ml_commons_client=self.mock_ml_commons_client,
                ml_connector=self.mock_ml_connector,
                model_group_id=self.model_group_id,
                model_name="Test Deploy Model"
            )
            
            model._deploy_model()
            
            # Verify register call
            expected_payload = {
                "name": "Test Deploy Model",
                "function_name": "remote",
                "description": RemoteMlModel.DEFAULT_DENSE_MODEL_DESCRIPTION,
                "connector_id": "test_connector_id",
                "model_group_id": self.model_group_id,
                "deploy": True,
            }
            
            self.mock_os_client.http.post.assert_called_once_with(
                url="/_plugins/_ml/models/_register",
                body=expected_payload,
            )
            
            # Verify task status check
            self.mock_os_client.http.get.assert_called_once_with(
                url="/_plugins/_ml/tasks/test_task_id"
            )
            
            # Verify sleep was called
            mock_sleep.assert_called_once_with(1)

    @patch('models.remote_ml_model.get_ml_base_uri')
    @patch('models.remote_ml_model.time.sleep')
    @patch('models.ml_model.logging')
    def test_deploy_model_failure(self, mock_logging, mock_sleep, mock_get_ml_base_uri):
        """Test model deployment failure"""
        mock_get_ml_base_uri.return_value = "/_plugins/_ml"
        
        # Mock HTTP responses
        register_response = {"task_id": "failed_task_id"}
        task_response = {"state": "FAILED"}
        
        self.mock_os_client.http.post.return_value = register_response
        self.mock_os_client.http.get.return_value = task_response
        
        with patch.object(RemoteMlModel, 'find_models', return_value=['deploy_fail_id']):
            model = RemoteMlModel(
                os_client=self.mock_os_client,
                ml_commons_client=self.mock_ml_commons_client,
                ml_connector=self.mock_ml_connector,
                model_group_id=self.model_group_id
            )
            
            with pytest.raises(Exception, match="Model deployment task failed_task_id is not COMPLETED!"):
                model._deploy_model()

    @patch('models.ml_model.logging')
    def test_clean_up_calls_parent_and_connector(self, mock_logging):
        """Test that clean_up calls both parent clean_up and connector clean_up"""
        with patch.object(RemoteMlModel, 'find_models', return_value=['cleanup_test_id']):
            with patch('models.ml_model.MlModel.clean_up') as mock_parent_cleanup:
                model = RemoteMlModel(
                    os_client=self.mock_os_client,
                    ml_commons_client=self.mock_ml_commons_client,
                    ml_connector=self.mock_ml_connector,
                    model_group_id=self.model_group_id
                )
                
                model.clean_up()
                
                # Verify parent clean_up was called
                mock_parent_cleanup.assert_called_once()
                
                # Verify connector clean_up was called
                self.mock_ml_connector.clean_up.assert_called_once()

    @patch('models.ml_model.logging')
    def test_remote_ml_model_constants(self, mock_logging):
        """Test RemoteMlModel constants"""
        assert hasattr(RemoteMlModel, 'DEFAULT_DENSE_MODEL_NAME')
        assert hasattr(RemoteMlModel, 'DEFAULT_DENSE_MODEL_DESCRIPTION')
        assert hasattr(RemoteMlModel, 'DEFAULT_SPARSE_MODEL_NAME')
        assert hasattr(RemoteMlModel, 'DEFAULT_SPARSE_MODEL_DESCRIPTION')
        
        # Verify they are strings
        assert isinstance(RemoteMlModel.DEFAULT_DENSE_MODEL_NAME, str)
        assert isinstance(RemoteMlModel.DEFAULT_DENSE_MODEL_DESCRIPTION, str)
        assert isinstance(RemoteMlModel.DEFAULT_SPARSE_MODEL_NAME, str)
        assert isinstance(RemoteMlModel.DEFAULT_SPARSE_MODEL_DESCRIPTION, str)
        
        # Verify they are not empty
        assert len(RemoteMlModel.DEFAULT_DENSE_MODEL_NAME) > 0
        assert len(RemoteMlModel.DEFAULT_DENSE_MODEL_DESCRIPTION) > 0
        assert len(RemoteMlModel.DEFAULT_SPARSE_MODEL_NAME) > 0
        assert len(RemoteMlModel.DEFAULT_SPARSE_MODEL_DESCRIPTION) > 0

    @patch('models.ml_model.logging')
    def test_remote_ml_model_str_representation(self, mock_logging):
        """Test string representation of RemoteMlModel"""
        with patch.object(RemoteMlModel, 'find_models', return_value=['str_test_id']):
            model = RemoteMlModel(
                os_client=self.mock_os_client,
                ml_commons_client=self.mock_ml_commons_client,
                ml_connector=self.mock_ml_connector,
                model_group_id=self.model_group_id,
                model_name="String Test Model"
            )
            
            expected_str = "<MlModel String Test Model str_test_id>"
            assert str(model) == expected_str
            assert repr(model) == expected_str

    @patch('models.remote_ml_model.get_ml_base_uri')
    @patch('models.remote_ml_model.time.sleep')
    @patch('models.ml_model.logging')
    def test_deploy_model_with_sparse_config(self, mock_logging, mock_sleep, mock_get_ml_base_uri):
        """Test model deployment with sparse configuration"""
        mock_get_ml_base_uri.return_value = "/_plugins/_ml"
        
        # Mock HTTP responses
        register_response = {"task_id": "sparse_task_id"}
        task_response = {"state": "COMPLETED"}
        
        self.mock_os_client.http.post.return_value = register_response
        self.mock_os_client.http.get.return_value = task_response
        
        model_configs = {"embedding_type": "sparse"}
        
        with patch.object(RemoteMlModel, 'find_models', return_value=['sparse_deploy_id']):
            model = RemoteMlModel(
                os_client=self.mock_os_client,
                ml_commons_client=self.mock_ml_commons_client,
                ml_connector=self.mock_ml_connector,
                model_group_id=self.model_group_id,
                model_configs=model_configs
            )
            
            model._deploy_model()
            
            # Verify the payload uses sparse model defaults
            expected_payload = {
                "name": RemoteMlModel.DEFAULT_SPARSE_MODEL_NAME,
                "function_name": "remote",
                "description": RemoteMlModel.DEFAULT_SPARSE_MODEL_DESCRIPTION,
                "connector_id": "test_connector_id",
                "model_group_id": self.model_group_id,
                "deploy": True,
            }
            
            self.mock_os_client.http.post.assert_called_once_with(
                url="/_plugins/_ml/models/_register",
                body=expected_payload,
            )

    @patch('models.ml_model.logging')
    def test_overrides_decorator(self, mock_logging):
        """Test that methods have @overrides decorator"""
        with patch.object(RemoteMlModel, 'find_models', return_value=['override_test_id']):
            model = RemoteMlModel(
                os_client=self.mock_os_client,
                ml_commons_client=self.mock_ml_commons_client,
                ml_connector=self.mock_ml_connector,
                model_group_id=self.model_group_id
            )
            
            # Verify that the overridden methods exist and are callable
            assert hasattr(model, '_register_model')
            assert callable(getattr(model, '_register_model'))
            assert hasattr(model, 'clean_up')
            assert callable(getattr(model, 'clean_up'))

    @patch('models.remote_ml_model.get_ml_base_uri')
    @patch('models.remote_ml_model.time.sleep')
    @patch('models.ml_model.logging')
    def test_deploy_model_task_status_variations(self, mock_logging, mock_sleep, mock_get_ml_base_uri):
        """Test deployment with different task status responses"""
        mock_get_ml_base_uri.return_value = "/_plugins/_ml"
        
        # Test various non-COMPLETED states
        non_completed_states = ["RUNNING", "PENDING", "FAILED", "CANCELLED"]
        
        for state in non_completed_states:
            register_response = {"task_id": f"task_{state.lower()}"}
            task_response = {"state": state}
            
            self.mock_os_client.http.post.return_value = register_response
            self.mock_os_client.http.get.return_value = task_response
            
            with patch.object(RemoteMlModel, 'find_models', return_value=[f'{state.lower()}_id']):
                model = RemoteMlModel(
                    os_client=self.mock_os_client,
                    ml_commons_client=self.mock_ml_commons_client,
                    ml_connector=self.mock_ml_connector,
                    model_group_id=self.model_group_id
                )
                
                with pytest.raises(Exception, match=f"Model deployment task task_{state.lower()} is not COMPLETED!"):
                    model._deploy_model()

    @patch('models.ml_model.logging')
    def test_model_configs_parameter_handling(self, mock_logging):
        """Test various model_configs parameter scenarios"""
        test_cases = [
            ({}, "dense"),  # Empty config defaults to dense
            ({"embedding_type": "dense"}, "dense"),
            ({"embedding_type": "sparse"}, "sparse"),
            ({"embedding_type": "invalid"}, "dense"),  # Invalid defaults to dense
            ({"other_param": "value"}, "dense"),  # Other params, no embedding_type
        ]
        
        for config, expected_type in test_cases:
            with patch.object(RemoteMlModel, 'find_models', return_value=['config_test_id']):
                model = RemoteMlModel(
                    os_client=self.mock_os_client,
                    ml_commons_client=self.mock_ml_commons_client,
                    ml_connector=self.mock_ml_connector,
                    model_group_id=self.model_group_id,
                    model_configs=config
                )
                
                if expected_type == "dense":
                    assert model._model_name == RemoteMlModel.DEFAULT_DENSE_MODEL_NAME
                    assert model._model_description == RemoteMlModel.DEFAULT_DENSE_MODEL_DESCRIPTION
                else:  # sparse
                    assert model._model_name == RemoteMlModel.DEFAULT_SPARSE_MODEL_NAME
                    assert model._model_description == RemoteMlModel.DEFAULT_SPARSE_MODEL_DESCRIPTION

    @patch('models.ml_model.logging')
    def test_connector_integration(self, mock_logging):
        """Test integration with ML connector"""
        with patch.object(RemoteMlModel, 'find_models', return_value=['connector_test_id']):
            model = RemoteMlModel(
                os_client=self.mock_os_client,
                ml_commons_client=self.mock_ml_commons_client,
                ml_connector=self.mock_ml_connector,
                model_group_id=self.model_group_id
            )
            
            # Verify connector is stored
            assert model._ml_connector == self.mock_ml_connector
            
            # Verify connector_id is called during deployment
            with patch.object(model, '_deploy_model'):
                model._register_model()
                # The connector_id should be called when _deploy_model is executed
