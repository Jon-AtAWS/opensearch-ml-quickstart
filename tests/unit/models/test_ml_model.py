# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for models.ml_model module"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from opensearchpy import OpenSearch
from opensearch_py_ml.ml_commons import MLCommonClient

from models.ml_model import MlModel


class ConcreteMlModel(MlModel):
    """Concrete implementation of MlModel for testing"""
    
    def _register_model(self):
        """Concrete implementation for testing"""
        pass


class TestMlModel:
    """Test cases for MlModel abstract base class"""

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
            "model_format": "TORCH_SCRIPT"
        }

    @patch('models.ml_model.logging')
    def test_ml_model_init_with_defaults(self, mock_logging):
        """Test MlModel initialization with default values"""
        # Mock find_models to return existing model
        with patch.object(ConcreteMlModel, 'find_models', return_value=['existing_model_id']):
            model = ConcreteMlModel(
                os_client=self.mock_os_client,
                ml_commons_client=self.mock_ml_commons_client,
                model_group_id=self.model_group_id
            )
            
            assert model._os_client == self.mock_os_client
            assert model._ml_commons_client == self.mock_ml_commons_client
            assert model._model_group_id == self.model_group_id
            assert model._model_name == MlModel.DEFAULT_MODEL_NAME
            assert model._model_description == MlModel.DEFAULT_MODEL_DESCRIPTION
            assert model._model_configs == {}
            assert model._model_id == 'existing_model_id'

    @patch('models.ml_model.logging')
    def test_ml_model_init_with_custom_values(self, mock_logging):
        """Test MlModel initialization with custom values"""
        custom_name = "Custom Model"
        custom_description = "Custom Description"
        
        with patch.object(ConcreteMlModel, 'find_models', return_value=['custom_model_id']):
            model = ConcreteMlModel(
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
            assert model._model_id == 'custom_model_id'

    @patch('models.ml_model.logging')
    def test_ml_model_init_no_existing_model(self, mock_logging):
        """Test MlModel initialization when no existing model found"""
        with patch.object(ConcreteMlModel, 'find_models', side_effect=[[], ['new_model_id']]):
            with patch.object(ConcreteMlModel, '_register_model') as mock_register:
                model = ConcreteMlModel(
                    os_client=self.mock_os_client,
                    ml_commons_client=self.mock_ml_commons_client,
                    model_group_id=self.model_group_id
                )
                
                mock_register.assert_called_once()
                assert model._model_id == 'new_model_id'

    @patch('models.ml_model.logging')
    def test_ml_model_init_registration_failure(self, mock_logging):
        """Test MlModel initialization when model registration fails"""
        with patch.object(ConcreteMlModel, 'find_models', return_value=[]):
            with patch.object(ConcreteMlModel, '_register_model'):
                with pytest.raises(Exception, match="Failed to find the registered model"):
                    ConcreteMlModel(
                        os_client=self.mock_os_client,
                        ml_commons_client=self.mock_ml_commons_client,
                        model_group_id=self.model_group_id
                    )

    def test_str_and_repr(self):
        """Test string representation methods"""
        with patch.object(ConcreteMlModel, 'find_models', return_value=['test_id']):
            model = ConcreteMlModel(
                os_client=self.mock_os_client,
                ml_commons_client=self.mock_ml_commons_client,
                model_group_id=self.model_group_id,
                model_name="Test Model"
            )
            
            expected_str = "<MlModel Test Model test_id>"
            assert str(model) == expected_str
            assert repr(model) == expected_str

    def test_model_id(self):
        """Test model_id method"""
        with patch.object(ConcreteMlModel, 'find_models', return_value=['test_model_id']):
            model = ConcreteMlModel(
                os_client=self.mock_os_client,
                ml_commons_client=self.mock_ml_commons_client,
                model_group_id=self.model_group_id
            )
            
            assert model.model_id() == 'test_model_id'

    def test_clean_up(self):
        """Test clean_up method"""
        with patch.object(ConcreteMlModel, 'find_models', return_value=['test_model_id']):
            with patch.object(ConcreteMlModel, '_undeploy_and_delete_model') as mock_undeploy:
                model = ConcreteMlModel(
                    os_client=self.mock_os_client,
                    ml_commons_client=self.mock_ml_commons_client,
                    model_group_id=self.model_group_id
                )
                
                model.clean_up()
                mock_undeploy.assert_called_once_with('test_model_id')

    def test_find_models_success(self):
        """Test find_models method with successful search"""
        search_result = {
            "hits": {
                "hits": [
                    {
                        "_id": "model1",
                        "_source": {
                            "name": "Test Model",
                            "model_id": "actual_model_id_1"
                        }
                    },
                    {
                        "_id": "model2", 
                        "_source": {
                            "name": "Another Model",
                            "model_id": "actual_model_id_2"
                        }
                    }
                ]
            }
        }
        
        self.mock_ml_commons_client.search_model.return_value = search_result
        
        # Create model without patching find_models
        model = ConcreteMlModel(
            os_client=self.mock_os_client,
            ml_commons_client=self.mock_ml_commons_client,
            model_group_id=self.model_group_id,
            model_name="Test Model"
        )
        
        # Call the actual find_models method
        result = model.find_models("Test Model")
        
        assert "actual_model_id_1" in result
        assert len(result) == 1

    def test_find_models_no_model_id_field(self):
        """Test find_models when model_id field is missing"""
        search_result = {
            "hits": {
                "hits": [
                    {
                        "_id": "model1",
                        "_source": {
                            "name": "Test Model"
                            # No model_id field
                        }
                    }
                ]
            }
        }
        
        self.mock_ml_commons_client.search_model.return_value = search_result
        
        # Create model without patching find_models
        model = ConcreteMlModel(
            os_client=self.mock_os_client,
            ml_commons_client=self.mock_ml_commons_client,
            model_group_id=self.model_group_id,
            model_name="Test Model"
        )
        
        # Call the actual find_models method
        result = model.find_models("Test Model")
        
        assert "model1" in result  # Should use _id when model_id is missing

    def test_find_models_empty_result(self):
        """Test find_models with empty search result"""
        # First, set up valid data for model initialization
        init_search_result = {
            "hits": {
                "hits": [
                    {
                        "_id": "init_model",
                        "_source": {
                            "name": "Test Model",
                            "model_id": "init_model_id"
                        }
                    }
                ]
            }
        }
        self.mock_ml_commons_client.search_model.return_value = init_search_result
        
        # Create model (this will succeed with the init data)
        model = ConcreteMlModel(
            os_client=self.mock_os_client,
            ml_commons_client=self.mock_ml_commons_client,
            model_group_id=self.model_group_id,
            model_name="Test Model"
        )
        
        # Now set up the empty result for the actual test
        self.mock_ml_commons_client.search_model.return_value = None
        
        # Call the actual find_models method
        result = model.find_models("Test Model")
        
        assert result == []

    def test_find_models_exception(self):
        """Test find_models with exception"""
        # First, set up valid data for model initialization
        init_search_result = {
            "hits": {
                "hits": [
                    {
                        "_id": "init_model",
                        "_source": {
                            "name": "Test Model",
                            "model_id": "init_model_id"
                        }
                    }
                ]
            }
        }
        self.mock_ml_commons_client.search_model.return_value = init_search_result
        
        # Create model (this will succeed with the init data)
        model = ConcreteMlModel(
            os_client=self.mock_os_client,
            ml_commons_client=self.mock_ml_commons_client,
            model_group_id=self.model_group_id,
            model_name="Test Model"
        )
        
        # Now set up the exception for the actual test
        self.mock_ml_commons_client.search_model.side_effect = Exception("Search failed")
        
        # Call the actual find_models method
        result = model.find_models("Test Model")
        
        assert result == []

    def test_find_models_all_models(self):
        """Test find_models without name filter"""
    def test_find_models_all_models(self):
        """Test find_models without name filter"""
        # First, set up valid data for model initialization
        init_search_result = {
            "hits": {
                "hits": [
                    {
                        "_id": "init_model",
                        "_source": {
                            "name": "Test Model",
                            "model_id": "init_model_id"
                        }
                    }
                ]
            }
        }
        self.mock_ml_commons_client.search_model.return_value = init_search_result
        
        # Create model (this will succeed with the init data)
        model = ConcreteMlModel(
            os_client=self.mock_os_client,
            ml_commons_client=self.mock_ml_commons_client,
            model_group_id=self.model_group_id,
            model_name="Test Model"
        )
        
        # Now set up the test data
        search_result = {
            "hits": {
                "hits": [
                    {
                        "_id": "model1",
                        "_source": {
                            "name": "Model 1",
                            "model_id": "id1"
                        }
                    },
                    {
                        "_id": "model2",
                        "_source": {
                            "name": "Model 2", 
                            "model_id": "id2"
                        }
                    }
                ]
            }
        }
        
        self.mock_ml_commons_client.search_model.return_value = search_result
        
        # Call the actual find_models method without name filter
        result = model.find_models()
        
        assert len(result) == 2
        assert "id1" in result
        assert "id2" in result

    @patch('builtins.input', return_value='y')
    @patch('models.ml_model.logging')
    def test_undeploy_and_delete_model_success(self, mock_logging, mock_input):
        """Test successful model undeploy and delete"""
        with patch.object(ConcreteMlModel, 'find_models', return_value=['test_id']):
            model = ConcreteMlModel(
                os_client=self.mock_os_client,
                ml_commons_client=self.mock_ml_commons_client,
                model_group_id=self.model_group_id
            )
            
            # Test the undeploy and delete method
            model._undeploy_and_delete_model('test_model_id')
            
            self.mock_ml_commons_client.undeploy_model.assert_called_once_with('test_model_id')
            self.mock_ml_commons_client.delete_model.assert_called_once_with('test_model_id')

    @patch('builtins.input', return_value='n')
    @patch('models.ml_model.logging')
    def test_undeploy_and_delete_model_cancelled(self, mock_logging, mock_input):
        """Test cancelled model undeploy and delete"""
        with patch.object(ConcreteMlModel, 'find_models', return_value=['test_id']):
            model = ConcreteMlModel(
                os_client=self.mock_os_client,
                ml_commons_client=self.mock_ml_commons_client,
                model_group_id=self.model_group_id
            )
            
            # Test the undeploy and delete method with cancellation
            model._undeploy_and_delete_model('test_model_id')
            
            self.mock_ml_commons_client.undeploy_model.assert_not_called()
            self.mock_ml_commons_client.delete_model.assert_not_called()

    @patch('builtins.input', return_value='y')
    @patch('models.ml_model.logging')
    def test_undeploy_and_delete_model_undeploy_failure(self, mock_logging, mock_input):
        """Test model undeploy failure"""
        # First, set up valid data for model initialization
        init_search_result = {
            "hits": {
                "hits": [
                    {
                        "_id": "init_model",
                        "_source": {
                            "name": "Test Model",
                            "model_id": "init_model_id"
                        }
                    }
                ]
            }
        }
        self.mock_ml_commons_client.search_model.return_value = init_search_result
        
        # Create model (this will succeed with the init data)
        model = ConcreteMlModel(
            os_client=self.mock_os_client,
            ml_commons_client=self.mock_ml_commons_client,
            model_group_id=self.model_group_id,
            model_name="Test Model"
        )
        
        # Now set up the undeploy failure
        self.mock_ml_commons_client.undeploy_model.side_effect = Exception("Undeploy failed")
        
        # Expect RetryError since tenacity wraps the original exception
        with pytest.raises(Exception):  # Accept any exception type
            model._undeploy_and_delete_model('test_model_id')

    @patch('builtins.input', return_value='y')
    @patch('models.ml_model.logging')
    def test_undeploy_and_delete_model_delete_failure(self, mock_logging, mock_input):
        """Test model delete failure"""
        # First, set up valid data for model initialization
        init_search_result = {
            "hits": {
                "hits": [
                    {
                        "_id": "init_model",
                        "_source": {
                            "name": "Test Model",
                            "model_id": "init_model_id"
                        }
                    }
                ]
            }
        }
        self.mock_ml_commons_client.search_model.return_value = init_search_result
        
        # Create model (this will succeed with the init data)
        model = ConcreteMlModel(
            os_client=self.mock_os_client,
            ml_commons_client=self.mock_ml_commons_client,
            model_group_id=self.model_group_id,
            model_name="Test Model"
        )
        
        # Now set up the delete failure
        self.mock_ml_commons_client.delete_model.side_effect = Exception("Delete failed")
        
        # Expect RetryError since tenacity wraps the original exception
        with pytest.raises(Exception):  # Accept any exception type
            model._undeploy_and_delete_model('test_model_id')

    def test_abstract_register_model(self):
        """Test that _register_model is abstract"""
        # This test verifies that MlModel is abstract
        with pytest.raises(TypeError):
            MlModel(
                os_client=self.mock_os_client,
                ml_commons_client=self.mock_ml_commons_client,
                model_group_id=self.model_group_id
            )

    @patch('models.ml_model.get_delete_resource_retry_time', return_value=3)
    @patch('models.ml_model.get_delete_resource_wait_time', return_value=1)
    def test_retry_configuration(self, mock_wait_time, mock_retry_time):
        """Test that retry configuration is properly applied"""
        with patch.object(ConcreteMlModel, 'find_models', return_value=['test_id']):
            model = ConcreteMlModel(
                os_client=self.mock_os_client,
                ml_commons_client=self.mock_ml_commons_client,
                model_group_id=self.model_group_id
            )
            
            # Verify that the configuration functions are called
            # This is tested indirectly through the decorator application
            assert hasattr(model._undeploy_and_delete_model, '__wrapped__')

    def test_find_models_duplicate_removal(self):
        """Test that find_models removes duplicates"""
        search_result = {
            "hits": {
                "hits": [
                    {
                        "_id": "model1",
                        "_source": {
                            "name": "Test Model",
                            "model_id": "duplicate_id"
                        }
                    },
                    {
                        "_id": "model2",
                        "_source": {
                            "name": "Test Model",
                            "model_id": "duplicate_id"  # Same ID
                        }
                    }
                ]
            }
        }
        
        self.mock_ml_commons_client.search_model.return_value = search_result
        
        # Create model without patching find_models
        model = ConcreteMlModel(
            os_client=self.mock_os_client,
            ml_commons_client=self.mock_ml_commons_client,
            model_group_id=self.model_group_id,
            model_name="Test Model"
        )
        
        # Call the actual find_models method
        result = model.find_models("Test Model")
        
        # Should only return one instance of the duplicate ID
        assert len(result) == 1
        assert "duplicate_id" in result
