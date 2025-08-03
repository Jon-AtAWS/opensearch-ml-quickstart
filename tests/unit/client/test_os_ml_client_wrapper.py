# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for client.os_ml_client_wrapper module"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from opensearchpy import OpenSearch
from opensearch_py_ml.ml_commons import MLCommonClient

from client.os_ml_client_wrapper import OsMlClientWrapper
from models import MlModel, MlModelGroup


class TestOsMlClientWrapper:
    """Test cases for OsMlClientWrapper class"""

    def setup_method(self):
        """Setup method run before each test"""
        # Create mock OpenSearch client
        self.mock_os_client = Mock(spec=OpenSearch)
        self.mock_os_client.ingest = Mock()
        self.mock_os_client.ingest.put_pipeline = Mock()
        self.mock_os_client.ingest.delete_pipeline = Mock()
        self.mock_os_client.indices = Mock()
        self.mock_os_client.indices.delete = Mock()
        
        # Create mock ML Commons client
        self.mock_ml_commons_client = Mock(spec=MLCommonClient)
        
        # Create mock model group
        self.mock_model_group = Mock(spec=MlModelGroup)
        self.mock_model_group.model_group_id.return_value = "test_group_id"
        
        # Patch the MLCommonClient and MlModelGroup constructors
        with patch('client.os_ml_client_wrapper.MLCommonClient') as mock_ml_client_class, \
             patch('client.os_ml_client_wrapper.MlModelGroup') as mock_model_group_class:
            
            mock_ml_client_class.return_value = self.mock_ml_commons_client
            mock_model_group_class.return_value = self.mock_model_group
            
            # Create wrapper instance
            self.wrapper = OsMlClientWrapper(self.mock_os_client)

    def test_init(self):
        """Test OsMlClientWrapper initialization"""
        assert self.wrapper.os_client == self.mock_os_client
        assert self.wrapper.ml_commons_client == self.mock_ml_commons_client
        assert self.wrapper.ml_model_group == self.mock_model_group
        assert self.wrapper.ml_model is None
        assert self.wrapper.index_name == ""
        assert self.wrapper.pipeline_name == ""

    def test_model_group_id(self):
        """Test model_group_id method"""
        result = self.wrapper.model_group_id()
        assert result == "test_group_id"
        self.mock_model_group.model_group_id.assert_called_once()

    def test_model_id_with_model(self):
        """Test model_id method when model is set"""
        # Setup mock model
        mock_model = Mock(spec=MlModel)
        mock_model.deploy = Mock(return_value=None)
        mock_model.delete = Mock(return_value=None)
        mock_model.model_id.return_value = "test_model_id"
        self.wrapper.ml_model = mock_model
        
        result = self.wrapper.model_id()
        assert result == "test_model_id"
        mock_model.model_id.assert_called_once()

    def test_model_id_without_model(self):
        """Test model_id method when model is None"""
        # ml_model is None by default
        with pytest.raises(AttributeError):
            self.wrapper.model_id()

    def test_dense_pipeline_config_default(self):
        """Test _dense_pipeline_config with default field map"""
        # Setup mock model
        mock_model = Mock(spec=MlModel)
        mock_model.deploy = Mock(return_value=None)
        mock_model.delete = Mock(return_value=None)
        mock_model.model_id.return_value = "test_model_id"
        self.wrapper.ml_model = mock_model
        
        result = self.wrapper._dense_pipeline_config()
        
        expected_config = {
            "description": "Pipeline for processing chunks",
            "processors": [
                {
                    "text_embedding": {
                        "model_id": "test_model_id",
                        "field_map": {"chunk": "chunk_vector"},
                    }
                }
            ],
        }
        assert result == expected_config

    def test_dense_pipeline_config_custom_field_map(self):
        """Test _dense_pipeline_config with custom field map"""
        # Setup mock model
        mock_model = Mock(spec=MlModel)
        mock_model.deploy = Mock(return_value=None)
        mock_model.delete = Mock(return_value=None)
        mock_model.model_id.return_value = "test_model_id"
        self.wrapper.ml_model = mock_model
        
        custom_field_map = {"text": "text_embedding"}
        result = self.wrapper._dense_pipeline_config(custom_field_map)
        
        expected_config = {
            "description": "Pipeline for processing chunks",
            "processors": [
                {
                    "text_embedding": {
                        "model_id": "test_model_id",
                        "field_map": {"text": "text_embedding"},
                    }
                }
            ],
        }
        assert result == expected_config

    def test_sparse_pipeline_config_default(self):
        """Test _sparse_pipeline_config with default field map"""
        # Setup mock model
        mock_model = Mock(spec=MlModel)
        mock_model.deploy = Mock(return_value=None)
        mock_model.delete = Mock(return_value=None)
        mock_model.model_id.return_value = "test_model_id"
        self.wrapper.ml_model = mock_model
        
        result = self.wrapper._sparse_pipeline_config()
        
        expected_config = {
            "description": "Pipeline for processing chunks",
            "processors": [
                {
                    "sparse_encoding": {
                        "model_id": "test_model_id",
                        "field_map": {"chunk": "chunk_vector"},
                    }
                }
            ],
        }
        assert result == expected_config

    def test_sparse_pipeline_config_custom_field_map(self):
        """Test _sparse_pipeline_config with custom field map"""
        # Setup mock model
        mock_model = Mock(spec=MlModel)
        mock_model.deploy = Mock(return_value=None)
        mock_model.delete = Mock(return_value=None)
        mock_model.model_id.return_value = "test_model_id"
        self.wrapper.ml_model = mock_model
        
        custom_field_map = {"text": "text_embedding"}
        result = self.wrapper._sparse_pipeline_config(custom_field_map)
        
        expected_config = {
            "description": "Pipeline for processing chunks",
            "processors": [
                {
                    "sparse_encoding": {
                        "model_id": "test_model_id",
                        "field_map": {"text": "text_embedding"},
                    }
                }
            ],
        }
        assert result == expected_config

    def test_setup_for_kNN_dense(self):
        """Test setup_for_kNN method for dense embeddings"""
        # Setup mock model
        mock_model = Mock(spec=MlModel)
        mock_model.deploy = Mock(return_value=None)
        mock_model.delete = Mock(return_value=None)
        mock_model.model_id.return_value = "test_model_id"
        
        # Setup field map
        field_map = {"chunk": "chunk_embedding"}
        
        # Call method
        self.wrapper.setup_for_kNN(
            ml_model=mock_model,
            index_name="test_index",
            pipeline_name="test_pipeline",
            pipeline_field_map=field_map,
            embedding_type="dense"
        )
        
        # Assertions
        assert self.wrapper.ml_model == mock_model
        assert self.wrapper.index_name == "test_index"
        assert self.wrapper.pipeline_name == "test_pipeline"
        
        # Verify pipeline creation
        expected_pipeline_config = {
            "description": "Pipeline for processing chunks",
            "processors": [
                {
                    "text_embedding": {
                        "model_id": "test_model_id",
                        "field_map": field_map,
                    }
                }
            ],
        }
        self.mock_os_client.ingest.put_pipeline.assert_called_once_with(
            "test_pipeline",
            body=expected_pipeline_config
        )

    def test_setup_for_kNN_sparse(self):
        """Test setup_for_kNN method for sparse embeddings"""
        # Setup mock model
        mock_model = Mock(spec=MlModel)
        mock_model.deploy = Mock(return_value=None)
        mock_model.delete = Mock(return_value=None)
        mock_model.model_id.return_value = "test_model_id"
        
        # Setup field map
        field_map = {"chunk": "chunk_embedding"}
        
        # Call method
        self.wrapper.setup_for_kNN(
            ml_model=mock_model,
            index_name="test_index",
            pipeline_name="test_pipeline",
            pipeline_field_map=field_map,
            embedding_type="sparse"
        )
        
        # Verify sparse pipeline creation
        expected_pipeline_config = {
            "description": "Pipeline for processing chunks",
            "processors": [
                {
                    "sparse_encoding": {
                        "model_id": "test_model_id",
                        "field_map": field_map,
                    }
                }
            ],
        }
        self.mock_os_client.ingest.put_pipeline.assert_called_once_with(
            "test_pipeline",
            body=expected_pipeline_config
        )

    def test_setup_for_kNN_invalid_embedding_type(self):
        """Test setup_for_kNN with invalid embedding type (defaults to dense)"""
        # Setup mock model
        mock_model = Mock(spec=MlModel)
        mock_model.deploy = Mock(return_value=None)
        mock_model.delete = Mock(return_value=None)
        mock_model.model_id.return_value = "test_model_id"
        
        # Call method with invalid embedding type (should default to dense)
        self.wrapper.setup_for_kNN(
            ml_model=mock_model,
            index_name="test_index",
            pipeline_name="test_pipeline",
            pipeline_field_map={"chunk": "chunk_embedding"},
            embedding_type="invalid"
        )
        
        # Should create dense pipeline (default behavior)
        expected_pipeline_config = {
            "description": "Pipeline for processing chunks",
            "processors": [
                {
                    "text_embedding": {
                        "model_id": "test_model_id",
                        "field_map": {"chunk": "chunk_embedding"},
                    }
                }
            ],
        }
        self.mock_os_client.ingest.put_pipeline.assert_called_once_with(
            "test_pipeline",
            body=expected_pipeline_config
        )

    @patch('builtins.input', return_value='y')
    def test_cleanup_kNN(self, mock_input):
        """Test cleanup_kNN method"""
        # Setup mock model
        mock_model = Mock(spec=MlModel)
        mock_model.deploy = Mock(return_value=None)
        mock_model.delete = Mock(return_value=None)
        mock_model.clean_up = Mock(return_value=None)
        self.wrapper.ml_model = mock_model
        self.wrapper.index_name = "test_index"
        self.wrapper.pipeline_name = "test_pipeline"
        
        # Call cleanup
        self.wrapper.cleanup_kNN(
            ml_model=mock_model,
            index_name="test_index", 
            pipeline_name="test_pipeline"
        )
        
        # Verify cleanup calls
        mock_model.clean_up.assert_called_once()
        self.mock_model_group.clean_up.assert_called_once()
        self.mock_os_client.ingest.delete_pipeline.assert_called_once_with("test_pipeline")
        self.mock_os_client.indices.delete.assert_called_once_with("test_index")

    @patch('builtins.input', return_value='y')
    def test_cleanup_kNN_with_exceptions(self, mock_input):
        """Test cleanup_kNN method when operations fail"""
        # Setup mock model that raises exception on clean_up
        mock_model = Mock(spec=MlModel)
        mock_model.deploy = Mock(return_value=None)
        mock_model.delete = Mock(return_value=None)
        mock_model.clean_up = Mock(side_effect=Exception("Model cleanup failed"))
        self.wrapper.ml_model = mock_model
        self.wrapper.index_name = "test_index"
        self.wrapper.pipeline_name = "test_pipeline"
        
        # Setup client methods to raise exceptions
        self.mock_os_client.ingest.delete_pipeline.side_effect = Exception("Pipeline delete failed")
        self.mock_os_client.indices.delete.side_effect = Exception("Index delete failed")
        
        # Call cleanup - should raise exception from model cleanup
        with pytest.raises(Exception, match="Model cleanup failed"):
            self.wrapper.cleanup_kNN(
                ml_model=mock_model,
                index_name="test_index", 
                pipeline_name="test_pipeline"
            )
        
        # Verify cleanup was attempted
        mock_model.clean_up.assert_called_once()
        # Model group cleanup should NOT be called since model cleanup failed
        self.mock_model_group.clean_up.assert_not_called()

    @patch('builtins.input', return_value='y')
    def test_cleanup_kNN_no_model(self, mock_input):
        """Test cleanup_kNN when no model is set"""
        # No model set
        self.wrapper.index_name = "test_index"
        self.wrapper.pipeline_name = "test_pipeline"
        
        # Call cleanup
        self.wrapper.cleanup_kNN(
            index_name="test_index", 
            pipeline_name="test_pipeline"
        )
        
        # Verify model group cleanup was called
        self.mock_model_group.clean_up.assert_called_once()
        # Pipeline and index deletion should be called
        self.mock_os_client.ingest.delete_pipeline.assert_called_once_with("test_pipeline")
        self.mock_os_client.indices.delete.assert_called_once_with("test_index")

    @patch('builtins.input', return_value='n')
    def test_cleanup_kNN_empty_names(self, mock_input):
        """Test cleanup_kNN when names are empty"""
        # Setup mock model
        mock_model = Mock(spec=MlModel)
        mock_model.deploy = Mock(return_value=None)
        mock_model.delete = Mock(return_value=None)
        mock_model.clean_up = Mock(return_value=None)
        self.wrapper.ml_model = mock_model
        # Names are empty by default
        
        # Call cleanup
        self.wrapper.cleanup_kNN(ml_model=mock_model)
        
        # Verify model cleanup was called
        mock_model.clean_up.assert_called_once()
        self.mock_model_group.clean_up.assert_called_once()
        # Pipeline and index deletion should not be called since user said 'n'
        self.mock_os_client.ingest.delete_pipeline.assert_not_called()
        self.mock_os_client.indices.delete.assert_not_called()
