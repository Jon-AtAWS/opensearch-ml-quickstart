# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for models.ml_model_group module"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from opensearchpy import OpenSearch
from opensearch_py_ml import ml_commons

from models.ml_model_group import MlModelGroup


class TestMlModelGroup:
    """Test cases for MlModelGroup class"""

    def setup_method(self):
        """Setup method run before each test"""
        # Create mock clients
        self.mock_os_client = Mock(spec=OpenSearch)
        self.mock_ml_commons_client = Mock(spec=ml_commons)
        
        # Mock HTTP responses
        self.mock_os_client.http = Mock()

    @patch('models.ml_model_group.logging')
    def test_ml_model_group_init_with_defaults(self, mock_logging):
        """Test MlModelGroup initialization with default values"""
        with patch.object(MlModelGroup, '_get_model_group_id', return_value='default_group_id'):
            group = MlModelGroup(
                os_client=self.mock_os_client,
                ml_commons_client=self.mock_ml_commons_client
            )
            
            assert group._os_client == self.mock_os_client
            assert group._ml_commons_client == self.mock_ml_commons_client
            assert group._model_group_name == MlModelGroup.DEFAULT_GROUP_NAME
            assert group._model_group_id == 'default_group_id'

    @patch('models.ml_model_group.logging')
    def test_ml_model_group_init_with_custom_name(self, mock_logging):
        """Test MlModelGroup initialization with custom group name"""
        custom_name = "custom_model_group"
        
        with patch.object(MlModelGroup, '_get_model_group_id', return_value='custom_group_id'):
            group = MlModelGroup(
                os_client=self.mock_os_client,
                ml_commons_client=self.mock_ml_commons_client,
                model_group_name=custom_name
            )
            
            assert group._model_group_name == custom_name
            assert group._model_group_id == 'custom_group_id'

    def test_str_and_repr(self):
        """Test string representation methods"""
        with patch.object(MlModelGroup, '_get_model_group_id', return_value='str_test_id'):
            group = MlModelGroup(
                os_client=self.mock_os_client,
                ml_commons_client=self.mock_ml_commons_client,
                model_group_name="Test Group"
            )
            
            expected_str = "<MlModelGroup Test Group str_test_id>"
            assert str(group) == expected_str
            assert repr(group) == expected_str

    def test_model_group_id(self):
        """Test model_group_id method"""
        with patch.object(MlModelGroup, '_get_model_group_id', return_value='test_group_id'):
            group = MlModelGroup(
                os_client=self.mock_os_client,
                ml_commons_client=self.mock_ml_commons_client
            )
            
            assert group.model_group_id() == 'test_group_id'

    @patch('models.ml_model_group.logging')
    def test_get_model_group_id_existing_group(self, mock_logging):
        """Test _get_model_group_id when group already exists"""
        with patch.object(MlModelGroup, '_find_model_group_id', return_value='existing_id'):
            group = MlModelGroup(
                os_client=self.mock_os_client,
                ml_commons_client=self.mock_ml_commons_client
            )
            
            assert group._model_group_id == 'existing_id'

    @patch('models.ml_model_group.logging')
    def test_get_model_group_id_new_group(self, mock_logging):
        """Test _get_model_group_id when group needs to be created"""
        with patch.object(MlModelGroup, '_find_model_group_id', return_value=None):
            with patch.object(MlModelGroup, '_register_model_group', return_value='new_group_id'):
                group = MlModelGroup(
                    os_client=self.mock_os_client,
                    ml_commons_client=self.mock_ml_commons_client
                )
                
                assert group._model_group_id == 'new_group_id'

    @patch('models.ml_model_group.get_ml_base_uri')
    @patch('models.ml_model_group.logging')
    def test_register_model_group(self, mock_logging, mock_get_ml_base_uri):
        """Test _register_model_group method"""
        mock_get_ml_base_uri.return_value = "/_plugins/_ml"
        register_response = {"model_group_id": "registered_group_id"}
        self.mock_os_client.http.post.return_value = register_response
        
        with patch.object(MlModelGroup, '_get_model_group_id', return_value='test_id'):
            group = MlModelGroup(
                os_client=self.mock_os_client,
                ml_commons_client=self.mock_ml_commons_client,
                model_group_name="Test Register Group"
            )
            
            # Call _register_model_group directly
            result = group._register_model_group()
            
            assert result == "registered_group_id"
            
            # Verify the HTTP call
            expected_body = {
                "name": "Test Register Group",
                "description": "This is a public model group",
            }
            self.mock_os_client.http.post.assert_called_once_with(
                url="/_plugins/_ml/model_groups/_register",
                body=expected_body,
            )

    @patch('models.ml_model_group.get_ml_base_uri')
    @patch('models.ml_model_group.logging')
    def test_get_all_model_groups_success(self, mock_logging, mock_get_ml_base_uri):
        """Test _get_all_model_groups with successful response"""
        mock_get_ml_base_uri.return_value = "/_plugins/_ml"
        search_response = {
            "hits": {
                "hits": [
                    {
                        "_id": "group1",
                        "_source": {"name": "Group 1"}
                    },
                    {
                        "_id": "group2", 
                        "_source": {"name": "Group 2"}
                    }
                ]
            }
        }
        self.mock_os_client.http.get.return_value = search_response
        
        with patch.object(MlModelGroup, '_get_model_group_id', return_value='test_id'):
            group = MlModelGroup(
                os_client=self.mock_os_client,
                ml_commons_client=self.mock_ml_commons_client
            )
            
            result = group._get_all_model_groups()
            
            assert len(result) == 2
            assert result[0]["_id"] == "group1"
            assert result[1]["_id"] == "group2"
            
            # Verify the HTTP call
            self.mock_os_client.http.get.assert_called_once_with(
                url="/_plugins/_ml/model_groups/_search",
                body={"size": 10000}
            )

    @patch('models.ml_model_group.logging')
    def test_get_all_model_groups_empty_response(self, mock_logging):
        """Test _get_all_model_groups with empty response"""
        self.mock_os_client.http.get.return_value = None
        
        with patch.object(MlModelGroup, '_get_model_group_id', return_value='test_id'):
            group = MlModelGroup(
                os_client=self.mock_os_client,
                ml_commons_client=self.mock_ml_commons_client
            )
            
            result = group._get_all_model_groups()
            assert result == []

    @patch('models.ml_model_group.logging')
    def test_get_all_model_groups_exception(self, mock_logging):
        """Test _get_all_model_groups with exception"""
        self.mock_os_client.http.get.side_effect = Exception("Search failed")
        
        with patch.object(MlModelGroup, '_get_model_group_id', return_value='test_id'):
            group = MlModelGroup(
                os_client=self.mock_os_client,
                ml_commons_client=self.mock_ml_commons_client
            )
            
            result = group._get_all_model_groups()
            assert result == []

    def test_get_all_model_group_ids(self):
        """Test _get_all_model_group_ids method"""
        mock_groups = [
            {"_id": "id1", "_source": {"name": "Group 1"}},
            {"_id": "id2", "_source": {"name": "Group 2"}},
            {"_id": "id3", "_source": {"name": "Group 3"}}
        ]
        
        with patch.object(MlModelGroup, '_get_model_group_id', return_value='test_id'):
            with patch.object(MlModelGroup, '_get_all_model_groups', return_value=mock_groups):
                group = MlModelGroup(
                    os_client=self.mock_os_client,
                    ml_commons_client=self.mock_ml_commons_client
                )
                
                result = group._get_all_model_group_ids()
                
                assert result == ["id1", "id2", "id3"]

    def test_find_model_group_id_found(self):
        """Test _find_model_group_id when group is found"""
        mock_groups = [
            {"_id": "id1", "_source": {"name": "Other Group"}},
            {"_id": "target_id", "_source": {"name": "Target Group"}},
            {"_id": "id3", "_source": {"name": "Another Group"}}
        ]
        
        with patch.object(MlModelGroup, '_get_model_group_id', return_value='test_id'):
            with patch.object(MlModelGroup, '_get_all_model_groups', return_value=mock_groups):
                group = MlModelGroup(
                    os_client=self.mock_os_client,
                    ml_commons_client=self.mock_ml_commons_client,
                    model_group_name="Target Group"
                )
                
                # Call _find_model_group_id directly
                result = group._find_model_group_id()
                
                assert result == "target_id"

    def test_find_model_group_id_not_found(self):
        """Test _find_model_group_id when group is not found"""
        mock_groups = [
            {"_id": "id1", "_source": {"name": "Other Group"}},
            {"_id": "id2", "_source": {"name": "Another Group"}}
        ]
        
        with patch.object(MlModelGroup, '_get_model_group_id', return_value='test_id'):
            with patch.object(MlModelGroup, '_get_all_model_groups', return_value=mock_groups):
                group = MlModelGroup(
                    os_client=self.mock_os_client,
                    ml_commons_client=self.mock_ml_commons_client,
                    model_group_name="Nonexistent Group"
                )
                
                # Call _find_model_group_id directly
                result = group._find_model_group_id()
                
                assert result is None

    @patch('builtins.input', return_value='y')
    @patch('models.ml_model_group.get_ml_base_uri')
    @patch('models.ml_model_group.logging')
    def test_delete_model_group_success(self, mock_logging, mock_get_ml_base_uri, mock_input):
        """Test successful model group deletion"""
        mock_get_ml_base_uri.return_value = "/_plugins/_ml"
        
        with patch.object(MlModelGroup, '_get_model_group_id', return_value='delete_test_id'):
            group = MlModelGroup(
                os_client=self.mock_os_client,
                ml_commons_client=self.mock_ml_commons_client
            )
            
            # Call _delete_model_group directly
            group._delete_model_group('delete_test_id')
            
            # Verify the HTTP call
            self.mock_os_client.http.delete.assert_called_once_with(
                url="/_plugins/_ml/model_groups/delete_test_id",
                body={},
            )

    @patch('builtins.input', return_value='n')
    @patch('models.ml_model_group.logging')
    def test_delete_model_group_cancelled(self, mock_logging, mock_input):
        """Test cancelled model group deletion"""
        with patch.object(MlModelGroup, '_get_model_group_id', return_value='cancel_test_id'):
            group = MlModelGroup(
                os_client=self.mock_os_client,
                ml_commons_client=self.mock_ml_commons_client
            )
            
            # Call _delete_model_group directly
            group._delete_model_group('cancel_test_id')
            
            # Verify no HTTP call was made
            self.mock_os_client.http.delete.assert_not_called()

    @patch('builtins.input', return_value='y')
    @patch('models.ml_model_group.get_ml_base_uri')
    @patch('models.ml_model_group.logging')
    def test_delete_model_group_exception(self, mock_logging, mock_get_ml_base_uri, mock_input):
        """Test model group deletion with exception"""
        mock_get_ml_base_uri.return_value = "/_plugins/_ml"
        self.mock_os_client.http.delete.side_effect = Exception("Delete failed")
        
        with patch.object(MlModelGroup, '_get_model_group_id', return_value='exception_test_id'):
            group = MlModelGroup(
                os_client=self.mock_os_client,
                ml_commons_client=self.mock_ml_commons_client
            )
            
            # Should not raise exception, just log error
            group._delete_model_group('exception_test_id')
            
            # Verify the HTTP call was attempted
            self.mock_os_client.http.delete.assert_called_once()

    @patch('models.ml_model_group.logging')
    def test_clean_up_with_model_group_id(self, mock_logging):
        """Test clean_up when model group ID exists"""
        with patch.object(MlModelGroup, '_get_model_group_id', return_value='cleanup_test_id'):
            with patch.object(MlModelGroup, '_delete_model_group') as mock_delete:
                group = MlModelGroup(
                    os_client=self.mock_os_client,
                    ml_commons_client=self.mock_ml_commons_client
                )
                
                group.clean_up()
                mock_delete.assert_called_once_with('cleanup_test_id')

    @patch('models.ml_model_group.logging')
    def test_clean_up_without_model_group_id(self, mock_logging):
        """Test clean_up when model group ID is None"""
        with patch.object(MlModelGroup, '_get_model_group_id', return_value=None):
            with patch.object(MlModelGroup, '_delete_model_group') as mock_delete:
                group = MlModelGroup(
                    os_client=self.mock_os_client,
                    ml_commons_client=self.mock_ml_commons_client
                )
                
                group.clean_up()
                mock_delete.assert_not_called()

    def test_default_group_name_constant(self):
        """Test DEFAULT_GROUP_NAME constant"""
        assert hasattr(MlModelGroup, 'DEFAULT_GROUP_NAME')
        assert isinstance(MlModelGroup.DEFAULT_GROUP_NAME, str)
        assert len(MlModelGroup.DEFAULT_GROUP_NAME) > 0

    @patch('models.ml_model_group.get_delete_resource_retry_time', return_value=3)
    @patch('models.ml_model_group.get_delete_resource_wait_time', return_value=1)
    def test_retry_configuration(self, mock_wait_time, mock_retry_time):
        """Test that retry configuration is properly applied"""
        with patch.object(MlModelGroup, '_get_model_group_id', return_value='retry_test_id'):
            group = MlModelGroup(
                os_client=self.mock_os_client,
                ml_commons_client=self.mock_ml_commons_client
            )
            
            # Verify that the configuration functions are called
            # This is tested indirectly through the decorator application
            assert hasattr(group._delete_model_group, '__wrapped__')

    @patch('models.ml_model_group.logging')
    def test_integration_workflow(self, mock_logging):
        """Test complete workflow from initialization to cleanup"""
        # Mock the workflow: no existing group -> register new -> cleanup
        with patch.object(MlModelGroup, '_find_model_group_id', return_value=None):
            with patch.object(MlModelGroup, '_register_model_group', return_value='workflow_id'):
                with patch.object(MlModelGroup, '_delete_model_group') as mock_delete:
                    group = MlModelGroup(
                        os_client=self.mock_os_client,
                        ml_commons_client=self.mock_ml_commons_client,
                        model_group_name="Workflow Test Group"
                    )
                    
                    # Verify initialization
                    assert group._model_group_name == "Workflow Test Group"
                    assert group._model_group_id == 'workflow_id'
                    
                    # Test cleanup
                    group.clean_up()
                    mock_delete.assert_called_once_with('workflow_id')

    @patch('models.ml_model_group.logging')
    def test_get_all_model_groups_malformed_response(self, mock_logging):
        """Test _get_all_model_groups with malformed response"""
        # Test response without hits
        malformed_response = {"total": 0}
        self.mock_os_client.http.get.return_value = malformed_response
        
        with patch.object(MlModelGroup, '_get_model_group_id', return_value='test_id'):
            group = MlModelGroup(
                os_client=self.mock_os_client,
                ml_commons_client=self.mock_ml_commons_client
            )
            
            result = group._get_all_model_groups()
            assert result == []

    @patch('models.ml_model_group.logging')
    def test_get_all_model_groups_nested_hits_missing(self, mock_logging):
        """Test _get_all_model_groups with missing nested hits"""
        # Test response with hits but no nested hits
        response_no_nested_hits = {"hits": {"total": 0}}
        self.mock_os_client.http.get.return_value = response_no_nested_hits
        
        with patch.object(MlModelGroup, '_get_model_group_id', return_value='test_id'):
            group = MlModelGroup(
                os_client=self.mock_os_client,
                ml_commons_client=self.mock_ml_commons_client
            )
            
            result = group._get_all_model_groups()
            assert result == []

    def test_find_model_group_id_empty_groups(self):
        """Test _find_model_group_id with empty groups list"""
        with patch.object(MlModelGroup, '_get_model_group_id', return_value='test_id'):
            with patch.object(MlModelGroup, '_get_all_model_groups', return_value=[]):
                group = MlModelGroup(
                    os_client=self.mock_os_client,
                    ml_commons_client=self.mock_ml_commons_client,
                    model_group_name="Any Group"
                )
                
                # Call _find_model_group_id directly
                result = group._find_model_group_id()
                
                assert result is None

    @patch('models.ml_model_group.logging')
    def test_model_group_name_case_sensitivity(self, mock_logging):
        """Test that model group name matching is case sensitive"""
        mock_groups = [
            {"_id": "id1", "_source": {"name": "test group"}},
            {"_id": "id2", "_source": {"name": "Test Group"}},
            {"_id": "id3", "_source": {"name": "TEST GROUP"}}
        ]
        
        with patch.object(MlModelGroup, '_get_model_group_id', return_value='test_id'):
            with patch.object(MlModelGroup, '_get_all_model_groups', return_value=mock_groups):
                group = MlModelGroup(
                    os_client=self.mock_os_client,
                    ml_commons_client=self.mock_ml_commons_client,
                    model_group_name="Test Group"  # Exact case match
                )
                
                # Call _find_model_group_id directly
                result = group._find_model_group_id()
                
                assert result == "id2"  # Should match exact case
