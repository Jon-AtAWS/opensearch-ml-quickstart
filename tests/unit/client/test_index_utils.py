# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for client.index_utils module"""

import pytest
from unittest.mock import Mock, patch, MagicMock, call
from opensearchpy import OpenSearch, helpers

from client.index_utils import (
    handle_index_creation,
    handle_data_loading,
    send_bulk_ignore_exceptions,
    get_index_size,
    SPACE_SEPARATOR
)
from data_process import QAndAFileReader


class TestHandleIndexCreation:
    """Test cases for handle_index_creation function"""

    def test_handle_index_creation_new_index(self):
        """Test creating a new index when it doesn't exist"""
        # Setup mock client
        mock_client = Mock(spec=OpenSearch)
        mock_client.indices = Mock()
        mock_client.indices.exists = Mock(return_value=False)
        mock_client.indices.create = Mock(return_value={"acknowledged": True})
        
        # Setup config
        config = {
            "index_name": "test_index",
            "index_settings": {"settings": {"number_of_shards": 1}}
        }
        
        # Call function
        handle_index_creation(mock_client, config, delete_existing=False)
        
        # Assertions - exists might be called multiple times in implementation
        assert mock_client.indices.exists.call_count >= 1
        mock_client.indices.exists.assert_any_call(index="test_index")
        mock_client.indices.create.assert_called_once_with(
            index="test_index", 
            body={"settings": {"number_of_shards": 1}}
        )
        mock_client.indices.delete.assert_not_called()

    def test_handle_index_creation_existing_index_no_delete(self):
        """Test when index exists and delete_existing is False"""
        # Setup mock client
        mock_client = Mock(spec=OpenSearch)
        mock_client.indices = Mock()
        mock_client.indices.exists = Mock(return_value=True)
        
        # Setup config
        config = {
            "index_name": "test_index",
            "index_settings": {"settings": {"number_of_shards": 1}}
        }
        
        # Call function
        handle_index_creation(mock_client, config, delete_existing=False)
        
        # Assertions - exists might be called multiple times in implementation
        assert mock_client.indices.exists.call_count >= 1
        mock_client.indices.exists.assert_any_call(index="test_index")
        mock_client.indices.create.assert_not_called()
        mock_client.indices.delete.assert_not_called()

    def test_handle_index_creation_existing_index_with_delete(self):
        """Test when index exists and delete_existing is True"""
        # Setup mock client
        mock_client = Mock(spec=OpenSearch)
        mock_client.indices = Mock()
        mock_client.indices.exists = Mock(side_effect=[True, False])  # First exists, then doesn't after deletion
        mock_client.indices.delete = Mock(return_value={"acknowledged": True})
        mock_client.indices.create = Mock(return_value={"acknowledged": True})
        
        # Setup config
        config = {
            "index_name": "test_index",
            "index_settings": {"settings": {"number_of_shards": 1}}
        }
        
        # Call function
        handle_index_creation(mock_client, config, delete_existing=True)
        
        # Assertions
        assert mock_client.indices.exists.call_count == 2
        mock_client.indices.delete.assert_called_once_with(index="test_index")
        mock_client.indices.create.assert_called_once_with(
            index="test_index", 
            body={"settings": {"number_of_shards": 1}}
        )

    def test_handle_index_creation_delete_nonexistent_index(self):
        """Test deleting an index that doesn't exist"""
        # Setup mock client
        mock_client = Mock(spec=OpenSearch)
        mock_client.indices = Mock()
        mock_client.indices.exists = Mock(side_effect=[False, False])  # Never exists
        mock_client.indices.create = Mock(return_value={"acknowledged": True})
        
        # Setup config
        config = {
            "index_name": "test_index",
            "index_settings": {"settings": {"number_of_shards": 1}}
        }
        
        # Call function
        handle_index_creation(mock_client, config, delete_existing=True)
        
        # Assertions
        mock_client.indices.delete.assert_not_called()
        mock_client.indices.create.assert_called_once()

    def test_handle_index_creation_create_failure(self):
        """Test when index creation fails"""
        # Setup mock client
        mock_client = Mock(spec=OpenSearch)
        mock_client.indices = Mock()
        mock_client.indices.exists = Mock(return_value=False)
        mock_client.indices.create = Mock(side_effect=Exception("Creation failed"))
        
        # Setup config
        config = {
            "index_name": "test_index",
            "index_settings": {"settings": {"number_of_shards": 1}}
        }
        
        # Call function - should not raise exception, just log error
        handle_index_creation(mock_client, config, delete_existing=False)
        
        # Verify create was attempted
        mock_client.indices.create.assert_called_once()


class TestHandleDataLoading:
    """Test cases for handle_data_loading function"""

    def test_handle_data_loading_normal(self):
        """Test normal data loading"""
        # Setup mocks
        mock_client = Mock(spec=OpenSearch)
        mock_dataset = Mock()
        
        # Mock dataset to return document count
        mock_dataset.load_data.return_value = 100
        
        config = {
            "categories": ["category1"],
            "index_name": "test_index",
            "bulk_send_chunk_size": 1000
        }
        
        # Call function
        handle_data_loading(mock_client, mock_dataset, config, no_load=False)
        
        # Assertions
        mock_dataset.load_data.assert_called_once_with(
            os_client=mock_client,
            index_name="test_index",
            filter_criteria=["category1"],
            bulk_chunk_size=1000
        )

    def test_handle_data_loading_no_load(self):
        """Test data loading when no_load is True"""
        # Setup mocks
        mock_client = Mock(spec=OpenSearch)
        mock_dataset = Mock()
        
        config = {
            "categories": ["category1"],
            "index_name": "test_index"
        }
        
        # Call function
        handle_data_loading(mock_client, mock_dataset, config, no_load=True)
        
        # Assertions - should not attempt to load data
        mock_dataset.load_data.assert_not_called()


class TestSendBulkIgnoreExceptions:
    """Test cases for send_bulk_ignore_exceptions function"""

    @patch('client.index_utils.helpers.bulk')
    def test_send_bulk_success(self, mock_bulk):
        """Test successful bulk send"""
        # Setup mocks
        mock_client = Mock(spec=OpenSearch)
        mock_bulk.return_value = (100, [])
        
        config = {"bulk_send_chunk_size": 1000}
        docs = [{"_index": "test", "_id": "1", "field": "value"}]
        
        # Call function
        result = send_bulk_ignore_exceptions(mock_client, config, docs)
        
        # Assertions
        mock_bulk.assert_called_once_with(
            mock_client,
            docs,
            chunk_size=1000,
            request_timeout=300,
            max_retries=10,
            raise_on_error=False,
        )
        assert result == (100, [])

    @patch('client.index_utils.helpers.bulk')
    def test_send_bulk_exception(self, mock_bulk):
        """Test bulk send with exception"""
        # Setup mocks
        mock_client = Mock(spec=OpenSearch)
        mock_bulk.side_effect = Exception("Bulk failed")
        
        config = {"bulk_send_chunk_size": 1000}
        docs = [{"_index": "test", "_id": "1", "field": "value"}]
        
        # Call function - should not raise exception
        result = send_bulk_ignore_exceptions(mock_client, config, docs)
        
        # Should return None when exception occurs
        assert result is None


class TestGetIndexSize:
    """Test cases for get_index_size function"""

    def test_get_index_size_exists(self):
        """Test getting size of existing index"""
        # Setup mock client
        mock_client = Mock(spec=OpenSearch)
        mock_client.indices = Mock()
        mock_client.indices.exists = Mock(return_value=True)
        mock_client.cat = Mock()
        mock_client.cat.indices = Mock(return_value="1024")
        
        # Call function
        result = get_index_size(mock_client, "test_index", "mb")
        
        # Assertions
        mock_client.indices.exists.assert_called_once_with(index="test_index")
        mock_client.cat.indices.assert_called_once_with(
            index="test_index",
            params={"bytes": "mb", "h": "pri.store.size"}
        )
        assert result == 1024

    def test_get_index_size_not_exists(self):
        """Test getting size of non-existent index"""
        # Setup mock client
        mock_client = Mock(spec=OpenSearch)
        mock_client.indices = Mock()
        mock_client.indices.exists = Mock(return_value=False)
        mock_client.cat = Mock()
        mock_client.cat.indices = Mock()
        
        # Call function
        result = get_index_size(mock_client, "test_index", "mb")
        
        # Assertions
        mock_client.indices.exists.assert_called_once_with(index="test_index")
        mock_client.cat.indices.assert_not_called()
        assert result == 0

    def test_get_index_size_exception(self):
        """Test getting size when cat.indices raises exception"""
        # Setup mock client
        mock_client = Mock(spec=OpenSearch)
        mock_client.indices = Mock()
        mock_client.indices.exists = Mock(return_value=True)
        mock_client.cat = Mock()
        mock_client.cat.indices = Mock(side_effect=Exception("API error"))
        
        # Call function
        result = get_index_size(mock_client, "test_index", "mb")
        
        # Should return 0 when exception occurs
        assert result == 0

    def test_get_index_size_different_units(self):
        """Test getting size with different units"""
        # Setup mock client
        mock_client = Mock(spec=OpenSearch)
        mock_client.indices = Mock()
        mock_client.indices.exists = Mock(return_value=True)
        mock_client.cat = Mock()
        mock_client.cat.indices = Mock(return_value="2048")
        
        # Call function with bytes
        result = get_index_size(mock_client, "test_index", "b")
        
        # Assertions
        mock_client.cat.indices.assert_called_with(
            index="test_index",
            params={"bytes": "b", "h": "pri.store.size"}
        )
        assert result == 2048
