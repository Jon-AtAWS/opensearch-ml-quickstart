# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for client.helper module"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from opensearchpy import OpenSearch

from client.helper import get_client, parse_version, check_client_version


class TestGetClient:
    """Test cases for get_client function"""

    @patch('client.helper.get_opensearch_config')
    @patch('client.helper.OpenSearch')
    @patch('client.helper.check_client_version')
    def test_get_client_os_with_port(self, mock_check_version, mock_opensearch, mock_get_config):
        """Test get_client for self-managed OpenSearch with port"""
        # Setup mock config
        mock_config = Mock()
        mock_config.host_url = "localhost"
        mock_config.port = 9200
        mock_config.username = "admin"
        mock_config.password = "password"
        mock_get_config.return_value = mock_config
        
        # Setup mock client
        mock_client = Mock(spec=OpenSearch)
        mock_client.cluster = Mock()
        mock_client.cluster.put_settings = Mock()
        mock_opensearch.return_value = mock_client
        
        # Call function
        result = get_client("os")
        
        # Assertions
        mock_get_config.assert_called_once_with("os")
        mock_opensearch.assert_called_once_with(
            hosts=[{"host": "localhost", "port": 9200}],
            http_auth=("admin", "password"),
            use_ssl=True,
            verify_certs=False,
            ssl_assert_hostname=False,
            ssl_show_warn=False,
            timeout=300,
        )
        mock_check_version.assert_called_once_with(mock_client)
        
        # Verify cluster settings were called
        expected_settings = {
            "plugins.ml_commons.memory_feature_enabled": True,
            "plugins.ml_commons.agent_framework_enabled": True,
            "plugins.ml_commons.rag_pipeline_feature_enabled": True,
            "plugins.ml_commons.trusted_connector_endpoints_regex": [
                "^https://bedrock-runtime\\..*[a-z0-9-]\\.amazonaws\\.com/.*$",
                "^https://runtime\\.sagemaker\\..*[a-z0-9-]\\.amazonaws\\.com/.*$"
            ],
            "plugins.ml_commons.allow_registering_model_via_url": True,
            "plugins.ml_commons.only_run_on_ml_node": True,
        }
        mock_client.cluster.put_settings.assert_called_once_with(
            body={"persistent": expected_settings}
        )
        
        assert result == mock_client

    @patch('client.helper.get_opensearch_config')
    @patch('client.helper.OpenSearch')
    @patch('client.helper.check_client_version')
    def test_get_client_aos_without_port(self, mock_check_version, mock_opensearch, mock_get_config):
        """Test get_client for Amazon OpenSearch Service without port"""
        # Setup mock config
        mock_config = Mock()
        mock_config.host_url = "https://search-domain.us-west-2.es.amazonaws.com"
        mock_config.port = None
        mock_config.username = "admin"
        mock_config.password = "password"
        mock_get_config.return_value = mock_config
        
        # Setup mock client
        mock_client = Mock(spec=OpenSearch)
        mock_client.cluster = Mock()
        mock_client.cluster.put_settings = Mock()
        mock_opensearch.return_value = mock_client
        
        # Call function
        result = get_client("aos")
        
        # Assertions
        mock_get_config.assert_called_once_with("aos")
        mock_opensearch.assert_called_once_with(
            hosts="https://search-domain.us-west-2.es.amazonaws.com",
            http_auth=("admin", "password"),
            use_ssl=True,
            verify_certs=False,
            ssl_assert_hostname=False,
            ssl_show_warn=False,
            timeout=300,
        )
        
        # Verify AOS-specific settings (no local model settings)
        expected_settings = {
            "plugins.ml_commons.memory_feature_enabled": True,
            "plugins.ml_commons.agent_framework_enabled": True,
            "plugins.ml_commons.rag_pipeline_feature_enabled": True,
            "plugins.ml_commons.trusted_connector_endpoints_regex": [
                "^https://bedrock-runtime\\..*[a-z0-9-]\\.amazonaws\\.com/.*$",
                "^https://runtime\\.sagemaker\\..*[a-z0-9-]\\.amazonaws\\.com/.*$"
            ]
        }
        mock_client.cluster.put_settings.assert_called_once_with(
            body={"persistent": expected_settings}
        )
        
        assert result == mock_client

    @patch('client.helper.get_opensearch_config')
    @patch('client.helper.OpenSearch')
    @patch('client.helper.check_client_version')
    def test_get_client_cluster_settings_failure(self, mock_check_version, mock_opensearch, mock_get_config):
        """Test get_client when cluster settings fail"""
        # Setup mock config
        mock_config = Mock()
        mock_config.host_url = "localhost"
        mock_config.port = 9200
        mock_config.username = "admin"
        mock_config.password = "password"
        mock_get_config.return_value = mock_config
        
        # Setup mock client with failing cluster settings
        mock_client = Mock(spec=OpenSearch)
        mock_client.cluster = Mock()
        mock_client.cluster.put_settings = Mock()
        mock_client.cluster.put_settings.side_effect = Exception("Cluster settings failed")
        mock_opensearch.return_value = mock_client
        
        # Call function and expect exception
        with pytest.raises(Exception, match="Cluster settings failed"):
            get_client("os")


class TestParseVersion:
    """Test cases for parse_version function"""

    def test_parse_version_simple(self):
        """Test parsing simple version string"""
        result = parse_version("2.13.0")
        assert result == (2, 13, 0)

    def test_parse_version_complex(self):
        """Test parsing complex version string"""
        result = parse_version("2.16.1")
        assert result == (2, 16, 1)

    def test_parse_version_single_digit(self):
        """Test parsing single digit version"""
        result = parse_version("1.0.0")
        assert result == (1, 0, 0)


class TestCheckClientVersion:
    """Test cases for check_client_version function"""

    @patch('client.helper.get_minimum_opensearch_version')
    def test_check_client_version_valid(self, mock_get_min_version):
        """Test check_client_version with valid version"""
        # Setup mock client
        mock_client = Mock(spec=OpenSearch)
        mock_client.info.return_value = {
            "version": {"number": "2.16.0"}
        }
        mock_get_min_version.return_value = "2.13.0"
        
        # Should not raise exception
        check_client_version(mock_client)
        
        mock_client.info.assert_called_once()
        mock_get_min_version.assert_called_once()

    @patch('client.helper.get_minimum_opensearch_version')
    def test_check_client_version_invalid(self, mock_get_min_version):
        """Test check_client_version with invalid version"""
        # Setup mock client
        mock_client = Mock(spec=OpenSearch)
        mock_client.info.return_value = {
            "version": {"number": "2.10.0"}
        }
        mock_get_min_version.return_value = "2.13.0"
        
        # Should raise ValueError
        with pytest.raises(ValueError, match="The minimum required version for opensearch cluster is 2.13.0"):
            check_client_version(mock_client)

    @patch('client.helper.get_minimum_opensearch_version')
    def test_check_client_version_equal(self, mock_get_min_version):
        """Test check_client_version with equal version"""
        # Setup mock client
        mock_client = Mock(spec=OpenSearch)
        mock_client.info.return_value = {
            "version": {"number": "2.13.0"}
        }
        mock_get_min_version.return_value = "2.13.0"
        
        # Should not raise exception
        check_client_version(mock_client)
