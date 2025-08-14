# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

import pytest
from unittest.mock import Mock, patch
from connectors.llm_connector import LlmConnector


class TestLlmConnectorModelName:
    """Test cases for LlmConnector model name functionality."""

    def test_fill_payload_includes_default_model_name(self):
        """Test _fill_in_connector_create_payload includes default model name."""
        # Create a minimal connector instance without full initialization
        connector = object.__new__(LlmConnector)
        connector._model_name = "anthropic.claude-3-5-sonnet-20241022-v2:0"
        connector._connector_name = "Test Connector"
        connector._connector_description = "Test Description"
        connector._connector_configs = {
            "region": "us-west-2",
            "access_key": "test_key",
            "secret_key": "test_secret"
        }
        connector._os_type = "os"
        
        # Mock payload structure
        payload = {
            "parameters": {}
        }
        
        connector._fill_in_connector_create_payload(payload)
        
        assert payload["parameters"]["model"] == "anthropic.claude-3-5-sonnet-20241022-v2:0"

    def test_fill_payload_includes_custom_model_name(self):
        """Test _fill_in_connector_create_payload includes custom model name."""
        # Create a minimal connector instance without full initialization
        connector = object.__new__(LlmConnector)
        connector._model_name = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
        connector._connector_name = "Test Connector"
        connector._connector_description = "Test Description"
        connector._connector_configs = {
            "region": "us-west-2",
            "access_key": "test_key",
            "secret_key": "test_secret"
        }
        connector._os_type = "os"
        
        # Mock payload structure
        payload = {
            "parameters": {}
        }
        
        connector._fill_in_connector_create_payload(payload)
        
        assert payload["parameters"]["model"] == "us.anthropic.claude-3-7-sonnet-20250219-v1:0"

    def test_fill_payload_includes_all_required_fields(self):
        """Test _fill_in_connector_create_payload includes all required fields."""
        # Create a minimal connector instance without full initialization
        connector = object.__new__(LlmConnector)
        connector._model_name = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
        connector._connector_name = "Test Connector"
        connector._connector_description = "Test Description"
        connector._connector_configs = {
            "region": "us-west-2",
            "access_key": "test_key",
            "secret_key": "test_secret",
            "connector_version": "2"
        }
        connector._os_type = "os"
        
        # Mock payload structure
        payload = {
            "parameters": {}
        }
        
        connector._fill_in_connector_create_payload(payload)
        
        # Verify all fields are set correctly
        assert payload["name"] == "Test Connector"
        assert payload["description"] == "Test Description"
        assert payload["parameters"]["region"] == "us-west-2"
        assert payload["parameters"]["model"] == "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
        assert payload["version"] == "2"
        assert payload["credential"]["access_key"] == "test_key"
        assert payload["credential"]["secret_key"] == "test_secret"
