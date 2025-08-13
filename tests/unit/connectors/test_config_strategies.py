# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

import pytest
from unittest.mock import patch
from connectors.config_strategies import (
    SagemakerOSStrategy, SagemakerAOSStrategy,
    BedrockOSStrategy, BedrockAOSStrategy,
    OpenAIOSStrategy, HuggingFaceOSStrategy,
    CONNECTOR_STRATEGIES
)


class TestConnectorStrategies:
    """Test cases for connector strategy classes."""

    @patch('connectors.config_strategies.get_raw_config_value')
    def test_sagemaker_os_strategy(self, mock_get_config):
        """Test SageMaker OS strategy configuration."""
        mock_config_values = {
            "AWS_ACCESS_KEY_ID": "test_key",
            "AWS_SECRET_ACCESS_KEY": "test_secret",
            "AWS_REGION": "us-west-2",
            "SAGEMAKER_CONNECTOR_VERSION": "1.0",
            "SAGEMAKER_SPARSE_URL": "sparse_url",
            "SAGEMAKER_DENSE_URL": "dense_url",
            "SAGEMAKER_DENSE_MODEL_DIMENSION": "384",
            "SAGEMAKER_PAYLOAD_FILE": None
        }
        mock_get_config.side_effect = lambda key: mock_config_values.get(key)
        
        strategy = SagemakerOSStrategy()
        config = strategy.get_config()
        
        assert config["access_key"] == "test_key"
        assert config["secret_key"] == "test_secret"
        assert len(strategy.get_required_fields()) == 7
        assert strategy.get_payload_filename("dense") == "sagemaker_dense.json"

    @patch('connectors.config_strategies.get_raw_config_value')
    def test_bedrock_os_strategy(self, mock_get_config):
        """Test Bedrock OS strategy configuration."""
        mock_config_values = {
            "AWS_ACCESS_KEY_ID": "test_key",
            "AWS_SECRET_ACCESS_KEY": "test_secret", 
            "AWS_REGION": "us-west-2",
            "BEDROCK_CONNECTOR_VERSION": "1.0",
            "BEDROCK_EMBEDDING_URL": "bedrock_url",
            "BEDROCK_MODEL_DIMENSION": "1536",
            "BEDROCK_PAYLOAD_FILE": None,
            "BEDROCK_LLM_PAYLOAD_FILE": None
        }
        mock_get_config.side_effect = lambda key: mock_config_values.get(key)
        
        strategy = BedrockOSStrategy()
        
        assert strategy.get_payload_filename("dense") == "bedrock_dense.json"
        assert strategy.get_payload_filename("llm") == "claude_3.5_sonnet_v2.json"
        
        with pytest.raises(ValueError, match="Bedrock doesn't support sparse embeddings"):
            strategy.get_payload_filename("sparse")

    @patch('connectors.config_strategies.get_raw_config_value')
    def test_custom_payload_filename(self, mock_get_config):
        """Test custom payload filename override."""
        mock_get_config.side_effect = lambda key: "custom_payload.json" if "PAYLOAD_FILE" in key else None
        
        strategy = SagemakerOSStrategy()
        assert strategy.get_payload_filename("dense") == "custom_payload.json"

    def test_strategy_registry(self):
        """Test that all expected strategies are registered."""
        expected_combinations = [
            ("sagemaker", "os"), ("sagemaker", "aos"),
            ("bedrock", "os"), ("bedrock", "aos"),
            ("openai", "os"), ("huggingface", "os")
        ]
        
        for combo in expected_combinations:
            assert combo in CONNECTOR_STRATEGIES
            
        assert len(CONNECTOR_STRATEGIES) == 6

    @patch('connectors.config_strategies.get_raw_config_value')
    def test_openai_strategy(self, mock_get_config):
        """Test OpenAI strategy configuration."""
        mock_config_values = {
            "OPENAI_API_KEY": "test_key",
            "OPENAI_MODEL_NAME": "gpt-4",
            "OPENAI_PAYLOAD_FILE": None
        }
        mock_get_config.side_effect = lambda key: mock_config_values.get(key)
        
        strategy = OpenAIOSStrategy()
        config = strategy.get_config()
        
        assert config["api_key"] == "test_key"
        assert config["model_name"] == "gpt-4"
        assert strategy.get_required_fields() == ["api_key", "model_name"]
        assert strategy.get_payload_filename() == "openai_llm.json"
