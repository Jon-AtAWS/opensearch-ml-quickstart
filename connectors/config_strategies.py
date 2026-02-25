# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

"""
Connector configuration strategies for different connector types and host types.
"""

from abc import ABC, abstractmethod
from typing import Dict, List
from configs.configuration_manager import get_raw_config_value


class ConnectorConfigStrategy(ABC):
    """Abstract base class for connector configuration strategies."""
    
    @abstractmethod
    def get_config(self) -> Dict[str, str]:
        """Get the configuration for this connector type and host type."""
        pass
    
    @abstractmethod
    def get_required_fields(self) -> List[str]:
        """Get the list of required configuration fields."""
        pass
    
    @abstractmethod
    def get_payload_filename(self, model_type: str = None) -> str:
        """Get the connector payload filename for this strategy.
        
        Args:
            model_type: Type of model - 'dense', 'sparse', or 'llm'
        """
        pass


class SagemakerOSStrategy(ConnectorConfigStrategy):
    """Configuration strategy for SageMaker on self-managed OpenSearch."""
    
    def get_config(self) -> Dict[str, str]:
        return {
            "access_key": get_raw_config_value("AWS_ACCESS_KEY_ID"),
            "secret_key": get_raw_config_value("AWS_SECRET_ACCESS_KEY"),
            "region": get_raw_config_value("AWS_REGION"),
            "connector_version": get_raw_config_value("SAGEMAKER_CONNECTOR_VERSION"),
            "sparse_url": get_raw_config_value("SAGEMAKER_SPARSE_URL"),
            "dense_url": get_raw_config_value("SAGEMAKER_DENSE_URL"),
            "model_dimensions": get_raw_config_value("SAGEMAKER_DENSE_MODEL_DIMENSION"),
        }
    
    def get_required_fields(self) -> List[str]:
        return ["access_key", "secret_key", "region", "connector_version", "sparse_url", "dense_url", "model_dimensions"]
    
    def get_payload_filename(self, model_type: str = None) -> str:
        return get_raw_config_value("SAGEMAKER_PAYLOAD_FILE") or f"sagemaker_{model_type}.json"


class SagemakerAOSStrategy(ConnectorConfigStrategy):
    """Configuration strategy for SageMaker on Amazon OpenSearch Service."""
    
    def get_config(self) -> Dict[str, str]:
        return {
            "dense_arn": get_raw_config_value("SAGEMAKER_DENSE_ARN"),
            "sparse_arn": get_raw_config_value("SAGEMAKER_SPARSE_ARN"),
            "connector_role_name": get_raw_config_value("SAGEMAKER_CONNECTOR_ROLE_NAME"),
            "create_connector_role_name": get_raw_config_value("SAGEMAKER_CREATE_CONNECTOR_ROLE_NAME"),
            "region": get_raw_config_value("AWS_REGION"),
            "connector_version": get_raw_config_value("SAGEMAKER_CONNECTOR_VERSION"),
            "sparse_url": get_raw_config_value("SAGEMAKER_SPARSE_URL"),
            "dense_url": get_raw_config_value("SAGEMAKER_DENSE_URL"),
            "model_dimensions": get_raw_config_value("SAGEMAKER_DENSE_MODEL_DIMENSION"),
        }
    
    def get_required_fields(self) -> List[str]:
        return list(self.get_config().keys())
    
    def get_payload_filename(self, model_type: str = None) -> str:
        return get_raw_config_value("SAGEMAKER_PAYLOAD_FILE") or f"sagemaker_{model_type}.json"


class BedrockOSStrategy(ConnectorConfigStrategy):
    """Configuration strategy for Bedrock on self-managed OpenSearch."""
    
    def get_config(self) -> Dict[str, str]:
        return {
            "access_key": get_raw_config_value("AWS_ACCESS_KEY_ID"),
            "secret_key": get_raw_config_value("AWS_SECRET_ACCESS_KEY"),
            "region": get_raw_config_value("AWS_REGION"),
            "connector_version": get_raw_config_value("BEDROCK_CONNECTOR_VERSION"),
            "dense_url": get_raw_config_value("BEDROCK_EMBEDDING_URL"),
            "model_dimensions": get_raw_config_value("BEDROCK_MODEL_DIMENSION"),
        }
    
    def get_required_fields(self) -> List[str]:
        return list(self.get_config().keys())
    
    def get_payload_filename(self, model_type: str = None) -> str:
        if model_type == "llm_predict":
            return get_raw_config_value("BEDROCK_LLM_PREDICT_PAYLOAD_FILE") or "bedrock_llm_predict.json"
        elif model_type == "llm_converse":
            return get_raw_config_value("BEDROCK_LLM_CONVERSE_PAYLOAD_FILE") or "bedrock_llm_converse.json"
        elif model_type == "llm_memory":
            return get_raw_config_value("BEDROCK_LLM_MEMORY_PAYLOAD_FILE") or "bedrock_llm_memory.json"
        elif model_type == "sparse":
            raise ValueError("Bedrock doesn't support sparse embeddings")
        return get_raw_config_value("BEDROCK_PAYLOAD_FILE") or "bedrock_dense.json"


class BedrockAOSStrategy(ConnectorConfigStrategy):
    """Configuration strategy for Bedrock on Amazon OpenSearch Service."""
    
    def get_config(self) -> Dict[str, str]:
        return {
            "dense_arn": get_raw_config_value("BEDROCK_ARN"),
            "connector_role_name": get_raw_config_value("BEDROCK_CONNECTOR_ROLE_NAME"),
            "create_connector_role_name": get_raw_config_value("BEDROCK_CREATE_CONNECTOR_ROLE_NAME"),
            "region": get_raw_config_value("AWS_REGION"),
            "connector_version": get_raw_config_value("BEDROCK_CONNECTOR_VERSION"),
            "model_dimensions": get_raw_config_value("BEDROCK_MODEL_DIMENSION"),
            "dense_url": get_raw_config_value("BEDROCK_EMBEDDING_URL"),
        }
    
    def get_required_fields(self) -> List[str]:
        return list(self.get_config().keys())
    
    def get_payload_filename(self, model_type: str = None) -> str:
        if model_type == "llm_predict":
            return get_raw_config_value("BEDROCK_LLM_PREDICT_PAYLOAD_FILE") or "bedrock_llm_predict.json"
        elif model_type == "llm_converse":
            return get_raw_config_value("BEDROCK_LLM_CONVERSE_PAYLOAD_FILE") or "bedrock_llm_converse.json"
        elif model_type == "llm_memory":
            return get_raw_config_value("BEDROCK_LLM_MEMORY_PAYLOAD_FILE") or "bedrock_llm_memory.json"
        elif model_type == "sparse":
            raise ValueError("Bedrock doesn't support sparse embeddings")
        return get_raw_config_value("BEDROCK_PAYLOAD_FILE") or "bedrock_dense.json"


# Example of how to add new LLM connectors
class OpenAIOSStrategy(ConnectorConfigStrategy):
    """Configuration strategy for OpenAI on self-managed OpenSearch."""
    
    def get_config(self) -> Dict[str, str]:
        return {
            "api_key": get_raw_config_value("OPENAI_API_KEY"),
            "organization": get_raw_config_value("OPENAI_ORGANIZATION"),
            "model_name": get_raw_config_value("OPENAI_MODEL_NAME"),
            "max_tokens": get_raw_config_value("OPENAI_MAX_TOKENS"),
            "temperature": get_raw_config_value("OPENAI_TEMPERATURE"),
        }
    
    def get_required_fields(self) -> List[str]:
        return ["api_key", "model_name"]
    
    def get_payload_filename(self, model_type: str = None) -> str:
        return get_raw_config_value("OPENAI_PAYLOAD_FILE") or "openai_llm.json"


class HuggingFaceOSStrategy(ConnectorConfigStrategy):
    """Configuration strategy for Hugging Face on self-managed OpenSearch."""
    
    def get_config(self) -> Dict[str, str]:
        return {
            "api_key": get_raw_config_value("HUGGINGFACE_API_KEY"),
            "model_name": get_raw_config_value("HUGGINGFACE_MODEL_NAME"),
            "endpoint_url": get_raw_config_value("HUGGINGFACE_ENDPOINT_URL"),
            "max_tokens": get_raw_config_value("HUGGINGFACE_MAX_TOKENS"),
        }
    
    def get_required_fields(self) -> List[str]:
        return ["model_name", "endpoint_url"]
    
    def get_payload_filename(self, model_type: str = None) -> str:
        return get_raw_config_value("HUGGINGFACE_PAYLOAD_FILE") or "huggingface_llm.json"


# Strategy registry
CONNECTOR_STRATEGIES = {
    ("sagemaker", "os"): SagemakerOSStrategy,
    ("sagemaker", "aos"): SagemakerAOSStrategy,
    ("bedrock", "os"): BedrockOSStrategy,
    ("bedrock", "aos"): BedrockAOSStrategy,
    ("openai", "os"): OpenAIOSStrategy,
    ("huggingface", "os"): HuggingFaceOSStrategy,
}
