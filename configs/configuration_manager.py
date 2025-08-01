# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

import os
from dataclasses import dataclass
from typing import Dict, Any, Optional, Union
from enum import Enum
from dynaconf import Dynaconf

# Configuration file path
CONFIG_FILE_PATH = os.path.join(os.path.dirname(__file__), "osmlqs.yaml")

class OpenSearchType(Enum):
    OS = "os"
    AOS = "aos"

class ModelProvider(Enum):
    LOCAL = "local"
    BEDROCK = "bedrock"
    SAGEMAKER = "sagemaker"

class ModelType(Enum):
    EMBEDDING = "embedding"
    LLM = "llm"

@dataclass
class OpenSearchConfig:
    """Configuration for OpenSearch cluster connection."""
    username: Optional[str] = None
    password: Optional[str] = None
    host_url: Optional[str] = None
    port: Optional[str] = None
    domain_name: Optional[str] = None  # AOS only
    region: Optional[str] = None  # AOS only
    aws_user_name: Optional[str] = None  # AOS only

@dataclass
class ModelConfig:
    """Configuration for a specific model setup."""
    # Authentication
    access_key: Optional[str] = None
    secret_key: Optional[str] = None
    role_arn: Optional[str] = None
    connector_role_name: Optional[str] = None
    create_connector_role_name: Optional[str] = None
    
    # Connection details
    region: Optional[str] = None
    connector_version: Optional[str] = None
    endpoint_url: Optional[str] = None
    sparse_url: Optional[str] = None
    dense_url: Optional[str] = None
    
    # Model-specific
    model_name: Optional[str] = None
    model_dimension: Optional[int] = None
    sparse_arn: Optional[str] = None
    dense_arn: Optional[str] = None
    llm_arn: Optional[str] = None
    
    # LLM-specific
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None

class ConfigurationManager:
    """Manages multi-dimensional configuration using Dynaconf."""
    
    def __init__(self, config_file_path: str = CONFIG_FILE_PATH):
        self.config_file_path = config_file_path
        self._setup_dynaconf()
        self._opensearch_configs = {}
        self._model_configurations = {}
        self._build_configurations()
    
    def _setup_dynaconf(self):
        """Initialize Dynaconf with the configuration file."""
        self.settings = Dynaconf(
            settings_files=[self.config_file_path],
            load_dotenv=False,  # We're using YAML, not .env
            environments=False,  # Disable environment-based configuration
            envvar_prefix=False,  # Don't use environment variable prefix
        )
    
    def _get_config_value(self, key: str, default=None) -> Optional[str]:
        """Get configuration value using Dynaconf, defaulting to None if not found."""
        value = self.settings.get(key, default)
        if value == "None" or value == "":
            return None
        return value
    
    def _safe_int_convert(self, value: Optional[str]) -> Optional[int]:
        """Safely convert string to int, returning None if conversion fails."""
        if value is None:
            return None
        try:
            return int(value)
        except (ValueError, TypeError):
            return None
    
    def _build_configurations(self):
        """Build the nested configuration structure using Dynaconf."""
        # OpenSearch configurations
        self._opensearch_configs = {
            OpenSearchType.OS: OpenSearchConfig(
                username=self._get_config_value("OS_USERNAME"),
                password=self._get_config_value("OS_PASSWORD"),
                host_url=self._get_config_value("OS_HOST_URL"),
                port=self._get_config_value("OS_PORT")
            ),
            OpenSearchType.AOS: OpenSearchConfig(
                username=self._get_config_value("AOS_USERNAME"),
                password=self._get_config_value("AOS_PASSWORD"),
                host_url=self._get_config_value("AOS_HOST_URL"),
                port=self._get_config_value("AOS_PORT"),
                domain_name=self._get_config_value("AOS_DOMAIN_NAME"),
                region=self._get_config_value("AOS_REGION"),
                aws_user_name=self._get_config_value("AOS_AWS_USER_NAME")
            )
        }
        
        # Model configurations - 3D nested structure
        self._model_configurations = {
            # OS configurations
            OpenSearchType.OS: {
                ModelProvider.LOCAL: {
                    ModelType.EMBEDDING: ModelConfig(
                        model_name="sentence-transformers/all-MiniLM-L6-v2",
                        model_dimension=384
                    ),
                    ModelType.LLM: ModelConfig(
                        model_name="local-llm-model"
                    )
                },
                ModelProvider.SAGEMAKER: {
                    ModelType.EMBEDDING: ModelConfig(
                        access_key=self._get_config_value("OS_SAGEMAKER_ACCESS_KEY"),
                        secret_key=self._get_config_value("OS_SAGEMAKER_SECRET_KEY"),
                        region=self._get_config_value("OS_SAGEMAKER_REGION"),
                        connector_version=self._get_config_value("OS_SAGEMAKER_CONNECTOR_VERSION"),
                        sparse_url=self._get_config_value("OS_SAGEMAKER_SPARSE_URL"),
                        dense_url=self._get_config_value("OS_SAGEMAKER_DENSE_URL"),
                        model_dimension=self._safe_int_convert(self._get_config_value("OS_SAGEMAKER_DENSE_MODEL_DIMENSION"))
                    ),
                    ModelType.LLM: ModelConfig(
                        access_key=self._get_config_value("OS_SAGEMAKER_ACCESS_KEY"),
                        secret_key=self._get_config_value("OS_SAGEMAKER_SECRET_KEY"),
                        region=self._get_config_value("OS_SAGEMAKER_REGION"),
                        connector_version=self._get_config_value("OS_SAGEMAKER_CONNECTOR_VERSION")
                    )
                },
                ModelProvider.BEDROCK: {
                    ModelType.EMBEDDING: ModelConfig(
                        access_key=self._get_config_value("OS_BEDROCK_ACCESS_KEY"),
                        secret_key=self._get_config_value("OS_BEDROCK_SECRET_KEY"),
                        region=self._get_config_value("OS_BEDROCK_REGION"),
                        connector_version=self._get_config_value("OS_BEDROCK_CONNECTOR_VERSION"),
                        endpoint_url=self._get_config_value("OS_BEDROCK_URL"),
                        model_name="amazon.titan-embed-text-v1",
                        model_dimension=self._safe_int_convert(self._get_config_value("OS_BEDROCK_MODEL_DIMENSION"))
                    ),
                    ModelType.LLM: ModelConfig(
                        access_key=self._get_config_value("OS_BEDROCK_ACCESS_KEY"),
                        secret_key=self._get_config_value("OS_BEDROCK_SECRET_KEY"),
                        region=self._get_config_value("OS_BEDROCK_REGION"),
                        connector_version=self._get_config_value("OS_BEDROCK_CONNECTOR_VERSION"),
                        model_name="anthropic.claude-3-5-sonnet-20241022-v2:0",
                        max_tokens=8000,
                        temperature=0.1
                    )
                }
            },
            
            # AOS configurations
            OpenSearchType.AOS: {
                ModelProvider.SAGEMAKER: {
                    ModelType.EMBEDDING: ModelConfig(
                        sparse_arn=self._get_config_value("AOS_SAGEMAKER_SPARSE_ARN"),
                        dense_arn=self._get_config_value("AOS_SAGEMAKER_DENSE_ARN"),
                        connector_role_name=self._get_config_value("AOS_SAGEMAKER_CONNECTOR_ROLE_NAME"),
                        create_connector_role_name=self._get_config_value("AOS_SAGEMAKER_CREATE_CONNECTOR_ROLE_NAME"),
                        region=self._get_config_value("AOS_SAGEMAKER_REGION"),
                        connector_version=self._get_config_value("AOS_SAGEMAKER_CONNECTOR_VERSION"),
                        sparse_url=self._get_config_value("AOS_SAGEMAKER_SPARSE_URL"),
                        dense_url=self._get_config_value("AOS_SAGEMAKER_DENSE_URL"),
                        model_dimension=self._safe_int_convert(self._get_config_value("AOS_SAGEMAKER_DENSE_MODEL_DIMENSION"))
                    ),
                    ModelType.LLM: ModelConfig(
                        connector_role_name=self._get_config_value("AOS_SAGEMAKER_CONNECTOR_ROLE_NAME"),
                        create_connector_role_name=self._get_config_value("AOS_SAGEMAKER_CREATE_CONNECTOR_ROLE_NAME"),
                        region=self._get_config_value("AOS_SAGEMAKER_REGION"),
                        connector_version=self._get_config_value("AOS_SAGEMAKER_CONNECTOR_VERSION")
                    )
                },
                ModelProvider.BEDROCK: {
                    ModelType.EMBEDDING: ModelConfig(
                        role_arn=self._get_config_value("AOS_BEDROCK_ARN"),
                        connector_role_name=self._get_config_value("AOS_BEDROCK_CONNECTOR_ROLE_NAME"),
                        create_connector_role_name=self._get_config_value("AOS_BEDROCK_CREATE_CONNECTOR_ROLE_NAME"),
                        region=self._get_config_value("AOS_BEDROCK_REGION"),
                        connector_version=self._get_config_value("AOS_BEDROCK_CONNECTOR_VERSION"),
                        endpoint_url=self._get_config_value("AOS_BEDROCK_URL"),
                        model_name="amazon.titan-embed-text-v1",
                        model_dimension=self._safe_int_convert(self._get_config_value("AOS_BEDROCK_MODEL_DIMENSION"))
                    ),
                    ModelType.LLM: ModelConfig(
                        llm_arn=self._get_config_value("AOS_BEDROCK_ARN"),
                        connector_role_name=self._get_config_value("AOS_BEDROCK_CONNECTOR_ROLE_NAME"),
                        create_connector_role_name=self._get_config_value("AOS_BEDROCK_CREATE_CONNECTOR_ROLE_NAME"),
                        region=self._get_config_value("AOS_BEDROCK_REGION"),
                        connector_version=self._get_config_value("AOS_BEDROCK_CONNECTOR_VERSION"),
                        model_name="anthropic.claude-3-5-sonnet-20241022-v2:0",
                        max_tokens=8000,
                        temperature=0.1
                    )
                }
            }
        }
    
    def get_opensearch_config(self, os_type: Union[OpenSearchType, str]) -> OpenSearchConfig:
        """Get OpenSearch cluster configuration."""
        if isinstance(os_type, str):
            os_type = OpenSearchType(os_type)
        
        if os_type not in self._opensearch_configs:
            raise ValueError(f"OpenSearch configuration not found for {os_type.value}")
        
        return self._opensearch_configs[os_type]
    
    def get_model_config(self, os_type: Union[OpenSearchType, str], 
                        provider: Union[ModelProvider, str], 
                        model_type: Union[ModelType, str]) -> ModelConfig:
        """Get model configuration for the specified combination."""
        # Convert strings to enums if needed
        if isinstance(os_type, str):
            os_type = OpenSearchType(os_type)
        if isinstance(provider, str):
            provider = ModelProvider(provider)
        if isinstance(model_type, str):
            model_type = ModelType(model_type)
        
        try:
            return self._model_configurations[os_type][provider][model_type]
        except KeyError:
            available = self.get_available_combinations()
            raise ValueError(f"Configuration not found for {os_type.value}/{provider.value}/{model_type.value}. Available: {available}")
    
    def get_available_combinations(self) -> list:
        """Get all available configuration combinations."""
        combinations = []
        for os_type, providers in self._model_configurations.items():
            for provider, model_types in providers.items():
                for model_type in model_types.keys():
                    combinations.append((os_type.value, provider.value, model_type.value))
        return combinations
    
    def get_raw_config_value(self, key: str, default=None):
        """Get raw configuration value directly from Dynaconf."""
        return self.settings.get(key, default)
    
    def list_all_config_keys(self) -> list:
        """List all configuration keys loaded by Dynaconf."""
        return list(self.settings.keys())
    
    def validate_all_configs(self) -> Dict[str, list]:
        """Validate all configurations and return any issues."""
        issues = {}
        
        # Validate OpenSearch configs
        for os_type, config in self._opensearch_configs.items():
            config_issues = []
            if not config.username:
                config_issues.append("Missing username")
            if not config.password:
                config_issues.append("Missing password")
            if not config.host_url:
                config_issues.append("Missing host_url")
            
            if config_issues:
                issues[f"opensearch_{os_type.value}"] = config_issues
        
        # Validate model configs
        for os_type, providers in self._model_configurations.items():
            for provider, model_types in providers.items():
                for model_type, config in model_types.items():
                    config_issues = []
                    key = f"{os_type.value}/{provider.value}/{model_type.value}"
                    
                    # Check authentication
                    if provider != ModelProvider.LOCAL:
                        if not config.access_key and not config.role_arn:
                            config_issues.append("Missing authentication (access_key or role_arn)")
                        if not config.region:
                            config_issues.append("Missing region")
                    
                    if config_issues:
                        issues[key] = config_issues
        
        return issues
    
    def reload_config(self):
        """Reload configuration from file."""
        self._setup_dynaconf()
        self._build_configurations()
    
    def get_config_info(self) -> Dict[str, Any]:
        """Get information about the configuration system."""
        return {
            "config_file": self.config_file_path,
            "config_exists": os.path.exists(self.config_file_path),
            "total_keys": len(self.list_all_config_keys()),
            "available_combinations": len(self.get_available_combinations()),
            "dynaconf_version": getattr(self.settings, '_dynaconf_version', 'unknown')
        }

# Global configuration manager instance
config_manager = ConfigurationManager()

# Convenience functions
def get_opensearch_config(os_type: str) -> OpenSearchConfig:
    """Get OpenSearch cluster configuration."""
    return config_manager.get_opensearch_config(os_type)

def get_model_config(os_type: str, provider: str, model_type: str) -> ModelConfig:
    """Get model configuration."""
    return config_manager.get_model_config(os_type, provider, model_type)

def get_embedding_config(os_type: str, provider: str) -> ModelConfig:
    """Get embedding model configuration."""
    return get_model_config(os_type, provider, "embedding")

def get_llm_config(os_type: str, provider: str) -> ModelConfig:
    """Get LLM model configuration."""
    return get_model_config(os_type, provider, "llm")

def get_raw_config_value(key: str, default=None):
    """Get raw configuration value directly from Dynaconf."""
    return config_manager.get_raw_config_value(key, default)

def reload_config():
    """Reload configuration from file."""
    config_manager.reload_config()

def get_available_combinations() -> list:
    """Get all available configuration combinations."""
    return config_manager.get_available_combinations()

def validate_all_configs() -> Dict[str, list]:
    """Validate all configurations and return any issues."""
    return config_manager.validate_all_configs()

def get_config_info() -> Dict[str, Any]:
    """Get information about the configuration system."""
    return config_manager.get_config_info()

def validate_config_for(config: Dict[str, Any], os_type: str, provider: str, model_type: str):
    """
    Validate that all required configuration parameters are present for a specific combination.
    
    Parameters:
        config (Dict[str, Any]): Configuration dictionary to validate
        os_type (str): OpenSearch type ("os" or "aos")
        provider (str): Model provider ("local", "bedrock", "sagemaker")
        model_type (str): Model type ("embedding" or "llm")
    
    Raises:
        ValueError: If any required parameters are missing, with a list of missing parameters
    """
    missing_params = []
    
    # Define required parameters based on the configuration combination
    required_opensearch_params = ["username", "password", "host_url"]
    
    # OpenSearch-specific requirements
    if os_type == "aos":
        required_opensearch_params.extend(["domain_name", "region"])
    
    # Check OpenSearch configuration
    opensearch_config = config.get("opensearch", {})
    for param in required_opensearch_params:
        if not opensearch_config.get(param):
            missing_params.append(f"opensearch.{param}")
    
    # Model-specific requirements
    model_config = config.get("model", {})
    
    if provider == "local":
        # Local models require minimal configuration
        required_model_params = ["model_name"]
        if model_type == "embedding":
            required_model_params.append("model_dimension")
    
    elif provider == "bedrock":
        if os_type == "os":
            # Self-managed OpenSearch with Bedrock requires AWS credentials
            required_model_params = ["access_key", "secret_key", "region"]
        else:  # aos
            # AOS with Bedrock requires IAM roles
            required_model_params = ["connector_role_name", "create_connector_role_name", "region"]
        
        # Common Bedrock requirements
        if model_type == "embedding":
            required_model_params.extend(["model_name", "model_dimension"])
        elif model_type == "llm":
            required_model_params.extend(["model_name"])
    
    elif provider == "sagemaker":
        if os_type == "os":
            # Self-managed OpenSearch with SageMaker requires AWS credentials
            required_model_params = ["access_key", "secret_key", "region"]
        else:  # aos
            # AOS with SageMaker requires IAM roles
            required_model_params = ["connector_role_name", "create_connector_role_name", "region"]
        
        # SageMaker-specific requirements
        if model_type == "embedding":
            if os_type == "os":
                required_model_params.extend(["sparse_url", "dense_url", "model_dimension"])
            else:  # aos
                required_model_params.extend(["sparse_arn", "dense_arn", "sparse_url", "dense_url", "model_dimension"])
    
    # Check model configuration
    for param in required_model_params:
        if not model_config.get(param):
            missing_params.append(f"model.{param}")
    
    # Check system constants (these should always be present)
    constants_config = config.get("constants", {})
    required_constants = ["ML_BASE_URI", "DELETE_RESOURCE_WAIT_TIME", "DELETE_RESOURCE_RETRY_TIME"]
    
    for param in required_constants:
        if not constants_config.get(param):
            missing_params.append(f"constants.{param}")
    
    # Raise exception if any parameters are missing
    if missing_params:
        combination = f"{os_type}/{provider}/{model_type}"
        raise ValueError(
            f"Missing required configuration parameters for {combination}: {', '.join(missing_params)}"
        )


def get_config_for(os_type: str, provider: str, model_type: str) -> Dict[str, Any]:
    """
    Get all configuration information for a particular opensearch type, model provider, and model type.
    
    This function returns a comprehensive configuration dictionary that includes:
    - OpenSearch cluster configuration
    - Model-specific configuration
    - System constants
    - Helper functions for backward compatibility
    
    Parameters:
        os_type (str): OpenSearch type ("os" or "aos")
        provider (str): Model provider ("local", "bedrock", "sagemaker")
        model_type (str): Model type ("embedding" or "llm")
    
    Returns:
        Dict[str, Any]: Complete configuration dictionary with all necessary values
    """
    # Get OpenSearch configuration
    opensearch_config = get_opensearch_config(os_type)
    
    # Get model configuration
    model_config = get_model_config(os_type, provider, model_type)
    
    # Build comprehensive configuration dictionary
    all_config = {
        # OpenSearch cluster configuration
        "opensearch": {
            "username": opensearch_config.username,
            "password": opensearch_config.password,
            "host_url": opensearch_config.host_url,
            "port": opensearch_config.port,
            "domain_name": opensearch_config.domain_name,
            "region": opensearch_config.region,
            "aws_user_name": opensearch_config.aws_user_name,
        },
        
        # Model configuration
        "model": {
            "access_key": model_config.access_key,
            "secret_key": model_config.secret_key,
            "role_arn": model_config.role_arn,
            "connector_role_name": model_config.connector_role_name,
            "create_connector_role_name": model_config.create_connector_role_name,
            "region": model_config.region,
            "connector_version": model_config.connector_version,
            "endpoint_url": model_config.endpoint_url,
            "sparse_url": model_config.sparse_url,
            "dense_url": model_config.dense_url,
            "model_name": model_config.model_name,
            "model_dimension": model_config.model_dimension,
            "sparse_arn": model_config.sparse_arn,
            "dense_arn": model_config.dense_arn,
            "llm_arn": model_config.llm_arn,
            "max_tokens": model_config.max_tokens,
            "temperature": model_config.temperature,
        },
        
        # System constants
        "constants": {
            "PROJECT_ROOT": get_project_root(),
            "BASE_MAPPING_PATH": get_base_mapping_path(),
            "QANDA_FILE_READER_PATH": get_qanda_file_reader_path(),
            "MINIMUM_OPENSEARCH_VERSION": get_minimum_opensearch_version(),
            "ML_BASE_URI": get_ml_base_uri(),
            "DELETE_RESOURCE_WAIT_TIME": get_delete_resource_wait_time(),
            "DELETE_RESOURCE_RETRY_TIME": get_delete_resource_retry_time(),
            "DEFAULT_LOCAL_MODEL_NAME": get_default_local_model_name(),
            "DEFAULT_MODEL_VERSION": get_default_model_version(),
            "DEFAULT_MODEL_FORMAT": get_default_model_format(),
            "PIPELINE_FIELD_MAP": {"chunk": "chunk_embedding"},
        },
        
        # Configuration metadata
        "metadata": {
            "os_type": os_type,
            "provider": provider,
            "model_type": model_type,
        }
    }
    
    # Remove None values for cleaner output
    def remove_none_values(d):
        if isinstance(d, dict):
            return {k: remove_none_values(v) for k, v in d.items() if v is not None}
        return d
    
    return remove_none_values(all_config)


def list_all_config_keys() -> list:
    """List all configuration keys loaded by Dynaconf."""
    return config_manager.list_all_config_keys()

# Additional configuration constants and convenience functions
def get_project_root() -> str:
    """Get the project root directory."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_base_mapping_path() -> str:
    """Get the path to the base mapping JSON file."""
    return os.path.join(get_project_root(), "mapping", "base_mapping.json")

def get_qanda_file_reader_path() -> str:
    """Get the path to the Q&A dataset."""
    return get_raw_config_value("QANDA_FILE_READER_PATH", "/Users/handler/datasets/amazon_pqa")

def get_minimum_opensearch_version() -> str:
    """Get the minimum required OpenSearch version."""
    return get_raw_config_value("MINIMUM_OPENSEARCH_VERSION", "2.19.0")

def get_ml_base_uri() -> str:
    """Get the ML base URI for OpenSearch ML Commons."""
    return get_raw_config_value("ML_BASE_URI", "/_plugins/_ml")

def get_delete_resource_wait_time() -> int:
    """Get the wait time for resource deletion operations."""
    return int(get_raw_config_value("DELETE_RESOURCE_WAIT_TIME", 5))

def get_delete_resource_retry_time() -> int:
    """Get the retry time for resource deletion operations."""
    return int(get_raw_config_value("DELETE_RESOURCE_RETRY_TIME", 5))

def get_default_local_model_name() -> str:
    """Get the default local model name."""
    return get_raw_config_value(
        "DEFAULT_LOCAL_MODEL_NAME", 
        "huggingface/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

def get_default_model_version() -> str:
    """Get the default model version."""
    return get_raw_config_value("DEFAULT_MODEL_VERSION", "1.0.1")

def get_default_model_format() -> str:
    """Get the default model format."""
    return get_raw_config_value("DEFAULT_MODEL_FORMAT", "TORCH_SCRIPT")

def get_pipeline_field_map() -> Dict[str, str]:
    """Get the pipeline field mapping."""
    return get_raw_config_value("PIPELINE_FIELD_MAP", {"chunk": "chunk_embedding"})

def get_pipeline_field_map() -> Dict[str, str]:
    """Get the pipeline field mapping."""
    return get_raw_config_value("PIPELINE_FIELD_MAP", {"chunk": "chunk_embedding"})

# Legacy compatibility constants (computed from configuration)
PROJECT_ROOT = get_project_root()
BASE_MAPPING_PATH = get_base_mapping_path()
QANDA_FILE_READER_PATH = get_qanda_file_reader_path()
MINIMUM_OPENSEARCH_VERSION = get_minimum_opensearch_version()
ML_BASE_URI = get_ml_base_uri()
DELETE_RESOURCE_WAIT_TIME = get_delete_resource_wait_time()
DELETE_RESOURCE_RETRY_TIME = get_delete_resource_retry_time()
DEFAULT_LOCAL_MODEL_NAME = get_default_local_model_name()
DEFAULT_MODEL_VERSION = get_default_model_version()
DEFAULT_MODEL_FORMAT = get_default_model_format()
