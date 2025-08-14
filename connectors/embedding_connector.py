# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

"""
Embedding Connector Module

This module provides the universal connector for all embedding model providers.
The EmbeddingConnector class handles connections to embedding models from various
providers (Bedrock, SageMaker, etc.) that convert text into vector representations
for semantic search.

The connector automatically adapts its behavior based on the provider and
OpenSearch deployment type specified in the configuration.
"""

import logging
from abc import abstractmethod
from typing import Dict, Any, Optional, List
from opensearchpy import OpenSearch

from configs.configuration_manager import validate_configs, get_ml_base_uri
from connectors.helper import get_remote_connector_configs, create_connector_with_iam_roles, create_connector_with_basic_auth
from .ml_connector import MlConnector


class EmbeddingConnector(MlConnector):
    """
    Universal connector for all embedding model providers.
    
    This class provides a unified interface for embedding connectors across different
    providers and OpenSearch deployment types:
    
    Supported providers:
    - bedrock: Amazon Bedrock embedding models (dense only)
    - sagemaker: Amazon SageMaker embedding endpoints (dense + sparse)
    
    Supported OpenSearch types:
    - aos: Amazon OpenSearch Service (uses IAM roles)
    - os: Self-managed OpenSearch (uses AWS credentials)
    
    The connector automatically loads appropriate configurations and adapts its
    behavior based on the provider and deployment type.
    """
    
    DEFAULT_CONNECTOR_NAME = "Embedding Model Connector"
    DEFAULT_CONNECTOR_DESCRIPTION = "Universal connector for embedding models"
    
    # Supported providers and embedding types
    SUPPORTED_PROVIDERS = {"bedrock", "sagemaker"}
    SUPPORTED_EMBEDDING_TYPES = {"dense", "sparse"}
    
    # Provider capabilities
    PROVIDER_CAPABILITIES = {
        "bedrock": {"dense"},  # Bedrock only supports dense embeddings
        "sagemaker": {"dense", "sparse"}  # SageMaker supports both
    }
    
    def __init__(
        self,
        os_client: OpenSearch,
        provider: str,
        os_type: str,
        opensearch_domain_url: Optional[str] = None,
        opensearch_domain_arn: Optional[str] = None,
        opensearch_username: Optional[str] = None,
        opensearch_password: Optional[str] = None,
        aws_user_name: Optional[str] = None,
        region: Optional[str] = None,
        connector_name: Optional[str] = None,
        connector_description: Optional[str] = None,
        connector_configs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the universal embedding connector.
        
        Args:
            os_client: OpenSearch client instance
            provider: Model provider ('bedrock' or 'sagemaker')
            os_type: OpenSearch deployment type ('aos' or 'os')
            opensearch_domain_url: OpenSearch domain URL (required for AOS)
            opensearch_domain_arn: OpenSearch domain ARN (required for AOS)
            opensearch_username: OpenSearch username (required for AOS)
            opensearch_password: OpenSearch password (required for AOS)
            aws_user_name: AWS IAM user name (required for AOS)
            region: AWS region (required for AOS)
            connector_name: Custom connector name (optional)
            connector_description: Custom connector description (optional)
            connector_configs: Override configurations (optional, will auto-load if not provided)
        
        Raises:
            ValueError: If provider, os_type, or other parameters are invalid
        """
        # Validate provider
        if provider not in self.SUPPORTED_PROVIDERS:
            raise ValueError(
                f"Unsupported provider: {provider}. "
                f"Supported providers: {', '.join(self.SUPPORTED_PROVIDERS)}"
            )
        
        # Validate os_type
        if os_type not in {"aos", "os"}:
            raise ValueError(f"os_type must be 'aos' or 'os', got: {os_type}")
        
        self._provider = provider
        self._os_type = os_type
        
        # Store AOS-specific parameters
        if os_type == "aos":
            if not all([opensearch_domain_url, opensearch_domain_arn, opensearch_username, 
                       opensearch_password, aws_user_name, region]):
                raise ValueError("AOS deployments require: opensearch_domain_url, opensearch_domain_arn, "
                               "opensearch_username, opensearch_password, aws_user_name, region")
            
            self._opensearch_domain_url = opensearch_domain_url
            self._opensearch_domain_arn = opensearch_domain_arn
            self._opensearch_username = opensearch_username
            self._opensearch_password = opensearch_password
            self._aws_user_name = aws_user_name
            self._region = region
        
        # Auto-load configuration and merge with provided configs
        auto_loaded_configs = get_remote_connector_configs(provider, os_type)
        if connector_configs is None:
            connector_configs = auto_loaded_configs
            logging.info(f"Auto-loaded {provider.title()} configuration for {os_type.upper()}")
        else:
            # Merge provided configs with auto-loaded ones, giving priority to provided configs
            merged_configs = auto_loaded_configs.copy()
            merged_configs.update(connector_configs)
            connector_configs = merged_configs
            logging.info(f"Merged provided configuration with auto-loaded {provider.title()} configuration for {os_type.upper()}")
        
        # Set default embedding type based on provider if not specified
        if "embedding_type" not in connector_configs:
            connector_configs["embedding_type"] = "dense"  # Default to dense
        
        # Validate embedding type is supported by provider
        embedding_type = connector_configs["embedding_type"]
        if embedding_type not in self.PROVIDER_CAPABILITIES[provider]:
            raise ValueError(
                f"Provider '{provider}' doesn't support '{embedding_type}' embeddings. "
                f"Supported types: {', '.join(self.PROVIDER_CAPABILITIES[provider])}"
            )
        
        # Set provider-specific default names if not provided
        if connector_name is None:
            connector_name = f"Amazon {provider.title()} Embedding Connector"
        if connector_description is None:
            connector_description = f"Connector to Amazon {provider.title()} embedding models"
        
        super().__init__(
            os_client=os_client,
            connector_name=connector_name,
            connector_description=connector_description,
            connector_configs=connector_configs
        )
        
        # Extract embedding-specific configurations
        self._model_dimensions = connector_configs.get("model_dimensions")
        self._max_tokens_per_chunk = connector_configs.get("max_tokens_per_chunk")
        
        logging.info(
            f"Initialized {self.__class__.__name__} for {provider.upper()} on {os_type.upper()} "
            f"with embedding type: {self._embedding_type}"
        )
    
    def _validate_embedding_configs(self) -> None:
        """
        Validate embedding-specific configurations based on provider and OS type.
        
        Raises:
            ValueError: If required configurations are missing
        """
        if self._provider == "bedrock":
            self._validate_bedrock_configs()
        elif self._provider == "sagemaker":
            self._validate_sagemaker_configs()
    
    def _validate_bedrock_configs(self) -> None:
        """Validate Bedrock-specific configurations."""
        if self._os_type == "aos":
            # AOS requires IAM role configurations
            required_args = [
                "dense_arn",  # Bedrock model ARN
                "connector_role_name",  # IAM role for connector
                "create_connector_role_name",  # IAM role for creation
                "region",  # AWS region
                "dense_url",  # Bedrock API URL
                "connector_version",  # Connector version
            ]
        else:  # os
            # Self-managed OpenSearch requires AWS credentials
            required_args = [
                "access_key",  # AWS access key
                "secret_key",  # AWS secret key
                "region",  # AWS region
                "dense_url",  # Bedrock API URL
                "connector_version",  # Connector version
            ]
        
        # Add common requirements
        required_args.extend(["model_dimensions"])
        
        validate_configs(self._connector_configs, required_args)
        
        # Validate model dimensions
        dimensions = self._connector_configs.get("model_dimensions")
        if not isinstance(dimensions, int) or dimensions <= 0:
            raise ValueError(f"model_dimensions must be a positive integer, got: {dimensions}")
    
    def _validate_sagemaker_configs(self) -> None:
        """Validate SageMaker-specific configurations."""
        if self._os_type == "aos":
            # AOS requires IAM role configurations
            required_args = [
                "connector_role_name",  # IAM role for connector
                "create_connector_role_name",  # IAM role for creation
                "region",  # AWS region
                "connector_version",  # Connector version
            ]
            
            # SageMaker AOS requires ARNs based on embedding type
            if self._embedding_type == "dense":
                required_args.extend(["dense_arn", "dense_url"])
            elif self._embedding_type == "sparse":
                required_args.extend(["sparse_arn", "sparse_url"])
            
        else:  # os
            # Self-managed OpenSearch requires AWS credentials
            required_args = [
                "access_key",  # AWS access key
                "secret_key",  # AWS secret key
                "region",  # AWS region
                "connector_version",  # Connector version
            ]
            
            # SageMaker OS requires URLs based on embedding type
            if self._embedding_type == "dense":
                required_args.append("dense_url")
            elif self._embedding_type == "sparse":
                required_args.append("sparse_url")
        
        # Add model dimensions for dense embeddings
        if self._embedding_type == "dense":
            required_args.append("model_dimensions")
        
        validate_configs(self._connector_configs, required_args)
        
        # Validate model dimensions for dense embeddings
        if self._embedding_type == "dense":
            dimensions = self._connector_configs.get("model_dimensions")
            if not isinstance(dimensions, int) or dimensions <= 0:
                raise ValueError(f"model_dimensions must be a positive integer, got: {dimensions}")
    
    def _validate_configs(self) -> None:
        """
        Validate all connector configurations.
        
        This method calls the provider-specific validation.
        """
        self._validate_embedding_configs()
    
    def _get_embedding_model_config(self) -> Dict[str, Any]:
        """
        Get embedding model configuration based on provider.
        
        Returns:
            Dictionary containing provider-specific model configuration
        """
        base_config = {
            "provider": self._provider,
            "model_type": "embedding",
            "embedding_type": self._embedding_type,
            "region": self._connector_configs.get("region"),
            "os_type": self._os_type,
        }
        
        if self._embedding_type == "dense":
            base_config["model_dimensions"] = self._connector_configs.get("model_dimensions")
        
        # Add provider-specific configurations
        if self._provider == "bedrock":
            if self._os_type == "aos":
                base_config.update({
                    "model_arn": self._connector_configs.get("dense_arn"),
                    "connector_role": self._connector_configs.get("connector_role_name"),
                    "authentication": "iam_role",
                })
            else:
                base_config.update({
                    "model_url": self._connector_configs.get("dense_url"),
                    "authentication": "aws_credentials",
                })
                
        elif self._provider == "sagemaker":
            if self._os_type == "aos":
                arn_key = f"{self._embedding_type}_arn"
                base_config.update({
                    "model_arn": self._connector_configs.get(arn_key),
                    "connector_role": self._connector_configs.get("connector_role_name"),
                    "authentication": "iam_role",
                })
            else:
                url_key = f"{self._embedding_type}_url"
                base_config.update({
                    "model_url": self._connector_configs.get(url_key),
                    "authentication": "aws_credentials",
                })
        
        return base_config
    
    def _get_connector_create_payload_filename(self) -> str:
        """
        Get the connector payload filename based on provider and embedding type.
        
        Returns:
            Filename for the connector payload
        """
        from .helper import get_connector_payload_filename
        return get_connector_payload_filename(self._provider, self._os_type, self._embedding_type)
    
    def _fill_in_connector_create_payload(self, connector_create_payload):
        """
        Fill in the connector creation payload with provider-specific values.
        
        Args:
            connector_create_payload: Base payload template
            
        Returns:
            Filled payload ready for connector creation
        """
        # Common payload fields
        connector_create_payload["name"] = self._connector_name
        connector_create_payload["description"] = self._connector_description
        connector_create_payload["version"] = self._connector_configs["connector_version"]
        connector_create_payload["parameters"]["region"] = self._connector_configs["region"]
        
        # Set service name based on provider
        service_name = self._provider  # "bedrock" or "sagemaker"
        connector_create_payload["parameters"]["service_name"] = service_name
        
        # Set the API URL based on provider and embedding type
        if self._provider == "bedrock":
            url = self._connector_configs.get("dense_url")
        else:  # sagemaker
            url_key = f"{self._embedding_type}_url"
            url = self._connector_configs.get(url_key)
        
        if not url:
            raise ValueError(f"{url_key if self._provider == 'sagemaker' else 'dense_url'} is required")
        
        connector_create_payload["actions"][0]["url"] = url
        
        # Add OS-specific configurations
        if self._os_type == "os":
            # For self-managed OpenSearch, add AWS credentials
            credential = {
                "access_key": self._connector_configs["access_key"],
                "secret_key": self._connector_configs["secret_key"],
            }
            connector_create_payload["credential"] = credential
        
        return connector_create_payload
    
    def _create_connector_with_payload(self, connector_create_payload):
        """
        Create the connector using the appropriate method based on OpenSearch type.
        
        Args:
            connector_create_payload: Filled connector payload
        """
        if self._os_type == "aos":
            # Use IAM roles for both OpenSearch access and outbound service access
            connector_role_inline_policy = self._get_connector_role_inline_policy()
            connector_role_name = self._connector_configs["connector_role_name"]
            create_connector_role_name = self._connector_configs["create_connector_role_name"]
            
            self._connector_id = create_connector_with_iam_roles(
                opensearch_domain_url=self._opensearch_domain_url,
                opensearch_domain_arn=self._opensearch_domain_arn,
                opensearch_username=self._opensearch_username,
                opensearch_password=self._opensearch_password,
                aws_user_name=self._aws_user_name,
                region=self._region,
                connector_role_inline_policy=connector_role_inline_policy,
                connector_role_name=connector_role_name,
                create_connector_role_name=create_connector_role_name,
                connector_payload=connector_create_payload,
                sleep_time_in_seconds=10,
            )
            
            logging.info(f"Created AOS {self._provider.title()} connector with ID: {self._connector_id}")
            
        else:  # os
            # Use basic auth for OpenSearch access, credentials in payload for outbound service access
            self._connector_id = create_connector_with_basic_auth(
                os_client=self._os_client,
                connector_payload=connector_create_payload,
            )
            
            logging.info(f"Created OS {self._provider.title()} connector with ID: {self._connector_id}")
    
    def _get_connector_role_inline_policy(self) -> Dict[str, Any]:
        """
        Get the IAM role inline policy based on provider.
        
        Returns:
            IAM policy document for the specific provider
        """
        if self._os_type != "aos":
            raise ValueError("Connector role policy is only needed for AOS deployments")
        
        if self._provider == "bedrock":
            return self._get_bedrock_iam_policy()
        elif self._provider == "sagemaker":
            return self._get_sagemaker_iam_policy()
    
    def _get_bedrock_iam_policy(self) -> Dict[str, Any]:
        """Get IAM policy for Bedrock access."""
        dense_arn = self._connector_configs.get("dense_arn")
        if not dense_arn:
            raise ValueError("dense_arn is required for AOS Bedrock connector")
        
        return {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Resource": [dense_arn],
                    "Action": ["bedrock:InvokeModel"],
                }
            ],
        }
    
    def _get_sagemaker_iam_policy(self) -> Dict[str, Any]:
        """Get IAM policy for SageMaker access."""
        resources = []
        
        # Add ARNs based on embedding type
        if self._embedding_type == "dense":
            dense_arn = self._connector_configs.get("dense_arn")
            if dense_arn:
                resources.append(dense_arn)
        elif self._embedding_type == "sparse":
            sparse_arn = self._connector_configs.get("sparse_arn")
            if sparse_arn:
                resources.append(sparse_arn)
        
        # For SageMaker, we might need both ARNs in some cases
        # Check if both are configured and add them
        dense_arn = self._connector_configs.get("dense_arn")
        sparse_arn = self._connector_configs.get("sparse_arn")
        if dense_arn and dense_arn not in resources:
            resources.append(dense_arn)
        if sparse_arn and sparse_arn not in resources:
            resources.append(sparse_arn)
        
        if not resources:
            raise ValueError("At least one SageMaker ARN is required for AOS SageMaker connector")
        
        return {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Resource": resources,
                    "Action": ["sagemaker:InvokeEndpoint"],
                }
            ],
        }
    
    # Public methods for provider information
    def get_provider(self) -> str:
        """Get the model provider."""
        return self._provider
    
    def get_os_type(self) -> str:
        """Get the OpenSearch deployment type."""
        return self._os_type
    
    def get_provider_capabilities(self) -> set:
        """Get the capabilities of the current provider."""
        return self.PROVIDER_CAPABILITIES[self._provider].copy()
    
    def supports_sparse_embeddings(self) -> bool:
        """Check if the current provider supports sparse embeddings."""
        return "sparse" in self.PROVIDER_CAPABILITIES[self._provider]
    
    def get_model_dimensions(self) -> Optional[int]:
        """
        Get the model dimensions for dense embeddings.
        
        Returns:
            Model dimensions as integer, or None for sparse embeddings
        """
        return self._model_dimensions
    
    def get_embedding_type(self) -> str:
        """
        Get the embedding type (dense or sparse).
        
        Returns:
            Embedding type as string
        """
        return self._embedding_type
    
    def get_max_tokens_per_chunk(self) -> Optional[int]:
        """
        Get the maximum tokens per chunk for processing.
        
        Returns:
            Maximum tokens per chunk, or None if not configured
        """
        return self._max_tokens_per_chunk
    
    def is_dense_embedding(self) -> bool:
        """
        Check if this is a dense embedding connector.
        
        Returns:
            True if dense embedding, False otherwise
        """
        return self._embedding_type == "dense"
    
    def is_sparse_embedding(self) -> bool:
        """
        Check if this is a sparse embedding connector.
        
        Returns:
            True if sparse embedding, False otherwise
        """
        return self._embedding_type == "sparse"
    
    def get_provider_model_info(self) -> Dict[str, Any]:
        """
        Get provider-specific model information.
        
        Returns:
            Dictionary containing provider model details
        """
        info = {
            "provider": self._provider,
            "os_type": self._os_type,
            "embedding_type": self._embedding_type,
            "region": self._connector_configs.get("region"),
            "connector_version": self._connector_configs.get("connector_version"),
            "capabilities": list(self.get_provider_capabilities()),
        }
        
        if self._embedding_type == "dense":
            info["model_dimensions"] = self._connector_configs.get("model_dimensions")
        
        # Add provider-specific details
        if self._os_type == "aos":
            info.update({
                "connector_role": self._connector_configs.get("connector_role_name"),
                "authentication": "iam_role",
            })
            
            if self._provider == "bedrock":
                info["model_arn"] = self._connector_configs.get("dense_arn")
            elif self._provider == "sagemaker":
                if self._embedding_type == "dense":
                    info["model_arn"] = self._connector_configs.get("dense_arn")
                elif self._embedding_type == "sparse":
                    info["model_arn"] = self._connector_configs.get("sparse_arn")
        else:
            info.update({
                "authentication": "aws_credentials",
            })
            
            if self._provider == "bedrock":
                info["model_url"] = self._connector_configs.get("dense_url")
            elif self._provider == "sagemaker":
                url_key = f"{self._embedding_type}_url"
                info["model_url"] = self._connector_configs.get(url_key)
        
        return info
    
    def get_connector_info(self) -> Dict[str, Any]:
        """
        Get comprehensive connector information.
        
        Returns:
            Dictionary containing complete connector information
        """
        base_info = {
            "connector_id": self.connector_id(),
            "connector_name": self._connector_name,
            "connector_description": self._connector_description,
            "embedding_type": self._embedding_type,
            "universal_connector": True,
            "supported_providers": list(self.SUPPORTED_PROVIDERS),
        }
        
        if self._model_dimensions:
            base_info["model_dimensions"] = self._model_dimensions
            
        if self._max_tokens_per_chunk:
            base_info["max_tokens_per_chunk"] = self._max_tokens_per_chunk
        
        # Add provider-specific information
        base_info["provider_info"] = self.get_provider_model_info()
        
        return base_info
    
    def __str__(self) -> str:
        """String representation of the embedding connector."""
        return (
            f"{self.__class__.__name__}("
            f"id={self.connector_id()}, "
            f"provider={self._provider}, "
            f"os_type={self._os_type}, "
            f"type={self._embedding_type}, "
            f"dimensions={self._model_dimensions})"
        )
    
    def __repr__(self) -> str:
        """Detailed string representation of the embedding connector."""
        return (
            f"{self.__class__.__name__}("
            f"connector_id='{self.connector_id()}', "
            f"name='{self._connector_name}', "
            f"provider='{self._provider}', "
            f"os_type='{self._os_type}', "
            f"embedding_type='{self._embedding_type}', "
            f"model_dimensions={self._model_dimensions}, "
            f"region='{self._connector_configs.get('region')}')"
        )
