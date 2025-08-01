# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

"""
Embedding Connector Module

This module provides the base class for all embedding model connectors.
Embedding connectors are responsible for connecting to embedding models
that convert text into vector representations for semantic search.

The EmbeddingConnector class serves as an abstract base for both dense
and sparse embedding connectors, handling common functionality like
embedding type validation and dimension configuration.
"""

import logging
from abc import abstractmethod
from typing import Dict, Any, Optional, List
from opensearchpy import OpenSearch

from .ml_connector import MlConnector


class EmbeddingConnector(MlConnector):
    """
    Abstract base class for all embedding model connectors.
    
    This class provides common functionality for embedding connectors including:
    - Embedding type validation (dense/sparse)
    - Model dimension handling
    - Common embedding-specific configuration validation
    - Standardized embedding connector interface
    
    Subclasses should implement OpenSearch-type specific functionality
    (e.g., AOS vs self-managed OpenSearch).
    """
    
    DEFAULT_CONNECTOR_NAME = "Embedding Model Connector"
    DEFAULT_CONNECTOR_DESCRIPTION = "Connector for embedding models that convert text to vectors"
    
    # Supported embedding types
    SUPPORTED_EMBEDDING_TYPES = {"dense", "sparse"}
    
    def __init__(
        self,
        os_client: OpenSearch,
        connector_name: Optional[str] = None,
        connector_description: Optional[str] = None,
        connector_configs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the embedding connector.
        
        Args:
            os_client: OpenSearch client instance
            connector_name: Name for the connector (optional)
            connector_description: Description for the connector (optional)
            connector_configs: Configuration dictionary for the connector
        """
        connector_configs = connector_configs or {}
        
        # Validate embedding type before calling parent constructor
        self._validate_embedding_type(connector_configs)
        
        super().__init__(
            os_client=os_client,
            connector_name=connector_name,
            connector_description=connector_description,
            connector_configs=connector_configs
        )
        
        # Extract embedding-specific configurations
        self._model_dimensions = connector_configs.get("model_dimensions")
        self._max_tokens_per_chunk = connector_configs.get("max_tokens_per_chunk")
        
        logging.info(f"Initialized {self.__class__.__name__} with embedding type: {self._embedding_type}")
    
    def _validate_embedding_type(self, connector_configs: Dict[str, Any]) -> None:
        """
        Validate that the embedding type is supported.
        
        Args:
            connector_configs: Configuration dictionary
            
        Raises:
            ValueError: If embedding type is not supported
        """
        embedding_type = connector_configs.get("embedding_type", "dense")
        if embedding_type not in self.SUPPORTED_EMBEDDING_TYPES:
            raise ValueError(
                f"Unsupported embedding type: {embedding_type}. "
                f"Supported types: {', '.join(self.SUPPORTED_EMBEDDING_TYPES)}"
            )
    
    @abstractmethod
    def _validate_embedding_configs(self) -> None:
        """
        Validate embedding-specific configurations.
        
        This method should be implemented by subclasses to validate
        configurations specific to their embedding model provider
        and OpenSearch deployment type.
        
        Raises:
            ValueError: If required configurations are missing or invalid
        """
        pass
    
    def _validate_configs(self) -> None:
        """
        Validate all connector configurations.
        
        This method calls both the base validation and embedding-specific validation.
        """
        # Call embedding-specific validation
        self._validate_embedding_configs()
        
        # Validate model dimensions for dense embeddings
        if self._embedding_type == "dense" and not self._model_dimensions:
            raise ValueError("model_dimensions is required for dense embedding connectors")
    
    @abstractmethod
    def _get_embedding_model_config(self) -> Dict[str, Any]:
        """
        Get embedding model specific configuration.
        
        Returns:
            Dictionary containing embedding model configuration
        """
        pass
    
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
    
    def get_connector_info(self) -> Dict[str, Any]:
        """
        Get comprehensive connector information.
        
        Returns:
            Dictionary containing connector information including embedding-specific details
        """
        base_info = {
            "connector_id": self.connector_id(),
            "connector_name": self._connector_name,
            "connector_description": self._connector_description,
            "embedding_type": self._embedding_type,
        }
        
        if self._model_dimensions:
            base_info["model_dimensions"] = self._model_dimensions
            
        if self._max_tokens_per_chunk:
            base_info["max_tokens_per_chunk"] = self._max_tokens_per_chunk
            
        return base_info
    
    def __str__(self) -> str:
        """String representation of the embedding connector."""
        return (
            f"{self.__class__.__name__}("
            f"id={self.connector_id()}, "
            f"type={self._embedding_type}, "
            f"dimensions={self._model_dimensions})"
        )
    
    def __repr__(self) -> str:
        """Detailed string representation of the embedding connector."""
        return (
            f"{self.__class__.__name__}("
            f"connector_id='{self.connector_id()}', "
            f"name='{self._connector_name}', "
            f"embedding_type='{self._embedding_type}', "
            f"model_dimensions={self._model_dimensions}, "
            f"max_tokens_per_chunk={self._max_tokens_per_chunk})"
        )
