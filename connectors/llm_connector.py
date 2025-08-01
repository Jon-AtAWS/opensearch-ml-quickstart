# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

"""
LLM Connector Module

This module provides the base class for all Large Language Model (LLM) connectors.
LLM connectors are responsible for connecting to language models that generate
text responses, typically used for conversational AI, text generation, and
question-answering tasks.

The LlmConnector class serves as an abstract base for all LLM connectors,
handling common functionality like model parameter configuration, response
formatting, and generation settings.
"""

import logging
from abc import abstractmethod
from typing import Dict, Any, Optional, List, Union
from opensearchpy import OpenSearch

from .ml_connector import MlConnector


class LlmConnector(MlConnector):
    """
    Abstract base class for all Large Language Model (LLM) connectors.
    
    This class provides common functionality for LLM connectors including:
    - Model parameter validation (temperature, max_tokens, etc.)
    - Response format handling
    - Common LLM-specific configuration validation
    - Standardized LLM connector interface
    
    Subclasses should implement OpenSearch-type specific functionality
    (e.g., AOS vs self-managed OpenSearch) and model provider specifics
    (e.g., Bedrock vs SageMaker).
    """
    
    DEFAULT_CONNECTOR_NAME = "Large Language Model Connector"
    DEFAULT_CONNECTOR_DESCRIPTION = "Connector for large language models that generate text responses"
    
    # Default model parameters
    DEFAULT_TEMPERATURE = 0.7
    DEFAULT_MAX_TOKENS = 1000
    DEFAULT_TOP_P = 0.9
    DEFAULT_TOP_K = 50
    
    def __init__(
        self,
        os_client: OpenSearch,
        connector_name: Optional[str] = None,
        connector_description: Optional[str] = None,
        connector_configs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the LLM connector.
        
        Args:
            os_client: OpenSearch client instance
            connector_name: Name for the connector (optional)
            connector_description: Description for the connector (optional)
            connector_configs: Configuration dictionary for the connector
        """
        connector_configs = connector_configs or {}
        
        super().__init__(
            os_client=os_client,
            connector_name=connector_name,
            connector_description=connector_description,
            connector_configs=connector_configs
        )
        
        # Extract LLM-specific configurations with defaults
        self._temperature = connector_configs.get("temperature", self.DEFAULT_TEMPERATURE)
        self._max_tokens = connector_configs.get("max_tokens", self.DEFAULT_MAX_TOKENS)
        self._top_p = connector_configs.get("top_p", self.DEFAULT_TOP_P)
        self._top_k = connector_configs.get("top_k", self.DEFAULT_TOP_K)
        self._stop_sequences = connector_configs.get("stop_sequences", [])
        self._system_prompt = connector_configs.get("system_prompt")
        
        # Model-specific configurations
        self._model_name = connector_configs.get("model_name")
        self._model_version = connector_configs.get("model_version")
        
        # Validate model parameters
        self._validate_model_parameters()
        
        logging.info(f"Initialized {self.__class__.__name__} with model: {self._model_name}")
    
    def _validate_model_parameters(self) -> None:
        """
        Validate LLM model parameters.
        
        Raises:
            ValueError: If model parameters are invalid
        """
        # Validate temperature
        if not 0.0 <= self._temperature <= 2.0:
            raise ValueError(f"Temperature must be between 0.0 and 2.0, got: {self._temperature}")
        
        # Validate max_tokens
        if self._max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got: {self._max_tokens}")
        
        # Validate top_p
        if not 0.0 <= self._top_p <= 1.0:
            raise ValueError(f"top_p must be between 0.0 and 1.0, got: {self._top_p}")
        
        # Validate top_k
        if self._top_k <= 0:
            raise ValueError(f"top_k must be positive, got: {self._top_k}")
    
    @abstractmethod
    def _validate_llm_configs(self) -> None:
        """
        Validate LLM-specific configurations.
        
        This method should be implemented by subclasses to validate
        configurations specific to their LLM provider and OpenSearch deployment type.
        
        Raises:
            ValueError: If required configurations are missing or invalid
        """
        pass
    
    def _validate_configs(self) -> None:
        """
        Validate all connector configurations.
        
        This method calls both the base validation and LLM-specific validation.
        """
        # Call LLM-specific validation
        self._validate_llm_configs()
    
    @abstractmethod
    def _get_llm_model_config(self) -> Dict[str, Any]:
        """
        Get LLM model specific configuration.
        
        Returns:
            Dictionary containing LLM model configuration
        """
        pass
    
    def get_model_parameters(self) -> Dict[str, Any]:
        """
        Get the current model parameters.
        
        Returns:
            Dictionary containing model parameters
        """
        params = {
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
            "top_p": self._top_p,
            "top_k": self._top_k,
        }
        
        if self._stop_sequences:
            params["stop_sequences"] = self._stop_sequences
            
        return params
    
    def get_model_name(self) -> Optional[str]:
        """
        Get the model name.
        
        Returns:
            Model name as string, or None if not configured
        """
        return self._model_name
    
    def get_model_version(self) -> Optional[str]:
        """
        Get the model version.
        
        Returns:
            Model version as string, or None if not configured
        """
        return self._model_version
    
    def get_system_prompt(self) -> Optional[str]:
        """
        Get the system prompt.
        
        Returns:
            System prompt as string, or None if not configured
        """
        return self._system_prompt
    
    def update_model_parameters(
        self,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
    ) -> None:
        """
        Update model parameters.
        
        Args:
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter (0.0 to 1.0)
            top_k: Top-k sampling parameter
            stop_sequences: List of stop sequences
        """
        if temperature is not None:
            if not 0.0 <= temperature <= 2.0:
                raise ValueError(f"Temperature must be between 0.0 and 2.0, got: {temperature}")
            self._temperature = temperature
        
        if max_tokens is not None:
            if max_tokens <= 0:
                raise ValueError(f"max_tokens must be positive, got: {max_tokens}")
            self._max_tokens = max_tokens
        
        if top_p is not None:
            if not 0.0 <= top_p <= 1.0:
                raise ValueError(f"top_p must be between 0.0 and 1.0, got: {top_p}")
            self._top_p = top_p
        
        if top_k is not None:
            if top_k <= 0:
                raise ValueError(f"top_k must be positive, got: {top_k}")
            self._top_k = top_k
        
        if stop_sequences is not None:
            self._stop_sequences = stop_sequences
        
        logging.info(f"Updated model parameters for {self.__class__.__name__}")
    
    def format_prompt(
        self,
        user_input: str,
        context: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Format a prompt for the LLM.
        
        Args:
            user_input: The user's input/question
            context: Optional context information (e.g., from RAG)
            conversation_history: Optional conversation history
            
        Returns:
            Formatted prompt string
        """
        prompt_parts = []
        
        # Add system prompt if configured
        if self._system_prompt:
            prompt_parts.append(f"System: {self._system_prompt}")
        
        # Add context if provided
        if context:
            prompt_parts.append(f"Context: {context}")
        
        # Add conversation history if provided
        if conversation_history:
            for turn in conversation_history:
                role = turn.get("role", "user")
                content = turn.get("content", "")
                prompt_parts.append(f"{role.title()}: {content}")
        
        # Add current user input
        prompt_parts.append(f"User: {user_input}")
        prompt_parts.append("Assistant:")
        
        return "\n\n".join(prompt_parts)
    
    def get_connector_info(self) -> Dict[str, Any]:
        """
        Get comprehensive connector information.
        
        Returns:
            Dictionary containing connector information including LLM-specific details
        """
        base_info = {
            "connector_id": self.connector_id(),
            "connector_name": self._connector_name,
            "connector_description": self._connector_description,
            "model_parameters": self.get_model_parameters(),
        }
        
        if self._model_name:
            base_info["model_name"] = self._model_name
            
        if self._model_version:
            base_info["model_version"] = self._model_version
            
        if self._system_prompt:
            base_info["system_prompt"] = self._system_prompt
            
        return base_info
    
    def __str__(self) -> str:
        """String representation of the LLM connector."""
        return (
            f"{self.__class__.__name__}("
            f"id={self.connector_id()}, "
            f"model={self._model_name}, "
            f"temp={self._temperature})"
        )
    
    def __repr__(self) -> str:
        """Detailed string representation of the LLM connector."""
        return (
            f"{self.__class__.__name__}("
            f"connector_id='{self.connector_id()}', "
            f"name='{self._connector_name}', "
            f"model_name='{self._model_name}', "
            f"temperature={self._temperature}, "
            f"max_tokens={self._max_tokens})"
        )
