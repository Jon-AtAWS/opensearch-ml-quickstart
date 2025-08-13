# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

"""
LLM Connector Module

This module provides the universal connector for Bedrock LLM models.
The LlmConnector class handles connections to Bedrock Claude models for
conversational AI and text generation tasks.
"""

import logging
from typing import Dict, Any, Optional
from opensearchpy import OpenSearch

from configs.configuration_manager import validate_configs, get_ml_base_uri
from connectors.helper import get_remote_connector_configs, create_connector_with_iam_roles, create_connector_with_basic_auth
from .ml_connector import MlConnector


class LlmConnector(MlConnector):
    """
    Universal connector for Bedrock LLM models.
    
    This class provides a unified interface for LLM connectors across different
    OpenSearch deployment types (AOS and self-managed OpenSearch).
    
    Only supports Bedrock Claude models.
    """
    
    DEFAULT_CONNECTOR_NAME = "Amazon Bedrock LLM Connector"
    DEFAULT_CONNECTOR_DESCRIPTION = "Connector to Amazon Bedrock language models"
    
    def __init__(
        self,
        os_client: OpenSearch,
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
        Initialize the LLM connector for Bedrock.
        
        Args:
            os_client: OpenSearch client instance
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
            ValueError: If os_type or required parameters are invalid
        """
        # Validate os_type
        if os_type not in {"aos", "os"}:
            raise ValueError(f"os_type must be 'aos' or 'os', got: {os_type}")
        
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
        
        # Auto-load configuration if not provided
        if connector_configs is None:
            connector_configs = get_remote_connector_configs("bedrock", os_type)
            logging.info(f"Auto-loaded Bedrock configuration for {os_type.upper()}")
        
        # Set default names if not provided
        if connector_name is None:
            connector_name = self.DEFAULT_CONNECTOR_NAME
        if connector_description is None:
            connector_description = self.DEFAULT_CONNECTOR_DESCRIPTION
        
        super().__init__(
            os_client=os_client,
            connector_name=connector_name,
            connector_description=connector_description,
            connector_configs=connector_configs
        )
        
        logging.info(f"Initialized {self.__class__.__name__} for Bedrock on {os_type.upper()}")
    
    def _validate_configs(self) -> None:
        """
        Validate LLM connector configurations.
        """
        if self._os_type == "aos":
            # AOS requires IAM role configurations
            required_args = [
                "llm_arn",  # Bedrock model ARN
                "connector_role_name",  # IAM role for connector
                "create_connector_role_name",  # IAM role for creation
                "region",  # AWS region
            ]
        else:  # os_type == "os"
            # Self-managed OpenSearch requires AWS credentials
            required_args = [
                "access_key",  # AWS access key
                "secret_key",  # AWS secret key
                "region",  # AWS region
            ]
        
        validate_configs(self._connector_configs, required_args)
    
    def _get_connector_create_payload_filename(self) -> str:
        """
        Get the connector payload filename for LLM.
        
        Returns:
            Filename for the LLM connector payload
        """
        from .helper import get_connector_payload_filename
        return get_connector_payload_filename(self._provider, self._os_type, "llm")
    
    def _fill_in_connector_create_payload(self, connector_create_payload):
        """
        Fill in the connector creation payload with configuration values.
        
        Args:
            connector_create_payload: Base payload template
            
        Returns:
            Filled payload ready for connector creation
        """
        # Common payload fields
        connector_create_payload["name"] = self._connector_name
        connector_create_payload["description"] = self._connector_description
        connector_create_payload["parameters"]["region"] = self._connector_configs["region"]
        connector_create_payload["version"] = self._connector_configs.get("connector_version", "1")
        
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
            # Use IAM roles for both OpenSearch access and Bedrock access
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
            
            logging.info(f"Created AOS Bedrock LLM connector with ID: {self._connector_id}")
            
        else:  # os_type == "os"
            # Use basic auth for OpenSearch access, credentials in payload for Bedrock access
            self._connector_id = create_connector_with_basic_auth(
                os_client=self._os_client,
                connector_payload=connector_create_payload,
            )
            
            logging.info(f"Created OS Bedrock LLM connector with ID: {self._connector_id}")
    
    def _get_connector_role_inline_policy(self) -> Dict[str, Any]:
        """
        Get the IAM role inline policy for Bedrock access.
        
        Returns:
            IAM policy document for Bedrock access
        """
        if self._os_type != "aos":
            raise ValueError("Connector role policy is only needed for AOS deployments")
        
        llm_arn = self._connector_configs["llm_arn"]
        return {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Resource": [llm_arn],
                    "Action": ["bedrock:InvokeModel"],
                }
            ],
        }
    
    def get_os_type(self) -> str:
        """Get the OpenSearch deployment type."""
        return self._os_type
    
    def __str__(self) -> str:
        """String representation of the LLM connector."""
        return (
            f"{self.__class__.__name__}("
            f"id={self.connector_id()}, "
            f"os_type={self._os_type})"
        )
    
    def __repr__(self) -> str:
        """Detailed string representation of the LLM connector."""
        return (
            f"{self.__class__.__name__}("
            f"connector_id='{self.connector_id()}', "
            f"name='{self._connector_name}', "
            f"os_type='{self._os_type}', "
            f"region='{self._connector_configs.get('region')}')"
        )
