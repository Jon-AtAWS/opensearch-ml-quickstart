# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

"""
Connector helper functions for retrieving remote connector configurations.
"""

from typing import Dict
from configs.configuration_manager import get_raw_config_value


def get_remote_connector_configs(connector_type: str, host_type: str) -> Dict[str, str]:
    """Get remote connector configurations for the specified connector and host type."""
    if connector_type not in {"sagemaker", "bedrock"}:
        raise ValueError(f"connector_type must be either sagemaker or bedrock")
    if host_type not in {"os", "aos"}:
        raise ValueError(f"host_type must either be os or aos")

    if connector_type == "sagemaker" and host_type == "os":
        configs = {
            "access_key": get_raw_config_value("OS_SAGEMAKER_ACCESS_KEY"),
            "secret_key": get_raw_config_value("OS_SAGEMAKER_SECRET_KEY"),
            "region": get_raw_config_value("OS_SAGEMAKER_REGION"),
            "connector_version": get_raw_config_value("OS_SAGEMAKER_CONNECTOR_VERSION"),
            "sparse_url": get_raw_config_value("OS_SAGEMAKER_SPARSE_URL"),
            "dense_url": get_raw_config_value("OS_SAGEMAKER_DENSE_URL"),
            "model_dimensions": get_raw_config_value("OS_SAGEMAKER_DENSE_MODEL_DIMENSION"),
        }
        # Validate that all required configs are present
        missing = [k for k, v in configs.items() if v is None or v == ""]
        if missing:
            raise ValueError(f"Missing required OS Sagemaker configurations: {missing}")
        return configs
    elif connector_type == "sagemaker" and host_type == "aos":
        configs = {
            "dense_arn": get_raw_config_value("AOS_SAGEMAKER_SPARSE_ARN"),
            "sparse_arn": get_raw_config_value("AOS_SAGEMAKER_DENSE_ARN"),
            "connector_role_name": get_raw_config_value("AOS_SAGEMAKER_CONNECTOR_ROLE_NAME"),
            "create_connector_role_name": get_raw_config_value("AOS_SAGEMAKER_CREATE_CONNECTOR_ROLE_NAME"),
            "region": get_raw_config_value("AOS_SAGEMAKER_REGION"),
            "connector_version": get_raw_config_value("AOS_SAGEMAKER_CONNECTOR_VERSION"),
            "sparse_url": get_raw_config_value("AOS_SAGEMAKER_SPARSE_URL"),
            "dense_url": get_raw_config_value("AOS_SAGEMAKER_DENSE_URL"),
            "model_dimensions": get_raw_config_value("AOS_SAGEMAKER_DENSE_MODEL_DIMENSION"),
        }
        # Validate that all required configs are present
        missing = [k for k, v in configs.items() if v is None or v == ""]
        if missing:
            raise ValueError(f"Missing required AOS Sagemaker configurations: {missing}")
        return configs
    elif connector_type == "bedrock" and host_type == "os":
        configs = {
            "access_key": get_raw_config_value("OS_BEDROCK_ACCESS_KEY"),
            "secret_key": get_raw_config_value("OS_BEDROCK_SECRET_KEY"),
            "region": get_raw_config_value("OS_BEDROCK_REGION"),
            "connector_version": get_raw_config_value("OS_BEDROCK_CONNECTOR_VERSION"),
            "dense_url": get_raw_config_value("OS_BEDROCK_URL"),
            "model_dimensions": get_raw_config_value("OS_BEDROCK_MODEL_DIMENSION"),
        }
        # Validate that all required configs are present
        missing = [k for k, v in configs.items() if v is None or v == ""]
        if missing:
            raise ValueError(f"Missing required OS Bedrock configurations: {missing}")
        return configs
    else:  # bedrock and aos
        configs = {
            "dense_arn": get_raw_config_value("AOS_BEDROCK_ARN"),
            "connector_role_name": get_raw_config_value("AOS_BEDROCK_CONNECTOR_ROLE_NAME"),
            "create_connector_role_name": get_raw_config_value("AOS_BEDROCK_CREATE_CONNECTOR_ROLE_NAME"),
            "region": get_raw_config_value("AOS_BEDROCK_REGION"),
            "connector_version": get_raw_config_value("AOS_BEDROCK_CONNECTOR_VERSION"),
            "model_dimensions": get_raw_config_value("AOS_BEDROCK_MODEL_DIMENSION"),
            "dense_url": get_raw_config_value("AOS_BEDROCK_URL"),
        }
        # Validate that all required configs are present
        missing = [k for k, v in configs.items() if v is None or v == ""]
        if missing:
            raise ValueError(f"Missing required AOS Bedrock configurations: {missing}")
        return configs
