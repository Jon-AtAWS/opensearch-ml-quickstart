# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Dict
from dotenv import load_dotenv

from .config import DEFAULT_ENV_PATH


def get_config(config_name, env_path=DEFAULT_ENV_PATH):
    load_dotenv(env_path)
    value = os.getenv(config_name)
    if value == "None":
        value = None
    return value


def validate_configs(configs, required_args):
    for required_arg in required_args:
        if required_arg not in configs or configs[required_arg] == None:
            raise ValueError(
                f"{required_arg} is missing or none, please specify {required_arg} in the configs"
            )


# if default value is none, the arg is required
def parse_arg_from_configs(configs, arg, default_value=None):
    if arg in configs:
        return configs[arg]
    elif default_value != None:
        return default_value
    else:
        raise ValueError(f"{arg} is missing, please specify {arg} in the configs")


def get_client_configs(host_type: str) -> Dict[str, str]:
    required_args = ["host_url", "username", "password"]
    if host_type == "os":
        configs = {
            "port": get_config("OS_PORT"),
            "host_url": get_config("OS_HOST_URL"),
            "username": get_config("OS_USERNAME"),
            "password": get_config("OS_PASSWORD"),
        }
        validate_configs(configs, required_args)
        return configs
    elif host_type == "aos":
        configs = {
            "port": get_config("AOS_PORT"),
            "host_url": get_config("AOS_HOST_URL"),
            "username": get_config("AOS_USERNAME"),
            "password": get_config("AOS_PASSWORD"),
            "domain_name": get_config("AOS_DOMAIN_NAME"),
            "region": get_config("AOS_REGION"),
            "aws_user_name": get_config("AOS_AWS_USER_NAME"),
        }
        validate_configs(configs, required_args)
        return configs
    else:
        raise ValueError("host_type must either be os or aos")


def get_remote_connector_configs(connector_type: str, host_type: str) -> Dict[str, str]:
    if connector_type not in {"sagemaker", "bedrock"}:
        raise ValueError(f"connector_type must be either sagemaker or bedrock")
    if host_type not in {"os", "aos"}:
        raise ValueError(f"host_type must either be os or aos")

    if connector_type == "sagemaker" and host_type == "os":
        configs = {
            "access_key": get_config("OS_SAGEMAKER_ACCESS_KEY"),
            "secret_key": get_config("OS_SAGEMAKER_SECRET_KEY"),
            "region": get_config("OS_SAGEMAKER_REGION"),
            "connector_version": get_config("OS_SAGEMAKER_CONNECTOR_VERSION"),
            "sparse_url": get_config("OS_SAGEMAKER_SPARSE_URL"),
            "dense_url": get_config("OS_SAGEMAKER_DENSE_URL"),
            "model_dimensions": get_config("OS_SAGEMAKER_DENSE_MODEL_DIMENSION"),
        }
        validate_configs(configs, list(configs.keys()))
        return configs
    elif connector_type == "sagemaker" and host_type == "aos":
        configs = {
            "dense_arn": get_config("AOS_SAGEMAKER_SPARSE_ARN"),
            "sparse_arn": get_config("AOS_SAGEMAKER_DENSE_ARN"),
            "connector_role_name": get_config("AOS_SAGEMAKER_CONNECTOR_ROLE_NAME"),
            "create_connector_role_name": get_config(
                "AOS_SAGEMAKER_CREATE_CONNECTOR_ROLE_NAME"
            ),
            "region": get_config("AOS_SAGEMAKER_REGION"),
            "connector_version": get_config("AOS_SAGEMAKER_CONNECTOR_VERSION"),
            "sparse_url": get_config("AOS_SAGEMAKER_SPARSE_URL"),
            "dense_url": get_config("AOS_SAGEMAKER_DENSE_URL"),
            "model_dimensions": get_config("AOS_SAGEMAKER_DENSE_MODEL_DIMENSION"),
        }
        validate_configs(configs, list(configs.keys()))
        return configs
    elif connector_type == "bedrock" and host_type == "os":
        configs = {
            "access_key": get_config("OS_BEDROCK_ACCESS_KEY"),
            "secret_key": get_config("OS_BEDROCK_SECRET_KEY"),
            "region": get_config("OS_BEDROCK_REGION"),
            "connector_version": get_config("OS_BEDROCK_CONNECTOR_VERSION"),
            "dense_url": get_config("OS_BEDROCK_URL"),
            "model_dimensions": get_config("OS_BEDROCK_MODEL_DIMENSION"),
        }
        validate_configs(configs, list(configs.keys()))
        return configs
    else:
        configs = {
            "dense_arn": get_config("AOS_BEDROCK_ARN"),
            "connector_role_name": get_config("AOS_BEDROCK_CONNECTOR_ROLE_NAME"),
            "create_connector_role_name": get_config(
                "AOS_BEDROCK_CREATE_CONNECTOR_ROLE_NAME"
            ),
            "region": get_config("AOS_BEDROCK_REGION"),
            "connector_version": get_config("AOS_BEDROCK_CONNECTOR_VERSION"),
            "model_dimensions": get_config("AOS_BEDROCK_MODEL_DIMENSION"),
            "dense_url": get_config("AOS_BEDROCK_URL"),
        }
        validate_configs(configs, list(configs.keys()))
        return configs
