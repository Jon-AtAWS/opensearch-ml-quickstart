# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Dict

from configs import validate_configs, get_config
from .ml_model_group import MlModelGroup
from .aos_connector_helper import AosConnectorHelper


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
            "url": get_config("OS_SAGEMAKER_URL"),
            "model_dimensions": get_config("OS_SAGEMAKER_MODEL_DIMENSION"),
        }
        validate_configs(configs, list(configs.keys()))
        return configs
    elif connector_type == "sagemaker" and host_type == "aos":
        configs = {
            "arn": get_config("AOS_SAGEMAKER_ARN"),
            "connector_role_name": get_config("AOS_SAGEMAKER_CONNECTOR_ROLE_NAME"),
            "create_connector_role_name": get_config(
                "AOS_SAGEMAKER_CREATE_CONNECTOR_ROLE_NAME"
            ),
            "region": get_config("AOS_SAGEMAKER_REGION"),
            "connector_version": get_config("AOS_SAGEMAKER_CONNECTOR_VERSION"),
            "url": get_config("AOS_SAGEMAKER_URL"),
            "model_dimensions": get_config("AOS_SAGEMAKER_MODEL_DIMENSION"),
        }
        validate_configs(configs, list(configs.keys()))
        return configs
    elif connector_type == "bedrock" and host_type == "os":
        configs = {
            "access_key": get_config("OS_BEDROCK_ACCESS_KEY"),
            "secret_key": get_config("OS_BEDROCK_SECRET_KEY"),
            "region": get_config("OS_BEDROCK_REGION"),
            "connector_version": get_config("OS_BEDROCK_CONNECTOR_VERSION"),
            "url": get_config("OS_BEDROCK_URL"),
            "model_dimensions": get_config("OS_BEDROCK_MODEL_DIMENSION"),
        }
        validate_configs(configs, list(configs.keys()))
        return configs
    else:
        configs = {
            "arn": get_config("AOS_BEDROCK_ARN"),
            "connector_role_name": get_config("AOS_BEDROCK_CONNECTOR_ROLE_NAME"),
            "create_connector_role_name": get_config(
                "AOS_BEDROCK_CREATE_CONNECTOR_ROLE_NAME"
            ),
            "region": get_config("AOS_BEDROCK_REGION"),
            "connector_version": get_config("AOS_BEDROCK_CONNECTOR_VERSION"),
            "model_dimensions": get_config("AOS_BEDROCK_MODEL_DIMENSION"),
            "url": get_config("AOS_BEDROCK_URL"),
        }
        validate_configs(configs, list(configs.keys()))
        return configs


def read_json_file(file_path):
    with open(file_path, "r") as file:
        return json.load(file)


def get_aos_connector_helper(configs) -> AosConnectorHelper:
    required_args = ["region", "username", "password", "domain_name", "aws_user_name"]
    validate_configs(configs, required_args)
    region = configs["region"]
    username = configs["username"]
    password = configs["password"]
    domain_name = configs["domain_name"]
    aws_username = configs["aws_user_name"]
    return AosConnectorHelper(region, domain_name, username, password, aws_username)


def get_ml_model_group(os_client, ml_commons_client) -> MlModelGroup:
    return MlModelGroup(
        os_client=os_client,
        ml_commons_client=ml_commons_client,
    )
