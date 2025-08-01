# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Dict

from opensearchpy import OpenSearch
from configs import validate_configs
from configs.configuration_manager import get_opensearch_config
from opensearch_py_ml.ml_commons import MLCommonClient

from .ml_model import MlModel
from .ml_model_group import MlModelGroup
from .local_ml_model import LocalMlModel
from .remote_ml_model import RemoteMlModel
from connectors import (
    OsBedrockMlConnector,
    OsSagemakerMlConnector,
    AosBedrockMlConnector,
    AosSagemakerMlConnector,
    AosConnectorHelper,
)


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


def get_ml_model(
    host_type,
    model_type,
    model_group_id,
    model_config: Dict[str, str],
    os_client: OpenSearch,
    ml_commons_client: MLCommonClient,
) -> MlModel:
    aos_connector_helper = None

    model_name = model_config.get("model_name", None)
    embedding_type = model_config.get("embedding_type", "dense")

    if model_type != "local":
        model_name = f"{host_type}_{model_type}_{embedding_type}"
        connector_name = f"{host_type}_{model_type}_{embedding_type}"

    if host_type == "aos":
        aos_config = get_opensearch_config("aos")
        aos_connector_helper = get_aos_connector_helper({
            "region": aos_config.region,
            "username": aos_config.username,
            "password": aos_config.password,
            "domain_name": aos_config.domain_name,
            "aws_user_name": aos_config.aws_user_name,
        })

    if model_type == "local":
        return LocalMlModel(
            os_client=os_client,
            ml_commons_client=ml_commons_client,
            model_group_id=model_group_id,
            model_name=model_name,
            model_configs=model_config,
        )
    elif model_type == "sagemaker" and host_type == "os":
        ml_connector = OsSagemakerMlConnector(
            os_client=os_client,
            connector_name=connector_name,
            connector_configs=model_config,
        )

    elif model_type == "sagemaker" and host_type == "aos":
        ml_connector = AosSagemakerMlConnector(
            os_client=os_client,
            connector_name=connector_name,
            aos_connector_helper=aos_connector_helper,
            connector_configs=model_config,
        )
    elif model_type == "bedrock" and host_type == "os":
        ml_connector = OsBedrockMlConnector(
            os_client=os_client,
            connector_name=connector_name,
            connector_configs=model_config,
        )
    elif model_type == "bedrock" and host_type == "aos":
        ml_connector = AosBedrockMlConnector(
            os_client=os_client,
            connector_name=connector_name,
            aos_connector_helper=aos_connector_helper,
            connector_configs=model_config,
        )
    return RemoteMlModel(
        os_client=os_client,
        ml_commons_client=ml_commons_client,
        ml_connector=ml_connector,
        model_group_id=model_group_id,
        model_name=model_name,
        model_configs=model_config,
    )
