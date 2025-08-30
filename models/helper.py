# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Dict

from opensearchpy import OpenSearch
from configs.configuration_manager import validate_configs, get_opensearch_config
from opensearch_py_ml.ml_commons import MLCommonClient

from .ml_model import MlModel
from .ml_model_group import MlModelGroup
from .local_ml_model import LocalMlModel
from .remote_ml_model import RemoteMlModel
from connectors import (
    EmbeddingConnector,
)


def get_ml_model_group(os_client, ml_commons_client) -> MlModelGroup:
    return MlModelGroup(
        os_client=os_client,
        ml_commons_client=ml_commons_client,
    )


def get_ml_model(
    host_type,
    model_host,
    model_group_id,
    model_config: Dict[str, str],
    os_client: OpenSearch,
    ml_commons_client: MLCommonClient,
) -> MlModel:
    model_name = model_config.get("model_name", None)
    embedding_type = model_config.get("embedding_type", "dense")

    if model_host != "local":
        model_name = f"{host_type}_{model_host}_{embedding_type}"
        connector_name = f"{host_type}_{model_host}_{embedding_type}"

    if model_host == "local":
        return LocalMlModel(
            os_client=os_client,
            ml_commons_client=ml_commons_client,
            model_group_id=model_group_id,
            model_name=model_name,
            model_configs=model_config,
        )
    elif model_host == "sagemaker" and host_type == "os":
        # Use the new universal EmbeddingConnector for OS SageMaker
        ml_connector = EmbeddingConnector(
            os_client=os_client,
            provider="sagemaker",
            os_type="os",
            connector_name=connector_name,
            connector_configs=model_config,
        )
    elif model_host == "bedrock" and host_type == "os":
        # Use the new universal EmbeddingConnector for OS Bedrock
        ml_connector = EmbeddingConnector(
            os_client=os_client,
            provider="bedrock",
            os_type="os",
            connector_name=connector_name,
            connector_configs=model_config,
        )
    elif model_host == "sagemaker" and host_type == "aos":
        # Use the new universal EmbeddingConnector for AOS SageMaker
        aos_config = get_opensearch_config("aos")
        ml_connector = EmbeddingConnector(
            os_client=os_client,
            provider="sagemaker",
            os_type="aos",
            opensearch_domain_url=aos_config.host_url,
            opensearch_domain_arn=f"arn:aws:es:{aos_config.region}:*:domain/{aos_config.domain_name}",
            opensearch_username=aos_config.username,
            opensearch_password=aos_config.password,
            aws_user_name=aos_config.aws_user_name,
            region=aos_config.region,
            connector_name=connector_name,
            connector_configs=model_config,
        )
    elif model_host == "bedrock" and host_type == "aos":
        # Use the new universal EmbeddingConnector for AOS Bedrock
        aos_config = get_opensearch_config("aos")
        ml_connector = EmbeddingConnector(
            os_client=os_client,
            provider="bedrock",
            os_type="aos",
            opensearch_domain_url=aos_config.host_url,
            opensearch_domain_arn=f"arn:aws:es:{aos_config.region}:*:domain/{aos_config.domain_name}",
            opensearch_username=aos_config.username,
            opensearch_password=aos_config.password,
            aws_user_name=aos_config.aws_user_name,
            region=aos_config.region,
            connector_name=connector_name,
            connector_configs=model_config,
        )
    else:
        raise ValueError(f"Unsupported combination: host_type='{host_type}', model_host='{model_host}'")
    
    return RemoteMlModel(
        os_client=os_client,
        ml_commons_client=ml_commons_client,
        ml_connector=ml_connector,
        model_group_id=model_group_id,
        model_name=model_name,
        model_configs=model_config,
    )
