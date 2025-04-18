# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

import logging
from unittest.mock import patch
from opensearch_py_ml.ml_commons import MLCommonClient

from configs import get_remote_connector_configs
from client import get_client, OsMlClientWrapper
from ml_models import (
    get_ml_model_group,
    LocalMlModel,
    OsBedrockMlConnector,
    OsSagemakerMlConnector,
    RemoteMlModel,
)


def test():
    logging.info("Testing os opensearch client...")
    os_client = get_client("os")
    client = OsMlClientWrapper(os_client)
    ml_commons_client = MLCommonClient(os_client=os_client)
    model_group_id = client.ml_model_group.model_group_id()

    model_group = get_ml_model_group(os_client, ml_commons_client)
    model_group_id = model_group.model_group_id()
    local_model_configs = {"model_group_id": model_group_id}
    local_dense_model = LocalMlModel(
        os_client,
        ml_commons_client,
        model_group_id=model_group_id,
        model_configs=local_model_configs,
    )
    local_sparse_model = LocalMlModel(
        os_client,
        ml_commons_client,
        model_group_id=model_group_id,
        model_configs=local_model_configs,
        model_name="amazon/neural-sparse/opensearch-neural-sparse-encoding-v1",
    )

    os_bedrock_configs = get_remote_connector_configs(
        host_type="os", connector_type="bedrock"
    )
    os_sagemaker_configs = get_remote_connector_configs(
        host_type="os", connector_type="sagemaker"
    )
    os_bedrock_ml_connector = OsBedrockMlConnector(
        os_client=os_client,
        connector_configs=os_bedrock_configs,
    )
    os_sagemaker_ml_connector = OsSagemakerMlConnector(
        os_client=os_client,
        connector_configs=os_sagemaker_configs,
    )
    os_bedrock_ml_model = RemoteMlModel(
        os_client,
        ml_commons_client,
        model_group_id=model_group_id,
        model_name="os_bedrock_model",
        ml_connector=os_bedrock_ml_connector,
    )
    os_sagemaker_ml_model = RemoteMlModel(
        os_client,
        ml_commons_client,
        model_group_id=model_group_id,
        model_name="os_sagemaker_model",
        ml_connector=os_sagemaker_ml_connector,
    )

    logging.info(f"Testing local dense model")
    logging.info("Setting up for kNN")
    client.setup_for_kNN(
        index_name="amazon_pqa",
        index_settings={"settings": {"number_of_shards": 1}},
        pipeline_name="amazon_pqa_pipeline",
        delete_existing=False,
        ml_model=local_dense_model,
    )
    with patch("builtins.input", return_value="y"):
        client.cleanup_kNN(index_name="amazon_pqa", pipeline_name="amazon_pqa_pipeline")

    logging.info(f"Testing local sparse model")
    logging.info("Setting up for kNN")
    client.setup_for_kNN(
        index_name="amazon_pqa",
        index_settings={"settings": {"number_of_shards": 1}},
        pipeline_name="amazon_pqa_pipeline",
        delete_existing=False,
        ml_model=local_sparse_model,
    )
    with patch("builtins.input", return_value="y"):
        client.cleanup_kNN(index_name="amazon_pqa", pipeline_name="amazon_pqa_pipeline")

    logging.info(f"Testing bedrock model")
    logging.info("Setting up for kNN")
    client.setup_for_kNN(
        index_name="amazon_pqa",
        index_settings={"settings": {"number_of_shards": 1}},
        pipeline_name="amazon_pqa_pipeline",
        delete_existing=True,
        ml_model=os_bedrock_ml_model,
    )
    with patch("builtins.input", return_value="y"):
        client.cleanup_kNN(index_name="amazon_pqa", pipeline_name="amazon_pqa_pipeline")

    logging.info(f"Testing sagemaker model")
    logging.info("Setting up for kNN")
    client.setup_for_kNN(
        index_name="amazon_pqa",
        index_settings={"settings": {"number_of_shards": 1}},
        pipeline_name="amazon_pqa_pipeline",
        delete_existing=True,
        ml_model=os_sagemaker_ml_model,
    )
    with patch("builtins.input", return_value="y"):
        client.cleanup_kNN(index_name="amazon_pqa", pipeline_name="amazon_pqa_pipeline")
