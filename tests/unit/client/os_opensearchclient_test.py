# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

import logging
from opensearch_py_ml.ml_commons import MLCommonClient

from client import get_client, OsMlClientWrapper
from ml_models import (
    get_ml_model_group,
    get_remote_model_configs,
    LocalMlModel,
    OsBedrockMlModel,
    OsSagemakerMlModel,
)


def test():
    logging.info("Testing os opensearch client...")
    os_client = get_client("os")
    client = OsMlClientWrapper(os_client)
    ml_commons_client = MLCommonClient(os_client=os_client)

    model_group = get_ml_model_group(os_client, ml_commons_client)
    model_group_id = model_group.model_group_id()
    local_model_configs = {"model_group_id": model_group_id}
    local_ml_model = LocalMlModel(
        os_client, ml_commons_client, model_configs=local_model_configs
    )

    logging.info(f"Testing local model")
    logging.info("Setting up for kNN")
    client.setup_for_kNN(
        index_name="amazon_pqa",
        index_settings={"settings": {"number_of_shards": 1}},
        pipeline_name="amazon_pqa_pipeline",
        delete_existing=False,
        ml_model=local_ml_model,
    )
    client.cleanup_kNN(index_name="amazon_pqa", pipeline_name="amazon_pqa_pipeline")

    os_bedrock_configs = get_remote_model_configs(host_type="os", model_type="bedrock")
    os_sagemaker_configs = get_remote_model_configs(
        host_type="os", model_type="sagemaker"
    )
    aos_bedrock_ml_model = OsBedrockMlModel(
        os_client, ml_commons_client, model_configs=os_bedrock_configs
    )
    aos_sagemaker_ml_model = OsSagemakerMlModel(
        os_client, ml_commons_client, model_configs=os_sagemaker_configs
    )

    logging.info(f"Testing bedrock model")
    logging.info("Setting up for kNN")
    client.setup_for_kNN(
        index_name="amazon_pqa",
        index_settings={"settings": {"number_of_shards": 1}},
        pipeline_name="amazon_pqa_pipeline",
        delete_existing=False,
        ml_model=aos_bedrock_ml_model,
    )
    client.cleanup_kNN(index_name="amazon_pqa", pipeline_name="amazon_pqa_pipeline")

    logging.info(f"Testing sagemaker model")
    logging.info("Setting up for kNN")
    client.setup_for_kNN(
        index_name="amazon_pqa",
        index_settings={"settings": {"number_of_shards": 1}},
        pipeline_name="amazon_pqa_pipeline",
        delete_existing=False,
        ml_model=aos_sagemaker_ml_model,
    )
    client.cleanup_kNN(index_name="amazon_pqa", pipeline_name="amazon_pqa_pipeline")
