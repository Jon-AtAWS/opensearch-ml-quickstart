# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

import json
import time
import logging
from typing import Dict
from unittest.mock import patch
from opensearchpy import helpers, OpenSearch

from configs import get_remote_connector_configs, tasks, BASE_MAPPING_PATH, QANDA_FILE_READER_PATH
from mapping import get_base_mapping, mapping_update
from client import get_index_size, get_client, get_client_configs, OsMlClientWrapper
from data_process import QAndAFileReader
from ml_models import (
    LocalMlModel,
    RemoteMlModel,
    OsBedrockMlConnector,
    OsSagemakerMlConnector,
    AosBedrockMlConnector,
    AosSagemakerMlConnector,
    get_aos_connector_helper,
)

INDEX_NAME = "amazon_pqa_test"
PIPELINE_NAME = "amazon_pqa_test"
TEST_TASK_NAME = "knn_768"
client_types = ["os", "aos"]
model_types = ["os_local", "os_sagemaker", "aos_sagemaker", "os_bedrock", "aos_bedrock"]

SPACE_SEPARATOR = " "
SEPARATOR = SPACE_SEPARATOR


def create_index_settings(base_mapping_path, task: Dict[str, str], model_config=dict()):
    settings = get_base_mapping(base_mapping_path)
    model_dimension = model_config.get("model_dimensions", task["model_dimensions"])
    if task["with_knn"]:
        pipeline_name = PIPELINE_NAME
        knn_settings = {
            "settings": {"index": {"knn": True}, "default_pipeline": pipeline_name},
            "mappings": {
                "properties": {
                    "chunk": {"type": "text", "index": False},
                    "chunk_embedding": {
                        "type": "knn_vector",
                        "dimension": model_dimension,
                        "method": {
                            "name": "hnsw",
                            "space_type": "l2",
                            "engine": "nmslib",
                            "parameters": {"ef_construction": 128, "m": 24},
                        },
                    },
                }
            },
        }
        mapping_update(settings, knn_settings)
    if "compression" in task and task["compression"] != "best_compression":
        compression_settings = {"settings": {"index": {"codec": task["compression"]}}}
        mapping_update(settings, compression_settings)
    return settings


def send_bulk_ignore_exceptions(client: OpenSearch, docs):
    try:
        status = helpers.bulk(
            client,
            docs,
            chunk_size=32,
            request_timeout=300,
            max_retries=10,
            raise_on_error=False,
        )
        return status
    except Exception as e:
        logging.error(f"Error sending bulk: {e}")


def load_to_opensearch(
    client: OsMlClientWrapper,
    index_name,
    index_settings,
    categories,
    max_cat_docs=10000,
    add_chunk=True,
):
    reader = QAndAFileReader(
        directory=QANDA_FILE_READER_PATH, max_number_of_docs=max_cat_docs
    )
    results = {}
    bytes = 0

    logging.info(f"Loading categories: {categories}")
    for category_name in categories:
        logging.info(f'Loading category "{category_name}"')
        number_of_docs = 0
        docs = []
        start_time = time.time()
        for doc in reader.questions_for_category(
            reader.amazon_pqa_category_name_to_constant(category_name), enriched=True
        ):
            doc["_index"] = index_name
            doc["_id"] = doc["question_id"]
            if add_chunk:
                doc["chunk"] = SPACE_SEPARATOR.join(
                    [doc["product_description"], doc["brand_name"], doc["item_name"]]
                )
                doc["chunk"] = SPACE_SEPARATOR.join(doc["chunk"].split()[:500])
                # documents less than 4 words are meaning less
                if len(doc["chunk"]) <= 4:
                    logging.info(f"Empty chunk for {doc}")
                    continue
            docs.append(doc)
            bytes += len(json.dumps(doc))
            number_of_docs += 1
            if number_of_docs % 2000 == 0:
                logging.info(f"Sending {number_of_docs} docs")
                send_bulk_ignore_exceptions(client.os_client, docs)
                docs = []
        if len(docs) > 0:
            send_bulk_ignore_exceptions(client.os_client, docs)
        end_time = time.time()
        results[category_name] = {
            "index_size": get_index_size(client.os_client, INDEX_NAME, unit="b"),
            "bytes": bytes,
            "bytes_on_disk": reader.file_size(category_name=category_name),
            "number_of_docs": number_of_docs,
            "wall_time": end_time - start_time,
        }
        client.delete_then_create_index(index_name=INDEX_NAME, settings=index_settings)
        logging.info(f"Results: {results}")
    return results


def get_ml_models_and_configs(
    os_client: OsMlClientWrapper,
    aos_client: OsMlClientWrapper,
    model_configs: Dict[str, str],
):
    ml_models = []
    local_model_name = model_configs["model_name"]
    bedrock_model_name = f"{local_model_name}_bedrock"
    sagemaker_model_name = f"{local_model_name}_sagemaker"
    aos_connector_helper = get_aos_connector_helper(get_client_configs("aos"))
    os_model_group_id = os_client.ml_model_group.model_group_id()
    aos_model_group_id = aos_client.ml_model_group.model_group_id()

    # local os
    local_model_configs = {}
    if "model_version" in model_configs:
        local_model_configs["model_version"] = model_configs["model_version"]

    model_configs = [
        local_model_configs,
        get_remote_connector_configs(host_type="os", connector_type="sagemaker"),
        get_remote_connector_configs(host_type="aos", connector_type="sagemaker"),
        get_remote_connector_configs(host_type="os", connector_type="bedrock"),
        get_remote_connector_configs(host_type="aos", connector_type="bedrock"),
    ]
    model_configs[0]["host_type"] = "os"
    model_configs[1]["host_type"] = "os"
    model_configs[2]["host_type"] = "aos"
    model_configs[3]["host_type"] = "os"
    model_configs[4]["host_type"] = "aos"

    ml_models.append(
        LocalMlModel(
            os_client=os_client.os_client,
            ml_commons_client=os_client.ml_commons_client,
            model_group_id=os_model_group_id,
            model_name=local_model_name,
            model_configs=model_configs[0],
        )
    )

    os_sagemaker_ml_connector = OsSagemakerMlConnector(
        os_client=os_client.os_client,
        connector_configs=model_configs[1],
    )

    aos_sagemaker_ml_connector = AosSagemakerMlConnector(
        os_client=aos_client.os_client,
        aos_connector_helper=aos_connector_helper,
        connector_configs=model_configs[2],
    )

    os_bedrock_ml_connector = OsBedrockMlConnector(
        os_client=os_client.os_client,
        connector_configs=model_configs[3],
    )

    aos_bedrock_ml_connector = AosBedrockMlConnector(
        os_client=aos_client.os_client,
        aos_connector_helper=aos_connector_helper,
        connector_configs=model_configs[4],
    )

    ml_models.append(
        RemoteMlModel(
            os_client=os_client.os_client,
            ml_commons_client=os_client.ml_commons_client,
            ml_connector=os_sagemaker_ml_connector,
            model_group_id=os_model_group_id,
            model_name=sagemaker_model_name,
        )
    )

    ml_models.append(
        RemoteMlModel(
            os_client=aos_client.os_client,
            ml_commons_client=aos_client.ml_commons_client,
            ml_connector=aos_sagemaker_ml_connector,
            model_group_id=aos_model_group_id,
            model_name=sagemaker_model_name,
        )
    )

    ml_models.append(
        RemoteMlModel(
            os_client=os_client.os_client,
            ml_commons_client=os_client.ml_commons_client,
            ml_connector=os_bedrock_ml_connector,
            model_group_id=os_model_group_id,
            model_name=bedrock_model_name,
        )
    )

    ml_models.append(
        RemoteMlModel(
            os_client=aos_client.os_client,
            ml_commons_client=aos_client.ml_commons_client,
            ml_connector=aos_bedrock_ml_connector,
            model_group_id=aos_model_group_id,
            model_name=bedrock_model_name,
        )
    )

    return ml_models, model_configs


def run_test(task: Dict[str, str]) -> Dict:
    cleanup = task["cleanup"]
    with_knn = task["with_knn"]
    categories = task["categories"]
    max_cat_docs = task["max_cat_docs"]
    pipeline_field_map = task["pipeline_field_map"]

    os_client = OsMlClientWrapper(get_client("os"))
    aos_client = OsMlClientWrapper(get_client("aos"))

    results = dict()

    if with_knn:
        logging.info("Setting up for KNN")
        ml_models, ml_model_configs = get_ml_models_and_configs(
            os_client, aos_client, task
        )
        for ml_model, ml_model_config, model_type in zip(
            ml_models, ml_model_configs, model_types
        ):
            client = aos_client if ml_model_config["host_type"] == "aos" else os_client
            index_settings = create_index_settings(
                base_mapping_path=BASE_MAPPING_PATH,
                task=task,
                model_config=ml_model_config,
            )
            client.setup_for_kNN(
                ml_model=ml_model,
                index_name=INDEX_NAME,
                pipeline_name=PIPELINE_NAME,
                index_settings=index_settings,
                pipeline_field_map=pipeline_field_map,
                delete_existing=True,
            )
            results[model_type] = load_to_opensearch(
                client,
                INDEX_NAME,
                index_settings=index_settings,
                max_cat_docs=max_cat_docs,
                add_chunk=with_knn,
                categories=categories,
            )
            if cleanup:
                with patch("builtins.input", return_value="y"):
                    client.cleanup_kNN(
                        index_name=INDEX_NAME, pipeline_name=PIPELINE_NAME
                    )
    else:
        logging.info("Setting up without KNN")
        clients = [os_client, aos_client]
        index_settings = create_index_settings(
            base_mapping_path=BASE_MAPPING_PATH, task=task
        )
        for client, client_type in zip(clients, client_types):
            client.setup_without_kNN(
                index_name=INDEX_NAME, index_settings=index_settings
            )
            results[client_type] = load_to_opensearch(
                client,
                INDEX_NAME,
                index_settings=index_settings,
                max_cat_docs=max_cat_docs,
                add_chunk=with_knn,
                categories=categories,
            )
    return results


def test():
    accumulated_results = dict()
    test_tasks = {TEST_TASK_NAME: tasks[TEST_TASK_NAME]}
    for task_name, task in test_tasks.items():
        logging.info(f'Running task "{task_name}"')
        results = run_test(task)
        accumulated_results[task_name] = results
        logging.info(f"------------------------------------------------------")
        logging.info(f"Accumulated results after {task_name}\n{accumulated_results}")

    for task_name, result in accumulated_results.items():
        logging.info(f"------------------------------------------------------")
        logging.info(task_name)
        logging.info(result)


if __name__ == "__main__":
    test()
