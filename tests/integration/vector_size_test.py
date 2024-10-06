# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

import json
import time
import logging
from typing import Dict
from opensearchpy import helpers, OpenSearch

from configs import get_config, tasks
from mapping import get_base_mapping, mapping_update
from client import get_index_size, get_client, get_client_configs, OsMlClientWrapper
from data_process import QAndAFileReader
from ml_models import (
    MlModel,
    LocalMlModel,
    OsBedrockMlModel,
    OsSagemakerMlModel,
    AosBedrockMlModel,
    AosSagemakerMlModel,
    get_connector_helper,
    get_remote_model_configs,
)

INDEX_NAME = "amazon_pqa_test"
PIPELINE_NAME = "amazon_pqa_test"
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
    reader = QAndAFileReader(directory=get_config("QANDA_FILE_READER_PATH"))
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
            if (max_cat_docs > 0) and (number_of_docs % max_cat_docs == 0):
                break
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


def get_ml_model(
    host_type, model_type, model_config: Dict[str, str], client: OsMlClientWrapper
) -> MlModel:
    helper = None

    model_name = model_config.get("model_name", None)

    if model_type != "local":
        model_name = f"{host_type}_{model_type}"

    if host_type == "aos":
        helper = get_connector_helper(get_client_configs("aos"))

    if model_type == "local":
        return LocalMlModel(
            os_client=client.os_client,
            ml_commons_client=client.ml_commons_client,
            model_name=model_name,
            model_configs=model_config,
        )
    elif model_type == "sagemaker" and host_type == "os":
        return OsSagemakerMlModel(
            os_client=client.os_client,
            ml_commons_client=client.ml_commons_client,
            model_name=model_name,
            model_configs=model_config,
        )
    elif model_type == "sagemaker" and host_type == "aos":
        return AosSagemakerMlModel(
            os_client=client.os_client,
            ml_commons_client=client.ml_commons_client,
            helper=helper,
            model_name=model_name,
            model_configs=model_config,
        )
    elif model_type == "bedrock" and host_type == "os":
        return OsBedrockMlModel(
            os_client=client.os_client,
            ml_commons_client=client.ml_commons_client,
            model_name=model_name,
            model_configs=model_config,
        )
    elif model_type == "bedrock" and host_type == "aos":
        return AosBedrockMlModel(
            os_client=client.os_client,
            ml_commons_client=client.ml_commons_client,
            helper=helper,
            model_name=model_name,
            model_configs=model_config,
        )


def get_ml_models_and_configs(
    os_client: OsMlClientWrapper,
    aos_client: OsMlClientWrapper,
    model_configs: Dict[str, str],
):
    ml_models = []
    local_model_name = model_configs["model_name"]
    bedrock_model_name = f"{local_model_name}_bedrock"
    sagemaker_model_name = f"{local_model_name}_sagemaker"
    helper = get_connector_helper(get_client_configs("aos"))

    # local os
    local_model_configs = {
        "model_group_id": os_client.ml_model_group.model_group_id(),
    }
    if "model_version" in model_configs:
        local_model_configs["model_version"] = (model_configs["model_version"],)

    model_configs = [
        local_model_configs,
        get_remote_model_configs(host_type="os", model_type="sagemaker"),
        get_remote_model_configs(host_type="aos", model_type="sagemaker"),
        get_remote_model_configs(host_type="os", model_type="bedrock"),
        get_remote_model_configs(host_type="aos", model_type="bedrock"),
    ]

    ml_models.append(
        LocalMlModel(
            os_client=os_client.os_client,
            ml_commons_client=os_client.ml_commons_client,
            model_name=local_model_name,
            model_configs=local_model_configs,
        )
    )

    ml_models.append(
        OsSagemakerMlModel(
            os_client=os_client.os_client,
            ml_commons_client=os_client.ml_commons_client,
            model_name=sagemaker_model_name,
            model_configs=model_configs[1],
        )
    )

    ml_models.append(
        AosSagemakerMlModel(
            os_client=aos_client.os_client,
            ml_commons_client=aos_client.ml_commons_client,
            helper=helper,
            model_name=sagemaker_model_name,
            model_configs=model_configs[2],
        )
    )

    ml_models.append(
        OsBedrockMlModel(
            os_client=os_client.os_client,
            ml_commons_client=os_client.ml_commons_client,
            model_name=bedrock_model_name,
            model_configs=model_configs[3],
        )
    )

    ml_models.append(
        AosBedrockMlModel(
            os_client=aos_client.os_client,
            ml_commons_client=aos_client.ml_commons_client,
            helper=helper,
            model_name=bedrock_model_name,
            model_configs=model_configs[4],
        )
    )

    return ml_models, model_configs


def run_test(task) -> Dict:
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
            client = (
                aos_client
                if isinstance(ml_model, AosBedrockMlModel)
                or isinstance(ml_model, AosSagemakerMlModel)
                else os_client
            )
            index_settings = create_index_settings(
                base_mapping_path=get_config("BASE_MAPPING_PATH"),
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
            os_client.cleanup_kNN(index_name=INDEX_NAME, pipeline_name=PIPELINE_NAME)
            aos_client.cleanup_kNN(index_name=INDEX_NAME, pipeline_name=PIPELINE_NAME)
    else:
        logging.info("Setting up without KNN")
        clients = [os_client, aos_client]
        index_settings = create_index_settings(
            base_mapping_path=get_config("BASE_MAPPING_PATH"), task=task
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
    for task_name, task in tasks.items():
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
