# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import json
import logging
from typing import Dict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs import BASE_MAPPING_PATH, PIPELINE_FIELD_MAP, QANDA_FILE_READER_PATH
from client import (
    OsMlClientWrapper,
    get_client,
)
from data_process import QAndAFileReader
from mapping import get_base_mapping, mapping_update
from ml_models import get_remote_connector_configs, MlModel
from main import get_ml_model, load_category

logging.basicConfig(
    format="%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)


def create_index_settings(base_mapping_path, index_config):
    settings = get_base_mapping(base_mapping_path)
    pipeline_name = index_config["pipeline_name"]
    knn_settings = {
        "settings": {"index": {"knn": True}, "default_pipeline": pipeline_name},
        "mappings": {
            "properties": {
                "chunk": {"type": "text", "index": False},
                "chunk_embedding": {
                    "type": "rank_features",
                },
            }
        },
    }
    mapping_update(settings, knn_settings)
    return settings


def load_dataset(
    client: OsMlClientWrapper,
    ml_model: MlModel,
    pqa_reader: QAndAFileReader,
    config: Dict[str, str],
    delete_existing: bool,
    index_name: str,
    pipeline_name: str,
):
    if delete_existing:
        logging.info(f"Deleting existing index {index_name}")
        client.delete_then_create_index(
            index_name=config["index_name"], settings=config["index_settings"]
        )

    logging.info("Setting up for KNN")
    client.setup_for_kNN(
        ml_model=ml_model,
        index_name=config["index_name"],
        pipeline_name=pipeline_name,
        index_settings=config["index_settings"],
        pipeline_field_map=config["pipeline_field_map"],
        delete_existing=delete_existing,
        embedding_type=config["embedding_type"],
    )

    for category in config["categories"]:
        load_category(
            client=client.os_client,
            pqa_reader=pqa_reader,
            category=category,
            config=config,
        )

    if config["cleanup"]:
        client.cleanup_kNN(
            ml_model=ml_model,
            index_name=config["index_name"],
            pipeline_name=pipeline_name,
        )


def main():
    host_type = "aos"
    model_type = "sagemaker"
    index_name = "sparse_search"
    dataset_path = QANDA_FILE_READER_PATH
    number_of_docs = -1
    client = OsMlClientWrapper(get_client(host_type))

    pqa_reader = QAndAFileReader(
        directory=dataset_path, max_number_of_docs=number_of_docs
    )

    categories = [
        "earbud headphones",
        "headsets",
        "diffusers",
        "mattresses",
        "mp3 and mp4 players",
        "sheet and pillowcase sets",
        "batteries",
        "casual",
        "costumes",
    ]
    config = {"with_knn": True, "pipeline_field_map": PIPELINE_FIELD_MAP}

    pipeline_name = "amazon_pqa_pipeline"
    embedding_type = "sparse"
    config["categories"] = categories
    config["index_name"] = index_name
    config["pipeline_name"] = pipeline_name
    config["embedding_type"] = embedding_type

    model_name = f"{host_type}_{model_type}"

    model_config = get_remote_connector_configs(
        host_type=host_type, connector_type=model_type
    )
    model_config["model_name"] = model_name
    model_config["embedding_type"] = embedding_type
    config.update(model_config)

    ml_model = get_ml_model(
        host_type=host_type,
        model_type=model_type,
        model_config=model_config,
        client=client,
    )

    config["index_settings"] = create_index_settings(
        base_mapping_path=BASE_MAPPING_PATH,
        index_config=config,
    )
    config["cleanup"] = False

    logging.info(f"Config:\n {json.dumps(config, indent=4)}")

    load_dataset(
        client,
        ml_model,
        pqa_reader,
        config,
        delete_existing=False,
        index_name=index_name,
        pipeline_name=pipeline_name,
    )

    query_text = input("Please input your search query text: ")
    search_query = {
        "_source": {"include": "chunk"},
        "query": {
            "neural_sparse": {
                "chunk_embedding": {
                    "query_text": query_text,
                    "model_id": ml_model.model_id(),
                }
            }
        },
    }
    search_results = client.os_client.search(index=index_name, body=search_query)
    hits = search_results["hits"]["hits"]
    hits = [hit["_source"]["chunk"] for hit in hits]
    hits = list(set(hits))
    for i, hit in enumerate(hits):
        print(f"{i + 1}th search result:\n {hit}")


if __name__ == "__main__":
    main()
