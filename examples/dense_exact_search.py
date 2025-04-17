# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import logging
from typing import Dict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs import (
    get_remote_connector_configs,
    BASE_MAPPING_PATH,
    PIPELINE_FIELD_MAP,
    QANDA_FILE_READER_PATH,
)
from client import (
    OsMlClientWrapper,
    get_client,
    load_category,
)
from data_process import QAndAFileReader
from mapping import get_base_mapping, mapping_update
from ml_models import get_ml_model, MlModel

logging.basicConfig(
    format="%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)


def create_index_settings(base_mapping_path, index_config):
    settings = get_base_mapping(base_mapping_path)
    pipeline_name = index_config["pipeline_name"]
    model_dimension = index_config["model_dimensions"]
    knn_settings = {
        "settings": {"index": {"knn": True}, "default_pipeline": pipeline_name},
        "mappings": {
            "properties": {
                "chunk": {"type": "text", "index": False},
                "chunk_embedding": {
                    "type": "knn_vector",
                    "dimension": model_dimension,
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
    index_name: str,
    pipeline_name: str,
):
    if client.os_client.indices.exists(index_name):
        logging.info(f"Index {index_name} already exists. Skipping loading dataset")
        return

    logging.info(f"Creating index {index_name}")
    client.idempotent_create_index(
        index_name=config["index_name"], settings=config["index_settings"]
    )

    logging.info("Setting up for KNN")
    client.setup_for_kNN(
        ml_model=ml_model,
        index_name=config["index_name"],
        pipeline_name=pipeline_name,
        index_settings=config["index_settings"],
        pipeline_field_map=config["pipeline_field_map"],
        embedding_type=config["embedding_type"],
    )

    for category in config["categories"]:
        load_category(
            client=client.os_client,
            pqa_reader=pqa_reader,
            category=category,
            config=config,
        )


def main():
    host_type = "aos"
    model_type = "sagemaker"
    index_name = "dense_exact_search"
    embedding_type = "dense"
    pipeline_name = "dense-ingest-pipeline"

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
    number_of_docs_per_category = 5000
    dataset_path = QANDA_FILE_READER_PATH

    client = OsMlClientWrapper(get_client(host_type))
    pqa_reader = QAndAFileReader(
        directory=dataset_path, max_number_of_docs=number_of_docs_per_category
    )

    config = {
        "with_knn": True,
        "pipeline_field_map": PIPELINE_FIELD_MAP,
        "categories": categories,
        "index_name": index_name,
        "pipeline_name": pipeline_name,
        "embedding_type": embedding_type,
    }

    model_name = f"{host_type}_{model_type}"
    model_config = get_remote_connector_configs(
        host_type=host_type, connector_type=model_type
    )
    model_config["model_name"] = model_name
    model_config["embedding_type"] = embedding_type
    ml_model = get_ml_model(
        host_type=host_type,
        model_type=model_type,
        model_config=model_config,
        os_client=client.os_client,
        ml_commons_client=client.ml_commons_client,
        model_group_id=client.ml_model_group.model_group_id(),
    )

    config.update(model_config)
    config["index_settings"] = create_index_settings(
        base_mapping_path=BASE_MAPPING_PATH,
        index_config=config,
    )

    load_dataset(
        client,
        ml_model,
        pqa_reader,
        config,
        index_name=index_name,
        pipeline_name=pipeline_name,
    )

    query_text = input("Please input your search query text: ")
    search_query = {
        "query": {
            "neural": {
                "chunk_embedding": {
                    "query_text": query_text,
                    "model_id": ml_model.model_id(),
                }
            }
        },
    }
    search_results = client.os_client.search(index=index_name, body=search_query)
    hits = search_results["hits"]["hits"]
    for hit in hits:
        print(
            "--------------------------------------------------------------------------------"
        )
        print(f'Category name: {hit["_source"]["category_name"]}')
        print()
        print(f'Item name: {hit["_source"]["item_name"]}')
        print()
        print(f'Production description: {hit["_source"]["product_description"]}')
        print()


if __name__ == "__main__":
    main()
