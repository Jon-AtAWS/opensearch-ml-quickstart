# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import json
import logging
from typing import Dict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs import get_config, DEFAULT_ENV_PATH, BASE_MAPPING_PATH
from client import (
    OsMlClientWrapper,
    get_client,
    get_client_configs,
)
from data_process import QAndAFileReader
from mapping import get_base_mapping, mapping_update
from ml_models import (get_remote_connector_configs, MlModel)
from main import get_ml_model, load_category


logging.basicConfig(
    format="%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)

SPACE_SEPARATOR = " "
SEPARATOR = SPACE_SEPARATOR


def create_index_settings(base_mapping_path, index_config, model_config=dict()):
    settings = get_base_mapping(base_mapping_path)
    model_dimension = model_config.get(
        "model_dimensions", index_config["model_dimensions"]
    )
    embedding_type = index_config.get("embedding_type", "dense")
    if index_config["with_knn"]:
        pipeline_name = index_config["pipeline_name"]
        if embedding_type == "dense":
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
        else:
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
    if (
        "compression" in index_config
        and index_config["compression"] != "best_compression"
    ):
        compression_settings = {
            "settings": {"index": {"codec": index_config["compression"]}}
        }
        mapping_update(settings, compression_settings)
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
    model_type = "bedrock"
    index_name = "hnsw_search"
    dataset_path = get_config("QANDA_FILE_READER_PATH")
    number_of_docs = 10
    client = OsMlClientWrapper(get_client(host_type))
    
    pqa_reader = QAndAFileReader(
        directory=dataset_path, max_number_of_docs=number_of_docs
    )

    categories = ["sheet and pillowcase sets"]
    config = {
        "with_knn": True,
    }

    pipeline_name = "amazon_pqa_pipeline"
    embedding_type = "dense"
    config["categories"] = categories
    config["index_name"] = index_name
    config["pipeline_name"] = pipeline_name
    config["embedding_type"] = embedding_type

    ml_model = None
    model_name = f"{host_type}_{model_type}"

    model_config = get_remote_connector_configs(host_type=host_type, connector_type=model_type)
    model_config["model_name"] = model_name
    model_config["embedding_type"] = embedding_type

    ml_model = get_ml_model(
        host_type=host_type,
        model_type=model_type,
        model_config=model_config,
        client=client,
    )

    config["index_settings"] = create_index_settings(
        base_mapping_path=BASE_MAPPING_PATH,
        index_config=config,
        model_config=model_config,
    )
    config["cleanup"] = False

    logging.info(f"Config:\n {json.dumps(config, indent=4)}")

    load_dataset(
        client,
        ml_model,
        pqa_reader,
        config,
        delete_existing=True,
        index_name=index_name,
        pipeline_name=pipeline_name,
    )


if __name__ == "__main__":
    main()
