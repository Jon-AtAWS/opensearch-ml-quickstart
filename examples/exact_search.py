# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import argparse
from typing import Dict
from opensearchpy import helpers, OpenSearch

from configs import tasks, get_config, DEFAULT_ENV_PATH
from client import (
    OsMlClientWrapper,
    get_client,
    get_client_configs,
)
from data_process import QAndAFileReader
from mapping import get_base_mapping, mapping_update
from ml_models import (
    MlModel,
    LocalMlModel,
    RemoteMlModel,
    OsBedrockMlConnector,
    AosBedrockMlConnector,
    OsSagemakerMlConnector,
    AosSagemakerMlConnector,
    get_aos_connector_helper,
    get_remote_connector_configs,
)
from ..main import get_ml_model, load_category

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


def get_args():
    parser = argparse.ArgumentParser(
        prog="main",
        description="This toolkit loads AmazonPQA data into an OpenSearch KNN index."
        "Your opensearch endpoint can be either opensource opensearch or "
        "amazon opensearch serivce. You can use local model and remote models "
        "from bedrock or sagemaker.",
    )
    parser.add_argument("-t", "--task", default="knn_768", action="store")
    parser.add_argument("-c", "--categories", default="all", action="store")
    parser.add_argument("-i", "--index_name", default="amazon_pqa", action="store")
    parser.add_argument(
        "-p", "--pipeline_name", default="amazon_pqa_pipeline", action="store"
    )
    parser.add_argument("-d", "--delete_existing", default=False, action="store_true")
    parser.add_argument("-n", "--number_of_docs", default=500, action="store", type=int)
    parser.add_argument(
        "-mt",
        "--model_type",
        choices=["local", "sagemaker", "bedrock"],
        default="local",
        action="store",
    )
    parser.add_argument("-ep", "--env_path", default=DEFAULT_ENV_PATH, action="store")
    parser.add_argument(
        "-ht", "--host_type", choices=["os", "aos"], default="os", action="store"
    )
    parser.add_argument("-et", "--embedding_type", default="dense", action="store")
    parser.add_argument("-cl", "--cleanup", default=False, action="store_true")
    parser.add_argument(
        "-dp",
        "--dataset_path",
        default=get_config("QANDA_FILE_READER_PATH"),
        action="store",
    )
    parser.add_argument(
        "-bmp",
        "--base_mapping_path",
        default=get_config("BASE_MAPPING_PATH"),
        action="store",
    )
    args = parser.parse_args()
    return args


def main():
    host_type = "os"
    model_type = "bedrock"
    index_name = "exact_search"
    dataset_path = get_config("QANDA_FILE_READER_PATH")
    number_of_docs = 10
    client = OsMlClientWrapper(get_client(host_type))
    
    pqa_reader = QAndAFileReader(
        directory=dataset_path, max_number_of_docs=number_of_docs
    )

    categories = ["sheet and pillowcase sets"]
    config = {
        "with_knn": True,
        "categories": categories,
        "max_cat_docs": -1,
        "cleanup": False,
        "compression": "best_compression",
        "pipeline_field_map": {"chunk": "chunk_embedding"},
        "model_name": "huggingface/sentence-transformers/msmarco-distilbert-base-tas-b",
        "model_version": "1.0.2",
        "model_dimensions": 768,
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

    base_mapping_path = get_config("BASE_MAPPING_PATH")
    config["index_settings"] = create_index_settings(
        base_mapping_path=base_mapping_path,
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
    i
    main()
