# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import json
import logging
from typing import Dict


import cmd_line_params
import print_utils


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

# ANSI escape sequence constants with improved colors
BOLD = "\033[1m"
RESET = "\033[0m"

# Headers
LIGHT_RED_HEADER = "\033[1;31m"
LIGHT_GREEN_HEADER = "\033[1;32m"
LIGHT_BLUE_HEADER = "\033[1;34m"
LIGHT_YELLOW_HEADER = "\033[1;33m"
LIGHT_PURPLE_HEADER = "\033[1;35m"


def create_index_settings(base_mapping_path, index_config):
    settings = get_base_mapping(base_mapping_path)
    pipeline_name = index_config["pipeline_name"]
    model_dimension = index_config["model_dimensions"]
    knn_settings = {
        "settings": {"index": {"knn": True}, "default_pipeline": pipeline_name},
        "mappings": {
            "properties": {
                "chunk": {"type": "text", "index": False},
                "chunk_sparse_embedding": {
                    "type": "rank_features",
                },
                "chunk_dense_embedding": {
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
    dense_ml_model: MlModel,
    sparse_ml_model: MlModel,
    pqa_reader: QAndAFileReader,
    config: Dict[str, str],
    index_name: str,
    pipeline_name: str,
):
    logging.info("Adding ingestion pipeline for hybrid search...")
    pipeline_config = {
        "description": "Pipeline for processing chunks",
        "processors": [
            {
                "sparse_encoding": {
                    "model_id": sparse_ml_model.model_id(),
                    "field_map": {"chunk": "chunk_sparse_embedding"},
                }
            },
            {
                "text_embedding": {
                    "model_id": dense_ml_model.model_id(),
                    "field_map": {"chunk": "chunk_dense_embedding"},
                }
            },
        ],
    }
    client.os_client.ingest.put_pipeline(pipeline_name, body=pipeline_config)

    if client.os_client.indices.exists(index_name):
        if config["delete_existing_index"]:
            logging.info(f"Deleting existing index {index_name}, then creating a new one")
            client.delete_then_create_index(index_name=index_name, settings=config["index_settings"])
        else:
            logging.info(f"Index {index_name} already exists. Skipping loading dataset")
            return
    else:
        logging.info(f"Creating new index {index_name}")
        client.idempotent_create_index(
            index_name=config["index_name"], settings=config["index_settings"]
        )

    for category in config["categories"]:
        load_category(
            client=client.os_client,
            pqa_reader=pqa_reader,
            category=category,
            config=config,
        )


def main():
    args = cmd_line_params.get_command_line_args()
    host_type = "aos"
    index_name = "hybrid_search"
    dense_model_type = "sagemaker"
    sparse_model_type = "sagemaker"
    ingest_pipeline_name = "hybrid-ingest-pipeline"
    search_pipeline_name = "hybrid-search-pipeline"

    client = OsMlClientWrapper(get_client(host_type))
    pqa_reader = QAndAFileReader(
        directory=QANDA_FILE_READER_PATH,
        max_number_of_docs=args.number_of_docs_per_category
    )

    config = {
        "with_knn": True,
        "pipeline_field_map": PIPELINE_FIELD_MAP,
        "categories": args.categories,
        "index_name": index_name,
        "pipeline_name": ingest_pipeline_name,
        "delete_existing_index": args.delete_existing_index,
        "bulk_send_chunk_size": args.bulk_send_chunk_size,
    }

    dense_model_name = f"{host_type}_{dense_model_type}"
    dense_model_config = get_remote_connector_configs(
        host_type=host_type, connector_type=dense_model_type
    )
    dense_model_config["model_name"] = dense_model_name
    dense_model_config["embedding_type"] = "dense"
    dense_ml_model = get_ml_model(
        host_type=host_type,
        model_type=dense_model_type,
        model_config=dense_model_config,
        os_client=client.os_client,
        ml_commons_client=client.ml_commons_client,
        model_group_id=client.ml_model_group.model_group_id(),
    )

    sparse_model_name = f"{host_type}_{sparse_model_type}"
    sparse_model_config = get_remote_connector_configs(
        host_type=host_type, connector_type=dense_model_type
    )
    sparse_model_config["model_name"] = sparse_model_name
    sparse_model_config["embedding_type"] = "sparse"
    sparse_ml_model = get_ml_model(
        host_type=host_type,
        model_type=sparse_model_type,
        model_config=sparse_model_config,
        os_client=client.os_client,
        ml_commons_client=client.ml_commons_client,
        model_group_id=client.ml_model_group.model_group_id(),
    )

    config["model_dimensions"] = dense_model_config["model_dimensions"]
    config["index_settings"] = create_index_settings(
        base_mapping_path=BASE_MAPPING_PATH,
        index_config=config,
    )

    load_dataset(
        client,
        dense_ml_model,
        sparse_ml_model,
        pqa_reader,
        config,
        index_name=index_name,
        pipeline_name=ingest_pipeline_name,
    )

    logging.info(f"Creating search pipeline {search_pipeline_name}")
    pipeline_config = {
        "description": "Post processor for hybrid search",
        "phase_results_processors": [
            {
                "normalization-processor": {
                    "normalization": {"technique": "min_max"},
                    "combination": {
                        "technique": "arithmetic_mean",
                        "parameters": {"weights": [0.5, 0.5]},
                    },
                }
            }
        ],
    }
    client.os_client.transport.perform_request(
        "PUT", f"/_search/pipeline/{search_pipeline_name}", body=pipeline_config
    )

    while True:
        query_text = input("Please input your search query text (or 'quit' to quit): ")
        if query_text == "quit":
            break
        search_query = {
            "size": 3,
            "query": {
                "hybrid": {
                    "queries": [
                        {
                            "neural": {
                                "chunk_dense_embedding": {
                                    "query_text": query_text,
                                    "model_id": dense_ml_model.model_id(),
                                }
                            }
                        },
                        {
                            "neural_sparse": {
                                "chunk_sparse_embedding": {
                                    "query_text": query_text,
                                    "model_id": sparse_ml_model.model_id(),
                                }
                            }
                        },
                    ]
                }
            },
        }
        print(f"{LIGHT_GREEN_HEADER}Search query:{RESET}")
        print(json.dumps(search_query, indent=4))
        print(f"{LIGHT_RED_HEADER}Search pipeline config:{RESET}")
        print(json.dumps(pipeline_config, indent=4))
        search_results = client.os_client.search(
            index=index_name, search_pipeline=search_pipeline_name, body=search_query
        )
        hits = search_results["hits"]["hits"]
        input("Press enter to see the search results: ")
        for hit_id, hit in enumerate(hits):
            print_utils.print_hit(hit_id, hit)


if __name__ == "__main__":
    main()
