# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import json
import logging
from typing import Dict


import cmd_line_interface


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
    index_utils,
)
from data_process import QAndAFileReader
from mapping import get_base_mapping, mapping_update
from models import get_ml_model, MlModel


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
    pipeline_name: str,
    args: str,
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

    index_utils.handle_index_creation(
        os_client=client.os_client,
        config=config,
        delete_existing=config["delete_existing_index"],
    )

    index_utils.handle_data_loading(
        os_client=client.os_client,
        pqa_reader=pqa_reader,
        config=config,
        no_load=args.no_load,
    )


def build_hybrid_query(
    query_text,
    dense_model_id=None,
    sparse_model_id=None,
    pipeline_config=None,
    **kwargs,
):
    """
    Build hybrid search query combining dense and sparse embeddings.

    Parameters:
        query_text (str): The search query text
        dense_model_id (str): Dense ML model ID for generating embeddings
        sparse_model_id (str): Sparse ML model ID for generating embeddings
        pipeline_config (dict): Search pipeline configuration to display
        **kwargs: Additional parameters (unused)

    Returns:
        dict: OpenSearch query dictionary
    """
    if not dense_model_id:
        raise ValueError("Dense model ID must be provided for hybrid search.")
    if not sparse_model_id:
        raise ValueError("Sparse model ID must be provided for hybrid search.")
    # Print pipeline config if provided
    if pipeline_config:
        print(f"{LIGHT_RED_HEADER}Search pipeline config:{RESET}")
        print(json.dumps(pipeline_config, indent=4))

    return {
        "size": 3,
        "query": {
            "hybrid": {
                "queries": [
                    {
                        "neural": {
                            "chunk_dense_embedding": {
                                "query_text": query_text,
                                "model_id": dense_model_id,
                            }
                        }
                    },
                    {
                        "neural_sparse": {
                            "chunk_sparse_embedding": {
                                "query_text": query_text,
                                "model_id": sparse_model_id,
                            }
                        }
                    },
                ]
            }
        },
    }


def main():
    args = cmd_line_interface.get_command_line_args()
    host_type = "aos"
    index_name = "hybrid_search"
    dense_model_type = "sagemaker"
    sparse_model_type = "sagemaker"
    ingest_pipeline_name = "hybrid-ingest-pipeline"
    search_pipeline_name = "hybrid-search-pipeline"

    client = OsMlClientWrapper(get_client(host_type))
    pqa_reader = QAndAFileReader(
        directory=QANDA_FILE_READER_PATH,
        max_number_of_docs=args.number_of_docs_per_category,
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
        pipeline_name=ingest_pipeline_name,
        args=args,
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

    logging.info("Setup complete! Starting interactive search interface...")

    # Start interactive search loop using the generic function
    cmd_line_interface.interactive_search_loop(
        client=client,
        index_name=index_name,
        model_info=f"Dense: {dense_ml_model.model_id()}, Sparse: {sparse_ml_model.model_id()}",
        query_builder_func=build_hybrid_query,
        dense_ml_model=dense_ml_model,
        sparse_ml_model=sparse_ml_model,
        pipeline_config=pipeline_config,
        search_params={"search_pipeline": search_pipeline_name},
    )


if __name__ == "__main__":
    main()
