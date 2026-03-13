# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

"""Hybrid (dense + sparse) search over the Multi-CPR Chinese passage retrieval dataset.

Combines dense vector search and sparse neural search with score normalization
for improved retrieval quality on Chinese text.

Usage:
    # Hybrid search on e-commerce domain with 500 passages
    python examples/multi_cpr_hybrid_search.py -o aos -c ecom -n 500

    # Search all domains
    python examples/multi_cpr_hybrid_search.py -o aos -c ecom video medical -n 1000

    # Single query
    python examples/multi_cpr_hybrid_search.py -o aos -c ecom -n 500 -q "尼康相机"

    # Skip loading if already indexed
    python examples/multi_cpr_hybrid_search.py -o aos --no-load -q "感冒吃什么药"
"""

import json
import logging
import os
import sys

import cmd_line_interface

from connectors.helper import get_remote_connector_configs
from client import OsMlClientWrapper, get_client
from configs.configuration_manager import get_multi_cpr_path
from data_process.multi_cpr_dataset import MultiCPRDataset
from models import get_ml_model, MlModel


logging.basicConfig(
    format="%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)

BOLD = "\033[1m"
RESET = "\033[0m"
LIGHT_RED_HEADER = "\033[1;31m"

INDEX_NAME = "multi_cpr_hybrid"
INGEST_PIPELINE_NAME = "multi-cpr-hybrid-ingest-pipeline"
SEARCH_PIPELINE_NAME = "multi-cpr-hybrid-search-pipeline"


def configure_index_for_hybrid_search(dataset, pipeline_name, dense_model_dimensions):
    """Configure index settings with both dense and sparse vector fields for hybrid search."""
    base_mapping = dataset.get_index_mapping()

    hybrid_settings = {
        "settings": {
            "index": {
                "knn": True,
                "mapping.dots_in_field_names_enabled": True,
            },
            "default_pipeline": pipeline_name,
        },
        "mappings": {
            "properties": {
                "chunk_sparse_embedding": {
                    "type": "rank_features",
                },
                "chunk_dense_embedding": {
                    "type": "knn_vector",
                    "dimension": dense_model_dimensions,
                },
            }
        },
    }

    index_settings = {"mappings": base_mapping}
    dataset.update_mapping(index_settings, hybrid_settings)
    return index_settings


def load_dataset(
    client, dense_ml_model, sparse_ml_model, dataset, index_name,
    pipeline_name, categories, bulk_chunk_size, delete_existing, no_load,
):
    logging.info("Adding ingestion pipeline for Multi-CPR hybrid search...")
    pipeline_config = {
        "description": "Pipeline for Multi-CPR hybrid search (dense + sparse)",
        "processors": [
            {
                "sparse_encoding": {
                    "model_id": sparse_ml_model.model_id(),
                    "field_map": {"chunk_text": "chunk_sparse_embedding"},
                }
            },
            {
                "text_embedding": {
                    "model_id": dense_ml_model.model_id(),
                    "field_map": {"chunk_text": "chunk_dense_embedding"},
                }
            },
        ],
    }
    client.os_client.ingest.put_pipeline(id=pipeline_name, body=pipeline_config)

    # Configure index settings for hybrid search
    index_settings = configure_index_for_hybrid_search(
        dataset, pipeline_name, 384  # Default dense model dimension
    )

    # Create index using dataset
    dataset.create_index(
        os_client=client.os_client,
        index_name=index_name,
        delete_existing=delete_existing,
        index_settings=index_settings,
    )

    # Load data using dataset
    if not no_load:
        total_docs = dataset.load_data(
            os_client=client.os_client,
            index_name=index_name,
            filter_criteria=categories,
            bulk_chunk_size=bulk_chunk_size,
        )
        logging.info(f"Loaded {total_docs} Multi-CPR passages")


def build_hybrid_query(
    query_text, dense_model_id=None, sparse_model_id=None, pipeline_config=None, **kwargs,
):
    """Build hybrid search query combining dense and sparse embeddings."""
    if not dense_model_id:
        raise ValueError("Dense model ID must be provided for hybrid search.")
    if not sparse_model_id:
        raise ValueError("Sparse model ID must be provided for hybrid search.")
    if pipeline_config:
        print(f"{LIGHT_RED_HEADER}Search pipeline config:{RESET}")
        print(json.dumps(pipeline_config, indent=4))

    return {
        "size": 5,
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


def print_multi_cpr_results(search_results, **kwargs):
    """Custom result printer for Multi-CPR passages."""
    hits = search_results.get("hits", {}).get("hits", [])
    total = search_results.get("hits", {}).get("total", {}).get("value", 0)
    print(f"\nFound {total} total matches, showing top {len(hits)} results:\n")
    for i, hit in enumerate(hits):
        src = hit.get("_source", {})
        score = hit.get("_score", 0)
        print("-" * 80)
        print(f"  Result {i + 1}  |  Score: {score:.4f}  |  Domain: {src.get('domain', 'N/A')}  |  PID: {src.get('pid', 'N/A')}")
        print(f"  Passage: {src.get('passage', 'N/A')[:200]}")
        print()
    print("-" * 80)


def main():
    args = cmd_line_interface.get_command_line_args()
    host_type = "aos"
    dense_model_host = "sagemaker"
    sparse_model_host = "sagemaker"

    client = OsMlClientWrapper(get_client(host_type))
    dataset = MultiCPRDataset(
        directory=get_multi_cpr_path(),
        max_number_of_docs=args.number_of_docs_per_category,
    )

    # Validate Multi-CPR domains
    valid_domains = dataset.get_available_filters()
    if args.categories is None or not all(c in valid_domains for c in args.categories):
        categories = valid_domains
        logging.info(f"Using all Multi-CPR domains: {categories}")
    else:
        categories = args.categories

    for c in categories:
        if c not in valid_domains:
            logging.error(f"Unknown domain '{c}'. Choose from: {valid_domains}")
            sys.exit(1)

    # Dense model setup
    dense_model_name = f"{host_type}_{dense_model_host}"
    dense_model_config = get_remote_connector_configs(
        host_type=host_type, connector_type=dense_model_host
    )
    dense_model_config["model_name"] = dense_model_name
    dense_model_config["embedding_type"] = "dense"
    dense_ml_model = get_ml_model(
        host_type=host_type,
        model_host=dense_model_host,
        model_config=dense_model_config,
        os_client=client.os_client,
        ml_commons_client=client.ml_commons_client,
        model_group_id=client.ml_model_group.model_group_id(),
    )

    # Sparse model setup
    sparse_model_name = f"{host_type}_{sparse_model_host}"
    sparse_model_config = get_remote_connector_configs(
        host_type=host_type, connector_type=sparse_model_host
    )
    sparse_model_config["model_name"] = sparse_model_name
    sparse_model_config["embedding_type"] = "sparse"
    sparse_ml_model = get_ml_model(
        host_type=host_type,
        model_host=sparse_model_host,
        model_config=sparse_model_config,
        os_client=client.os_client,
        ml_commons_client=client.ml_commons_client,
        model_group_id=client.ml_model_group.model_group_id(),
    )

    load_dataset(
        client, dense_ml_model, sparse_ml_model, dataset,
        INDEX_NAME, INGEST_PIPELINE_NAME, categories,
        args.bulk_send_chunk_size, args.delete_existing_index, args.no_load,
    )

    logging.info(f"Creating search pipeline {SEARCH_PIPELINE_NAME}")
    pipeline_config = {
        "description": "Post processor for Multi-CPR hybrid search",
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
        "PUT", f"/_search/pipeline/{SEARCH_PIPELINE_NAME}", body=pipeline_config
    )

    logging.info("Setup complete! Starting interactive Multi-CPR hybrid search...")

    cmd_line_interface.interactive_search_loop(
        client=client,
        index_name=INDEX_NAME,
        model_info=f"Dense: {dense_ml_model.model_id()}, Sparse: {sparse_ml_model.model_id()}",
        query_builder_func=build_hybrid_query,
        result_processor_func=print_multi_cpr_results,
        dense_ml_model=dense_ml_model,
        sparse_ml_model=sparse_ml_model,
        pipeline_config=pipeline_config,
        search_params={"search_pipeline": SEARCH_PIPELINE_NAME},
        question=args.question,
    )


if __name__ == "__main__":
    main()
