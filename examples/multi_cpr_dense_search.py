# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

"""Dense vector search over the Multi-CPR Chinese passage retrieval dataset.

Usage:
    # Search e-commerce domain, 500 passages, local OpenSearch
    python examples/multi_cpr_search.py -o os -c ecom -n 500

    # Search all three domains on AOS
    python examples/multi_cpr_search.py -o aos -c ecom video medical -n 1000

    # Non-interactive single query
    python examples/multi_cpr_search.py -o os -c medical -n 200 -q "感冒吃什么药"
"""

import logging
import sys

import cmd_line_interface

from client import OsMlClientWrapper, get_client
from configs.configuration_manager import get_client_configs
from connectors.helper import get_remote_connector_configs
from data_process.multi_cpr_dataset import MultiCPRDataset
from models import get_ml_model

logging.basicConfig(
    format="%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)

# Default dataset path – override via MULTI_CPR_PATH env var or osmlqs.yaml
import os
MULTI_CPR_PATH = os.environ.get("MULTI_CPR_PATH", "~/datasets/multi_cpr")

INDEX_NAME = "multi_cpr_dense"
PIPELINE_NAME = "multi-cpr-dense-ingest-pipeline"


def configure_index(dataset, pipeline_name, model_dimensions):
    """Build index settings with kNN vector field for dense search."""
    base_mapping = dataset.get_index_mapping()

    knn_settings = {
        "settings": {
            "index": {
                "knn": True,
                "knn.algo_param.ef_search": 100,
            },
            "default_pipeline": pipeline_name,
        },
        "mappings": {
            "properties": {
                "chunk_embedding": {
                    "type": "knn_vector",
                    "dimension": model_dimensions,
                    "method": {
                        "name": "hnsw",
                        "space_type": "l2",
                        "engine": "faiss",
                        "parameters": {"ef_construction": 128, "m": 24},
                    },
                },
            }
        },
    }

    index_settings = {"mappings": base_mapping}
    dataset.update_mapping(index_settings, knn_settings)
    return index_settings


def build_query(query_text, model_id=None, **kwargs):
    """Neural search query for dense HNSW vector search."""
    if not model_id:
        raise ValueError("model_id is required for dense search")
    return {
        "size": 5,
        "query": {
            "neural": {
                "chunk_embedding": {
                    "query_text": query_text,
                    "model_id": model_id,
                }
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

    host_type = args.opensearch_type
    # Use Bedrock for multilingual (Chinese) embedding support
    model_host = "bedrock"
    embedding_type = "dense"

    logging.info(f"Multi-CPR dense search | host={host_type} model={model_host}")

    client = OsMlClientWrapper(get_client(host_type))
    dataset = MultiCPRDataset(
        directory=MULTI_CPR_PATH,
        max_number_of_docs=args.number_of_docs_per_category,
    )

    # Override default Amazon PQA categories with Multi-CPR domains.
    # cmd_line_interface defaults to PQA categories when -c is not given,
    # so we detect that and fall back to all Multi-CPR domains instead.
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

    # Model configuration
    model_config = get_remote_connector_configs(
        host_type=host_type, connector_type=model_host
    )
    model_name = f"{host_type}_{model_host}"
    model_config["model_name"] = model_name
    model_config["embedding_type"] = embedding_type

    ml_model = get_ml_model(
        host_type=host_type,
        model_host=model_host,
        model_config=model_config,
        os_client=client.os_client,
        ml_commons_client=client.ml_commons_client,
        model_group_id=client.ml_model_group.model_group_id(),
    )

    # Index setup
    index_settings = configure_index(
        dataset, PIPELINE_NAME, model_config.get("model_dimensions", 384)
    )
    dataset.create_index(
        os_client=client.os_client,
        index_name=INDEX_NAME,
        delete_existing=args.delete_existing_index,
        index_settings=index_settings,
    )

    client.setup_for_kNN(
        ml_model=ml_model,
        index_name=INDEX_NAME,
        pipeline_name=PIPELINE_NAME,
        pipeline_field_map={"chunk_text": "chunk_embedding"},
        embedding_type=embedding_type,
    )

    # Load data
    if not args.no_load:
        total = dataset.load_data(
            os_client=client.os_client,
            index_name=INDEX_NAME,
            filter_criteria=categories,
            bulk_chunk_size=args.bulk_send_chunk_size,
        )
        logging.info(f"Loaded {total} passages into '{INDEX_NAME}'")

    # Interactive search
    cmd_line_interface.interactive_search_loop(
        client=client,
        index_name=INDEX_NAME,
        model_info=ml_model.model_id(),
        query_builder_func=build_query,
        result_processor_func=print_multi_cpr_results,
        ml_model=ml_model,
        question=args.question,
    )


if __name__ == "__main__":
    main()
