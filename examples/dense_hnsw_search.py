# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import sys

import cmd_line_interface

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from client import OsMlClientWrapper, get_client
from configs.configuration_manager import get_client_configs
from connectors.helper import get_remote_connector_configs
from data_process.amazon_pqa_dataset import AmazonPQADataset
from models import get_ml_model

logging.basicConfig(
    format="%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)


def configure_index_for_dense_search(dataset, pipeline_name, model_dimensions):
    """Configure index settings with kNN and vector fields."""
    base_mapping = dataset.get_index_mapping()
    
    knn_settings = {
        "settings": {
            "index": {
                "knn": True,
                "knn.algo_param.ef_search": 100
            },
            "default_pipeline": pipeline_name
        },
        "mappings": {
            "properties": {
                "chunk_embedding": {
                    "type": "knn_vector",
                    "dimension": model_dimensions,
                    "method": {
                        "name": "hnsw",
                        "space_type": "l2",
                        "engine": "lucene",
                        "parameters": {
                            "ef_construction": 128,
                            "m": 24
                        }
                    }
                }
            }
        }
    }
    
    index_settings = {"mappings": base_mapping}
    dataset.update_mapping(index_settings, knn_settings)
    return index_settings


def build_dense_hnsw_query(query_text, model_id=None, **kwargs):
    """
    Build neural search query for HNSW vector search.

    Parameters:
        query_text (str): The search query text
        model_id (str): ML model ID for generating embeddings
        **kwargs: Additional parameters (unused)

    Returns:
        dict: OpenSearch query dictionary
    """
    if not model_id:
        raise ValueError("Model ID must be provided for dense HNSW search.")
    return {
        "size": 3,
        "query": {
            "neural": {
                "chunk_embedding": {
                    "query_text": query_text,
                    "model_id": model_id,
                }
            }
        },
    }


def main():
    args = cmd_line_interface.get_command_line_args()

    if args.opensearch_type != "aos":
        logging.error(
            "This example is designed for Amazon OpenSearch Service (AOS) only."
        )
        sys.exit(1)

    # This example uses a dense model, hosted on Amazon SageMaker and an Amazon
    # OpenSearch Service domain.
    host_type = "aos"
    model_host = "sagemaker"
    index_name = "dense_hnsw_search"
    embedding_type = "dense"
    pipeline_name = "dense-hnsw-ingest-pipeline"

    logging.info(f"Initializing dense HNSW search with {model_host.upper()} on {host_type.upper()}")

    client = OsMlClientWrapper(get_client(host_type))
    dataset = AmazonPQADataset(max_number_of_docs=args.number_of_docs_per_category)

    config = {
        "with_knn": True,
        "pipeline_field_map": {"chunk_text": "chunk_embedding"},  # Example-specific field mapping
        "index_name": index_name,
        "pipeline_name": pipeline_name,
        "embedding_type": embedding_type,
        "categories": args.categories,
        "delete_existing_index": args.delete_existing_index,
        "bulk_send_chunk_size": args.bulk_send_chunk_size,
    }

    model_name = f"{host_type}_{model_host}"
    model_config = get_remote_connector_configs(
        host_type=host_type, connector_type=model_host
    )
    model_config["model_name"] = model_name
    model_config["embedding_type"] = embedding_type
    config.update(model_config)

    ml_model = get_ml_model(
        host_type=host_type,
        model_host=model_host,
        model_config=model_config,
        os_client=client.os_client,
        ml_commons_client=client.ml_commons_client,
        model_group_id=client.ml_model_group.model_group_id(),
    )

    # Configure index settings with kNN and vector fields
    index_settings = configure_index_for_dense_search(
        dataset, pipeline_name, model_config.get("model_dimensions", 384)
    )

    # Create index using dataset with custom settings
    dataset.create_index(
        os_client=client.os_client,
        index_name=index_name,
        delete_existing=args.delete_existing_index,
        index_settings=index_settings
    )

    logging.info("Setting up for KNN")
    client.setup_for_kNN(
        ml_model=ml_model,
        index_name=index_name,
        pipeline_name=pipeline_name,
        pipeline_field_map=config["pipeline_field_map"],
        embedding_type=embedding_type,
    )

    # Load data using dataset
    if not args.no_load:
        total_docs = dataset.load_data(
            os_client=client.os_client,
            index_name=index_name,
            filter_criteria=args.categories,
            bulk_chunk_size=args.bulk_send_chunk_size
        )
        logging.info(f"Loaded {total_docs} documents")

    logging.info("Setup complete! Starting interactive search interface...")

    # Start interactive search loop using the generic function
    cmd_line_interface.interactive_search_loop(
        client=client,
        index_name=index_name,
        model_info=ml_model.model_id(),
        query_builder_func=build_dense_hnsw_query,
        ml_model=ml_model,
        question=args.question,
    )


if __name__ == "__main__":
    main()
