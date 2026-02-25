# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

"""
Hybrid Local Search Example

This module demonstrates hybrid search combining lexical (keyword) and dense vector search
using a local OpenSearch cluster with a locally deployed Hugging Face model. It combines
traditional BM25 scoring with semantic vector similarity for enhanced search relevance.

The example:
1. Loads a Hugging Face sentence transformer model locally into OpenSearch
2. Creates embeddings using the local model for semantic search
3. Combines lexical (BM25) and dense vector search in a hybrid query
4. Provides interactive search interface with both search types
5. Uses search pipeline for result normalization and combination

Features:
- Local model deployment (huggingface/sentence-transformers/msmarco-distilbert-base-tas-b)
- Hybrid search combining lexical and semantic matching
- Interactive search interface with result scoring
- Automatic pipeline setup for result combination
"""

import json
import logging
import os
import sys

import cmd_line_interface

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from client import OsMlClientWrapper, get_client
from configs.configuration_manager import get_qanda_file_reader_path
from data_process.amazon_pqa_dataset import AmazonPQADataset
from models import get_ml_model

logging.basicConfig(
    format="%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)

# ANSI color constants for pipeline display
LIGHT_RED_HEADER = "\033[1;31m"
RESET = "\033[0m"


def configure_index_for_hybrid_search(dataset, pipeline_name, model_dimensions):
    """Configure index settings with kNN and vector fields for hybrid search."""
    base_mapping = dataset.get_index_mapping()
    
    hybrid_settings = {
        "settings": {"index": {"knn": True}, "default_pipeline": pipeline_name},
        "mappings": {
            "properties": {
                "chunk_embedding": {
                    "type": "knn_vector",
                    "dimension": model_dimensions,
                    "method": {
                        "name": "hnsw",
                        "space_type": "l2",
                        "engine": "lucene",
                        "parameters": {"ef_construction": 128, "m": 24},
                    },
                },
            }
        },
    }
    
    index_settings = {"mappings": base_mapping}
    dataset.update_mapping(index_settings, hybrid_settings)
    return index_settings


def build_hybrid_local_query(query_text, model_id=None, pipeline_config=None, **kwargs):
    """
    Build hybrid search query combining lexical (BM25) and dense vector search.

    Parameters:
        query_text (str): The search query text
        model_id (str): Local ML model ID for generating embeddings
        pipeline_config (dict): Search pipeline configuration to display
        **kwargs: Additional parameters (unused)

    Returns:
        dict: OpenSearch hybrid query dictionary
    """
    if not model_id:
        raise ValueError("Model ID must be provided for hybrid local search.")
    # Print pipeline config if provided
    if pipeline_config:
        print(f"{LIGHT_RED_HEADER}Search pipeline config:{RESET}")
        print(json.dumps(pipeline_config, indent=4))

    return {
        "size": 5,
        "query": {
            "hybrid": {
                "queries": [
                    {"match": {"chunk": {"query": query_text, "boost": 1.0}}},
                    {
                        "neural": {
                            "chunk_embedding": {
                                "query_text": query_text,
                                "model_id": model_id,
                                "k": 10,
                                "boost": 1.0,
                            }
                        }
                    },
                ]
            }
        },
    }


def main():
    """
    Main function to run hybrid local search example.

    This function:
    1. Initializes local OpenSearch client and Hugging Face model
    2. Configures hybrid index settings for lexical and vector search
    3. Loads dataset with vector embeddings
    4. Sets up search pipeline for result combination
    5. Provides interactive hybrid search interface
    """
    args = cmd_line_interface.get_command_line_args()

    if args.opensearch_type != "os":
        logging.error(
            "This example is designed for local OpenSearch clusters only. Use --opensearch-type os"
        )
        sys.exit(1)

    # This example uses a local Hugging Face model deployed in OpenSearch
    host_type = "os"
    model_host = "local"
    index_name = "hybrid_local_search"
    embedding_type = "dense"
    ingest_pipeline_name = "hybrid-local-ingest-pipeline"
    search_pipeline_name = "hybrid-local-search-pipeline"

    # Initialize OpenSearch client and dataset
    client = OsMlClientWrapper(get_client(host_type))
    dataset = AmazonPQADataset(
        directory=get_qanda_file_reader_path(),
        max_number_of_docs=args.number_of_docs_per_category
    )

    config = {
        "with_knn": True,
        "pipeline_field_map": {"chunk_text": "chunk_embedding"},  # Example-specific field mapping
        "categories": args.categories,
        "index_name": index_name,
        "pipeline_name": ingest_pipeline_name,
        "embedding_type": embedding_type,
        "delete_existing_index": args.delete_existing_index,
        "bulk_send_chunk_size": args.bulk_send_chunk_size,
    }

    # Initialize local Hugging Face model using configuration
    from configs.configuration_manager import get_local_dense_embedding_model_name, get_local_dense_embedding_model_version, get_local_dense_embedding_model_format, get_local_dense_embedding_model_dimension
    
    model_name = get_local_dense_embedding_model_name()
    model_config = {
        "model_name": model_name,
        "model_version": get_local_dense_embedding_model_version(),
        "model_dimensions": get_local_dense_embedding_model_dimension(),
        "embedding_type": embedding_type,
        "model_format": get_local_dense_embedding_model_format(),
    }

    logging.info(f"Initializing local Hugging Face model: {model_name}")

    ml_model = get_ml_model(
        host_type=host_type,
        model_host=model_host,
        model_group_id=client.ml_model_group.model_group_id(),
        model_config=model_config,
        os_client=client.os_client,
        ml_commons_client=client.ml_commons_client,
    )

    config.update(model_config)
    
    # Configure index settings with hybrid search support
    index_settings = configure_index_for_hybrid_search(
        dataset, ingest_pipeline_name, model_config.get("model_dimensions", 384)
    )

    # Create index using dataset with custom settings
    dataset.create_index(
        os_client=client.os_client,
        index_name=index_name,
        delete_existing=args.delete_existing_index,
        index_settings=index_settings
    )

    logging.info("Setting up k-NN pipeline for dense embeddings")
    client.setup_for_kNN(
        ml_model=ml_model,
        index_name=index_name,
        pipeline_name=ingest_pipeline_name,
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

    # Create search pipeline for hybrid result combination
    logging.info(f"Creating hybrid search pipeline: {search_pipeline_name}")
    pipeline_config = {
        "description": "Hybrid search pipeline combining lexical and dense vector results",
        "phase_results_processors": [
            {
                "normalization-processor": {
                    "normalization": {"technique": "min_max"},
                    "combination": {
                        "technique": "arithmetic_mean",
                        "parameters": {
                            "weights": [0.4, 0.6]
                        },  # Favor semantic search slightly
                    },
                }
            }
        ],
    }

    client.os_client.transport.perform_request(
        "PUT", f"/_search/pipeline/{search_pipeline_name}", body=pipeline_config
    )

    logging.info("Setup complete! Starting interactive hybrid search interface...")

    # Start interactive search loop using the generic function
    cmd_line_interface.interactive_search_loop(
        client=client,
        index_name=index_name,
        model_info=f"Local Hybrid: Lexical + {ml_model.model_id()}",
        query_builder_func=build_hybrid_local_query,
        ml_model=ml_model,
        pipeline_config=pipeline_config,
        search_params={"search_pipeline": search_pipeline_name},
        question=args.question,
    )


if __name__ == "__main__":
    main()
