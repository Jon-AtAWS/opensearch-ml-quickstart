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
from client import OsMlClientWrapper, get_client, index_utils
from configs import (BASE_MAPPING_PATH, PIPELINE_FIELD_MAP,
                     QANDA_FILE_READER_PATH, get_remote_connector_configs)
from data_process import QAndAFileReader
from mapping import get_base_mapping, mapping_update
from ml_models import get_ml_model

logging.basicConfig(
    format="%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)

# ANSI color constants for pipeline display
LIGHT_RED_HEADER = "\033[1;31m"
RESET = "\033[0m"


def create_index_settings(base_mapping_path, index_config):
    """
    Create OpenSearch index settings for hybrid lexical and dense vector search.
    
    Parameters:
        base_mapping_path (str): Path to base mapping configuration
        index_config (dict): Configuration containing pipeline and model settings
    
    Returns:
        dict: Updated index settings with k-NN vector configuration and text fields
    """
    settings = get_base_mapping(base_mapping_path)
    pipeline_name = index_config["pipeline_name"]
    model_dimension = index_config["model_dimensions"]
    
    # Configure hybrid settings for both lexical and vector search
    hybrid_settings = {
        "settings": {
            "index": {"knn": True}, 
            "default_pipeline": pipeline_name
        },
        "mappings": {
            "properties": {
                "chunk": {
                    "type": "text",
                    "analyzer": "standard"  # Enable lexical search on chunk field
                },
                "chunk_embedding": {
                    "type": "knn_vector",
                    "dimension": model_dimension,
                    "method": {
                        "name": "hnsw",
                        "space_type": "l2",
                        "engine": "nmslib",
                        "parameters": {
                            "ef_construction": 128,
                            "m": 24
                        }
                    }
                }
            }
        },
    }
    mapping_update(settings, hybrid_settings)
    return settings


def build_hybrid_local_query(query_text, ml_model=None, pipeline_config=None, **kwargs):
    """
    Build hybrid search query combining lexical (BM25) and dense vector search.
    
    Parameters:
        query_text (str): The search query text
        ml_model: Local ML model instance for generating embeddings
        pipeline_config (dict): Search pipeline configuration to display
        **kwargs: Additional parameters (unused)
    
    Returns:
        dict: OpenSearch hybrid query dictionary
    """
    # Print pipeline config if provided
    if pipeline_config:
        print(f"{LIGHT_RED_HEADER}Search pipeline config:{RESET}")
        print(json.dumps(pipeline_config, indent=4))
    
    return {
        "size": 5,
        "query": {
            "hybrid": {
                "queries": [
                    {
                        "match": {
                            "chunk": {
                                "query": query_text,
                                "boost": 1.0
                            }
                        }
                    },
                    {
                        "neural": {
                            "chunk_embedding": {
                                "query_text": query_text,
                                "model_id": ml_model.model_id(),
                                "k": 10,
                                "boost": 1.0
                            }
                        }
                    }
                ]
            }
        }
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
    model_type = "local"
    index_name = "hybrid_local_search"
    embedding_type = "dense"
    ingest_pipeline_name = "hybrid-local-ingest-pipeline"
    search_pipeline_name = "hybrid-local-search-pipeline"

    # Initialize OpenSearch client and data reader
    client = OsMlClientWrapper(get_client(host_type))
    pqa_reader = QAndAFileReader(
        directory=QANDA_FILE_READER_PATH,
        max_number_of_docs=args.number_of_docs_per_category
    )


    ping = client.os_client.transport.perform_request(
        "GET", f"/"
    )
    logging.info(f"Connected to OpenSearch: {json.dumps(ping, indent=2)}")


    config = {
        "with_knn": True,
        "pipeline_field_map": PIPELINE_FIELD_MAP,
        "categories": args.categories,
        "index_name": index_name,
        "pipeline_name": ingest_pipeline_name,
        "embedding_type": embedding_type,
        "delete_existing_index": args.delete_existing_index,
        "bulk_send_chunk_size": args.bulk_send_chunk_size,
    }

    # Initialize local Hugging Face model
    model_name = "huggingface/sentence-transformers/all-MiniLM-L6-v2"
    model_config = {
        "model_name": model_name,
        "model_version": "1.0.1",
        "model_dimensions": 384,
        "embedding_type": embedding_type,
        "model_format": "TORCH_SCRIPT",
    }
    
    logging.info(f"Initializing local Hugging Face model: {model_name}")
    
    ml_model = get_ml_model(
        host_type=host_type,
        model_type=model_type,
        model_group_id=client.ml_model_group.model_group_id(),
        model_config=model_config,
        os_client=client.os_client,
        ml_commons_client=client.ml_commons_client,
    )
    
    config.update(model_config)
    config["index_settings"] = create_index_settings(
        base_mapping_path=BASE_MAPPING_PATH,
        index_config=config,
    )

    # Handle index creation
    index_utils.handle_index_creation(
        os_client=client.os_client,
        config=config,
        delete_existing=config["delete_existing_index"],
    )

    logging.info("Setting up k-NN pipeline for dense embeddings")
    client.setup_for_kNN(
        ml_model=ml_model,
        index_name=config["index_name"],
        pipeline_name=ingest_pipeline_name,
        pipeline_field_map=config["pipeline_field_map"],
        embedding_type=config["embedding_type"],
    )

    # Load data into the index
    index_utils.handle_data_loading(
        os_client=client.os_client,
        pqa_reader=pqa_reader,
        config=config,
        no_load=args.no_load,
    )

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
                        "parameters": {"weights": [0.4, 0.6]}  # Favor semantic search slightly
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
        search_params={'search_pipeline': search_pipeline_name}
    )


if __name__ == "__main__":
    main()
