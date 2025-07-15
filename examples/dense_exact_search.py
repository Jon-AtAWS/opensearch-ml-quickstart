# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

"""
Dense Exact Search Example

This module demonstrates dense vector search using OpenSearch with exact k-NN.
It uses ML models to generate embeddings for semantic search capabilities,
allowing users to find documents based on meaning rather than exact keyword matches.

The example:
1. Loads data from Amazon PQA dataset
2. Creates embeddings using remote ML models (Bedrock/SageMaker)
3. Stores vectors in OpenSearch with k-NN configuration
4. Provides interactive semantic search interface
"""

import logging
import os
import sys

import cmd_line_params
import print_utils

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


def create_index_settings(base_mapping_path, index_config):
    """
    Create OpenSearch index settings for dense vector search.
    
    Parameters:
        base_mapping_path (str): Path to base mapping configuration
        index_config (dict): Configuration containing pipeline and model settings
    
    Returns:
        dict: Updated index settings with k-NN vector configuration
    """
    settings = get_base_mapping(base_mapping_path)
    pipeline_name = index_config["pipeline_name"]
    model_dimension = index_config["model_dimensions"]
    # Configure k-NN settings for vector search
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


def interactive_search_loop(client, index_name, ml_model):
    """
    Provide an interactive search interface for user queries.
    
    Parameters:
        client (OsMlClientWrapper): OpenSearch ML client wrapper
        index_name (str): Name of the index to search
        ml_model: ML model instance for generating embeddings
    """
    print_utils.print_search_interface_header(index_name, ml_model.model_id())
    
    while True:
        try:
            query_text = print_utils.print_search_prompt()
            
            if query_text.lower().strip() in ['quit', 'exit', 'q']:
                print_utils.print_goodbye()
                break
                
            if not query_text.strip():
                print_utils.print_empty_query_warning()
                continue
            
            # Build neural search query for semantic matching
            search_query = {
                "size": 3,
                "query": {
                    "neural": {
                        "chunk_embedding": {
                            "query_text": query_text,
                            "model_id": ml_model.model_id(),
                        }
                    }
                },
            }
            
            print_utils.print_executing_search()
            print_utils.print_query(search_query)

            # Execute semantic search and display results
            search_results = client.os_client.search(index=index_name, body=search_query)
            
            # Print search results using the print_utils function
            print_utils.print_search_results(search_results)
                    
        except KeyboardInterrupt:
            print_utils.print_search_interrupted()
            break
        except Exception as e:
            logging.error(f"Search error: {e}")
            print_utils.print_search_error(e)


def main():
    """
    Main function to run dense exact search example.
    
    This function:
    1. Initializes OpenSearch client and ML model
    2. Configures k-NN index settings
    3. Loads dataset with vector embeddings
    4. Provides interactive semantic search interface
    """
    args = cmd_line_params.get_command_line_args()
    
    if args.opensearch_type != "aos":
        logging.error(
            "This example is designed for Amazon OpenSearch Service (AOS) only."
        )
        sys.exit(1)

    # This example uses a dense model, hosted on Amazon SageMaker and an Amazon
    # OpenSearch Service domain.
    host_type = "aos"
    model_type = "sagemaker"
    index_name = "dense_exact_search"
    embedding_type = "dense"
    pipeline_name = "dense-ingest-pipeline"

    # Initialize OpenSearch client and data reader
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
        "pipeline_name": pipeline_name,
        "embedding_type": embedding_type,
        "delete_existing_index": args.delete_existing_index,
        "bulk_send_chunk_size": args.bulk_send_chunk_size,
    }

    # Initialize ML model for embeddings
    model_config = get_remote_connector_configs(
        host_type=host_type, connector_type=model_type
    )
    model_config["model_name"] = f"{host_type}_{model_type}"
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

    # Handle index creation ensures the index exists and creates and applies the
    # mapping.
    index_utils.handle_index_creation(
        os_client=client.os_client,
        config=config,
        delete_existing=config["delete_existing_index"],
    )

    # Set up k-NN pipeline for automatic embedding generation
    client.setup_for_kNN(
        ml_model=ml_model,
        index_name=config["index_name"],
        pipeline_name=pipeline_name,
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

    logging.info("Setup complete! Starting interactive search interface...")
    
    # Start interactive search loop
    interactive_search_loop(client, index_name, ml_model)


if __name__ == "__main__":
    main()
