# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

"""
Dense Exact Search Example

This module demonstrates dense vector search using OpenSearch with exact k-NN.
It uses the universal EmbeddingConnector to generate embeddings for semantic search
capabilities, allowing users to find documents based on meaning rather than exact
keyword matches.

The example:
1. Loads data from Amazon PQA dataset
2. Creates embeddings using the universal EmbeddingConnector (supports Bedrock/SageMaker)
3. Stores vectors in OpenSearch with k-NN configuration
4. Provides interactive semantic search interface
"""

import logging
import os
import sys

import cmd_line_interface

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from client import OsMlClientWrapper, get_client, index_utils
from configs.configuration_manager import (
    get_base_mapping_path,
    get_pipeline_field_map,
    get_qanda_file_reader_path,
    get_client_configs,
)
from connectors import EmbeddingConnector
from data_process import QAndAFileReader
from mapping import get_base_mapping, mapping_update
from models import get_ml_model

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


def build_dense_exact_query(query_text, model_id=None, **kwargs):
    """
    Build neural search query for dense exact vector search.

    Parameters:
        query_text (str): The search query text
        model_id (str): ML model ID for generating embeddings
        **kwargs: Additional parameters (unused)

    Returns:
        dict: OpenSearch query dictionary
    """
    if not model_id:
        raise ValueError("Model ID must be provided for dense exact search.")
    
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
    """
    Main function to run dense exact search example.

    This function:
    1. Initializes OpenSearch client and EmbeddingConnector
    2. Configures k-NN index settings
    3. Loads dataset with vector embeddings
    4. Provides interactive semantic search interface
    """
    args = cmd_line_interface.get_command_line_args()

    if args.opensearch_type != "aos":
        logging.error(
            "This example is designed for Amazon OpenSearch Service (AOS) only."
        )
        sys.exit(1)

    # Configuration for dense exact search
    os_type = "aos"  # Amazon OpenSearch Service
    provider = "sagemaker"  # Using SageMaker for this example (can be changed to "bedrock")
    index_name = "dense_exact_search"
    embedding_type = "dense"
    pipeline_name = "dense-ingest-pipeline"

    logging.info(f"Initializing dense exact search with {provider.upper()} on {os_type.upper()}")

    # Initialize OpenSearch client and data reader
    client = OsMlClientWrapper(get_client(os_type))
    pqa_reader = QAndAFileReader(
        directory=get_qanda_file_reader_path(),
        max_number_of_docs=args.number_of_docs_per_category,
    )

    # Get AOS client configs for domain info
    aos_configs = get_client_configs("aos")

    # Initialize the universal EmbeddingConnector
    try:
        logging.info(f"Attempting to initialize EmbeddingConnector...")
        logging.info(f"  - Provider: {provider}")
        logging.info(f"  - OS Type: {os_type}")
        
        embedding_connector = EmbeddingConnector(
            os_client=client.os_client,
            provider=provider,
            os_type=os_type,
            opensearch_domain_url=aos_configs["host_url"],
            opensearch_domain_arn=f"arn:aws:es:{aos_configs['region']}:*:domain/{aos_configs['domain_name']}",
            opensearch_username=aos_configs["username"],
            opensearch_password=aos_configs["password"],
            aws_user_name=aos_configs["aws_user_name"],
            region=aos_configs["region"],
        )
        
        logging.info(f"✓ EmbeddingConnector initialized successfully")
        logging.info(f"  - Provider: {embedding_connector.get_provider()}")
        logging.info(f"  - OS Type: {embedding_connector.get_os_type()}")
        logging.info(f"  - Embedding Type: {embedding_connector.get_embedding_type()}")
        logging.info(f"  - Model Dimensions: {embedding_connector.get_model_dimensions()}")
        
        # Try to get connector ID safely
        try:
            connector_id = embedding_connector.connector_id()
            logging.info(f"  - Connector ID: {connector_id}")
        except Exception as e:
            logging.error(f"Failed to get connector ID: {e}")
            raise
        
    except Exception as e:
        logging.error(f"Failed to initialize EmbeddingConnector: {e}")
        logging.error("Please ensure your configuration is properly set up.")
        
        # Add more detailed error information
        import traceback
        logging.error("Full error traceback:")
        logging.error(traceback.format_exc())
        sys.exit(1)

    # Get connector configuration for index setup
    try:
        connector_info = embedding_connector.get_provider_model_info()
        logging.info(f"✓ Retrieved connector model info: {list(connector_info.keys())}")
    except Exception as e:
        logging.error(f"Failed to get connector model info: {e}")
        sys.exit(1)
    
    config = {
        "with_knn": True,
        "pipeline_field_map": get_pipeline_field_map(),
        "categories": args.categories,
        "index_name": index_name,
        "pipeline_name": pipeline_name,
        "embedding_type": embedding_type,
        "delete_existing_index": args.delete_existing_index,
        "bulk_send_chunk_size": args.bulk_send_chunk_size,
        "model_dimensions": embedding_connector.get_model_dimensions(),
    }

    # Create ML model using the embedding connector
    # Note: This maintains compatibility with the existing model system
    # while using the new connector architecture
    try:
        connector_id = embedding_connector.connector_id()
        model_config = {
            "model_name": f"{os_type}_{provider}",
            "embedding_type": embedding_type,
            "model_dimensions": embedding_connector.get_model_dimensions(),
            "connector_id": connector_id,
        }
        
        # Add provider-specific configuration
        model_config.update(connector_info)
        
        logging.info(f"✓ Model configuration prepared with connector ID: {connector_id}")
        
    except Exception as e:
        logging.error(f"Failed to prepare model configuration: {e}")
        sys.exit(1)
    
    ml_model = get_ml_model(
        host_type=os_type,
        model_host=provider,
        model_config=model_config,
        os_client=client.os_client,
        ml_commons_client=client.ml_commons_client,
        model_group_id=client.ml_model_group.model_group_id(),
    )
    
    config.update(model_config)

    # Create index settings with k-NN configuration
    config["index_settings"] = create_index_settings(
        base_mapping_path=get_base_mapping_path(),
        index_config=config,
    )

    # Handle index creation - ensures the index exists and creates/applies the mapping
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
    logging.info(f"Using {provider.upper()} embeddings with {embedding_connector.get_model_dimensions()} dimensions")

    # Start interactive search loop using the generic function
    cmd_line_interface.interactive_search_loop(
        client=client,
        index_name=index_name,
        model_info=ml_model.model_id(),
        query_builder_func=build_dense_exact_query,
        ml_model=ml_model,
        question=args.question,
    )


if __name__ == "__main__":
    main()
