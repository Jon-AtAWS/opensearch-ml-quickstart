# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

"""
Dense S3 Search Example

This module demonstrates dense vector search using OpenSearch with s3.
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
from client import OsMlClientWrapper, get_client
from configs.configuration_manager import get_client_configs, get_qanda_file_reader_path
from connectors import EmbeddingConnector
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
        "settings": {"index": {"knn": True}, "default_pipeline": pipeline_name},
        "mappings": {
            "properties": {
                "chunk_embedding": {
                    "type": "knn_vector",
                    "dimension": model_dimensions,
                    "method": {"engine": "s3vector"},
                }
            }
        },
    }

    index_settings = {"mappings": base_mapping}
    dataset.update_mapping(index_settings, knn_settings)
    return index_settings


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
    provider = (
        "bedrock"  # Using Bedrock for this example (can be changed to "sagemaker")
    )
    index_name = "dense_s3_search"
    embedding_type = "dense"
    pipeline_name = "dense-exact-ingest-pipeline"

    logging.info(
        f"Initializing dense exact search with {provider.upper()} on {os_type.upper()}"
    )

    # Initialize OpenSearch client and dataset
    client = OsMlClientWrapper(get_client(os_type))
    dataset = AmazonPQADataset(dataset_path = get_qanda_file_reader_path(), max_number_of_docs=args.number_of_docs_per_category)

    # Get AOS client configs for domain info
    aos_configs = get_client_configs("aos")

    # Initialize the universal EmbeddingConnector
    try:
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

    # Get connector configuration for index setup
    try:
        connector_info = embedding_connector.get_provider_model_info()
    except Exception as e:
        logging.error(f"Failed to get connector model info: {e}")
        sys.exit(1)

    config = {
        "with_knn": True,
        "pipeline_field_map": {
            "chunk_text": "chunk_embedding"
        },  # Example-specific field mapping
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

        logging.info(
            f"✓ Model configuration prepared with connector ID: {connector_id}"
        )

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

    # Configure index settings with kNN and vector fields
    index_settings = configure_index_for_dense_search(
        dataset, pipeline_name, embedding_connector.get_model_dimensions()
    )

    # Create index using dataset with custom settings
    dataset.create_index(
        os_client=client.os_client,
        index_name=index_name,
        delete_existing=args.delete_existing_index,
        index_settings=index_settings,
    )

    # Set up k-NN pipeline for automatic embedding generation
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
            bulk_chunk_size=args.bulk_send_chunk_size,
        )
        logging.info(f"Loaded {total_docs} documents")

    logging.info("Setup complete! Starting interactive search interface...")
    logging.info(
        f"Using {provider.upper()} embeddings with {embedding_connector.get_model_dimensions()} dimensions"
    )

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
