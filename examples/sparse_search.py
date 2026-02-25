# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import sys

import cmd_line_interface

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from client import OsMlClientWrapper, get_client
from connectors.helper import get_remote_connector_configs
from data_process.amazon_pqa_dataset import AmazonPQADataset
from models.helper import get_ml_model

logging.basicConfig(
    format="%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)


def configure_index_for_sparse_search(dataset, pipeline_name):
    """Configure index settings with sparse vector field for neural sparse search."""
    base_mapping = dataset.get_index_mapping()
    
    sparse_settings = {
        "settings": {"index": {"knn": True}, "default_pipeline": pipeline_name},
        "mappings": {
            "properties": {
                "chunk_sparse_embedding": {
                    "type": "rank_features",
                },
            }
        },
    }
    
    index_settings = {"mappings": base_mapping}
    dataset.update_mapping(index_settings, sparse_settings)
    return index_settings


def load_dataset(
    client: OsMlClientWrapper,
    ml_model,
    dataset: AmazonPQADataset,
    index_name: str,
    pipeline_name: str,
    categories: list,
    bulk_chunk_size: int,
    delete_existing: bool,
    no_load: bool,
):
    logging.info("Adding ingestion pipeline for sparse search...")
    
    # Create ingest pipeline with sparse embedding
    pipeline_config = {
        "description": "Sparse embedding pipeline",
        "processors": [
            {
                "sparse_encoding": {
                    "model_id": ml_model.model_id(),
                    "field_map": {"chunk_text": "chunk_sparse_embedding"}
                }
            }
        ]
    }
    
    client.os_client.ingest.put_pipeline(id=pipeline_name, body=pipeline_config)
    
    # Configure and create index
    index_settings = configure_index_for_sparse_search(dataset, pipeline_name)
    dataset.create_index(
        client.os_client, index_name, delete_existing, index_settings
    )
    
    # Load data using dataset
    if not no_load:
        total_docs = dataset.load_data(
            os_client=client.os_client,
            index_name=index_name,
            filter_criteria=categories,
            bulk_chunk_size=bulk_chunk_size
        )
        logging.info(f"Loaded {total_docs} documents")


def build_sparse_query(query_text, model_id=None, **kwargs):
    """
    Build neural sparse search query.

    Parameters:
        query_text (str): The search query text
        model_id (str): ML model ID for generating embeddings
        **kwargs: Additional parameters (unused)

    Returns:
        dict: OpenSearch query dictionary
    """
    if not model_id:
        raise ValueError("Model ID must be provided for sparse search.")
    return {
        "size": 3,
        "query": {
            "neural_sparse": {
                "chunk_sparse_embedding": {
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

    # Configuration
    host_type = "aos"
    model_type = "sagemaker"
    index_name = "sparse_search"
    ingest_pipeline_name = "sparse-ingest-pipeline"

    # Initialize client and dataset
    client = OsMlClientWrapper(get_client(host_type))
    dataset = AmazonPQADataset(max_number_of_docs=args.number_of_docs_per_category)

    # Get model configuration and create model
    model_name = f"{host_type}_{model_type}"
    model_config = get_remote_connector_configs(
        host_type=host_type, connector_type=model_type
    )
    model_config["model_name"] = model_name
    model_config["embedding_type"] = "sparse"
    
    ml_model = get_ml_model(
        host_type=host_type,
        model_host=model_type,
        model_config=model_config,
        os_client=client.os_client,
        ml_commons_client=client.ml_commons_client,
        model_group_id=client.ml_model_group.model_group_id(),
    )

    # Load dataset and setup index
    load_dataset(
        client,
        ml_model,
        dataset,
        index_name,
        ingest_pipeline_name,
        args.categories,
        args.bulk_send_chunk_size,
        args.delete_existing_index,
        args.no_load,
    )

    logging.info("Setup complete! Starting interactive search interface...")

    # Start interactive search loop
    cmd_line_interface.interactive_search_loop(
        client=client,
        index_name=index_name,
        model_info=ml_model.model_id(),
        query_builder_func=build_sparse_query,
        ml_model=ml_model,
        question=args.question,
    )


if __name__ == "__main__":
    main()
