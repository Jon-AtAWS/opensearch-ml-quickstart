# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

"""
Workflow Example Module

This module demonstrates using OpenSearch's template API to create and execute
workflows for automated index setup, then provides an interactive search interface.
It integrates with existing MLModel classes and uses the AmazonPQADataset for data loading.

Features:
1. Uses OpenSearch workflow templates for automated setup
2. Creates, provisions, and executes workflows before data loading
3. Creates a dense vector index named "workflow_dense"
4. Integrates with existing ML model infrastructure
5. Provides interactive query loop for user searches
6. Handles both template-based and fallback manual setup
7. Verifies index creation after workflow execution
"""

import json
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


def create_workflow_template(
    index_name, pipeline_name, model_id, model_dimension, dataset
):
    """
    Create a workflow template for automated OpenSearch setup.

    This template automates the creation of:
    - Index with dense vector configuration using dataset mapping
    - Ingest pipeline for text embedding
    - Proper mappings and settings from dataset configuration

    Parameters:
        index_name (str): Name of the index to create
        pipeline_name (str): Name of the ingest pipeline
        model_id (str): ID of the ML model for embeddings
        model_dimension (int): Dimension of the embedding vectors
        dataset: Dataset instance for mapping configuration

    Returns:
        dict: Complete workflow template configuration
    """
    # Create index settings using dataset mapping
    index_settings = create_index_settings(dataset, pipeline_name, model_dimension)

    return {
        "name": f"{index_name}_workflow_template",
        "description": f"Workflow template for {index_name} dense vector search setup",
        "use_case": "SEMANTIC_SEARCH",
        "version": {"template": "1.0.0", "compatibility": ["2.12.0", "3.0.0"]},
        "workflows": {
            "provision": {
                "nodes": [
                    {
                        "id": "create_index_node",
                        "type": "create_index",
                        "user_inputs": {
                            "index_name": index_name,
                            "configurations": index_settings,
                        },
                    },
                    {
                        "id": "create_pipeline_node",
                        "type": "create_ingest_pipeline",
                        "previous_node_inputs": {"create_index_node": "index_name"},
                        "user_inputs": {
                            "pipeline_id": pipeline_name,
                            "configurations": {
                                "description": f"Text embedding pipeline for {index_name}",
                                "processors": [
                                    {
                                        "text_embedding": {
                                            "model_id": model_id,
                                            "field_map": {"chunk_text": "chunk_embedding"},
                                        }
                                    }
                                ],
                            },
                        },
                    },
                ]
            }
        },
    }


def create_index_settings(dataset, pipeline_name, model_dimension):
    """
    Create index settings by using dataset mapping and updating with vector configuration.

    Parameters:
        dataset: Dataset instance for base mapping
        pipeline_name (str): Name of the ingest pipeline
        model_dimension (int): Dimension of the embedding vectors

    Returns:
        dict: Index settings with dense vector configuration
    """
    # Get base mapping from dataset
    base_mapping = dataset.get_index_mapping()
    
    # Create the vector field configuration to add to the base mapping
    dense_vector_settings = {
        "settings": {"index": {"knn": True}, "default_pipeline": pipeline_name},
        "mappings": {
            "properties": {
                "chunk_embedding": {
                    "type": "knn_vector",
                    "dimension": model_dimension,
                    "method": {
                        "name": "hnsw",
                        "space_type": "l2",
                        "engine": "faiss",
                        "parameters": {"ef_construction": 128, "m": 24},
                    },
                }
            }
        },
    }

    # Create index settings and update with vector configuration
    index_settings = {"mappings": base_mapping}
    dataset.update_mapping(index_settings, dense_vector_settings)
    return index_settings


def verify_index_creation(client, index_name):
    """
    Verify that the index was created successfully.

    Parameters:
        client (OsMlClientWrapper): OpenSearch ML client wrapper
        index_name (str): Name of the index to verify

    Returns:
        bool: True if index exists and is ready, False otherwise
    """
    try:
        # Check if index exists
        exists_response = client.os_client.indices.exists(index=index_name)
        if not exists_response:
            logging.warning(f"Index {index_name} does not exist")
            return False

        # Get index information
        index_info = client.os_client.indices.get(index=index_name)
        logging.info(f"Index {index_name} exists and is ready")

        # Check if the index has the expected vector field
        mappings = index_info[index_name]["mappings"]
        if "chunk_embedding" in mappings.get("properties", {}):
            logging.info("Vector field 'chunk_embedding' found in index mappings")
            return True
        else:
            logging.warning(
                "Vector field 'chunk_embedding' not found in index mappings"
            )
            return False

    except Exception as e:
        logging.error(f"Error verifying index creation: {e}")
        return False


def setup_and_run_workflow(client, workflow_template):
    """
    Set up, provision, and execute the workflow template.

    Parameters:
        client (OsMlClientWrapper): OpenSearch ML client wrapper
        workflow_template (dict): Workflow template configuration

    Returns:
        tuple: (success: bool, workflow_id: str or None)
    """
    try:
        logging.info("Creating workflow template...")

        # Create workflow template
        template_response = client.os_client.transport.perform_request(
            "POST", "/_plugins/_flow_framework/workflow", body=workflow_template
        )

        workflow_id = template_response.get("workflow_id")
        if not workflow_id:
            logging.error("Failed to get workflow ID from template creation response")
            return False, None

        logging.info(f"Created workflow template with ID: {workflow_id}")

        # Provision the workflow
        logging.info("Provisioning workflow...")
        provision_response = client.os_client.transport.perform_request(
            "POST", f"/_plugins/_flow_framework/workflow/{workflow_id}/_provision"
        )

        logging.info(f"Workflow provisioning initiated: {provision_response}")

        workflow_id = provision_response.get("workflow_id", None)
        if not workflow_id:
            logging.error("Failed to get workflow ID from provisioning response")
            return False, None

        state = client.os_client.transport.perform_request(
            "GET",
            f"/_plugins/_flow_framework/workflow/{workflow_id}/_status",
        )
        while state.get("state") != "COMPLETED":
            logging.info(f"Workflow state: {state.get('state')}")
            if state.get("state") == "FAILED":
                logging.error(f"Workflow provisioning failed: {state}")
                return False, None
            state = client.os_client.transport.perform_request(
                "GET",
                f"/_plugins/_flow_framework/workflow/{workflow_id}/_status",
            )

        return True, workflow_id

    except Exception as e:
        logging.error(f"Workflow template setup failed: {e}")
        return False, None


def build_workflow_query(query_text, model_id=None, **kwargs):
    """
    Build neural search query for dense vector search in workflow example.

    Parameters:
        query_text (str): The search query text
        model_id (str): ML model ID for generating embeddings
        **kwargs: Additional parameters (unused)

    Returns:
        dict: OpenSearch query dictionary
    """
    if not model_id:
        raise ValueError("Model ID must be provided for workflow search.")
    return {
        "size": 5,
        "query": {
            "neural": {
                "chunk_embedding": {
                    "query_text": query_text,
                    "model_id": model_id,
                    "k": 10,
                }
            }
        },
    }


def main():
    """
    Main function to run the workflow example.

    This function:
    1. Sets up command line arguments and configuration
    2. Initializes OpenSearch client and ML model
    3. Creates workflow template for index setup
    4. Loads data using AmazonPQADataset
    5. Provides interactive search interface
    """
    # Parse command line arguments
    args = cmd_line_interface.get_command_line_args()

    if args.opensearch_type != "aos":
        logging.error(
            "This example is designed for Amazon OpenSearch Service (AOS) only."
        )
        sys.exit(1)

    # Configuration
    host_type = "aos"  # Amazon OpenSearch Service
    model_type = "bedrock"   # Using Bedrock for ML model hosting
    index_name = "workflow_dense"  # Required index name
    pipeline_name = "workflow-dense-pipeline"

    logging.info(f"Starting workflow example with index: {index_name}")

    # Initialize OpenSearch client and dataset
    client = OsMlClientWrapper(get_client(host_type))
    dataset = AmazonPQADataset(max_number_of_docs=args.number_of_docs_per_category)

    # Set up ML model using existing MLModel classes
    model_name = f"{host_type}_{model_type}"
    model_config = get_remote_connector_configs(
        host_type=host_type, connector_type=model_type
    )
    model_config["model_name"] = model_name
    model_config["embedding_type"] = "dense"

    logging.info("Initializing ML model...")
    ml_model = get_ml_model(
        host_type=host_type,
        model_host=model_type,
        model_config=model_config,
        os_client=client.os_client,
        ml_commons_client=client.ml_commons_client,
        model_group_id=client.ml_model_group.model_group_id(),
    )

    # Create workflow template
    workflow_template = create_workflow_template(
        index_name=index_name,
        pipeline_name=pipeline_name,
        model_id=ml_model.model_id(),
        model_dimension=model_config["model_dimensions"],
        dataset=dataset,
    )

    # if the index already exists and the cmd line does not specify to delete
    # it, then the workflow provision fails when it tries to create the index.
    # So check for the existence, and if not deleting it, then don't run the
    # workflow. This code can't use dataset.create_index (which
    # does the above), since the workflow will create the index.
    index_exists = client.os_client.indices.exists(index=index_name)
    if index_exists and not args.delete_existing_index:
        logging.warning(f"Index {index_name} already exists. Skipping workflow setup.")
        template_success, workflow_id = True, None
    elif index_exists:
        logging.info(f"Deleting existing index: {index_name}")
        client.os_client.indices.delete(index=index_name)
        template_success, workflow_id = setup_and_run_workflow(
            client, workflow_template
        )
    else:
        # Index doesn't exist, run the workflow to create it
        logging.info(f"Index {index_name} doesn't exist. Running workflow to create it.")
        template_success, workflow_id = setup_and_run_workflow(
            client, workflow_template
        )

    if template_success:
        logging.info(
            f"Successfully set up and executed workflow template: {workflow_id}"
        )

        # Verify that the index was created successfully
        if verify_index_creation(client, index_name):
            logging.info("Index and pipeline are ready for data loading")
        else:
            logging.warning(
                "Index verification failed, but continuing with data loading"
            )
    else:
        logging.error("Failed to set up and execute the workflow template. Exiting")
        sys.exit(1)

    # Load data using dataset
    logging.info("Loading data into the index...")
    if not getattr(args, "no_load", False):
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
        query_builder_func=build_workflow_query,
        ml_model=ml_model,
        question=args.question,
    )


if __name__ == "__main__":
    main()
