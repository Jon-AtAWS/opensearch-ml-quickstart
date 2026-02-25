# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

"""
Workflow with Template Module

This module demonstrates using OpenSearch's built-in semantic_search workflow template
to create and execute workflows for automated index setup, then provides an interactive 
search interface. It integrates with existing MLModel classes and uses the 
AmazonPQADataset for data loading.

Features:
1. Uses OpenSearch's built-in semantic_search workflow template
2. Creates, provisions, and executes workflows before data loading
3. Creates a dense vector index named "workflow_with_template"
4. Integrates with existing ML model infrastructure
5. Provides interactive query loop for user searches
6. Handles both template-based and fallback manual setup
7. Verifies index creation after workflow execution
"""

import json
import logging
import os
import sys
from opensearchpy import OpenSearch

import cmd_line_interface

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from client import OsMlClientWrapper, get_client
from configs.configuration_manager import get_qanda_file_reader_path
from connectors.helper import get_remote_connector_configs
from data_process.amazon_pqa_dataset import AmazonPQADataset
from models.helper import get_ml_model

logging.basicConfig(
    format="%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)

# Constants for index and pipeline names
INDEX_NAME = "workflow_with_template"
PIPELINE_NAME = "workflow_with_template_pipeline"


def semantic_search_workflow_parameters(
    index_name, pipeline_name, model_id, model_dimension
):
    """
    Create a dict with the parameters for OpenSearch's SEMANTIC_FLOW workflow.

    Parameters:
        index_name (str): Name of the index to create
        pipeline_name (str): Name of the ingest pipeline
        model_id (str): ID of the ML model for embeddings
        model_dimension (int): Dimension of the embedding vectors

    Returns:
        dict: Workflow configuration using semantic_search template
    """
    return {
        "template.name": f"{index_name}_semantic_search_workflow",
        "template.description": f"Semantic search workflow for {index_name} using built-in template",
        "create_index.name": index_name,
        "create_index.mappings.method.engine": "lucene",  # Updated from faiss to lucene
        "create_index.mappings.method.space_type": "l2",
        "create_index.mappings.method.name": "hnsw",
        "create_ingest_pipeline.pipeline_id": pipeline_name,
        "create_ingest_pipeline.description": f"Ingest pipeline for {index_name}",
        "text_embedding.field_map.input": "chunk_text",  # Updated to match dataset
        "text_embedding.field_map.output": "chunk_embedding",
        "create_ingest_pipeline.model_id": model_id,
        "text_embedding.field_map.output.dimension": model_dimension,
    }


def add_additional_field_mappings(os_client: OpenSearch, index_name, dataset):
    """
    Add additional field mappings from dataset to the index.

    Parameters:
        os_client: OpenSearch client
        index_name (str): Name of the index
        dataset: Dataset instance for mapping configuration

    Returns:
        bool: True if successful, False otherwise
    """
    logging.info(f"Adding additional field mappings to {index_name}")

    # Check if the index exists before trying to add mappings
    if not os_client.indices.exists(index=index_name):
        logging.error(f"Index {index_name} does not exist. Cannot add field mappings.")
        return False

    # Get the dataset mapping
    properties = {"mappings": dataset.get_index_mapping()}

    if not properties.get("mappings"):
        logging.warning("No mappings found in the dataset, not adding any")
        return True

    logging.info(f"Adding {json.dumps(properties, indent=2)}")
    
    try:
        response = os_client.transport.perform_request(
            "PUT", f"/{index_name}/_mapping", body=properties["mappings"]
        )
        if not response.get("acknowledged"):
            logging.error("Failed to add additional field mappings")
            return False

        logging.info("Successfully added additional field mappings")
        return True
    except Exception as e:
        logging.error(f"Error adding field mappings: {e}")
        return False


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


def provision_semantic_search_workflow(client, workflow_config):
    """
    Set up, provision, and execute the semantic search workflow using built-in template.

    Parameters:
        client (OsMlClientWrapper): OpenSearch ML client wrapper
        workflow_config (dict): Workflow configuration for semantic search

    Returns:
        tuple: (success: bool, workflow_id: str or None)
    """
    try:
        logging.info("Creating workflow using built-in semantic_search template...")

        # Use the built-in semantic_search template
        template_response = client.os_client.transport.perform_request(
            "POST",
            "/_plugins/_flow_framework/workflow?use_case=semantic_search",
            body=workflow_config,
        )

        workflow_id = template_response.get("workflow_id")
        if not workflow_id:
            logging.error("Failed to get workflow ID from template creation response")
            return False, None

        logging.info(f"Created semantic search workflow with ID: {workflow_id}")

        # Provision the workflow
        logging.info("Provisioning semantic search workflow...")
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
        logging.error(f"Semantic search workflow setup failed: {e}")
        return False, None


def build_workflow_template_query(query_text, model_id=None, **kwargs):
    """
    Build neural search query for dense vector search in workflow with template example.

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
    Main function to run the workflow with template example.

    This function:
    1. Sets up command line arguments and configuration
    2. Initializes OpenSearch client and ML model
    3. Creates semantic search workflow using built-in template
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

    # Configuration using constants
    host_type = "aos"  # Amazon OpenSearch Service
    model_type = "sagemaker"  # Use SageMaker for embeddings
    index_name = INDEX_NAME  # Use constant for index name
    pipeline_name = PIPELINE_NAME  # Use constant for pipeline name

    logging.info(f"Starting workflow with template example using index: {index_name}")

    # Initialize OpenSearch client and dataset
    client = OsMlClientWrapper(get_client(host_type))
    dataset = AmazonPQADataset(
        directory=get_qanda_file_reader_path(),
        max_number_of_docs=args.number_of_docs_per_category
    )

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

    workflow_config = semantic_search_workflow_parameters(
        index_name=index_name,
        pipeline_name=pipeline_name,
        model_id=ml_model.model_id(),
        model_dimension=model_config["model_dimensions"],
    )

    # if the index already exists and the cmd line does not specify to delete
    # it, then the workflow provision fails when it tries to create the index.
    # So check for the existence, and if not deleting it, then don't run the
    # workflow. This code can't use dataset.create_index (which
    # does the above), since the workflow will create the index.
    index_exists = client.os_client.indices.exists(index=index_name)
    template_success = False
    if index_exists and not args.delete_existing_index:
        logging.warning(f"Index {index_name} already exists. Skipping workflow setup.")
        template_success, workflow_id = True, None
    elif index_exists:
        logging.info(f"Deleting existing index: {index_name}")
        client.os_client.indices.delete(index=index_name)
        template_success, workflow_id = provision_semantic_search_workflow(
            client, workflow_config=workflow_config
        )
    else:
        # Index doesn't exist, run the workflow to create it
        logging.info(f"Index {index_name} doesn't exist. Running workflow to create it.")
        template_success, workflow_id = provision_semantic_search_workflow(
            client, workflow_config=workflow_config
        )

    if template_success:
        logging.info(
            f"Successfully set up and executed semantic search workflow: {workflow_id}"
        )

        # Verify that the index was created successfully
        if verify_index_creation(client, index_name):
            logging.info("Index and pipeline are ready for data loading")
            
            # Only add additional field mappings if the index exists and was created successfully
            if not add_additional_field_mappings(client.os_client, index_name, dataset):
                logging.warning("Continuing without additional field mappings")
        else:
            logging.warning(
                "Index verification failed, but continuing with data loading"
            )
    else:
        logging.error("Failed to set up and execute the semantic search workflow. Exiting")
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
        query_builder_func=build_workflow_template_query,
        ml_model=ml_model,
        question=args.question,
    )


if __name__ == "__main__":
    main()
