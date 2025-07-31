# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

"""
Workflow with Template Module

This module demonstrates using OpenSearch's built-in semantic_search workflow template
to create and execute workflows for automated index setup, then provides an interactive 
search interface. It integrates with existing MLModel classes and uses the 
index_utils.handle_data_loading function.

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
from client import OsMlClientWrapper, get_client, index_utils
from configs import (
    BASE_MAPPING_PATH,
    PIPELINE_FIELD_MAP,
    QANDA_FILE_READER_PATH,
    get_remote_connector_configs,
)
from data_process import QAndAFileReader
from mapping import get_base_mapping
from ml_models import get_ml_model

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
        "create_index.mappings.method.engine": "faiss",
        "create_index.mappings.method.space_type": "l2",
        "create_index.mappings.method.name": "hnsw",
        "create_ingest_pipeline.pipeline_id": pipeline_name,
        "create_ingest_pipeline.description": f"Ingest pipeline for {index_name}",
        "text_embedding.field_map.input": "chunk",
        "text_embedding.field_map.output": "chunk_embedding",
        "create_ingest_pipeline.model_id": model_id,
        "text_embedding.field_map.output.dimension": model_dimension,
    }


def add_additional_field_mappings(os_client: OpenSearch, index_name):
    """
    Get the base mapping and pull out the additional fields

    Parameters:
        index_name (str): Name of the index
        base_mapping_path (str): Path to base mapping configuration
        index_config (dict): Index configuration parameters

    Returns:
        dict: Index settings with dense vector configuration
    """
    logging.info(f"Adding additional field mappings to {index_name}")

    # Read the base mapping from the specified path
    index_config = get_base_mapping(BASE_MAPPING_PATH)
    properties = index_config.get("mappings", None)

    if not properties:
        logging.warning("No mappings found in the base mapping, not adding any")
        return True

    logging.info(f"Adding {json.dumps(properties, indent=2)}")
    response = os_client.transport.perform_request(
        "PUT", f"/{index_name}/_mapping", body=properties
    )
    if not response.get("acknowledged"):
        logging.error("Failed to add additional field mappings")
        return False

    logging.info("Successfully added additional field mappings")
    return True


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

        return True, workflow_id  # Continue anyway, as resources might still be created

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
    4. Loads data using existing index_utils.handle_data_loading
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
    embedding_type = "dense"
    pipeline_name = PIPELINE_NAME  # Use constant for pipeline name

    logging.info(f"Starting workflow with template example using index: {index_name}")

    # Initialize OpenSearch client and data reader
    client = OsMlClientWrapper(get_client(host_type))
    pqa_reader = QAndAFileReader(
        directory=QANDA_FILE_READER_PATH,
        max_number_of_docs=args.number_of_docs_per_category,
    )

    # Configuration dictionary
    config = {
        "with_knn": True,
        "pipeline_field_map": PIPELINE_FIELD_MAP,
        "index_name": index_name,
        "pipeline_name": pipeline_name,
        "embedding_type": embedding_type,
        "categories": args.categories,
        "delete_existing_index": args.delete_existing_index,
        "bulk_send_chunk_size": args.bulk_send_chunk_size,
    }

    # Set up ML model using existing MLModel classes
    model_name = f"{host_type}_{model_type}"
    model_config = get_remote_connector_configs(
        host_type=host_type, connector_type=model_type
    )
    model_config["model_name"] = model_name
    model_config["embedding_type"] = embedding_type
    config.update(model_config)

    logging.info("Initializing ML model...")
    ml_model = get_ml_model(
        host_type=host_type,
        model_type=model_type,
        model_config=model_config,
        os_client=client.os_client,
        ml_commons_client=client.ml_commons_client,
        model_group_id=client.ml_model_group.model_group_id(),
    )

    workflow_config = semantic_search_workflow_parameters(
        index_name=index_name,
        pipeline_name=pipeline_name,
        model_id=ml_model.model_id(),
        model_dimension=config["model_dimensions"],
    )

    # if the index already exists and the cmd line does not specify to delete
    # it, then the workflow provision fails when it tries to create the index.
    # So check for the existence, and if not deleting it, then don't run the
    # workflow. This code can't use index_utils.handle_index_creation (which
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

    if not add_additional_field_mappings(client.os_client, index_name=index_name):
        logging.warning("Continuing without additional field mappings")

    if template_success:
        logging.info(
            f"Successfully set up and executed semantic search workflow: {workflow_id}"
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

    # Load data using existing index_utils.handle_data_loading
    logging.info("Loading data into the index...")
    index_utils.handle_data_loading(
        os_client=client.os_client,
        pqa_reader=pqa_reader,
        config=config,
        no_load=getattr(args, "no_load", False),
    )

    logging.info("Setup complete! Starting interactive search interface...")

    # Start interactive search loop using the generic function
    cmd_line_interface.interactive_search_loop(
        client=client,
        index_name=index_name,
        model_info=ml_model.model_id(),
        query_builder_func=build_workflow_template_query,
        ml_model=ml_model,
    )


if __name__ == "__main__":
    main()
