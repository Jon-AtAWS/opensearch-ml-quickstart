# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

"""
Conversational Agent Example

This module demonstrates OpenSearch's conversational agent capabilities using
the ML Commons conversational agent API. It combines semantic search with
large language models to provide contextual, conversational responses.

The example:
1. Loads data from Amazon PQA dataset with embeddings
2. Sets up a conversational agent with LLM model
3. Creates tools for the agent to search the knowledge base
4. Provides interactive conversational interface
"""

import json
import logging
import os
import sys

import cmd_line_interface

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import agent_tools
from client import OsMlClientWrapper, get_client, index_utils
from configs.configuration_manager import (
    get_base_mapping_path,
    get_client_configs,
    get_pipeline_field_map,
    get_qanda_file_reader_path,
    config_override,  # Add this import
)
from connectors.helper import get_remote_connector_configs, get_raw_config_value
from data_process import QAndAFileReader
from mapping import get_base_mapping, mapping_update
from models import (
    RemoteMlModel,
    get_ml_model,
)
from connectors import LlmConnector

logging.basicConfig(
    format="%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)


def create_index_settings(base_mapping_path, index_config):
    """
    Create OpenSearch index settings for conversational agent.

    Parameters:
        base_mapping_path (str): Path to base mapping configuration
        index_config (dict): Configuration containing pipeline and model settings

    Returns:
        dict: Updated index settings with dense vector configuration
    """
    settings = get_base_mapping(base_mapping_path)
    pipeline_name = index_config["pipeline_name"]
    model_dimension = index_config["model_dimensions"]
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


def create_llm_model(client: OsMlClientWrapper):
    """
    Create and deploy LLM model for conversational agent using the LlmConnector.

    Parameters:
        client (OsMlClientWrapper): OpenSearch ML client wrapper

    Returns:
        str: Model ID of the deployed LLM model
    """
    # Get base connector configs for local OpenSearch + Bedrock
    connector_configs = get_remote_connector_configs("bedrock", "os")

    logging.info(f"LLM connector configs:\n{connector_configs}")

    # Create the LLM connector (uses basic auth for local OpenSearch, credentials for Bedrock)
    llm_connector = LlmConnector(
        os_client=client.os_client,
        os_type="os",
        connector_configs=connector_configs,
        llm_type="converse",  # Use converse API for agent workflows
        model_name="us.anthropic.claude-3-7-sonnet-20250219-v1:0",  # Claude 3.5 Sonnet v2
    )

    logging.info(f"LLM connector ID: {llm_connector.connector_id()}")

    # Create the remote ML model using the connector
    model_group_id = client.ml_model_group.model_group_id()
    llm_model = RemoteMlModel(
        os_client=client.os_client,
        ml_commons_client=client.ml_commons_client,
        ml_connector=llm_connector,
        model_group_id=model_group_id,
        model_name="Amazon Bedrock Claude for Agent",
    )

    llm_model_id = llm_model.model_id()
    logging.info(f"LLM model ID: {llm_model_id}")
    return llm_model_id


def create_conversational_agent(
    client: OsMlClientWrapper,
    index_name: str,
    embedding_model_id: str,
    llm_model_id: str,
):
    """
    Create a conversational agent with the specified LLM model and tools.

    Parameters:
        client (OsMlClientWrapper): OpenSearch ML client wrapper
        llm_model_id (str): ID of the LLM model
        tools (list): List of tools for the agent

    Returns:
        str: Agent ID
    """
    agent_config = {
        "name": "RAG Agent",
        "type": "conversational",
        "description": "this is a test agent",
        "app_type": "rag",
        "llm": {
            "model_id": llm_model_id,
            "parameters": {
                "max_iteration": 20,
                "system_prompt":
                    "You are a helpful assistant that can answer questions about products "
                    "in your knowledge base. "
                    "  "
                    "The knowledge base contains user questions and answers, with one "
                    "search document per user question. Each search document also contains "
                    "product information such as item name, product description, and brand name. "
                    "  "
                    "You have tools that search based on matching the user question "
                    "to the question in the search result, as well as lexical and semantic "
                    "search against the product information. "
                    "  "
                    "Because the knowledge base is organized by user questions, you may not "
                    "get a broadly diverse range of product information in the search results, "
                    "so try variants of the user question to get a wider range of products. "
                    "  "
                    "First evaluate whether the user is asking a broad question about products, "
                    "or a specific question about a product. If the question is broad, you will "
                    "use the lexical and semantic search tool to find products that are similar "
                    "to the user's query. If it seems that the user question is about product "
                    "features or use, you will use the Q&A search tool to find questions users "
                    "have asked about products. "
                    "  "
                    "In summarizing the search results include whether you approached the "
                    "question as a broad product search or a specific product question. ",
                "prompt": "${parameters.question}"
            }
        },
        "memory": {
            "type": "conversation_index"
        },
        "parameters": {
            "_llm_interface": "bedrock/converse/claude"
        },
        "tools": [
            agent_tools.get_products_tool_semantic(index_name, embedding_model_id),  # Include the semantic search tool
            agent_tools.get_products_tool_lexical(index_name),  # Include the lexical search tool
            agent_tools.get_products_qna_lexical(index_name),  # Include the Q&A lexical search tool
            agent_tools.list_index_tool(),  # Include the list index tool
            agent_tools.index_mapping_tool(),  # Include the index mapping tool
        ]
    }

    logging.info(f"Creating conversational agent with config: {json.dumps(agent_config, indent=2)}")

    try:        
        # Use OpenSearch client transport layer
        response = client.os_client.transport.perform_request(
            "POST", "/_plugins/_ml/agents/_register", body=agent_config
        )
        # response is already a dict, no need for .json() or .raise_for_status()
        response_data = response
    except Exception as e:
        logging.error(f"Failed to create conversational agent: {e}")
        raise

    agent_id = response_data["agent_id"]
    logging.info(f"Created conversational agent with ID: {agent_id}")
    return agent_id


def execute_agent_query(client, agent_id, query_body):
    """
    Execute a query against the conversational agent.

    Parameters:
        client (OsMlClientWrapper): OpenSearch ML client wrapper
        agent_id (str): ID of the conversational agent
        query_body (dict): Query parameters

    Returns:
        dict: Agent response
    """
    logging.info(f"Executing query for agent {agent_id}: {query_body}")
    try:
        response = client.os_client.transport.perform_request(
            "POST", f"/_plugins/_ml/agents/{agent_id}/_execute", 
            body=query_body
        )
    except Exception as e:
        logging.error(f"Failed to execute agent query: {e}")
        raise
    return response


def build_agent_query(query_text, agent_id=None, **kwargs):
    """
    Build conversational agent query.

    Parameters:
        query_text (str): The user's question
        agent_id (str): ID of the conversational agent
        **kwargs: Additional parameters (unused)

    Returns:
        dict: Agent execution request
    """
    if not agent_id:
        raise ValueError("Agent ID must be provided for conversational agent.")

    return {
        "parameters": {
            "question": query_text,
            "verbose": True
        }
    }


def main():
    """
    Main function to run conversational agent example.

    This function:
    1. Sets up the knowledge base with embeddings
    2. Creates and deploys LLM model
    3. Creates conversational agent with search tools
    4. Provides interactive conversational interface
    """
    args = cmd_line_interface.get_command_line_args()

    if args.opensearch_type != "os":
        logging.error(
            "This example is designed for self-managed OpenSearch (OS) only to support local models."
        )
        sys.exit(1)

    host_type = "os"
    model_type = "local"
    index_name = "conversational_agent_knowledge_base"
    embedding_type = "dense"
    ingest_pipeline_name = "agent-dense-ingest-pipeline"

    # Conversational agent requires OpenSearch 3.1+
    with config_override(MINIMUM_OPENSEARCH_VERSION="3.1.0"):
        client = OsMlClientWrapper(get_client(host_type))
    
    pqa_reader = QAndAFileReader(
        directory=get_qanda_file_reader_path(),
        max_number_of_docs=args.number_of_docs_per_category,
    )

    config = {
        "with_knn": True,
        "pipeline_field_map": get_pipeline_field_map(),
        "categories": args.categories,
        "index_name": index_name,
        "pipeline_name": ingest_pipeline_name,
        "embedding_type": embedding_type,
        "delete_existing_index": args.delete_existing_index,
        "bulk_send_chunk_size": args.bulk_send_chunk_size,
    }

    # Create local dense embedding model for knowledge base using configuration
    from configs.configuration_manager import get_local_dense_embedding_model_name, get_local_dense_embedding_model_version, get_local_dense_embedding_model_format, get_local_dense_embedding_model_dimension
    
    model_name = get_local_dense_embedding_model_name()
    model_config = {
        "model_name": model_name,
        "model_version": get_local_dense_embedding_model_version(),
        "model_dimensions": get_local_dense_embedding_model_dimension(),
        "embedding_type": embedding_type,
        "model_format": get_local_dense_embedding_model_format(),
    }
    embedding_ml_model = get_ml_model(
        host_type=host_type,
        model_type=model_type,
        model_config=model_config,
        os_client=client.os_client,
        ml_commons_client=client.ml_commons_client,
        model_group_id=client.ml_model_group.model_group_id(),
    )

    config.update(model_config)
    config["index_settings"] = create_index_settings(
        base_mapping_path=get_base_mapping_path(),
        index_config=config,
    )

    # Set up knowledge base
    index_utils.handle_index_creation(
        os_client=client.os_client,
        config=config,
        delete_existing=config["delete_existing_index"],
    )

    logging.info("Setting up knowledge base with embeddings...")
    client.setup_for_kNN(
        ml_model=embedding_ml_model,
        index_name=config["index_name"],
        pipeline_name=ingest_pipeline_name,
        pipeline_field_map=config["pipeline_field_map"],
        embedding_type=config["embedding_type"],
    )

    # Load data into knowledge base
    index_utils.handle_data_loading(
        os_client=client.os_client,
        pqa_reader=pqa_reader,
        config=config,
        no_load=args.no_load,
    )

    # Create LLM model for agent
    logging.info("Creating LLM model for conversational agent...")
    llm_model_id = create_llm_model(client)

    # Create conversational agent
    logging.info("Creating conversational agent...")
    agent_id = create_conversational_agent(
        client,
        index_name=index_name,
        embedding_model_id=embedding_ml_model.model_id(),
        llm_model_id=llm_model_id,
    )

    logging.info(f"Conversational agent created with ID: {agent_id}")
    logging.info("Setup complete")

    # Check if a specific query was provided via command line
    if hasattr(args, 'question') and args.question:
        logging.info(f"Executing single query: {args.question}")
        try:
            agent_query = build_agent_query(args.question, agent_id)
            results = execute_agent_query(client, agent_id, agent_query)
            cmd_line_interface.process_and_print_agent_results(results)
        except Exception as e:
            logging.error(f"Error executing query: {e}")
        return

    logging.info(
        "Starting interactive agent interface..."
    )

    # Start interactive agent loop using the generic function from cmd_line_interface
    cmd_line_interface.interactive_agent_loop(
        client=client,
        agent_id=agent_id,
        model_info=f"Agent: {agent_id}, LLM: {llm_model_id}, Embedding: {embedding_ml_model.model_id()}",
        build_agent_query_func=build_agent_query,
        agent_executor_func=execute_agent_query,
    )


if __name__ == "__main__":
    main()
