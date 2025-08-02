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
from client import OsMlClientWrapper, get_client, index_utils
from configs.configuration_manager import (
    get_base_mapping_path,
    get_client_configs,
    get_pipeline_field_map,
    get_qanda_file_reader_path,
)
from connectors.helper import get_remote_connector_configs
from data_process import QAndAFileReader
from mapping import get_base_mapping, mapping_update
from models import (
    MlModel,
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
    )

    logging.info(f"LLM connector ID: {llm_connector.connector_id()}")

    # Create the remote ML model using the connector
    model_group_id = client.ml_model_group.model_group_id()
    llm_model = RemoteMlModel(
        os_client=client.os_client,
        ml_commons_client=client.ml_commons_client,
        ml_connector=llm_connector,
        model_group_id=model_group_id,
        model_name="Amazon Bedrock Claude 3.5 Sonnet for Agent",
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
        "name": "Amazon PQA Conversational Agent",
        "type": "conversational",
        "description": "An intelligent agent that can search and answer questions about Amazon products",
        "llm": {
            "model_id": llm_model_id,
            "parameters": {
                "max_iterations": 5,
                "stop_when_no_tool_found": True,
                "response_filter": "$.completion",
            },
        },
        "memory": {
            "type": "conversation_index",
        },
        "tools": [
            {
                "type": "VectorDBTool",
                "name": "knowledge_base_search",
                "description": "Search the Amazon PQA knowledge base for product information, questions, and answers",
                "parameters": {
                    "model_id": embedding_model_id,
                    "index": [index_name],
                    "input": "${parameters.question}",
                    "embedding_field": "chunk_embedding",
                    "source_field": "chunk",
                    "doc_size": 5,
                },
            },
            {
                "type": "MLModelTool",
                "description": "Use the LLM model to generate answers based on search results",
                "parameters": {
                    "model_id": llm_model_id,
                    "prompt": " ".join(
                        [
                            "\n\nHuman:You are a professional data analyst.",
                            "You will always answer a question based on the given context first.",
                            "If the answer is not directly shown in the context, you will analyze",
                            "the data and find the answer. If you don't know the answer, just say",
                            "you don't know.\n\n",
                            "Context:\n${parameters.VectorDBTool.output}\n\n",
                            "Human:${parameters.question}\n\nAssistant:",
                        ]
                    ),
                },
            },
        ],
    }

    logging.info(f"Creating conversational agent: {json.dumps(agent_config, indent=2)}")

    try:
        response = client.os_client.transport.perform_request(
            "POST", "/_plugins/_ml/agents/_register", body=agent_config
        )
    except Exception as e:
        logging.error(f"Failed to create conversational agent: {e}")
        raise

    agent_id = response["agent_id"]
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
    logging.info(
        f"Executing query for agent {agent_id}: {json.dumps(query_body, indent=2)}"
    )
    try:
        response = client.os_client.transport.perform_request(
            "POST", f"/_plugins/_ml/agents/{agent_id}/_execute", body=query_body
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

    return {"parameters": {"question": query_text}}


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

    # Create local dense embedding model for knowledge base
    model_name = "huggingface/sentence-transformers/all-MiniLM-L6-v2"
    model_config = {
        "model_name": model_name,
        "model_version": "1.0.1",
        "model_dimensions": 384,
        "embedding_type": embedding_type,
        "model_format": "TORCH_SCRIPT",
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

    logging.info(
        "Setup complete! Starting interactive conversational agent interface..."
    )

    # Start interactive agent loop using the generic function from cmd_line_interface
    cmd_line_interface.interactive_agent_loop(
        client=client,
        agent_id=agent_id,
        model_info=f"Agent: {agent_id}, LLM: {llm_model_id}, Embedding: {embedding_ml_model.model_id()}",
        agent_executor_func=execute_agent_query,
    )


if __name__ == "__main__":
    main()
