# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import os
import sys
import uuid
from typing import Dict

import cmd_line_interface

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from client import OsMlClientWrapper, get_client, index_utils
from configs.configuration_manager import (
    get_base_mapping_path,
    get_client_configs,
    get_pipeline_field_map,
    get_qanda_file_reader_path,
)
from data_process.amazon_pqa_dataset import AmazonPQADataset
from connectors import LlmConnector
from connectors.helper import get_remote_connector_configs, get_raw_config_value
from mapping import get_base_mapping, mapping_update
from models import (
    MlModel,
    RemoteMlModel,
    get_ml_model,
)

logging.basicConfig(
    format="%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)


def create_index_settings(base_mapping_path, index_config):
    settings = get_base_mapping(base_mapping_path)
    pipeline_name = index_config["pipeline_name"]
    knn_settings = {
        "settings": {"index": {"knn": True}, "default_pipeline": pipeline_name},
        "mappings": {
            "properties": {
                "chunk": {"type": "text", "index": False},
                "chunk_embedding": {
                    "type": "rank_features",
                },
            }
        },
    }
    mapping_update(settings, knn_settings)
    return settings


def create_text_gen_model():
    """
    Create a text generation model using the LlmConnector.
    
    Returns:
        str: The model ID of the created LLM model
    """
    client = OsMlClientWrapper(get_client("aos"))
    
    # Get LLM connector configs
    from configs import get_model_config
    from dataclasses import asdict
    llm_config = get_model_config("aos", "bedrock", "llm")
    connector_configs = asdict(llm_config)

    logging.info(f"connector_configs:\n{connector_configs}")
    
    # Get AOS client configs for domain info
    aos_configs = get_client_configs("aos")

    # Create the LLM connector
    llm_connector = LlmConnector(
        os_client=client.os_client,
        os_type="aos",
        opensearch_domain_url=aos_configs["host_url"],
        opensearch_domain_arn=f"arn:aws:es:{aos_configs['region']}:*:domain/{aos_configs['domain_name']}",
        opensearch_username=aos_configs["username"],
        opensearch_password=aos_configs["password"],
        aws_user_name=aos_configs["aws_user_name"],
        region=aos_configs["region"],
        connector_configs=connector_configs,
        llm_type="predict",  # Use predict API for conversational search
        model_name=connector_configs["model_name"],  # Use model name from config
    )
    
    logging.info(f"connector id of the llm connector: {llm_connector.connector_id()}")
    
    # Create the remote ML model using the connector
    model_group_id = client.ml_model_group.model_group_id()
    llm_model = RemoteMlModel(
        os_client=client.os_client,
        ml_commons_client=client.ml_commons_client,
        ml_connector=llm_connector,
        model_group_id=model_group_id,
        model_name="Amazon Bedrock Claude",
    )
    
    llm_model_id = llm_model.model_id()
    logging.info(f"Model id of the llm model: {llm_model_id}")
    return llm_model_id


def process_conversational_results(search_results, **kwargs):
    """
    Custom result processor for conversational search that handles both search results and LLM answers.

    Parameters:
        search_results (dict): OpenSearch search response
        **kwargs: Additional parameters (unused)
    """
    import logging

    # Print standard search results
    cmd_line_interface.print_search_results(search_results)

    # Extract and print LLM answer
    if (
        "ext" in search_results
        and "retrieval_augmented_generation" in search_results["ext"]
    ):
        cmd_line_interface.print_answer(
            search_results["ext"]["retrieval_augmented_generation"]["answer"]
        )
    else:
        logging.warning("No LLM answer found in response")


def build_conversational_query(query_text, model_id=None, memory_id=None, **kwargs):
    """
    Build conversational search query with RAG parameters.

    Parameters:
        query_text (str): The search query text
        model_id (str): ML model ID for generating embeddings
        memory_id (str): ID of the conversation memory
        **kwargs: Additional parameters (unused)

    Returns:
        dict: OpenSearch query dictionary with RAG extensions
    """
    if not model_id:
        raise ValueError("Model ID must be provided for conversational search.")
    if not memory_id:
        raise ValueError("Memory ID must be provided for conversational search.")
    return {
        "size": 3,
        "query": {
            "neural_sparse": {
                "chunk_embedding": {
                    "query_text": query_text,
                    "model_id": model_id,
                }
            }
        },
        "ext": {
            "generative_qa_parameters": {
                "llm_model": "bedrock/claude",
                "llm_question": query_text,
                "llm_response_field": "response",
                "memory_id": memory_id,
                "context_size": 10,
                "message_size": 10,
                "timeout": 30,
            }
        },
    }


def main():
    args = cmd_line_interface.get_command_line_args()

    host_type = "aos"
    model_host = "sagemaker"
    embedding_type = "sparse"
    index_name = "amazon_pqa_qa_emebedding"
    ingest_pipeline_name = "sparse-ingest-pipeline"
    search_pipeline_name = "conversational-search-pipeline"

    if args.opensearch_type != "aos":
        logging.error(
            "This example is designed for Amazon OpenSearch Service (AOS) only."
        )
        sys.exit(1)

    client = OsMlClientWrapper(get_client(host_type))

    config = {
        "with_knn": True,
        "pipeline_field_map": {"chunk_text": "chunk_embedding"},  # Use chunk_text for AmazonPQADataset
        "categories": args.categories,
        "index_name": index_name,
        "pipeline_name": ingest_pipeline_name,
        "embedding_type": embedding_type,
        "delete_existing_index": args.delete_existing_index,
        "bulk_send_chunk_size": args.bulk_send_chunk_size,
    }

    # Create embedding model for sparse search
    from connectors.helper import get_remote_connector_configs
    
    model_name = f"{host_type}_{model_host}"
    model_config = get_remote_connector_configs(
        host_type=host_type, connector_type=model_host
    )
    model_config["model_name"] = model_name
    model_config["embedding_type"] = embedding_type
    ml_model = get_ml_model(
        host_type=host_type,
        model_host=model_host,
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

    index_utils.handle_index_creation(
        os_client=client.os_client,
        config=config,
        delete_existing=config["delete_existing_index"],
    )

    logging.info("Setting up for KNN")
    client.setup_for_kNN(
        ml_model=ml_model,
        index_name=config["index_name"],
        pipeline_name=ingest_pipeline_name,
        pipeline_field_map=config["pipeline_field_map"],
        embedding_type=config["embedding_type"],
    )

    # Load data using dataset abstraction
    if not args.no_load:
        dataset = AmazonPQADataset(max_number_of_docs=args.number_of_docs_per_category)
        total_docs = dataset.load_data(
            os_client=client.os_client,
            index_name=config["index_name"],
            filter_criteria=args.categories,
            bulk_chunk_size=args.bulk_send_chunk_size
        )
        print(f"Loaded {total_docs} documents")

    # Create text generation model using the new universal LlmConnector
    text_gen_model_id = create_text_gen_model()

    # Create search pipeline for conversational search
    response = client.os_client.transport.perform_request(
        "PUT",
        f"/_search/pipeline/{search_pipeline_name}",
        body={
            "response_processors": [
                {
                    "retrieval_augmented_generation": {
                        "tag": "conversation demo",
                        "description": "Demo pipeline Using Bedrock Connector",
                        "model_id": f"{text_gen_model_id}",
                        "context_field_list": [
                            "item_name",
                            "product_description",
                            "chunk_text",
                        ],
                        "system_prompt": "You are a helpful assistant",
                        "user_instructions": "Generate a concise and informative answer in less than 100 words for the given question",
                    }
                }
            ]
        },
    )

    # Create conversation memory
    uuid_str = str(uuid.uuid4())[:8]
    conversation_name = f"conversation-{uuid_str}"
    response = client.os_client.transport.perform_request(
        "POST", "/_plugins/_ml/memory/", body={"name": conversation_name}
    )
    memory_id = response["memory_id"]
    logging.info(f"Conversation Memory ID: {memory_id}")

    logging.info(
        "Setup complete! Starting interactive conversational search interface..."
    )

    # Start interactive search loop using the generic function
    cmd_line_interface.interactive_search_loop(
        client=client,
        index_name=index_name,
        model_info=f"Embedding: {ml_model.model_id()}, LLM: {text_gen_model_id}",
        query_builder_func=build_conversational_query,
        result_processor_func=process_conversational_results,
        ml_model=ml_model,
        memory_id=memory_id,
        search_params={"search_pipeline": search_pipeline_name},
        question=args.question,
    )


if __name__ == "__main__":
    main()
