# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

"""Multi-CPR conversational search with retrieval-augmented generation (RAG).

This example demonstrates conversational search over the Multi-CPR Chinese dataset
using sparse embeddings for retrieval and LLM for answer generation.

Usage:
    # Interactive conversational search on e-commerce domain
    python examples/multi_cpr_conversational_search.py -o aos -c ecom -n 500

    # Search all domains with conversational AI
    python examples/multi_cpr_conversational_search.py -o aos -c ecom video medical -n 1000

    # Single conversational query
    python examples/multi_cpr_conversational_search.py -o aos -c medical -n 200 -q "感冒吃什么药"
"""

import json
import logging
import os
import sys
import uuid
from typing import Dict

import cmd_line_interface

from client import OsMlClientWrapper, get_client, index_utils
from configs.configuration_manager import (
    get_base_mapping_path,
    get_client_configs,
    get_multi_cpr_path,
)
from data_process.multi_cpr_dataset import MultiCPRDataset
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


def create_index_settings(dataset, index_config):
    """Create index settings with Multi-CPR specific mapping and sparse embedding configuration."""
    # Get Multi-CPR specific mapping
    base_mapping = dataset.get_index_mapping()
    
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
    
    # Combine Multi-CPR mapping with sparse embedding settings
    settings = {"mappings": base_mapping}
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
        model_name="Amazon Bedrock Claude for Multi-CPR",
    )
    
    llm_model_id = llm_model.model_id()
    logging.info(f"Model id of the llm model: {llm_model_id}")
    return llm_model_id


def process_conversational_results(search_results, **kwargs):
    """
    Custom result processor for Multi-CPR conversational search.

    Parameters:
        search_results (dict): OpenSearch search response
        **kwargs: Additional parameters (unused)
    """
    import logging

    # Print Multi-CPR specific search results
    print_multi_cpr_search_results(search_results)

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


def print_multi_cpr_search_results(search_results):
    """Print Multi-CPR specific search results."""
    hits = search_results.get("hits", {}).get("hits", [])
    total = search_results.get("hits", {}).get("total", {}).get("value", 0)
    
    print(f"\n📚 Found {total} relevant passages from Multi-CPR dataset:")
    print("=" * 80)
    
    for i, hit in enumerate(hits):
        src = hit.get("_source", {})
        score = hit.get("_score", 0)
        
        print(f"\n🔍 Result {i + 1} | Score: {score:.4f}")
        print(f"📂 Domain: {src.get('domain', 'N/A')}")
        print(f"🆔 PID: {src.get('pid', 'N/A')}")
        print(f"📝 Passage: {src.get('passage', 'N/A')[:200]}...")
        print("-" * 60)
    
    print("=" * 80)


def build_conversational_query(query_text, model_id=None, memory_id=None, **kwargs):
    """
    Build conversational search query with RAG parameters for Multi-CPR.

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
    index_name = "multi_cpr_conversational"
    ingest_pipeline_name = "multi-cpr-sparse-ingest-pipeline"
    search_pipeline_name = "multi-cpr-conversational-search-pipeline"

    if args.opensearch_type != "aos":
        logging.error(
            "This example is designed for Amazon OpenSearch Service (AOS) only."
        )
        sys.exit(1)

    client = OsMlClientWrapper(get_client(host_type))

    # Initialize Multi-CPR dataset
    dataset = MultiCPRDataset(
        directory=get_multi_cpr_path(),
        max_number_of_docs=args.number_of_docs_per_category,
    )

    # Override default Amazon PQA categories with Multi-CPR domains
    valid_domains = dataset.get_available_filters()
    if args.categories is None or not all(c in valid_domains for c in args.categories):
        categories = valid_domains
        logging.info(f"Using all Multi-CPR domains: {categories}")
    else:
        categories = args.categories

    for c in categories:
        if c not in valid_domains:
            logging.error(f"Unknown domain '{c}'. Choose from: {valid_domains}")
            sys.exit(1)

    config = {
        "with_knn": True,
        "pipeline_field_map": {"chunk_text": "chunk_embedding"},  # Use chunk_text for Multi-CPR
        "categories": categories,
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
        dataset=dataset,
        index_config=config,
    )

    index_utils.handle_index_creation(
        os_client=client.os_client,
        config=config,
        delete_existing=config["delete_existing_index"],
    )

    logging.info("Setting up for KNN with Multi-CPR dataset")
    client.setup_for_kNN(
        ml_model=ml_model,
        index_name=config["index_name"],
        pipeline_name=ingest_pipeline_name,
        pipeline_field_map=config["pipeline_field_map"],
        embedding_type=config["embedding_type"],
    )

    # Load Multi-CPR data
    if not args.no_load:
        total_docs = dataset.load_data(
            os_client=client.os_client,
            index_name=config["index_name"],
            filter_criteria=categories,
            bulk_chunk_size=args.bulk_send_chunk_size
        )
        print(f"Loaded {total_docs} Multi-CPR documents")

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
                        "tag": "Multi-CPR conversation demo",
                        "description": "Multi-CPR conversational search using Bedrock Connector",
                        "model_id": f"{text_gen_model_id}",
                        "context_field_list": [
                            "domain",
                            "pid", 
                            "passage",
                            "chunk_text",
                        ],
                        "system_prompt": "你是一个专门处理中文内容的智能助手。你可以访问来自电商、视频和医疗领域的文档段落。",
                        "user_instructions": "基于提供的上下文信息，用中文为给定问题生成简洁且信息丰富的答案。答案应控制在150字以内，并引用段落中的相关信息。",
                    }
                }
            ]
        },
    )

    # Create conversation memory
    uuid_str = str(uuid.uuid4())[:8]
    conversation_name = f"multi-cpr-conversation-{uuid_str}"
    response = client.os_client.transport.perform_request(
        "POST", "/_plugins/_ml/memory/", body={"name": conversation_name}
    )
    memory_id = response["memory_id"]
    logging.info(f"Multi-CPR Conversation Memory ID: {memory_id}")

    logging.info(
        "Setup complete! Starting interactive Multi-CPR conversational search interface..."
    )
    
    print("\n" + "="*80)
    print("🤖 Multi-CPR Conversational Search Interface")
    print("📚 Domains: E-commerce (电商), Video (视频), Medical (医疗)")
    print("💬 Ask questions in Chinese or English!")
    print("="*80)

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