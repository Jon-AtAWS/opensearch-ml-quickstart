# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

"""
Multi-CPR Conversational Agent Example

Demonstrates OpenSearch's conversational agent capabilities using the Multi-CPR
Chinese passage retrieval dataset. Combines semantic search with LLM to provide
contextual, conversational responses across e-commerce, video, and medical domains.

Usage:
    # Interactive agent on e-commerce domain
    python examples/multi_cpr_conversational_agent.py -o os -c ecom -n 500

    # All domains
    python examples/multi_cpr_conversational_agent.py -o os -c ecom video medical -n 1000

    # Single query
    python examples/multi_cpr_conversational_agent.py -o os -c medical -n 200 -q "感冒吃什么药"
"""

import json
import logging
import os
import sys
import uuid

import cmd_line_interface
import multi_cpr_agent_tools
from agent_tools import list_index_tool, index_mapping_tool

from client import OsMlClientWrapper, get_client, index_utils
from configs.configuration_manager import (
    get_local_dense_embedding_model_name,
    get_local_dense_embedding_model_version,
    get_local_dense_embedding_model_format,
    get_local_dense_embedding_model_dimension,
    get_multi_cpr_path,
    config_override,
)
from connectors.helper import get_remote_connector_configs
from data_process.multi_cpr_dataset import MultiCPRDataset
from mapping import mapping_update
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

INDEX_NAME = "multi_cpr_agent_knowledge_base"
INGEST_PIPELINE_NAME = "multi-cpr-agent-dense-ingest-pipeline"


def create_index_settings(dataset, index_config):
    """Create index settings with Multi-CPR mapping and dense vector configuration."""
    base_mapping = dataset.get_index_mapping()
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
    settings = {"mappings": base_mapping}
    mapping_update(settings, knn_settings)
    return settings


def create_llm_model(client: OsMlClientWrapper):
    """Create and deploy LLM model for conversational agent."""
    connector_configs = get_remote_connector_configs("bedrock", "os")
    logging.info(f"LLM connector configs:\n{connector_configs}")

    llm_connector = LlmConnector(
        os_client=client.os_client,
        os_type="os",
        connector_configs=connector_configs,
        llm_type="converse",
        model_name="us.anthropic.claude-sonnet-4-20250514-v1:0",  # Claude Sonnet 4
    )

    logging.info(f"LLM connector ID: {llm_connector.connector_id()}")

    model_group_id = client.ml_model_group.model_group_id()
    llm_model = RemoteMlModel(
        os_client=client.os_client,
        ml_commons_client=client.ml_commons_client,
        ml_connector=llm_connector,
        model_group_id=model_group_id,
        model_name="Amazon Bedrock Claude for Multi-CPR Agent",
    )

    llm_model_id = llm_model.model_id()
    logging.info(f"LLM model ID: {llm_model_id}")
    return llm_model_id


def create_conversational_agent(client, index_name, embedding_model_id, llm_model_id):
    """Create a conversational agent with Multi-CPR specific tools and Chinese system prompt."""
    agent_config = {
        "name": "Multi-CPR RAG Agent",
        "type": "conversational",
        "description": "Multi-CPR中文段落检索对话代理",
        "app_type": "rag",
        "llm": {
            "model_id": llm_model_id,
            "parameters": {
                "max_iteration": 20,
                "system_prompt":
                    "你是一个专门处理中文内容的智能助手，能够回答关于电商产品、视频内容和医疗健康的问题。"
                    "  "
                    "你的知识库包含来自Multi-CPR数据集的中文段落，涵盖三个领域："
                    "电商(ecom)：产品描述、商品信息；"
                    "视频(video)：视频内容、娱乐信息；"
                    "医疗(medical)：健康知识、医疗信息。"
                    "  "
                    "你有语义搜索工具和关键词搜索工具来查找相关段落。"
                    "语义搜索适合理解查询含义的场景，关键词搜索适合包含特定名称或术语的查询。"
                    "  "
                    "首先判断用户的问题属于哪个领域，然后选择合适的搜索工具。"
                    "如果初次搜索结果不够理想，尝试用不同的关键词重新搜索。"
                    "  "
                    "回答时请用中文，并引用搜索结果中的相关信息。"
                    "如果搜索结果与问题不相关，请说明原因并建议用户换一种方式提问。",
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
            multi_cpr_agent_tools.get_passages_tool_semantic(index_name, embedding_model_id),
            multi_cpr_agent_tools.get_passages_tool_lexical(index_name),
            multi_cpr_agent_tools.get_domain_filter_tool(index_name),
            multi_cpr_agent_tools.list_index_tool(),
            multi_cpr_agent_tools.index_mapping_tool(),
        ]
    }

    logging.info(f"Creating Multi-CPR conversational agent with config: {json.dumps(agent_config, indent=2, ensure_ascii=False)}")

    try:
        response = client.os_client.transport.perform_request(
            "POST", "/_plugins/_ml/agents/_register", body=agent_config
        )
    except Exception as e:
        logging.error(f"Failed to create conversational agent: {e}")
        raise

    agent_id = response["agent_id"]
    logging.info(f"Created Multi-CPR conversational agent with ID: {agent_id}")
    return agent_id


def execute_agent_query(client, agent_id, query_body):
    """Execute a query against the conversational agent."""
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
    """Build conversational agent query."""
    if not agent_id:
        raise ValueError("Agent ID must be provided for conversational agent.")

    query = {
        "parameters": {
            "question": query_text,
            "verbose": True
        }
    }

    if "memory_id" in kwargs:
        query["parameters"]["memory_id"] = kwargs["memory_id"]

    return query


def main():
    args = cmd_line_interface.get_command_line_args()

    if args.opensearch_type != "os":
        logging.error(
            "This example is designed for self-managed OpenSearch (OS) only to support local models."
        )
        sys.exit(1)

    host_type = "os"
    model_host = "local"
    embedding_type = "dense"

    # Initialize Multi-CPR dataset
    dataset = MultiCPRDataset(
        directory=get_multi_cpr_path(),
        max_number_of_docs=args.number_of_docs_per_category,
    )

    # Validate Multi-CPR domains
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

    # Conversational agent requires OpenSearch 3.1+
    with config_override(MINIMUM_OPENSEARCH_VERSION="3.3.0"):
        client = OsMlClientWrapper(get_client(host_type))

    config = {
        "with_knn": True,
        "pipeline_field_map": {"chunk_text": "chunk_embedding"},
        "categories": categories,
        "index_name": INDEX_NAME,
        "pipeline_name": INGEST_PIPELINE_NAME,
        "embedding_type": embedding_type,
        "delete_existing_index": args.delete_existing_index,
        "bulk_send_chunk_size": args.bulk_send_chunk_size,
    }

    # Create local dense embedding model
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

    # Set up knowledge base
    index_utils.handle_index_creation(
        os_client=client.os_client,
        config=config,
        delete_existing=config["delete_existing_index"],
    )

    logging.info("Setting up Multi-CPR knowledge base with embeddings...")
    client.setup_for_kNN(
        ml_model=embedding_ml_model,
        index_name=config["index_name"],
        pipeline_name=INGEST_PIPELINE_NAME,
        pipeline_field_map=config["pipeline_field_map"],
        embedding_type=config["embedding_type"],
    )

    # Load Multi-CPR data
    if not args.no_load:
        total_docs = dataset.load_data(
            os_client=client.os_client,
            index_name=config["index_name"],
            filter_criteria=categories,
            bulk_chunk_size=args.bulk_send_chunk_size,
        )
        print(f"Loaded {total_docs} Multi-CPR passages")

    # Create LLM model for agent
    logging.info("Creating LLM model for Multi-CPR conversational agent...")
    llm_model_id = create_llm_model(client)

    # Create conversational agent
    logging.info("Creating Multi-CPR conversational agent...")
    agent_id = create_conversational_agent(
        client,
        index_name=INDEX_NAME,
        embedding_model_id=embedding_ml_model.model_id(),
        llm_model_id=llm_model_id,
    )

    logging.info(f"Multi-CPR conversational agent created with ID: {agent_id}")
    logging.info("Setup complete")

    # Single query mode
    if hasattr(args, 'question') and args.question:
        logging.info(f"Executing single query: {args.question}")
        try:
            agent_query = build_agent_query(args.question, agent_id)
            results = execute_agent_query(client, agent_id, agent_query)
            cmd_line_interface.process_and_print_agent_response(results)
        except Exception as e:
            logging.error(f"Error executing query: {e}")
        return

    logging.info("Starting interactive Multi-CPR agent interface...")

    # Create conversation memory
    uuid_str = str(uuid.uuid4())[:8]
    conversation_name = f"multi-cpr-conversation-{uuid_str}"
    response = client.os_client.transport.perform_request(
        "POST", "/_plugins/_ml/memory/", body={"name": conversation_name}
    )
    memory_id = response["memory_id"]
    logging.info(f"Multi-CPR Conversation Memory ID: {memory_id}")

    print("\n" + "="*80)
    print("🤖 Multi-CPR 对话式智能代理")
    print("📚 领域: 电商(ecom) | 视频(video) | 医疗(medical)")
    print("💬 请用中文或英文提问!")
    print("="*80)

    # Start interactive agent loop
    cmd_line_interface.interactive_agent_loop(
        client=client,
        agent_id=agent_id,
        model_info=f"Agent: {agent_id}, LLM: {llm_model_id}, Embedding: {embedding_ml_model.model_id()}",
        build_agent_query_func=build_agent_query,
        agent_executor_func=execute_agent_query,
        memory_id=memory_id,
    )


if __name__ == "__main__":
    main()
