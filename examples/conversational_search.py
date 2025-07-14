# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import os
import sys
import uuid
from typing import Dict

import cmd_line_params
import print_utils

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from client import (OsMlClientWrapper, get_client, get_client_configs,
                    index_utils)
from configs import (BASE_MAPPING_PATH, PIPELINE_FIELD_MAP,
                     QANDA_FILE_READER_PATH, get_remote_connector_configs)
from data_process import QAndAFileReader
from mapping import get_base_mapping, mapping_update
from ml_models import (AosLlmConnector, MlModel, RemoteMlModel,
                       get_aos_connector_helper, get_ml_model)

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
    client = OsMlClientWrapper(get_client("aos"))
    connector_configs = get_remote_connector_configs(
        host_type="aos", connector_type="bedrock"
    )
    connector_configs[
        "llm_arn"
    ] = "arn:aws:bedrock:*::foundation-model/anthropic.claude-3-5-sonnet-20241022-v2:0"
    connector_configs["connector_role_name"] = "bedrock_llm_connector_role"
    connector_configs[
        "create_connector_role_name"
    ] = "bedrock_llm_create_connector_role"
    logging.info(f"connector_configs:\n{connector_configs}")
    aos_connector_helper = get_aos_connector_helper(get_client_configs("aos"))
    aosLlmConnector = AosLlmConnector(
        os_client=client.os_client,
        connector_configs=connector_configs,
        aos_connector_helper=aos_connector_helper,
    )
    logging.info(f"connector id of the llm connector: {aosLlmConnector.connector_id()}")
    model_group_id = client.ml_model_group.model_group_id()
    llm_model = RemoteMlModel(
        os_client=client.os_client,
        ml_commons_client=client.ml_commons_client,
        ml_connector=aosLlmConnector,
        model_group_id=model_group_id,
        model_name="Amazon Bedrock claude 3.5",
    )
    llm_model_id = llm_model.model_id()
    logging.info(f"Model id of the llm model: {llm_model_id}")
    return llm_model_id


def main():
    args = cmd_line_params.get_command_line_args()

    host_type = "aos"
    model_type = "sagemaker"
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
    pqa_reader = QAndAFileReader(
        directory=QANDA_FILE_READER_PATH,
        max_number_of_docs=args.number_of_docs_per_category
    )

    config = {
        "with_knn": True,
        "pipeline_field_map": PIPELINE_FIELD_MAP,
        "categories": args.categories,
        "index_name": index_name,
        "pipeline_name": ingest_pipeline_name,
        "embedding_type": embedding_type,
        "delete_existing_index": args.delete_existing_index,
        "bulk_send_chunk_size": args.bulk_send_chunk_size,
    }

    model_name = f"{host_type}_{model_type}"
    model_config = get_remote_connector_configs(
        host_type=host_type, connector_type=model_type
    )
    model_config["model_name"] = model_name
    model_config["embedding_type"] = embedding_type
    ml_model = get_ml_model(
        host_type=host_type,
        model_type=model_type,
        model_config=model_config,
        os_client=client.os_client,
        ml_commons_client=client.ml_commons_client,
        model_group_id=client.ml_model_group.model_group_id(),
    )

    config.update(model_config)
    config["index_settings"] = create_index_settings(
        base_mapping_path=BASE_MAPPING_PATH,
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

    index_utils.handle_data_loading(
        os_client=client.os_client,
        pqa_reader=pqa_reader,
        config=config,
        no_load=args.no_load,
    )

    text_gen_model_id = create_text_gen_model()

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
                            "chunk",
                        ],
                        "system_prompt": "You are a helpful assistant",
                        "user_instructions": "Generate a concise and informative answer in less than 100 words for the given question",
                    }
                }
            ]
        },
    )

    uuid_str = str(uuid.uuid4())[:8]
    conversation_name = f"conversation-{uuid_str}"
    response = client.os_client.transport.perform_request(
        "POST", "/_plugins/_ml/memory/", body={"name": conversation_name}
    )
    memory_id = response["memory_id"]
    logging.info(f"Conversation Memory ID: {memory_id}")

    while True:
        question = input("Please input your question (or 'quit' to quit): ")
        if question == "quit":
            break
        search_query = {
            "size": 3,
            "query": {
                "neural_sparse": {
                    "chunk_embedding": {
                        "query_text": question,
                        "model_id": ml_model.model_id(),
                    }
                }
            },
            "ext": {
                "generative_qa_parameters": {
                    "llm_model": "bedrock/claude",
                    "llm_question": question,
                    "llm_response_field": "response",
                    "memory_id": memory_id,
                    "context_size": 10,
                    "message_size": 10,
                    "timeout": 30,
                }
            },
        }
        print_utils.print_query(search_query)

        response = client.os_client.search(
            index=index_name, search_pipeline=search_pipeline_name, body=search_query
        )
        hits = response["hits"]["hits"]
        input("Press enter to see the search results: ")
        for hit_id, hit in enumerate(hits):
            print_utils.print_hit(hit_id, hit)
        print_utils.print_answer(response["ext"]["retrieval_augmented_generation"]["answer"])


if __name__ == "__main__":
    main()
