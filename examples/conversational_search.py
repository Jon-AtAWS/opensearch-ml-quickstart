# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import logging
from typing import Dict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs import (
    get_remote_connector_configs,
    BASE_MAPPING_PATH,
    PIPELINE_FIELD_MAP,
    QANDA_FILE_READER_PATH,
)
from client import get_client, get_client_configs, load_category, OsMlClientWrapper
from data_process import QAndAFileReader
from mapping import get_base_mapping, mapping_update
from ml_models import (
    get_ml_model,
    get_aos_connector_helper,
    MlModel,
    RemoteMlModel,
    AosLlmConnector,
)

logging.basicConfig(
    format="%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)


def create_index_settings(base_mapping_path, index_config):
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
                    "method": {
                        "name": "hnsw",
                        "space_type": "l2",
                        "engine": "nmslib",
                        "parameters": {"ef_construction": 128, "m": 24},
                    },
                },
            }
        },
    }
    mapping_update(settings, knn_settings)
    return settings


def load_dataset(
    client: OsMlClientWrapper,
    ml_model: MlModel,
    pqa_reader: QAndAFileReader,
    config: Dict[str, str],
    index_name: str,
    pipeline_name: str,
):
    if client.os_client.indices.exists(index_name):
        logging.info(f"Index {index_name} already exists. Skipping loading dataset")
        return

    logging.info(f"Creating index {index_name}")
    client.idempotent_create_index(
        index_name=config["index_name"], settings=config["index_settings"]
    )

    logging.info("Setting up for KNN")
    client.setup_for_kNN(
        ml_model=ml_model,
        index_name=config["index_name"],
        pipeline_name=pipeline_name,
        index_settings=config["index_settings"],
        pipeline_field_map=config["pipeline_field_map"],
        embedding_type=config["embedding_type"],
    )

    for category in config["categories"]:
        load_category(
            client=client.os_client,
            pqa_reader=pqa_reader,
            category=category,
            config=config,
        )


def create_llm_model():
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
    host_type = "aos"
    model_type = "sagemaker"
    embedding_type = "dense"
    index_name = "conversational_search"
    ingest_pipeline_name = "sparse-ingest-pipeline"
    search_pipeline_name = "conversational-search-pipeline"

    categories = [
        "earbud headphones",
        "headsets",
        "diffusers",
        "mattresses",
        "mp3 and mp4 players",
        "sheet and pillowcase sets",
        "batteries",
        "casual",
        "costumes",
    ]
    number_of_docs_per_category = 5000
    dataset_path = QANDA_FILE_READER_PATH

    client = OsMlClientWrapper(get_client(host_type))
    pqa_reader = QAndAFileReader(
        directory=dataset_path, max_number_of_docs=number_of_docs_per_category
    )

    config = {
        "with_knn": True,
        "pipeline_field_map": PIPELINE_FIELD_MAP,
        "categories": categories,
        "index_name": index_name,
        "pipeline_name": ingest_pipeline_name,
        "embedding_type": embedding_type,
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

    load_dataset(
        client,
        ml_model,
        pqa_reader,
        config,
        index_name=index_name,
        pipeline_name=ingest_pipeline_name,
    )

    llm_model_id = create_llm_model()

    response = client.os_client.transport.perform_request(
        "PUT",
        f"/_search/pipeline/{search_pipeline_name}",
        body={
            "response_processors": [
                {
                    "retrieval_augmented_generation": {
                        "tag": "conversation demo",
                        "description": "Demo pipeline Using Bedrock Connector",
                        "model_id": f"{llm_model_id}",
                        "context_field_list": ["chunk"],
                        "system_prompt": "You are a helpful assistant",
                        "user_instructions": "Generate a concise and informative answer in less than 100 words for the given question",
                    }
                }
            ]
        },
    )

    conversation_name = f"conversation-{categories[0]}"
    response = client.os_client.transport.perform_request(
        "POST", "/_plugins/_ml/memory/", body={"name": conversation_name}
    )
    memory_id = response["memory_id"]
    logging.info(f"Conversation Memory ID: {memory_id}")

    while True:
        question = input("Please input your question: ")
        response = client.os_client.search(
            index=index_name,
            search_pipeline=search_pipeline_name,
            body={
                "size": 3,
                "query": {
                    "neural": {
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
                        "context_size": 3,
                        "message_size": 5,
                        "timeout": 30,
                    }
                },
            },
        )
        hits = response["hits"]["hits"]
        for hit_id, hit in enumerate(hits):
            print(
                "--------------------------------------------------------------------------------"
            )
            print(
                f'Item {hit_id + 1} name: {hit["_source"]["item_name"]} ({hit["_source"]["category_name"]})'
            )
            print()
            if hit["_source"]["product_description"]:
                print(
                    f'Production description: {hit["_source"]["product_description"]}'
                )
            print()
            print(f'Question: {hit["_source"]["question_text"]}')
            for answer_id, answer in enumerate(hit["_source"]["answers"]):
                print(f'Answer {answer_id + 1}: {answer["answer_text"]}')
            print()
        print(
            "--------------------------------------------------------------------------------"
        )
        print(
            "LLM Answer:", response["ext"]["retrieval_augmented_generation"]["answer"]
        )


if __name__ == "__main__":
    main()
