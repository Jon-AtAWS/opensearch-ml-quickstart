# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import sys

import cmd_line_params
import print_utils

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from client import OsMlClientWrapper, get_client, index_utils
from configs import (BASE_MAPPING_PATH, PIPELINE_FIELD_MAP,
                     QANDA_FILE_READER_PATH, get_remote_connector_configs)
from data_process import QAndAFileReader
from mapping import get_base_mapping, mapping_update
from ml_models import get_ml_model

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


def main():
    args = cmd_line_params.get_command_line_args()

    if args.opensearch_type != "aos":
        logging.error(
            "This example is designed for Amazon OpenSearch Service (AOS) only."
        )
        sys.exit(1)

    # This example uses a sparse model, hosted on Amazon SageMaker and an Amazon
    # OpenSearch Service domain.
    host_type = "aos"
    model_type = "sagemaker"
    index_name = "sparse_search"
    embedding_type = "sparse"
    pipeline_name = "sparse-ingest-pipeline"

    client = OsMlClientWrapper(get_client(host_type))
    pqa_reader = QAndAFileReader(
        directory=QANDA_FILE_READER_PATH,
        max_number_of_docs=args.number_of_docs_per_category
    )

    config = {
        "with_knn": True,
        "index_name": index_name,
        "pipeline_name": pipeline_name,
        "pipeline_field_map": PIPELINE_FIELD_MAP,
        "embedding_type": embedding_type,
        "categories": args.categories,
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
        pipeline_name=pipeline_name,
        pipeline_field_map=config["pipeline_field_map"],
        embedding_type=config["embedding_type"],
    )

    index_utils.handle_data_loading(
        os_client=client.os_client,
        pqa_reader=pqa_reader,
        config=config,
        no_load=args.no_load,
    )

    while True:
        query_text = input("Please input your search query text (or 'quit' to quit): ")
        if query_text == "quit":
            break
        search_query = {
            "size": 3,
            "query": {
                "neural_sparse": {
                    "chunk_embedding": {
                        "query_text": query_text,
                        "model_id": ml_model.model_id(),
                    }
                }
            },
        }
        print_utils.print_query(search_query)
        
        search_results = client.os_client.search(index=index_name, body=search_query)
        hits = search_results["hits"]["hits"]
        input("Press enter to see the search results: ")
        for hit_id, hit in enumerate(hits):
            print_utils.print_hit(hit_id, hit)


if __name__ == "__main__":
    main()
