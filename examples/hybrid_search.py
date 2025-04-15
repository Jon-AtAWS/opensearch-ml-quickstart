# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import json
import logging
from typing import Dict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs import BASE_MAPPING_PATH, PIPELINE_FIELD_MAP, QANDA_FILE_READER_PATH
from client import (
    OsMlClientWrapper,
    get_client,
)
from data_process import QAndAFileReader
from mapping import get_base_mapping, mapping_update
from ml_models import get_remote_connector_configs, MlModel
from main import get_ml_model, load_category

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
                "chunk_sparse_embedding": {
                    "type": "rank_features",
                },
                "chunk_dense_embedding": {
                    "type": "knn_vector",
                    "dimension": model_dimension,
                },
            }
        },
    }
    mapping_update(settings, knn_settings)
    return settings


def load_dataset(
    client: OsMlClientWrapper,
    dense_ml_model: MlModel,
    sparse_ml_model: MlModel,
    pqa_reader: QAndAFileReader,
    config: Dict[str, str],
    delete_existing: bool,
    index_name: str,
    pipeline_name: str,
):
    if delete_existing:
        logging.info(f"Deleting existing index {index_name}")
        client.delete_then_create_index(
            index_name=config["index_name"], settings=config["index_settings"]
        )

    logging.info("Adding pipeline...")

    pipeline_config = {
        "description": "Pipeline for processing chunks",
        "processors": [
            {
                "sparse_encoding": {
                    "model_id": sparse_ml_model.model_id(),
                    "field_map": {"chunk": "chunk_sparse_embedding"},
                }
            },
            {
                "text_embedding": {
                    "model_id": dense_ml_model.model_id(),
                    "field_map": {"chunk": "chunk_dense_embedding"},
                }
            },
        ],
    }
    client.os_client.ingest.put_pipeline(pipeline_name, body=pipeline_config)

    for category in config["categories"]:
        load_category(
            client=client.os_client,
            pqa_reader=pqa_reader,
            category=category,
            config=config,
        )


def main():
    host_type = "aos"
    index_name = "hybrid_search"
    dense_model_type = "sagemaker"
    sparse_model_type = "sagemaker"
    dataset_path = QANDA_FILE_READER_PATH
    number_of_docs = 500
    client = OsMlClientWrapper(get_client(host_type))

    pqa_reader = QAndAFileReader(
        directory=dataset_path, max_number_of_docs=number_of_docs
    )

    categories = ["sheet and pillowcase sets"]
    config = {"with_knn": True, "pipeline_field_map": PIPELINE_FIELD_MAP}

    pipeline_name = "amazon_pqa_pipeline"
    config["categories"] = categories
    config["index_name"] = index_name
    config["pipeline_name"] = pipeline_name

    dense_model_name = f"{host_type}_{dense_model_type}"
    sparse_model_name = f"{host_type}_{sparse_model_type}"

    dense_model_config = get_remote_connector_configs(
        host_type=host_type, connector_type=dense_model_type
    )
    sparse_model_config = get_remote_connector_configs(
        host_type=host_type, connector_type=dense_model_type
    )
    dense_model_config["model_name"] = dense_model_name
    sparse_model_config["model_name"] = sparse_model_name

    dense_model_config["embedding_type"] = "dense"
    sparse_model_config["embedding_type"] = "sparse"
    config["model_dimensions"] = dense_model_config["model_dimensions"]

    dense_ml_model = get_ml_model(
        host_type=host_type,
        model_type=dense_model_type,
        model_config=dense_model_config,
        client=client,
    )

    sparse_ml_model = get_ml_model(
        host_type=host_type,
        model_type=sparse_model_type,
        model_config=sparse_model_config,
        client=client,
    )

    print("index_config:\n", config)
    config["index_settings"] = create_index_settings(
        base_mapping_path=BASE_MAPPING_PATH,
        index_config=config,
    )
    config["cleanup"] = False

    logging.info(f"Config:\n {json.dumps(config, indent=4)}")

    load_dataset(
        client,
        dense_ml_model,
        sparse_ml_model,
        pqa_reader,
        config,
        delete_existing=True,
        index_name=index_name,
        pipeline_name=pipeline_name,
    )

    search_pipeline_id = "hybrid-search-pipeline"
    logging.info(f"Creating search pipeline {search_pipeline_id}")
    pipeline_config = {
        "description": "Post processor for hybrid search",
        "phase_results_processors": [
            {
                "normalization-processor": {
                    "normalization": {"technique": "min_max"},
                    "combination": {
                        "technique": "arithmetic_mean",
                        "parameters": {"weights": [0.5, 0.5]},
                    },
                }
            }
        ],
    }
    client.os_client.transport.perform_request(
        "PUT", f"/_search/pipeline/{search_pipeline_id}", body=pipeline_config
    )

    query_text = input("Please input your search query text: ")
    search_query = {
        "_source": {"include": "chunk"},
        "query": {
            "hybrid": {
                "queries": [
                    {
                        "neural": {
                            "chunk_dense_embedding": {
                                "query_text": query_text,
                                "model_id": dense_ml_model.model_id(),
                            }
                        }
                    },
                    {
                        "neural_sparse": {
                            "chunk_sparse_embedding": {
                                "query_text": query_text,
                                "model_id": sparse_ml_model.model_id(),
                            }
                        }
                    },
                ]
            }
        },
    }

    search_results = client.os_client.search(
        index=index_name, body=search_query, search_pipeline=search_pipeline_id
    )
    hits = search_results["hits"]["hits"]
    hits = [hit["_source"]["chunk"] for hit in hits]
    hits = list(set(hits))
    for i, hit in enumerate(hits):
        print(f"{i + 1}th search result:\n {hit}")


if __name__ == "__main__":
    main()
