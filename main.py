# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import argparse

from configs import (
    tasks,
    get_remote_connector_configs,
    DEFAULT_ENV_PATH,
    BASE_MAPPING_PATH,
    QANDA_FILE_READER_PATH,
)
from client import (
    get_client,
    load_dataset,
    OsMlClientWrapper,
)
from ml_models import get_ml_model
from data_process import QAndAFileReader
from mapping import get_base_mapping, mapping_update

logging.basicConfig(
    format="%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)


def create_index_settings(base_mapping_path, index_config, model_config=dict()):
    settings = get_base_mapping(base_mapping_path)
    model_dimension = model_config.get(
        "model_dimensions", index_config["model_dimensions"]
    )
    embedding_type = index_config.get("embedding_type", "dense")
    if index_config["with_knn"]:
        pipeline_name = index_config["pipeline_name"]
        if embedding_type == "dense":
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
        else:
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
    if (
        "compression" in index_config
        and index_config["compression"] != "best_compression"
    ):
        compression_settings = {
            "settings": {"index": {"codec": index_config["compression"]}}
        }
        mapping_update(settings, compression_settings)
    return settings


def get_args():
    parser = argparse.ArgumentParser(
        prog="main",
        description="This toolkit loads AmazonPQA data into an OpenSearch KNN index."
        "Your opensearch endpoint can be either opensource opensearch or "
        "amazon opensearch serivce. You can use local model and remote models "
        "from bedrock or sagemaker.",
    )
    parser.add_argument("-t", "--task", default="knn_768", action="store")
    parser.add_argument("-c", "--categories", default="all", action="store")
    parser.add_argument("-i", "--index_name", default="amazon_pqa", action="store")
    parser.add_argument(
        "-p", "--pipeline_name", default="amazon_pqa_pipeline", action="store"
    )
    parser.add_argument("-d", "--delete_existing", default=False, action="store_true")
    parser.add_argument("-n", "--number_of_docs", default=500, action="store", type=int)
    parser.add_argument(
        "-mt",
        "--model_type",
        choices=["local", "sagemaker", "bedrock"],
        default="local",
        action="store",
    )
    parser.add_argument("-ep", "--env_path", default=DEFAULT_ENV_PATH, action="store")
    parser.add_argument(
        "-ht", "--host_type", choices=["os", "aos"], default="os", action="store"
    )
    parser.add_argument("-et", "--embedding_type", default="dense", action="store")
    parser.add_argument("-cl", "--cleanup", default=False, action="store_true")
    parser.add_argument(
        "-dp",
        "--dataset_path",
        default=QANDA_FILE_READER_PATH,
        action="store",
    )
    parser.add_argument(
        "-bmp",
        "--base_mapping_path",
        default=BASE_MAPPING_PATH,
        action="store",
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    if args.model_type == "local" and args.host_type == "aos":
        raise ValueError(f"local model on aos is not supported")

    client = OsMlClientWrapper(get_client(args.host_type))
    pqa_reader = QAndAFileReader(
        directory=args.dataset_path, max_number_of_docs=args.number_of_docs
    )

    if args.task not in tasks.keys():
        raise ValueError(
            f'Config "{args.task}" not found.\nValid values {tasks.keys()}'
        )
    config = tasks[args.task]

    # Overwrite the test case categories with cmd line args
    categories = config["categories"]
    if args.categories.lower() == "all":
        categories = [name for name in pqa_reader.amazon_pqa_category_names()]
    else:
        categories = args.categories.split(",")
        if type(categories) is str:
            categories = [categories]

    config["categories"] = categories
    config["index_name"] = args.index_name
    config["pipeline_name"] = args.pipeline_name
    config["embedding_type"] = args.embedding_type

    ml_model = None
    model_config = None
    model_name = config["model_name"]

    if args.model_type != "local":
        model_name = f"{args.host_type}_{args.model_type}"

    model_config = (
        {
            "model_version": config["model_version"],
        }
        if args.model_type == "local"
        else get_remote_connector_configs(
            host_type=args.host_type, connector_type=args.model_type
        )
    )
    model_config["model_name"] = model_name
    model_config["embedding_type"] = args.embedding_type

    ml_model = get_ml_model(
        host_type=args.host_type,
        model_type=args.model_type,
        model_config=model_config,
        os_client=client.os_client,
        ml_commons_client=client.ml_commons_client,
        model_group_id=client.ml_model_group.model_group_id(),
    )

    config["index_settings"] = create_index_settings(
        base_mapping_path=args.base_mapping_path,
        index_config=config,
        model_config=model_config,
    )
    config["cleanup"] = config["cleanup"] or args.cleanup

    logging.info(f"Config:\n {json.dumps(config, indent=4)}")

    load_dataset(
        client,
        ml_model,
        pqa_reader,
        config,
        args.delete_existing,
        args.index_name,
        pipeline_name=args.pipeline_name,
    )


if __name__ == "__main__":
    main()
