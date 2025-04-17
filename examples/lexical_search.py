# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import json
import logging
from typing import Dict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs import BASE_MAPPING_PATH, QANDA_FILE_READER_PATH
from client import (
    OsMlClientWrapper,
    get_client,
)
from data_process import QAndAFileReader
from mapping import get_base_mapping
from main import load_category

logging.basicConfig(
    format="%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)


def load_dataset(
    client: OsMlClientWrapper,
    pqa_reader: QAndAFileReader,
    config: Dict[str, str],
    delete_existing: bool,
    index_name: str,
):
    if delete_existing:
        logging.info(f"Deleting existing index {index_name}")
        client.delete_then_create_index(
            index_name=config["index_name"], settings=config["index_settings"]
        )

        for category in config["categories"]:
            load_category(
                client=client.os_client,
                pqa_reader=pqa_reader,
                category=category,
                config=config,
            )
    else:
        logging.info("Skipping index setup")


def main():
    host_type = "aos"
    index_name = "lexical_search"
    dataset_path = QANDA_FILE_READER_PATH
    number_of_docs = 5000
    client = OsMlClientWrapper(get_client(host_type))

    pqa_reader = QAndAFileReader(
        directory=dataset_path, max_number_of_docs=number_of_docs
    )

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

    config = {
        "categories": categories,
        "index_name": index_name,
        "index_settings": get_base_mapping(BASE_MAPPING_PATH),
    }

    # logging.info(f"Config:\n {json.dumps(config, indent=4)}")

    load_dataset(
        client,
        pqa_reader,
        config,
        delete_existing=False,
        index_name=index_name,
    )

    query_text = input("Please input your search query text: ")
    search_query = {
        "query": {"match": {"chunk": query_text}},
    }
    search_results = client.os_client.search(index=index_name, body=search_query)
    hits = search_results["hits"]["hits"]
    for hit in hits:
        print('--------------------------------------------------------------------------------')
        print(f'Category name: {hit["_source"]["category_name"]}')
        print()
        print(f'Item name: {hit["_source"]["item_name"]}')
        print()
        print(f'Production description: {hit["_source"]["product_description"]}')
        print()


if __name__ == "__main__":
    main()