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
    get_client,
    load_category,
    OsMlClientWrapper,
)
from data_process import QAndAFileReader
from mapping import get_base_mapping

logging.basicConfig(
    format="%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)

# ANSI escape sequence constants with improved colors
BOLD = "\033[1m"
RESET = "\033[0m"

# Headers
LIGHT_RED_HEADER = "\033[1;31m"
LIGHT_GREEN_HEADER = "\033[1;32m"
LIGHT_BLUE_HEADER = "\033[1;34m"
LIGHT_YELLOW_HEADER = "\033[1;33m"
LIGHT_PURPLE_HEADER = "\033[1;35m"


def load_dataset(
    client: OsMlClientWrapper,
    pqa_reader: QAndAFileReader,
    config: Dict[str, str],
    index_name: str,
):
    if client.os_client.indices.exists(index_name):
        logging.info(f"Index {index_name} already exists. Skipping loading dataset")
        return

    logging.info(f"Creating index {index_name}")
    client.idempotent_create_index(
        index_name=config["index_name"], settings=config["index_settings"]
    )

    for category in config["categories"]:
        load_category(
            client=client.os_client,
            pqa_reader=pqa_reader,
            category=category,
            config=config,
        )


def main():
    host_type = "aos"
    index_name = "lexical_search"

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
    dataset_path = QANDA_FILE_READER_PATH
    number_of_docs_per_category = 5000

    client = OsMlClientWrapper(get_client(host_type))
    pqa_reader = QAndAFileReader(
        directory=dataset_path, max_number_of_docs=number_of_docs_per_category
    )

    config = {
        "categories": categories,
        "index_name": index_name,
        "index_settings": get_base_mapping(BASE_MAPPING_PATH),
    }

    load_dataset(
        client,
        pqa_reader,
        config,
        index_name=index_name,
    )

    while True:
        query_text = input("Please input your search query text (or 'quit' to quit): ")
        if query_text == "quit":
            break
        search_query = {
            "size": 3,
            "query": {"match": {"chunk": query_text}},
        }
        print(f"{LIGHT_GREEN_HEADER}Search query:{RESET}")
        print(json.dumps(search_query, indent=4))
        search_results = client.os_client.search(index=index_name, body=search_query)
        hits = search_results["hits"]["hits"]
        input("Press enter to see the search results: ")
        for hit_id, hit in enumerate(hits):
            print(
                "--------------------------------------------------------------------------------"
            )
            print()
            print(
                f'{LIGHT_PURPLE_HEADER}Item {hit_id + 1} category:{RESET} {hit["_source"]["category_name"]}'
            )
            print(
                f'{LIGHT_YELLOW_HEADER}Item {hit_id + 1} product name:{RESET} {hit["_source"]["item_name"]}'
            )
            print()
            if hit["_source"]["product_description"]:
                print(f"{LIGHT_BLUE_HEADER}Production description:{RESET}")
                print(hit["_source"]["product_description"])
                print()
            print(
                f'{LIGHT_RED_HEADER}Question:{RESET} {hit["_source"]["question_text"]}'
            )
            for answer_id, answer in enumerate(hit["_source"]["answers"]):
                print(
                    f'{LIGHT_GREEN_HEADER}Answer {answer_id + 1}:{RESET} {answer["answer_text"]}'
                )
            print()
        print(
            "--------------------------------------------------------------------------------"
        )


if __name__ == "__main__":
    main()
