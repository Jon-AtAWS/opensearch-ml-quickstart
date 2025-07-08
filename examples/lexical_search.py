# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import json
import logging
from typing import Dict


import cmd_line_params
import print_utils


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs import BASE_MAPPING_PATH, QANDA_FILE_READER_PATH
from client import (
    get_client,
    OsMlClientWrapper,
    load_category,
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
    client.handle_index_creation(
        index_name=index_name,
        index_settings=config["index_settings"],
        delete_existing=config["delete_existing_index"],
    )

    for category in config["categories"]:
        load_category(
            client=client.os_client,
            pqa_reader=pqa_reader,
            category=category,
            config=config,
        )


def main():
    args = cmd_line_params.get_command_line_args()
    host_type = "aos"
    index_name = "lexical_search"

    client = OsMlClientWrapper(get_client(host_type))
    pqa_reader = QAndAFileReader(
        directory=QANDA_FILE_READER_PATH,
        max_number_of_docs=args.number_of_docs_per_category,
    )

    config = {
        "categories": args.categories,
        "index_name": index_name,
        "index_settings": get_base_mapping(BASE_MAPPING_PATH),
        "delete_existing_index": args.delete_existing_index,
        "bulk_send_chunk_size": args.bulk_send_chunk_size,
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
            print_utils.print_hit(hit_id, hit)

if __name__ == "__main__":
    main()
