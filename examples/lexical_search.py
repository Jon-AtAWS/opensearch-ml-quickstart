# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

"""
Lexical Search Example

This module demonstrates lexical (keyword-based) search using OpenSearch.
It loads data from the Amazon PQA dataset and provides an interactive
search interface using traditional text matching.

See cmd_line_params.py for command line parameters.

Consider:
 - c, --categories: Comma-separated list of categories to load from the dataset.

This example is designed to be run in a terminal environment.
"""

import logging
import os
import sys

import cmd_line_params
import print_utils

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from client import OsMlClientWrapper, get_client, index_utils
from configs import BASE_MAPPING_PATH, QANDA_FILE_READER_PATH
from data_process import QAndAFileReader
from mapping import get_base_mapping

logging.basicConfig(
    format="%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)


def main():
    """
    Main function to run lexical search example.
    
    This function:
    1. Parses command line arguments
    2. Initializes OpenSearch client and data reader
    3. Loads dataset into OpenSearch index
    4. Provides interactive search interface
    """
    args = cmd_line_params.get_command_line_args()

    # The index name for lexical search. Could be a command line argument. For
    # simplicity, use a fixed name here.
    index_name = "lexical_search"

    # Initialize OpenSearch client and data reader
    client = OsMlClientWrapper(get_client(args.opensearch_type))
    pqa_reader = QAndAFileReader(
        directory=QANDA_FILE_READER_PATH,
        max_number_of_docs=args.number_of_docs_per_category,
    )

    # Set the configuration
    config = {
        "categories": args.categories,
        "index_name": index_name,
        "index_settings": get_base_mapping(BASE_MAPPING_PATH),
        "delete_existing_index": args.delete_existing_index,
        "bulk_send_chunk_size": args.bulk_send_chunk_size,
    }

    # Handle index creation ensures the index exists and creates and applies the
    # mapping.
    index_utils.handle_index_creation(
        os_client=client.os_client,
        config=config,
        delete_existing=config["delete_existing_index"],
    )

    # Load data into the index
    index_utils.handle_data_loading(
        os_client=client.os_client,
        pqa_reader=pqa_reader,
        config=config,
        no_load=args.no_load,
    )

    # Interactive search loop
    while True:
        query_text = input("Please input your search query text (or 'quit' to quit): ")
        if query_text == "quit":
            break
        # Build lexical search query
        search_query = {
            "size": 3,
            "query": {"match": {"chunk": query_text}},
        }
        print_utils.print_query(search_query)

        # Execute search and display results
        search_results = client.os_client.search(index=index_name, body=search_query)
        hits = search_results["hits"]["hits"]
        input("Press enter to see the search results: ")
        for hit_id, hit in enumerate(hits):
            print_utils.print_hit(hit_id, hit)

if __name__ == "__main__":
    main()
