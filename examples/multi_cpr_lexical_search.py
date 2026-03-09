# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

"""Lexical (keyword) search over the Multi-CPR Chinese passage retrieval dataset.

No ML model is needed — this uses OpenSearch's built-in text matching.

Usage:
    # Index 5000 ecom passages and search interactively
    python examples/multi_cpr_lexical_search.py -o aos -c ecom -n 5000

    # Single query
    python examples/multi_cpr_lexical_search.py -o aos -c ecom -n 5000 -q "尼康相机"

    # Skip loading if already indexed
    python examples/multi_cpr_lexical_search.py -o aos -c ecom --no-load
"""

import logging
import os
import sys

import cmd_line_interface

from client import OsMlClientWrapper, get_client
from data_process.multi_cpr_dataset import MultiCPRDataset

logging.basicConfig(
    format="%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)

MULTI_CPR_PATH = os.environ.get("MULTI_CPR_PATH", "~/datasets/multi_cpr")
INDEX_NAME = "multi_cpr_lexical"


def build_lexical_query(query_text, **kwargs):
    """Keyword-based search against the passage field."""
    return {
        "size": 5,
        "query": {
            "match": {
                "passage": query_text,
            }
        },
    }


def print_multi_cpr_results(search_results, **kwargs):
    """Custom result printer for Multi-CPR passages."""
    hits = search_results.get("hits", {}).get("hits", [])
    total = search_results.get("hits", {}).get("total", {}).get("value", 0)
    print(f"\nFound {total} total matches, showing top {len(hits)} results:\n")
    for i, hit in enumerate(hits):
        src = hit.get("_source", {})
        score = hit.get("_score", 0)
        print("-" * 80)
        print(f"  Result {i + 1}  |  Score: {score:.4f}  |  Domain: {src.get('domain', 'N/A')}  |  PID: {src.get('pid', 'N/A')}")
        print(f"  Passage: {src.get('passage', 'N/A')[:200]}")
        print()
    print("-" * 80)


def main():
    args = cmd_line_interface.get_command_line_args()

    logging.info(f"Multi-CPR lexical search | host={args.opensearch_type}")

    client = OsMlClientWrapper(get_client(args.opensearch_type))
    dataset = MultiCPRDataset(
        directory=MULTI_CPR_PATH,
        max_number_of_docs=args.number_of_docs_per_category,
    )

    # Validate domains
    valid_domains = dataset.get_available_filters()
    if args.categories is None or not all(c in valid_domains for c in args.categories):
        categories = valid_domains
        logging.info(f"Using all Multi-CPR domains: {categories}")
    else:
        categories = args.categories

    # Lexical search uses a plain mapping — no pipeline, no kNN
    dataset.create_index(
        os_client=client.os_client,
        index_name=INDEX_NAME,
        delete_existing=args.delete_existing_index,
    )

    if not args.no_load:
        total = dataset.load_data(
            os_client=client.os_client,
            index_name=INDEX_NAME,
            filter_criteria=categories,
            bulk_chunk_size=args.bulk_send_chunk_size,
        )
        logging.info(f"Loaded {total} passages into '{INDEX_NAME}'")

    cmd_line_interface.interactive_search_loop(
        client=client,
        index_name=INDEX_NAME,
        model_info="Lexical Search (no model)",
        query_builder_func=build_lexical_query,
        result_processor_func=print_multi_cpr_results,
        question=args.question,
    )


if __name__ == "__main__":
    main()
