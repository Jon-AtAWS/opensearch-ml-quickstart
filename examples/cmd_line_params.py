"""
Command Line Parameters Module

This module handles command line argument parsing for OpenSearch ML quickstart
examples. Provides configuration options for data loading, indexing, and search
operations.
"""

import argparse
import os
import sys


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_process import QAndAFileReader


# Default categories from Amazon PQA dataset for testing
DEFAULT_CATEGORIES = [
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


def get_command_line_args():
    """
    Parse command line arguments for OpenSearch ML quickstart examples.
    
    Returns:
        argparse.Namespace: Parsed command line arguments with the following
        attributes:
            - force_index_creation (bool): Force index creation
            - delete_existing_index (bool): Delete existing index before
              creating new one
            - categories (list): List of Amazon PQA categories to process
            - bulk_send_chunk_size (int): Batch size for bulk document
              operations
            - number_of_docs_per_category (int): Maximum documents per category
            - opensearch_type (str): Connect to a local (os) or remote (aos)
              OpenSearch instance
    """
    parser = argparse.ArgumentParser(
        description="Run AI-powered search examples with OpenSearch ML Quickstart"
    )
    parser.add_argument(
        "-d", "--delete-existing-index",
        default=False,
        action="store_true",
        help="Delete the index if it already exists",
    )
    parser.add_argument(
        "-c", "--categories",
        nargs="+",
        default=None,
        help="List of categories to load into the index",
    )
    parser.add_argument(
        "-s", "--bulk-send-chunk-size",
        type=int,
        default=100,
        help="Chunk size for bulk sending documents to OpenSearch",
    )
    parser.add_argument(
        "-n", "--number-of-docs-per-category",
        type=int,
        default=5000,
        help="Number of documents to load per category",
    )
    parser.add_argument(
        "-o", "--opensearch-type",
        choices=["os", "aos"],
        default="aos",
        help="Type of OpenSearch instance to connect to: local=os or remote=aos",
    )
    parser.add_argument(
        '--no-load',
        action='store_true',
        default=False,
        help="Skip loading data into the index",
    )

    args = parser.parse_args()

    # Set default categories if none specified
    if args.categories is None:
        args.categories = DEFAULT_CATEGORIES
    elif args.categories == "all":
        args.categories = QAndAFileReader.AMAZON_PQA_CATEGORY_MAP.keys()
        
    return args