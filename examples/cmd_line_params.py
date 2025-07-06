import argparse
import os
import sys


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_process import QAndAFileReader


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
    parser = argparse.ArgumentParser(
        description="Run AI-powered search examples with OpenSearch ML Quickstart"
    )
    parser.add_argument(
        "-f", "--force-index-creation",
        default=False,
        action="store_true",
        help="Delete/recreate the index even if it already exists",
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
    args = parser.parse_args()
    
    if args.categories is None:
        args.categories = DEFAULT_CATEGORIES
    elif args.categories == "all":
        args.categories = QAndAFileReader.AMAZON_PQA_CATEGORY_MAP.keys()
        
    return args