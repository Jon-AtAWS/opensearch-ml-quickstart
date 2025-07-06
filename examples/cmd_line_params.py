import argparse


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
        default=DEFAULT_CATEGORIES,
        help="List of categories to load into the index",
    )
    parser.add_argument(
        "-c", "--bulk-send-chunk-size",
        type=int,
        default=100,
        help="Chunk size for bulk sending documents to OpenSearch",
    )
    return parser.parse_args()