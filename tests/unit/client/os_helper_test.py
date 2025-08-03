# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

"""Legacy OS helper test - moved to test_integration.py"""

import logging
from client import get_client, get_index_size


def test():
    """Legacy test function - use test_integration.py for new tests"""
    logging.info("Testing os client util...")
    try:
        client = get_client("os")
        size = get_index_size(client, index_name="amazon_pqa", unit="b")
        logging.info(f"OS index size: {size}")
    except Exception as e:
        logging.warning(f"OS client test failed: {e}")


# For backward compatibility
if __name__ == "__main__":
    test()
