# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

"""Testing cat_indices_reader"""

import logging
from client import get_client, get_index_size


def test():
    logging.info("Testing aos client util...")
    client = get_client("aos")
    logging.info(get_index_size(client, index_name="amazon_pqa", unit="b"))
