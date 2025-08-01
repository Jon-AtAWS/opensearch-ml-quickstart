# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

import logging
from unittest.mock import patch
from opensearch_py_ml.ml_commons import MLCommonClient

from client import get_client
from models import get_ml_model_group, LocalMlModel


def test():
    logging.info("Testing os local ml model...")
    os_client = get_client("os")
    ml_commons_client = MLCommonClient(os_client=os_client)

    model_group = get_ml_model_group(os_client, ml_commons_client)
    model_group_id = model_group.model_group_id()
    logging.info(f"Model Group Id {model_group_id}")
    model_configs = {"model_group_id": model_group_id}

    logging.info("Creating os local dense ml model...")
    model = LocalMlModel(
        os_client=os_client,
        ml_commons_client=ml_commons_client,
        model_group_id=model_group_id,
        model_configs=model_configs,
    )

    logging.info("Cleaning up...")
    with patch("builtins.input", return_value="y"):
        model.clean_up()
        model_group.clean_up()

    logging.info("Creating os local sparse ml model...")
    model = LocalMlModel(
        os_client=os_client,
        ml_commons_client=ml_commons_client,
        model_group_id=model_group_id,
        model_configs=model_configs,
        model_name="amazon/neural-sparse/opensearch-neural-sparse-encoding-v1",
    )

    logging.info("Cleaning up...")
    with patch("builtins.input", return_value="y"):
        model.clean_up()
        model_group.clean_up()
