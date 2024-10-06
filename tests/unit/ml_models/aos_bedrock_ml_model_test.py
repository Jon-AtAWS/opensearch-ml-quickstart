# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

import logging
from unittest.mock import patch
from opensearch_py_ml.ml_commons import MLCommonClient

from client import get_client, get_client_configs
from ml_models import get_connector_helper, get_remote_model_configs, AosBedrockMlModel


def test():
    logging.info("Testing aos bedrock ml model...")
    os_client = get_client("aos")
    ml_commons_client = MLCommonClient(os_client=os_client)

    logging.info("Creating aos bedrock ml model...")
    aos_bedrock_configs = get_remote_model_configs(
        host_type="aos", model_type="bedrock"
    )
    helper = get_connector_helper(get_client_configs("aos"))
    model = AosBedrockMlModel(
        os_client=os_client,
        ml_commons_client=ml_commons_client,
        helper=helper,
        model_configs=aos_bedrock_configs,
    )

    logging.info("Cleaning up...")
    with patch("builtins.input", return_value="y"):
        model.clean_up()
