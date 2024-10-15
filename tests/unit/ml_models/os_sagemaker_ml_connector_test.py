# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

import logging
from unittest.mock import patch
from opensearch_py_ml.ml_commons import MLCommonClient

from client import get_client
from ml_models import get_remote_model_configs, OsSagemakerMlConnector, RemoteMlModel


def test():
    logging.info("Testing os sagemaker ml model...")
    os_client = get_client("os")
    ml_commons_client = MLCommonClient(os_client=os_client)

    logging.info("Creating os sagemaker ml model...")
    os_sagemaker_configs = get_remote_model_configs(
        host_type="os", model_type="sagemaker"
    )
    ml_connector = OsSagemakerMlConnector(
        os_client=os_client,
        connector_configs=os_sagemaker_configs
    )
    model = RemoteMlModel(
        os_client=os_client,
        ml_commons_client=ml_commons_client,
        ml_connector=ml_connector,
    )

    logging.info("Cleaning up..")
    with patch("builtins.input", return_value="y"):
        model.clean_up()
