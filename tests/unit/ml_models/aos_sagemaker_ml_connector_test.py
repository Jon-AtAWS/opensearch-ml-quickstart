# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

import logging
from unittest.mock import patch
from opensearch_py_ml.ml_commons import MLCommonClient

from configs import get_remote_connector_configs
from client import get_client, get_client_configs
from ml_models import (
    get_aos_connector_helper,
    AosSagemakerMlConnector,
    RemoteMlModel,
    MlModelGroup,
)


def test():
    logging.info("Testing aos sagemaker ml model...")
    os_client = get_client("aos")
    ml_commons_client = MLCommonClient(os_client=os_client)
    ml_model_group = MlModelGroup(
        os_client=os_client,
        ml_commons_client=ml_commons_client,
    )

    logging.info("Creating aos sagemaker ml model...")
    aos_connector_helper = get_aos_connector_helper(get_client_configs("aos"))
    aos_sagemaker_connector_configs = get_remote_connector_configs(
        host_type="aos", connector_type="sagemaker"
    )
    ml_connector = AosSagemakerMlConnector(
        os_client=os_client,
        connector_configs=aos_sagemaker_connector_configs,
        aos_connector_helper=aos_connector_helper,
    )
    model = RemoteMlModel(
        os_client=os_client,
        ml_commons_client=ml_commons_client,
        ml_connector=ml_connector,
        model_group_id=ml_model_group.model_group_id(),
    )

    logging.info("Cleaning up...")
    with patch("builtins.input", return_value="y"):
        model.clean_up()
