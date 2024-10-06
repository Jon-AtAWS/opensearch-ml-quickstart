# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

import logging
from unittest.mock import patch
from opensearch_py_ml.ml_commons import MLCommonClient

from client import get_client
from ml_models import LocalMlModel, MlModelGroup


def test():
    logging.info("Testing ml model group...")
    os_client = get_client("os")
    ml_commons_client = MLCommonClient(os_client=os_client)
    model_group = MlModelGroup(
        os_client=os_client,
        ml_commons_client=ml_commons_client,
    )
    model_group_id = model_group.model_group_id()
    logging.info(f"Model Group Id {model_group_id}")

    model = LocalMlModel(
        os_client=os_client,
        ml_commons_client=ml_commons_client,
        model_name="huggingface/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_configs={"model_group_id": model_group_id},
    )

    logging.info("Cleaning up...")
    with patch("builtins.input", return_value="y"):
        model.clean_up()
        model_group.clean_up()
