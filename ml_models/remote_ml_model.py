# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

import time
from overrides import overrides
from opensearchpy import OpenSearch
from opensearch_py_ml.ml_commons import MLCommonClient

from .ml_model import MlModel
from .ml_connector import MlConnector
from configs import parse_arg_from_configs, ML_BASE_URI


class RemoteMlModel(MlModel):
    DEFAULT_MODEL_NAME = "Remote Model"
    DEFAULT_MODEL_DESCRIPTION = "This is a Remote model"

    def __init__(
        self,
        os_client: OpenSearch,
        ml_commons_client: MLCommonClient,
        ml_connector: MlConnector,
        model_group_id,
        model_name=DEFAULT_MODEL_NAME,
        model_description=DEFAULT_MODEL_DESCRIPTION,
        model_configs=dict(),
    ) -> None:
        # TODO: as a best practice, move the parent class consturctor function first
        self._ml_connector = ml_connector
        super().__init__(
            os_client,
            ml_commons_client,
            model_group_id,
            model_name,
            model_description,
            model_configs,
        )

    @overrides
    def _register_model(self):
        self._deploy_model()

    @overrides
    def clean_up(self):
        super().clean_up()
        self._ml_connector.clean_up()

    def _deploy_model(self):
        model_deploy_payload = {
            "name": self._model_name,
            "function_name": "remote",
            "description": self._model_description,
            "connector_id": self._ml_connector.connector_id(),
            "model_group_id": self._model_group_id,
            "deploy": True,
        }
        response = self._os_client.http.post(
            url=f"{ML_BASE_URI}/models/_register",
            body=model_deploy_payload,
        )
        task_id = response["task_id"]

        # validate model deployment task
        time.sleep(1)
        response = self._os_client.http.get(url=f"{ML_BASE_URI}/tasks/{task_id}")
        state = response["state"]
        if state != "COMPLETED":
            raise Exception(f"Model deployment task {task_id} is not COMPLETED!")
