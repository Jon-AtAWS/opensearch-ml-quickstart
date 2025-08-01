# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

import time
from overrides import overrides
from opensearchpy import OpenSearch
from opensearch_py_ml.ml_commons import MLCommonClient

from .ml_model import MlModel
from connectors import MlConnector
from configs.configuration_manager import get_ml_base_uri


class RemoteMlModel(MlModel):
    DEFAULT_DENSE_MODEL_NAME = "Remote Dense Model"
    DEFAULT_DENSE_MODEL_DESCRIPTION = "This is a remote dense model"

    DEFAULT_SPARSE_MODEL_NAME = "Remote Sparse Model"
    DEFAULT_SPARSE_MODEL_DESCRIPTION = "This is a remote sparse model"

    def __init__(
        self,
        os_client: OpenSearch,
        ml_commons_client: MLCommonClient,
        ml_connector: MlConnector,
        model_group_id,
        model_name=None,
        model_description=None,
        model_configs=dict(),
    ) -> None:
        # TODO: as a best practice, move the parent class consturctor function first
        self._ml_connector = ml_connector
        embedding_type = model_configs.get("embedding_type", "dense")
        # Set default name and description based on embedding type
        if embedding_type == "sparse":
            default_name = self.DEFAULT_SPARSE_MODEL_NAME
            default_description = self.DEFAULT_SPARSE_MODEL_DESCRIPTION
        else:  # default to dense
            default_name = self.DEFAULT_DENSE_MODEL_NAME
            default_description = self.DEFAULT_DENSE_MODEL_DESCRIPTION

        # Use provided values or defaults
        model_name = model_name if model_name is not None else default_name
        model_description = (
            model_description if model_description is not None else default_description
        )
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
            url=f"{get_ml_base_uri()}/models/_register",
            body=model_deploy_payload,
        )
        task_id = response["task_id"]

        # validate model deployment task
        time.sleep(1)
        response = self._os_client.http.get(url=f"{get_ml_base_uri()}/tasks/{task_id}")
        state = response["state"]
        if state != "COMPLETED":
            raise Exception(f"Model deployment task {task_id} is not COMPLETED!")
