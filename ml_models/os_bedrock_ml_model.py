# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

import os
import time
from overrides import overrides
from opensearchpy import OpenSearch
from opensearch_py_ml.ml_commons import MLCommonClient

from .helper import read_json_file
from .remote_ml_model import RemoteMlModel
from configs import validate_configs, ML_BASE_URI


class OsBedrockMlModel(RemoteMlModel):
    DEFAULT_MODEL = "Bedrock model"

    def __init__(
        self,
        os_client: OpenSearch,
        ml_commons_client: MLCommonClient,
        model_name=DEFAULT_MODEL,
        model_configs=None,
    ) -> None:
        super().__init__(os_client, ml_commons_client, model_name, model_configs)

    @overrides
    def _validate_configs(self):
        required_args = ["access_key", "secret_key", "region", "url"]
        validate_configs(self.model_configs, required_args)

    @overrides
    def _set_up_connector(self):
        url = self.model_configs["url"]
        region = self.model_configs["region"]
        access_key = self.model_configs["access_key"]
        secret_key = self.model_configs["secret_key"]
        connector_version = self.model_configs["connector_version"]

        current_dir = os.path.dirname(os.path.abspath(__file__))
        create_connector_payload_path = os.path.join(
            current_dir, "connector_payloads", "os_bedrock.json"
        )
        connector_create_payload = read_json_file(create_connector_payload_path)
        connector_create_payload["version"] = connector_version
        connector_create_payload["parameters"]["region"] = region
        connector_create_payload["credential"]["access_key"] = access_key
        connector_create_payload["credential"]["secret_key"] = secret_key
        connector_create_payload["actions"][0]["url"] = url

        response = self._os_client.http.post(
            url=f"{ML_BASE_URI}/connectors/_create",
            body=connector_create_payload,
        )

        self._connector_id = response["connector_id"]

    @overrides
    def _deploy_model(self):
        model_deploy_payload = {
            "name": self._model_name,
            "function_name": "remote",
            "description": f"Bedrock embedding model {self._model_name}",
            "connector_id": self._connector_id,
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
