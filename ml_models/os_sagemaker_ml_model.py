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


class OsSagemakerMlModel(RemoteMlModel):
    DEFAULT_MODEL_NAME = "Sagemaker Model"
    DEFAULT_MODEL_DESCRIPTION = "This is a Sagemaker model"

    def __init__(
        self,
        os_client: OpenSearch,
        ml_commons_client: MLCommonClient,
        model_name=DEFAULT_MODEL_NAME,
        model_description=DEFAULT_MODEL_DESCRIPTION,
        model_configs=dict(),
    ) -> None:
        super().__init__(
            os_client, ml_commons_client, model_name, model_description, model_configs
        )

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
            current_dir, "connector_payloads", "os_sagemaker.json"
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
