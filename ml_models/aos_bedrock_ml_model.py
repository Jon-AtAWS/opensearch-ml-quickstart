# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

import os
from overrides import overrides
from opensearchpy import OpenSearch
from opensearch_py_ml.ml_commons import MLCommonClient

from configs import validate_configs
from .remote_ml_model import RemoteMlModel
from .ai_connector_helper import AiConnectorHelper
from .helper import read_json_file


class AosBedrockMlModel(RemoteMlModel):
    DEFAULT_MODEL = "AOS Bedrock Model"
    DEFAULT_MODEL_DESCRIPTION = "This is an AOS Bedrock model"

    def __init__(
        self,
        os_client: OpenSearch,
        ml_commons_client: MLCommonClient,
        helper: AiConnectorHelper,
        model_name=DEFAULT_MODEL,
        model_description=DEFAULT_MODEL_DESCRIPTION,
        model_configs=dict(),
    ) -> None:
        self.helper = helper
        self._model_description = model_description
        super().__init__(os_client, ml_commons_client, model_name, model_configs)

    @overrides
    def _validate_configs(self):
        required_args = [
            "arn",
            "connector_role_name",
            "create_connector_role_name",
            "region",
        ]
        validate_configs(self.model_configs, required_args)

    @overrides
    def _set_up_connector(self):
        url = self.model_configs["url"]
        arn = self.model_configs["arn"]
        region = self.model_configs["region"]
        connector_version = self.model_configs["connector_version"]
        connector_role_name = self.model_configs["connector_role_name"]
        create_connector_role_name = self.model_configs["create_connector_role_name"]

        connector_role_inline_policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Action": ["bedrock:InvokeModel"],
                    "Effect": "Allow",
                    "Resource": arn,
                }
            ],
        }

        current_dir = os.path.dirname(os.path.abspath(__file__))
        create_connector_payload_path = os.path.join(
            current_dir, "connector_payloads", "aos_bedrock_connector.json"
        )
        create_connector_payload = read_json_file(create_connector_payload_path)
        create_connector_payload["version"] = connector_version
        create_connector_payload["parameters"]["region"] = region
        create_connector_payload["actions"][0]["url"] = url

        self._connector_id = self.helper.create_connector_with_role(
            connector_role_inline_policy,
            connector_role_name,
            create_connector_role_name,
            create_connector_payload,
            sleep_time_in_seconds=10,
        )

    @overrides
    def _deploy_model(self):
        self.helper.create_and_deploy_model(
            model_name=self._model_name,
            description=self._model_description,
            connector_id=self._connector_id,
        )
