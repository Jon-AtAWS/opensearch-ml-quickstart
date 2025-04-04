# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

from overrides import overrides

from configs import validate_configs
from .aos_ml_connector import AosMlConnector


class AosLlmConnector(AosMlConnector):
    DEFAULT_CONNECTOR_NAME = "Amazon OpenSearch Service LLM Connector"
    DEFAULT_CONNECTOR_DESCRIPTION = "The connector to LLM in Amazon OpenSearch Service "

    @overrides
    def get_connector_role_inline_policy(self):
        llm_arn = self._connector_configs["llm_arn"]
        return {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Resource": [llm_arn],
                    "Action": ["bedrock:InvokeModel"],
                }
            ],
        }

    @overrides
    def _validate_configs(self):
        required_args = [
            "arn",
            "connector_role_name",
            "create_connector_role_name",
            "region",
        ]
        validate_configs(self._connector_configs, required_args)

    @overrides
    def _fill_in_connector_create_payload(self, connector_create_payload):
        region = self._connector_configs["region"]
        connector_version = self._connector_configs["connector_version"]

        connector_create_payload["name"] = self._connector_name
        connector_create_payload["description"] = self._connector_description
        connector_create_payload["parameters"]["region"] = region
        connector_create_payload["version"] = connector_version

        print(
            f"aos_llm_connector connector_create_payload:\n{connector_create_payload}"
        )
        return connector_create_payload

    @overrides
    def _get_connector_create_payload_filename(self):
        return "claude_3.5_sonnet_v2.json"
