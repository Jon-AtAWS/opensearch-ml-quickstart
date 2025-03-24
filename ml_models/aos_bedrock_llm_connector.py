# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

from overrides import overrides

from .aos_ml_connector import AosMlConnector


class AosBedrockLLMConnector(AosMlConnector):
    DEFAULT_CONNECTOR_NAME = "Amazon Bedrock Connector"
    DEFAULT_CONNECTOR_DESCRIPTION = "The connector to Bedrock embedding model"

    @overrides
    def get_connector_role_inline_policy(self):
        dense_arn = self._connector_configs["dense_arn"]
        return {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Resource": [dense_arn],
                    "Action": ["bedrock:InvokeModel"],
                }
            ],
        }

    @overrides
    def _get_connector_create_payload_filename(self):
        return "claude_3.5_sonnet_v2.json"
