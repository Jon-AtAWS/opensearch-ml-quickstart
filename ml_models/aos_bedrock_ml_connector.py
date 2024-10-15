# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

from overrides import overrides

from .aos_ml_connector import AosMlConnector


class AosBedrockMlConnector(AosMlConnector):
    DEFAULT_CONNECTOR_NAME = "Amazon Bedrock Connector"
    DEFAULT_CONNECTOR_DESCRIPTION = "The connector to Bedrock embedding model"

    @overrides
    def get_connector_role_inline_policy(self):
        arn = self._connector_configs["arn"]
        return {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Action": ["bedrock:InvokeModel"],
                    "Effect": "Allow",
                    "Resource": arn,
                }
            ],
        }

    @overrides
    def _get_connector_create_payload_filename(self):
        return "aos_bedrock.json"
