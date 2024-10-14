# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

from overrides import overrides

from .aos_ml_connector import AosMlConnector


class AosBedrockMlConnector(AosMlConnector):
    # TODO: verify that default connector name takes effect
    DEFAULT_CONNECTOR_NAME = "Bedrock connector"
    DEFAULT_CONNECTOR_DESCRIPTION = "This is a Bedrock connector"

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
    def _get_connector_create_payload(self):
        return "aos_bedrock.json"
