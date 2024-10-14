# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

from overrides import overrides

from .aos_ml_connector import AosMlConnector

class AosSagemakerMlModel(AosMlConnector):
    DEFAULT_MODEL_NAME = "Sagemaker Model"
    DEFAULT_MODEL_DESCRIPTION = "This is a Sagemaker model"

    @overrides
    def get_connector_role_inline_policy(self):
        arn = self._connector_configs["arn"]
        return {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": ["sagemaker:InvokeEndpoint"],
                    "Resource": [arn],
                }
            ],
        }
    
    @overrides
    def _get_connector_create_payload(self):
        return "aos_sagemaker.json"