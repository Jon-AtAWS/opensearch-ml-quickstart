# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

from overrides import overrides

from .aos_ml_connector import AosMlConnector


class AosSagemakerMlConnector(AosMlConnector):
    DEFAULT_CONNECTOR_NAME = "Amazon Sagemaker Connector"
    DEFAULT_CONNECTOR_DESCRIPTION = "The connector to Sagemaker embedding model"

    @overrides
    def get_connector_role_inline_policy(self):
        dense_arn = self._connector_configs["dense_arn"]
        sparse_arn = self._connector_configs["sparse_arn"]
        return {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Resource": [dense_arn, sparse_arn],
                    "Action": ["sagemaker:InvokeEndpoint"],
                }
            ],
        }

    @overrides
    def _get_connector_create_payload_filename(self):
        return (
            "sagemaker_sparse.json"
            if self._embedding_type == "sparse"
            else "sagemaker_dense.json"
        )
