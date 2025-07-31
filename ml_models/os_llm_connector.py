# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

from overrides import overrides

from configs import validate_configs
from .os_ml_connector import OsMlConnector


class OsLlmConnector(OsMlConnector):
    DEFAULT_CONNECTOR_NAME = "Open-source OpenSearch LLM Connector"
    DEFAULT_CONNECTOR_DESCRIPTION = "The connector to LLM in Open-source OpenSearch"

    @overrides
    def _get_connector_create_payload_filename(self):
        return (
            "sagemaker_sparse.json"
            if self._embedding_type == "sparse"
            else "sagemaker_dense.json"
        )

    @overrides
    def _get_connector_create_payload_filename(self):
        return "claude_3.5_sonnet_v2.json"

    @overrides
    def _fill_in_connector_create_payload(self, connector_create_payload):
        url = (
            self._connector_configs["dense_url"]
            if self._embedding_type == "dense"
            else self._connector_configs["sparse_url"]
        )
        region = self._connector_configs["region"]
        connector_version = self._connector_configs["connector_version"]
        credential = {
            "access_key": self._connector_configs["access_key"],
            "secret_key": self._connector_configs["secret_key"],
        }

        connector_create_payload["name"] = self._connector_name
        connector_create_payload["description"] = self._connector_description
        connector_create_payload["version"] = connector_version
        connector_create_payload["parameters"]["region"] = region
        connector_create_payload["credential"] = credential

        return connector_create_payload
