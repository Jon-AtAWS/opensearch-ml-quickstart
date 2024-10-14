# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

from overrides import overrides

from .ml_connector import MlConnector
from configs import validate_configs, ML_BASE_URI


class OsMlConnector(MlConnector):

    @overrides
    def _validate_configs(self):
        required_args = ["access_key", "secret_key", "region", "url"]
        validate_configs(self._connector_configs, required_args)

    @overrides
    def _create_connector_with_payload(self, connector_create_payload):
        response = self._os_client.http.post(
            url=f"{ML_BASE_URI}/connectors/_create",
            body=connector_create_payload,
        )
        self._connector_id = response["connector_id"]

    @overrides
    def _fill_in_connector_create_payload(self, connector_create_payload):
        url = self._connector_configs["url"]
        region = self._connector_configs["region"]
        access_key = self._connector_configs["access_key"]
        secret_key = self._connector_configs["secret_key"]
        connector_version = self._connector_configs["connector_version"]

        connector_create_payload["version"] = connector_version
        connector_create_payload["parameters"]["region"] = region
        connector_create_payload["credential"]["access_key"] = access_key
        connector_create_payload["credential"]["secret_key"] = secret_key
        connector_create_payload["actions"][0]["url"] = url

        return connector_create_payload