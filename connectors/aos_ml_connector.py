# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

from abc import abstractmethod
from overrides import overrides
from opensearchpy import OpenSearch

from configs import validate_configs
from .ml_connector import MlConnector
from .aos_connector_helper import AosConnectorHelper


class AosMlConnector(MlConnector):
    def __init__(
        self,
        os_client: OpenSearch,
        aos_connector_helper: AosConnectorHelper,
        connector_name=None,
        connector_description=None,
        connector_configs=dict(),
    ) -> None:
        # TODO: as a best practice, move the parent class consturctor function first
        self._aos_connector_helper = aos_connector_helper
        super().__init__(
            os_client, connector_name, connector_description, connector_configs
        )

    @overrides
    def _validate_configs(self):
        required_args = [
            "arn",
            "connector_role_name",
            "create_connector_role_name",
            "region",
            "url",
        ]
        validate_configs(self._connector_configs, required_args)

    @abstractmethod
    def get_connector_role_inline_policy(self):
        pass

    @overrides
    def _create_connector_with_payload(self, connector_create_payload):
        connector_role_inline_policy = self.get_connector_role_inline_policy()
        connector_role_name = self._connector_configs["connector_role_name"]
        create_connector_role_name = self._connector_configs[
            "create_connector_role_name"
        ]

        self._connector_id = self._aos_connector_helper.create_connector_with_role(
            connector_role_inline_policy,
            connector_role_name,
            create_connector_role_name,
            connector_create_payload,
            sleep_time_in_seconds=10,
        )

    @overrides
    def _fill_in_connector_create_payload(self, connector_create_payload):
        url = (
            self._connector_configs["dense_url"]
            if self._embedding_type == "dense"
            else self._connector_configs["sparse_url"]
        )
        region = self._connector_configs["region"]
        connector_version = self._connector_configs["connector_version"]

        connector_create_payload["name"] = self._connector_name
        connector_create_payload["description"] = self._connector_description
        connector_create_payload["actions"][0]["url"] = url
        connector_create_payload["parameters"]["region"] = region
        connector_create_payload["version"] = connector_version

        return connector_create_payload
