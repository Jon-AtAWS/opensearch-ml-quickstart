# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

from overrides import overrides

from .os_ml_connector import OsMlConnector


class OsBedrockMlConnector(OsMlConnector):
    DEFAULT_CONNECTOR_NAME = "Amazon Bedrock Connector"
    DEFAULT_CONNECTOR_DESCRIPTION = "The connector to Bedrock embedding model"

    @overrides
    def _get_connector_create_payload_filename(self):
        return "os_bedrock.json"
