# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

from overrides import overrides

from .os_ml_connector import OsMlConnector


class OsBedrockMlConnector(OsMlConnector):
    # TODO: verify that default connector name takes effect
    DEFAULT_CONNECTOR_NAME = "Bedrock connector"
    DEFAULT_CONNECTOR_DESCRIPTION = "This is a Bedrock connector"

    @overrides
    def _get_connector_create_payload_filename(self):
        return "os_bedrock.json"
