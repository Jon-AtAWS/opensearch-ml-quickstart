# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

from overrides import overrides

from .os_ml_connector import OsMlConnector


class OsSagemakerMlConnector(OsMlConnector):
    DEFAULT_CONNECTOR_NAME = "Amazon Sagemaker Connector"
    DEFAULT_CONNECTOR_DESCRIPTION = "The connector to Sagemaker embedding model"

    @overrides
    def _get_connector_create_payload_filename(self):
        return "os_sagemaker.json"
