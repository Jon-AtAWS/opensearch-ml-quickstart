# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

from overrides import overrides

from .os_ml_connector import OsMlConnector


class OsSagemakerMlConnector(OsMlConnector):
    # TODO: verify that default connector name takes effect
    DEFAULT_CONNECTOR_NAME = "Sagemaker connector"
    DEFAULT_CONNECTOR_DESCRIPTION = "This is a Sagemaker connector"

    @overrides
    def _get_connector_create_payload(self):
        return "os_sagemaker.json"
