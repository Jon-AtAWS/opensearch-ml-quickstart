# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

from overrides import overrides
from opensearchpy import OpenSearch
from opensearch_py_ml.ml_commons import MLCommonClient

from .ml_model import MlModel
from configs import (
    parse_arg_from_configs,
    DEFAULT_LOCAL_MODEL_NAME,
    DEFAULT_MODEL_FORMAT,
    DEFAULT_MODEL_VERSION,
)


# class for local ml model
class LocalMlModel(MlModel):
    DEFAULT_LOCAL_MODEL_DESCRIPTION = "This is a local ML Model"

    def __init__(
        self,
        os_client: OpenSearch,
        ml_commons_client: MLCommonClient,
        model_name=DEFAULT_LOCAL_MODEL_NAME,
        model_description=DEFAULT_LOCAL_MODEL_DESCRIPTION,
        model_configs=dict(),
    ) -> None:
        super().__init__(
            os_client, ml_commons_client, model_name, model_description, model_configs
        )

    @overrides
    def _register_model(self):
        model_version = parse_arg_from_configs(
            self.model_configs, "model_version", DEFAULT_MODEL_VERSION
        )
        model_format = parse_arg_from_configs(
            self.model_configs, "model_format", DEFAULT_MODEL_FORMAT
        )
        model_group_id = parse_arg_from_configs(self.model_configs, "model_group_id")
        self._ml_commons_client.register_pretrained_model(
            model_name=self._model_name,
            model_version=model_version,
            model_format=model_format,
            model_group_id=model_group_id,
        )
