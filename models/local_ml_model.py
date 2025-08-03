# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

from overrides import overrides
from opensearchpy import OpenSearch
from opensearch_py_ml.ml_commons import MLCommonClient

from .ml_model import MlModel
from configs.configuration_manager import (
    get_local_embedding_model_name,
    get_local_embedding_model_format,
    get_local_embedding_model_version,
)


# class for local ml model
class LocalMlModel(MlModel):
    DEFAULT_LOCAL_MODEL_DESCRIPTION = "This is a local ML Model"

    def __init__(
        self,
        os_client: OpenSearch,
        ml_commons_client: MLCommonClient,
        model_group_id,
        model_name=None,
        model_description=DEFAULT_LOCAL_MODEL_DESCRIPTION,
        model_configs=dict(),
    ) -> None:
        # Use configuration manager default if model_name not provided
        if model_name is None:
            model_name = get_local_embedding_model_name()
            
        super().__init__(
            os_client,
            ml_commons_client,
            model_group_id,
            model_name,
            model_description,
            model_configs,
        )

    @overrides
    def _register_model(self):
        # Get model version from config or use default from configuration manager
        model_version = self._model_configs.get("model_version", get_local_embedding_model_version())
        
        # Get model format from config or use default from configuration manager
        model_format = self._model_configs.get("model_format", get_local_embedding_model_format())
        
        self._ml_commons_client.register_pretrained_model(
            model_name=self._model_name,
            model_version=model_version,
            model_format=model_format,
            model_group_id=self._model_group_id,
            deploy_model=True,
            wait_until_deployed=True,
        )
