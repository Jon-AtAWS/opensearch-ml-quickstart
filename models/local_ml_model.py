# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

import logging
from overrides import overrides
from opensearchpy import OpenSearch
from opensearch_py_ml.ml_commons import MLCommonClient

from .ml_model import MlModel
from configs.configuration_manager import (
    get_local_dense_embedding_model_name,
    get_local_dense_embedding_model_format,
    get_local_dense_embedding_model_version,
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
            model_name = get_local_dense_embedding_model_name()
            
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
        model_version = self._model_configs.get("model_version", get_local_dense_embedding_model_version())
        
        # Get model format from config or use default from configuration manager
        model_format = self._model_configs.get("model_format", get_local_dense_embedding_model_format())
        
        self._ml_commons_client.register_pretrained_model(
            model_name=self._model_name,
            model_version=model_version,
            model_format=model_format,
            model_group_id=self._model_group_id,
            deploy_model=True,
            wait_until_deployed=True,
        )
        
    def _check_and_redeploy_if_needed(self, model_id):
        """Check if model is deployed and redeploy if necessary."""
        try:
            # First verify the model actually exists
            try:
                model_info = self._ml_commons_client.get_model_info(model_id)
            except Exception as e:
                if "404" in str(e) or "NotFoundError" in str(e):
                    logging.error(f"Model {model_id} not found. This indicates a stale reference. Skipping deployment check.")
                    return
                raise
                
            model_state = model_info.get("model_state", "UNKNOWN")
            logging.info(f"Model {model_id} current state: {model_state}")
            
            if model_state == "DEPLOY_FAILED":
                logging.warning(f"Model {model_id} deployment failed. Attempting to redeploy...")
                self._ml_commons_client.deploy_model(model_id, wait_until_deployed=True)
                logging.info(f"Successfully redeployed model {model_id}")
            elif model_state != "DEPLOYED":
                logging.warning(f"Model {model_id} is in state {model_state}. Attempting to deploy...")
                self._ml_commons_client.deploy_model(model_id, wait_until_deployed=True)
                logging.info(f"Successfully deployed model {model_id}")
            else:
                logging.info(f"Model {model_id} is already deployed")
        except Exception as e:
            logging.error(f"Failed to check/redeploy model {model_id}: {e}")
            raise
