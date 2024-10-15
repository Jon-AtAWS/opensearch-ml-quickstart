# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

import logging
from abc import ABC, abstractmethod
from opensearchpy import OpenSearch
from opensearch_py_ml.ml_commons import MLCommonClient
from tenacity import retry, stop_after_attempt, wait_fixed

from configs import DELETE_MODEL_WAIT_TIME, DELETE_MODEL_RETRY_TIME


# parent abstract class for all ml models
class MlModel(ABC):
    DEFAULT_MODEL_NAME = "Machine Learning Model"
    DEFAULT_MODEL_DESCRIPTION = "This is a Machine Learning model"

    def __init__(
        self,
        os_client: OpenSearch,
        ml_commons_client: MLCommonClient,
        model_name=DEFAULT_MODEL_NAME,
        model_description=DEFAULT_MODEL_DESCRIPTION,
        model_configs=dict(),
    ) -> None:
        self._os_client = os_client
        self._ml_commons_client = ml_commons_client
        self._model_name = model_name
        self._model_description = model_description
        self.model_configs = model_configs
        self._model_id = self._get_model_id()
        logging.info(f"MlModel id {self._model_id}")

    def _get_model_id(self):
        model_ids = self.find_models(self._model_name)
        if len(model_ids) == 0:
            logging.info(f"Registering model {self._model_name}")
            self._register_model()

        # in case of duplicate model names, find the first model id
        model_ids = self.find_models(self._model_name)
        if len(model_ids) == 0:
            raise Exception("Failed to find the registered model")
        return model_ids[0]

    def __str__(self) -> str:
        return f"<MlModel {self._model_name} {self._model_id}>"

    def __repr__(self) -> str:
        return self.__str__()

    def model_id(self):
        return self._model_id

    def clean_up(self):
        self._undeploy_and_delete_model(self._model_id)

    @abstractmethod
    def _register_model(self):
        pass

    # name == None to return all model ids, list to find all of those
    # returns a list of model_ids
    def find_models(self, *names):
        try:
            search_result = self._ml_commons_client.search_model(
                input_json={
                    "size": 10000,
                    "_source": {
                        "includes": ["name", "model_id"],
                        "excludes": ["content", "model_content"],
                    },
                }
            )
            if not search_result:
                return []
            if "hits" not in search_result or "hits" not in search_result["hits"]:
                return []
            ret = []
            for hit in search_result["hits"]["hits"]:
                if not names or hit["_source"]["name"] in names:
                    # for local model, hit["_id"] is not the actual model id
                    if "model_id" in hit["_source"]:
                        ret.append(hit["_source"]["model_id"])
                    else:
                        ret.append(hit["_id"])
            return list(set(ret))
        except Exception as e:
            logging.error(f"find_models failed due to exception: {e}")
            # return an empty list with any exception
            return []

    @retry(
        stop=stop_after_attempt(DELETE_MODEL_RETRY_TIME),
        wait=wait_fixed(DELETE_MODEL_WAIT_TIME),
    )
    def _undeploy_and_delete_model(self, model_id):
        user_input = (
            input(f"Do you want to undeploy and delete the model {model_id}? (y/n): ")
            .strip()
            .lower()
        )

        if user_input != "y":
            logging.info("Undeploy and delete model canceled.")
            return

        try:
            logging.info(f"Undeploying model {model_id}")
            self._ml_commons_client.undeploy_model(model_id)
            logging.info(f"Undeployed model {model_id}")
        except Exception as e:
            logging.error(f"Undeploying model {model_id} failed due to exception {e}")
            raise e

        try:
            logging.info(f"Deleting model {model_id}")
            self._ml_commons_client.delete_model(model_id)
            logging.info(f"Deleted model {model_id}")
        except Exception as e:
            logging.error(f"Deleting model {model_id} failed due to exception {e}")
            raise e

    def unload_and_delete_all_loaded_models(self):
        logging.info("Deleting all loaded models")
        model_ids = self.find_models()
        for model_id in model_ids:
            self._undeploy_and_delete_model(model_id)
