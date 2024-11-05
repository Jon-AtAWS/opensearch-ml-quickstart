# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

import logging
from opensearchpy import OpenSearch
from opensearch_py_ml import ml_commons

from configs import ML_BASE_URI


class MlModelGroup:
    DEFAULT_GROUP_NAME = "default_model_group"

    def __init__(
        self,
        os_client: OpenSearch,
        ml_commons_client: ml_commons,
        model_group_name=DEFAULT_GROUP_NAME,
    ) -> None:
        self._os_client = os_client
        self._ml_commons_client = ml_commons_client
        self._model_group_name = model_group_name
        self._model_group_id = self._get_model_group_id()

    def __str__(self) -> str:
        return f"<MlModelGroup {self._model_group_name} {self._model_group_id}>"

    def __repr__(self) -> str:
        return self.__str__()

    def clean_up(self):
        if self._model_group_id:
            self._delete_model_group(self._model_group_id)
        else:
            logging.info(
                f"Model group {self.DEFAULT_GROUP_NAME} does not exist, cannot delete"
            )

    def model_group_id(self):
        return self._model_group_id

    def _get_model_group_id(self):
        model_group_id = self._find_model_group_id()
        if not model_group_id:
            logging.info("Registering new model group")
            model_group_id = self._register_model_group()
            logging.info(f"Registered MlModelGroup {model_group_id}")
        return model_group_id

    def _register_model_group(self):
        logging.info(f"Registering model group {self._model_group_name}")
        register_result = self._os_client.http.post(
            url=f"{ML_BASE_URI}/model_groups/_register",
            body={
                "name": self._model_group_name,
                "description": "This is a public model group",
            },
        )
        return register_result["model_group_id"]

    def _get_all_model_groups(self):
        try:
            search_result = self._os_client.http.get(
                url=f"{ML_BASE_URI}/model_groups/_search", body={"size": 10000}
            )
            if not search_result:
                return []
            if "hits" not in search_result or "hits" not in search_result["hits"]:
                return []
            return search_result["hits"]["hits"]
        except Exception as e:
            logging.error(f"_get_all_model_groups failed due to exception: {e}")
            # return an empty list with any exception
            return []

    def _get_all_model_group_ids(self):
        model_groups = self._get_all_model_groups()
        return [model_group["_id"] for model_group in model_groups]

    def _find_model_group_id(self):
        model_groups = self._get_all_model_groups()
        for model_group in model_groups:
            if model_group["_source"]["name"] == self._model_group_name:
                return model_group["_id"]
        return None

    def _delete_model_group(self, model_group_id):
        user_input = (
            input(f"Do you want to delete the model group {model_group_id}? (y/n): ")
            .strip()
            .lower()
        )

        if user_input != "y":
            logging.info("Delete model group canceled.")
            return

        try:
            logging.info(f"Deleting model group {model_group_id}")
            self._os_client.http.delete(
                url=f"{ML_BASE_URI}/model_groups/{model_group_id}",
                body={},
            )
            logging.info(f"Deleted model group {model_group_id}")
        except Exception as e:
            logging.error(
                f"Deleting model group {model_group_id} failed due to exception {e}"
            )
