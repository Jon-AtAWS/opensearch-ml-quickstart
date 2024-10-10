# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

import logging
from abc import abstractmethod
from overrides import overrides
from opensearchpy import OpenSearch
from opensearch_py_ml.ml_commons import MLCommonClient

from .ml_model import MlModel
from .ml_model_group import MlModelGroup
from configs import ML_BASE_URI


class RemoteMlModel(MlModel):
    DEFAULT_MODEL_NAME = "Remote Model"
    DEFAULT_MODEL_DESCRIPTION = "This is a Remote model"

    def __init__(
        self,
        os_client: OpenSearch,
        ml_commons_client: MLCommonClient,
        model_name=DEFAULT_MODEL_NAME,
        model_description=DEFAULT_MODEL_DESCRIPTION,
        model_configs=dict(),
    ) -> None:
        super().__init__(
            os_client, ml_commons_client, model_name, model_description, model_configs
        )
        self._connector_id = self._get_connector_id()

    def _get_connector_id(self):
        try:
            search_result = self._ml_commons_client.search_model(
                input_json={"query": {"ids": {"values": [self._model_id]}}}
            )
            hit = search_result["hits"]["hits"][0]
            connector_id = hit["_source"]["connector_id"]
            return connector_id
        except Exception as e:
            raise Exception(
                f"Failed to find the registered connector due to exception {e}"
            )

    @overrides
    def _register_model(self):
        self._validate_configs()
        self._set_up_connector()
        self._deploy_model()

    @overrides
    def clean_up(self):
        super().clean_up()
        self._delete_connector(self._connector_id)

    @abstractmethod
    def _validate_configs(self):
        pass

    @abstractmethod
    def _set_up_connector(self):
        pass

    @abstractmethod
    def _deploy_model(self):
        pass

    def _search_connectors(self, search_query):
        if not isinstance(search_query, dict):
            raise ValueError("search_query needs to be a dictionary")

        return self._os_client.http.post(
            url=f"{ML_BASE_URI}/connectors/_search", body=search_query
        )

    def _find_connectors(self):
        # returns all connector ids
        try:
            search_query = {"query": {"match_all": {}}}
            search_result = self._search_connectors(search_query)
            if not search_result:
                return []
            if "hits" not in search_result or "hits" not in search_result["hits"]:
                return []
            ret = []
            for hit in search_result["hits"]["hits"]:
                ret.append(hit["_id"])
            return ret
        except Exception as e:
            logging.error(
                f"remote_ml_model _find_connectors failed due to exception: {e}"
            )
            # return an empty list with any exception
            return []

    def _delete_connector(self, connector_id):
        user_input = (
            input(f"Do you want to delete the connector {connector_id}? (y/n): ")
            .strip()
            .lower()
        )

        if user_input != "y":
            logging.info("Delete connector canceled.")
            return

        try:
            logging.info(f"Deleting connector {connector_id}")
            self._os_client.http.delete(url=f"{ML_BASE_URI}/connectors/{connector_id}")
            logging.info(f"Deleted connector {connector_id}")
        except Exception as e:
            logging.error(
                f"Deleting connector {connector_id} failed due to exception {e}"
            )

    @overrides
    def unload_and_delete_all_loaded_models(self):
        super().unload_and_delete_all_loaded_models()
        logging.info("Deleting all connectors")
        connector_ids = self._find_connectors()
        for connector_id in connector_ids:
            self._delete_connector(connector_id)

        logging.info("Deleting the model group")
        ml_model_group = MlModelGroup(
            self._os_client, self._ml_commons_client, self._model_name
        )
        ml_model_group.clean_up()
