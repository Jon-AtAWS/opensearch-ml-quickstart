# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

import os
import logging
from abc import ABC, abstractmethod
from opensearchpy import OpenSearch

from configs import ML_BASE_URI
from .helper import read_json_file


# parent abstract class for all connectors
class MlConnector(ABC):
    DEFAULT_CONNECTOR_NAME = "Machine Learning connector"
    DEFAULT_CONNECTOR_DESCRIPTION = "This is a Machine Learning connector"

    def __init__(
        self,
        os_client: OpenSearch,
        connector_name=None,
        connector_description=None,
        connector_configs=dict(),
    ) -> None:
        self._os_client = os_client
        self._connector_name = (
            connector_name if connector_name else self.DEFAULT_CONNECTOR_NAME
        )
        self._connector_description = (
            connector_description
            if connector_description
            else self.DEFAULT_CONNECTOR_DESCRIPTION
        )
        self._connector_configs = connector_configs
        self._connector_id = self._get_connector_id()
        logging.info(f"Connector id {self._connector_id}")

    def _get_connector_id(self):
        connector_ids = self._find_connectors(self._connector_name)
        if len(connector_ids) == 0:
            logging.info(f"Creating connector {self._connector_name}")
            self.set_up()

        # in case of duplicate connector names, find the first connector id
        connector_ids = self._find_connectors(self._connector_name)
        if len(connector_ids) == 0:
            raise Exception("Failed to find the created connector")
        return connector_ids[0]

    def __str__(self) -> str:
        return f"<Connector {self._connector_name} {self._connector_id}>"

    def __repr__(self) -> str:
        return self.__str__()

    def connector_id(self):
        return self._connector_id

    @abstractmethod
    def _validate_configs(self):
        pass

    @abstractmethod
    def _get_connector_create_payload_filename(self):
        pass

    def _read_connector_create_payload(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        create_connector_payload_path = os.path.join(
            current_dir,
            "connector_payloads",
            self._get_connector_create_payload_filename(),
        )
        connector_create_payload = read_json_file(create_connector_payload_path)
        return connector_create_payload

    @abstractmethod
    def _fill_in_connector_create_payload(self, connector_create_payload):
        pass

    def _get_connector_create_payload(self):
        connector_create_payload = self._read_connector_create_payload()
        connector_create_payload = self._fill_in_connector_create_payload(
            connector_create_payload
        )
        return connector_create_payload

    @abstractmethod
    def _create_connector_with_payload(self, connector_create_payload):
        pass

    def set_up(self):
        connector_create_payload = self._get_connector_create_payload()
        self._create_connector_with_payload(connector_create_payload)

    def clean_up(self):
        self._delete_connector(self._connector_id)

    def _search_connectors(self, search_query):
        if not isinstance(search_query, dict):
            raise ValueError("search_query needs to be a dictionary")

        return self._os_client.http.post(
            url=f"{ML_BASE_URI}/connectors/_search", body=search_query
        )

    # name == None to return all connector ids
    def _find_connectors(self, connector_name=None):
        try:
            search_query = {"size": 10000, "_source": {"includes": ["name"]}}
            search_result = self._search_connectors(search_query)
            if not search_result:
                return []
            if "hits" not in search_result or "hits" not in search_result["hits"]:
                return []
            ret = []
            for hit in search_result["hits"]["hits"]:
                source = hit["_source"]
                if not connector_name or hit["_source"]["name"] == connector_name:
                    ret.append(hit["_id"])
            return ret
        except Exception as e:
            logging.error(f"MlConnector _find_connectors failed due to exception: {e}")
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

    def delete_all_connectors(self):
        logging.info("Deleting all connectors")
        connector_ids = self._find_connectors()
        for connector_id in connector_ids:
            self._delete_connector(connector_id)
