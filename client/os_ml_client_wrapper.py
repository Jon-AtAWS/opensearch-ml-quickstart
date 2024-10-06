# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

import logging
from opensearchpy import OpenSearch
from opensearch_py_ml.ml_commons import MLCommonClient

from ml_models import MlModel, MlModelGroup


class OsMlClientWrapper:
    """
    OsMlClientWrapper is a wrapper on OpenSearchClient and MLClient. Both 
    clients connect to an opensearch host, sets up and cleans up knn.
    """

    DEFAULT_PIPELINE_FIELD_MAP = {"chunk": "chunk_vector"}

    def __init__(self, os_client: OpenSearch) -> None:
        self.os_client = os_client
        self.ml_commons_client = MLCommonClient(os_client=self.os_client)
        self.ml_model_group = MlModelGroup(
            self.os_client,
            self.ml_commons_client,
        )
        self.ml_model = None
        self.index_name = ""
        self.pipeline_name = ""

    def model_id(self):
        return self.ml_model.model_id()

    def model_group_id(self):
        return self.ml_model_group.model_group_id()

    def _pipeline_config(self, pipeline_field_map=None):
        if not pipeline_field_map:
            pipeline_field_map = self.DEFAULT_PIPELINE_FIELD_MAP
        config = {
            "description": "Pipeline for processing chunks",
            "processors": [
                {
                    "text_embedding": {
                        "model_id": f"{self.model_id()}",
                        "field_map": pipeline_field_map,
                    }
                }
            ],
        }
        logging.info(config)
        return config

    def _add_neural_pipeline(self, pipeline_name="", pipeline_field_map=None):
        if not pipeline_name:
            raise ValueError("add_neural_pipeline: pipeline name must be specified")
        pipeline_config = self._pipeline_config(pipeline_field_map=pipeline_field_map)
        logging.info(f"pipeline_config: {pipeline_field_map}")
        logging.info("Adding neural pipeline...")
        self.os_client.ingest.put_pipeline(pipeline_name, body=pipeline_config)

    def idempotent_create_index(self, index_name="", settings=None):
        """
        Create the index with settings.
        """
        if not index_name:
            raise ValueError("idempotent_create_index: index name must be specified")
        if not settings:
            raise ValueError("idempotent_create_index: settings must be specified")
        try:
            response = self.os_client.indices.create(index_name, body=settings)
            logging.info(
                f"idempotent_create_index response: {response}",
            )
        except Exception as e:
            logging.error(f"Error creating index {index_name} due to exception: {e}")

    def delete_then_create_index(self, index_name="", settings=None):
        """
        Delete the index and then create the index with settings.
        """
        if not index_name:
            raise ValueError("delete_then_create_index: index name must be specified")
        if not settings:
            raise ValueError("delete_then_create_index: settings must be specified")
        if self.os_client.indices.exists(index_name):
            logging.info(f"Deleting index {index_name}")
            self.os_client.indices.delete(index=index_name)
        self.idempotent_create_index(index_name=index_name, settings=settings)

    def setup_for_kNN(
        self,
        ml_model: MlModel,
        index_name="",
        index_settings="",
        pipeline_name=None,
        pipeline_field_map=None,
        delete_existing=False,
    ):
        """
        Sets up a kNN index with ingestion pipeline and model.
        """
        logging.info(
            f"Setup for KNN; ml model: {ml_model.model_id()}; ml model group: {self.ml_model_group.model_group_id()}"
        )
        self.index_name = index_name
        self.pipeline_name = pipeline_name
        try:
            # TODO: Should this go in .env? Maybe just the trusted endpoints?
            self.os_client.cluster.put_settings(
                body={
                    "persistent": {
                        "plugins.ml_commons.model_access_control_enabled": False,
                        "plugins.ml_commons.allow_registering_model_via_url": True,
                    }
                }
            )
        except Exception as e:
            logging.info(f"Setting up cluster settings failed due to exception {e}")

        try:
            self.os_client.cluster.put_settings(
                body={
                    "persistent": {
                        "plugins.ml_commons.only_run_on_ml_node": False,
                    }
                }
            )
        except Exception as e:
            # AOS cannot apply allow_registering_model_via_url setting
            self.os_client.cluster.put_settings(
                body={
                    "persistent": {
                        "plugins.ml_commons.only_run_on_ml_node": False,
                    }
                }
            )

        self.ml_model = ml_model
        if delete_existing:
            self.delete_then_create_index(
                index_name=index_name, settings=index_settings
            )
        else:
            self.idempotent_create_index(index_name=index_name, settings=index_settings)
        self._add_neural_pipeline(
            pipeline_name=pipeline_name, pipeline_field_map=pipeline_field_map
        )

    def setup_without_kNN(self, index_name="", index_settings=""):
        """
        Sets up the index without KNN.
        """
        self.delete_then_create_index(index_name=index_name, settings=index_settings)

    def cleanup_kNN(self, ml_model=None, index_name=None, pipeline_name=None):
        """
        Cleans up the knn model, model_group, pipeline and index.
        """
        if ml_model:
            self.ml_model = ml_model
        if self.ml_model:
            self.ml_model.clean_up()
        self.ml_model = None
        self.ml_model_group.clean_up()
        try:
            self.os_client.ingest.delete_pipeline(pipeline_name)
        except Exception as e:
            logging.info(
                f"Deleting pipeline {pipeline_name} failed due to exception {e}"
            )

        try:
            self.os_client.indices.delete(index_name)
        except Exception as e:
            logging.info(f"Deleting index {index_name} failed due to exception {e}")
