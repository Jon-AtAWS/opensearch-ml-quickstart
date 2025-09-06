# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

import logging
from opensearchpy import OpenSearch
from opensearch_py_ml.ml_commons import MLCommonClient
from models import MlModel, MlModelGroup

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import index_utils


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

    def _dense_pipeline_config(self, pipeline_field_map=None):
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

    def _sparse_pipeline_config(self, pipeline_field_map=None):
        if not pipeline_field_map:
            pipeline_field_map = self.DEFAULT_PIPELINE_FIELD_MAP
        config = {
            "description": "Pipeline for processing chunks",
            "processors": [
                {
                    "sparse_encoding": {
                        "model_id": f"{self.model_id()}",
                        "field_map": pipeline_field_map,
                    }
                }
            ],
        }
        logging.info(config)
        return config

    def _add_dense_pipeline(self, pipeline_name="", pipeline_field_map=None):
        if not pipeline_name:
            raise ValueError("_add_dense_pipeline: pipeline name must be specified")
        pipeline_config = self._dense_pipeline_config(
            pipeline_field_map=pipeline_field_map
        )
        logging.info(f"{pipeline_name}: {pipeline_field_map}")
        logging.info("Adding dense pipeline...")
        self.os_client.ingest.put_pipeline(id=pipeline_name, body=pipeline_config)

    def _add_sparse_pipeline(self, pipeline_name="", pipeline_field_map=None):
        if not pipeline_name:
            raise ValueError("_add_sparse_pipeline: pipeline name must be specified")
        pipeline_config = self._sparse_pipeline_config(
            pipeline_field_map=pipeline_field_map
        )
        logging.info(f"sparse_pipeline_config: {pipeline_field_map}")
        logging.info("Adding sparse pipeline...")
        self.os_client.ingest.put_pipeline(id=pipeline_name, body=pipeline_config)

    def setup_for_kNN(
        self,
        ml_model: MlModel,
        index_name="",
        pipeline_name=None,
        pipeline_field_map=None,
        embedding_type="dense",
    ):
        """
        Sets up a kNN index with ingestion pipeline and model.
        """
        logging.info(
            f"Setup for KNN; ml model: {ml_model.model_id()}; ml model group: {self.ml_model_group.model_group_id()}"
        )
        self.index_name = index_name
        self.pipeline_name = pipeline_name
        self.ml_model = ml_model

        if embedding_type == "sparse":
            self._add_sparse_pipeline(
                pipeline_name=pipeline_name, pipeline_field_map=pipeline_field_map
            )
        else:
            self._add_dense_pipeline(
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

        user_input = (
            input(f"Do you want to delete the pipeline {pipeline_name}? (y/n): ")
            .strip()
            .lower()
        )
        if user_input == "y":
            logging.info(f"Deleting pipeline {pipeline_name}")
            try:
                self.os_client.ingest.delete_pipeline(pipeline_name)
                logging.info(f"Deleted pipeline {pipeline_name}")
            except Exception as e:
                logging.info(
                    f"Deleting pipeline {pipeline_name} failed due to exception {e}"
                )
        else:
            logging.info("Delete pipeline canceled.")

        user_input = (
            input(f"Do you want to delete the index {index_name}? (y/n): ")
            .strip()
            .lower()
        )
        if user_input == "y":
            logging.info(f"Deleting index {index_name}")
            try:
                self.os_client.indices.delete(index_name)
                logging.info(f"Deleted index {index_name}")
            except Exception as e:
                logging.info(f"Deleting index {index_name} failed due to exception {e}")
        else:
            logging.info("Delete index canceled.")
