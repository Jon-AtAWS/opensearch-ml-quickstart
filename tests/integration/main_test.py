# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

import os
import time
import logging
import subprocess
from opensearchpy import OpenSearch

from client import get_client
from configs import ML_BASE_URI

INDEX_NAME = "amazon_pqa"
PIPELINE_NAME = "amazon_pqa"
OS_CLIENT: OpenSearch = get_client("os")
AOS_CLIENT: OpenSearch = get_client("aos")
TEXT_FIELD = "chunk"
EMBEDDING_FIELD = "chunk_embedding"


def validate_embedding(client: OpenSearch, index_name):
    # every document with text field must have the embedding field
    assert client.indices.exists(index_name)
    query = {"size": 10000}
    client.indices.refresh(index=index_name)
    search_response = client.search(index=index_name, body=query)
    hits = search_response["hits"]["hits"]
    # sometimes a few document cannot be ingested into the index
    # we only need 400 documents to avoid flakey test
    assert len(hits) >= 400
    for hit in hits:
        source = hit["_source"]
        if TEXT_FIELD in source:
            assert EMBEDDING_FIELD in source


def validate_clean_up(client: OpenSearch, index_name):
    # index deleted
    assert not client.indices.exists(index_name)
    # pipeline deleted
    get_pipeline_response = None
    try:
        get_pipeline_response = client.ingest.get_pipeline(id=PIPELINE_NAME)
        assert False
    except Exception:
        assert get_pipeline_response == None


# This function is used to validate cleaning up.
# It get the number of connectors, model and model group from the cluster
def get_resources_cnt(client: OpenSearch):
    # connector
    query = {"size": 10000}
    connector_search_result = client.http.get(
        url=f"{ML_BASE_URI}/connectors/_search", body=query
    )
    assert "hits" in connector_search_result
    assert "hits" in connector_search_result["hits"]
    connector_cnt = len(connector_search_result["hits"]["hits"])
    # model
    model_search_result = client.http.get(
        url=f"{ML_BASE_URI}/models/_search", body=query
    )
    assert "hits" in model_search_result
    assert "hits" in model_search_result["hits"]
    model_cnt = len(model_search_result["hits"]["hits"])
    # model group
    model_group_search_result = client.http.get(
        url=f"{ML_BASE_URI}/model_groups/_search", body=query
    )
    assert "hits" in model_group_search_result
    assert "hits" in model_group_search_result["hits"]
    model_group_cnt = len(model_group_search_result["hits"]["hits"])
    return connector_cnt, model_cnt, model_group_cnt


def run_test(is_os_client: bool, model_type, embedding_type="dense"):
    host_type = "os" if is_os_client else "aos"
    client = OS_CLIENT if is_os_client else AOS_CLIENT
    clean_up_task = "sparse_encoding_v1" if embedding_type == "sparse" else "knn_768"
    no_clean_up_task = clean_up_task + "_no_cleanup"
    prev_connector_cnt, prev_model_cnt, prev_model_group_cnt = get_resources_cnt(client)
    process = subprocess.Popen(
        args=[
            "python3",
            "main.py",
            f"-t={no_clean_up_task}",
            f"-mt={model_type}",
            f"-ht={host_type}",
            f"-et={embedding_type}",
            "-c=headlight bulbs",
            "-d",
        ],
        cwd=os.path.curdir,
    )
    process.communicate(input="")
    validate_embedding(client, INDEX_NAME)
    process = subprocess.Popen(
        args=[
            "python3",
            "main.py",
            f"-t={clean_up_task}",
            f"-mt={model_type}",
            f"-ht={host_type}",
            f"-et={embedding_type}",
            "-c=headlight bulbs",
            "-d",
        ],
        stdin=subprocess.PIPE,
        cwd=os.path.curdir,
        text=True,
    )
    process.communicate(input="y\ny\ny\n")
    time.sleep(5)
    validate_clean_up(client, INDEX_NAME)
    curr_connector_cnt, curr_model_cnt, curr_model_group_cnt = get_resources_cnt(client)
    # some resources may already exist and then gets deleted by cleanup
    assert prev_connector_cnt >= curr_connector_cnt
    assert prev_model_cnt >= curr_model_cnt
    assert prev_model_group_cnt >= curr_model_group_cnt
    process = subprocess.Popen(
        args=[
            "python3",
            "main.py",
            f"-t={clean_up_task}",
            f"-mt={model_type}",
            f"-ht={host_type}",
            f"-et={embedding_type}",
            "-c=headlight bulbs",
            "-d",
            "-cl",
        ],
        stdin=subprocess.PIPE,
        cwd=os.path.curdir,
        text=True,
    )
    process.communicate(input="y\ny\ny\ny\n")
    time.sleep(5)
    validate_clean_up(client, INDEX_NAME)
    curr_connector_cnt, curr_model_cnt, curr_model_group_cnt = get_resources_cnt(client)
    # some resources may already exist and then gets deleted by cleanup
    assert prev_connector_cnt >= curr_connector_cnt
    assert prev_model_cnt >= curr_model_cnt
    assert prev_model_group_cnt >= curr_model_group_cnt


def test():
    logging.info("Testing main with os local dense model...")
    run_test(True, "local")
    logging.info("Testing main with os local sparse model...")
    run_test(True, "local", "sparse")
    logging.info("Testing main with os bedrock model...")
    run_test(True, "bedrock")
    logging.info("Testing main with os sagemaker model...")
    run_test(True, "sagemaker")

    logging.info("Testing main with aos bedrock model...")
    run_test(False, "bedrock")
    logging.info("Testing main with aos sagemaker model...")
    run_test(False, "sagemaker")


if __name__ == "__main__":
    test()
