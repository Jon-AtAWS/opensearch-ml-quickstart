# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

import os
import time
import logging
import subprocess
from opensearchpy import OpenSearch

from client import get_client

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


def run_test(is_os_client: bool, model_type):
    host_type = "os" if is_os_client else "aos"
    client = OS_CLIENT if is_os_client else AOS_CLIENT
    process = subprocess.Popen(
        args=[
            "python3",
            "main.py",
            "-t=knn_768_no_cleanup",
            f"-mt={model_type}",
            f"-ht={host_type}",
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
            "-t=knn_768",
            f"-mt={model_type}",
            f"-ht={host_type}",
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
    process = subprocess.Popen(
        args=[
            "python3",
            "main.py",
            "-t=knn_768",
            f"-mt={model_type}",
            f"-ht={host_type}",
            "-c=headlight bulbs",
            "-d",
            "cl",
        ],
        stdin=subprocess.PIPE,
        cwd=os.path.curdir,
        text=True,
    )
    process.communicate(input="y\ny\ny\n")
    time.sleep(5)
    validate_clean_up(client, INDEX_NAME)


def test():
    logging.info("Testing main with os local model...")
    run_test(True, "local")
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