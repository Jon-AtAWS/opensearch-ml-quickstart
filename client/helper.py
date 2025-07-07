# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Dict
from opensearchpy import helpers, OpenSearch

from ml_models import MlModel
from data_process import QAndAFileReader
from configs import get_client_configs, MINIMUM_OPENSEARCH_VERSION

from .os_ml_client_wrapper import OsMlClientWrapper


def get_client(host_type: str) -> OpenSearch:
    configs = get_client_configs(host_type)
    port = configs["port"]
    host_url = configs["host_url"]
    username = configs["username"]
    password = configs["password"]
    hosts = [{"host": host_url, "port": port}] if port else host_url
    client = OpenSearch(
        hosts=hosts,
        http_auth=(username, password),
        use_ssl=True,
        verify_certs=False,
        ssl_assert_hostname=False,
        ssl_show_warn=False,
    )
    check_client_version(client)
    return client


def get_index_size(client: OpenSearch, index_name, unit="mb"):
    """Get the index size from the opensearch client"""
    if not client.indices.exists(index=index_name):
        return 0
    return int(
        client.cat.indices(
            index=index_name, params={"bytes": f"{unit}", "h": "pri.store.size"}
        )
    )


def parse_version(version: str):
    return tuple(map(int, version.split(".")))


def check_client_version(client: OpenSearch):
    """
    Checks if the given version is at least the mininum version
    """
    info = client.info()
    version = info["version"]["number"]
    if parse_version(version) < parse_version(MINIMUM_OPENSEARCH_VERSION):
        raise ValueError(
            f"The mininum required version for opensearch cluster is {MINIMUM_OPENSEARCH_VERSION}"
        )


def send_bulk_ignore_exceptions(client: OpenSearch, config: Dict[str, str], docs):
    logging.info(f"Sending {config['bulk_send_chunk_size']} docs over the wire")
    try:
        status = helpers.bulk(
            client,
            docs,
            chunk_size=config['bulk_send_chunk_size'],
            request_timeout=300,
            max_retries=10,
            raise_on_error=False,
        )
        return status
    except Exception as e:
        logging.error(f"Error sending bulk: {e}")


def load_category(client: OpenSearch, pqa_reader: QAndAFileReader, category, config):
    SPACE_SEPARATOR = " "
    logging.info(f'Loading category "{category}"')
    docs = []
    number_of_docs = 0
    for doc in pqa_reader.questions_for_category(
        pqa_reader.amazon_pqa_category_name_to_constant(category), enriched=True
    ):
        doc["_index"] = config["index_name"]
        doc["_id"] = doc["question_id"]
        doc["chunk"] = SPACE_SEPARATOR.join(
            [doc["product_description"], doc["brand_name"], doc["item_name"]]
        )
        # limit the document token count to 500 tokens from embedding models
        doc["chunk"] = SPACE_SEPARATOR.join(doc["chunk"].split()[:500])
        # documents less than 4 words are meaningless
        if len(doc["chunk"]) <= 4:
            logging.info(f"Empty chunk for {doc}")
            continue
        docs.append(doc)
        number_of_docs += 1
        if number_of_docs % 2000 == 0:
            logging.info(f"Sending {number_of_docs} docs")
            send_bulk_ignore_exceptions(client, config, docs)
            docs = []
    if len(docs) > 0:
        logging.info(f'Category "{category}" complete. Sending {number_of_docs} docs')
        send_bulk_ignore_exceptions(client, config, docs)


def load_dataset(
    client: OsMlClientWrapper,
    ml_model: MlModel,
    pqa_reader: QAndAFileReader,
    config: Dict[str, str],
    delete_existing: bool,
    index_name: str,
    pipeline_name: str,
):
    if delete_existing:
        logging.info(f"Deleting existing index {index_name}")
        client.delete_then_create_index(
            index_name=config["index_name"], settings=config["index_settings"]
        )

    logging.info("Setting up for KNN")
    client.setup_for_kNN(
        ml_model=ml_model,
        index_name=config["index_name"],
        pipeline_name=pipeline_name,
        index_settings=config["index_settings"],
        pipeline_field_map=config["pipeline_field_map"],
        delete_existing=delete_existing,
        embedding_type=config["embedding_type"],
    )

    for category in config["categories"]:
        load_category(
            client=client.os_client,
            pqa_reader=pqa_reader,
            category=category,
            config=config,
        )

    if config["cleanup"]:
        client.cleanup_kNN(
            ml_model=ml_model,
            index_name=config["index_name"],
            pipeline_name=pipeline_name,
        )
