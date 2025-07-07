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
