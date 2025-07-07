# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

import logging
from opensearchpy import OpenSearch
from configs import get_client_configs, MINIMUM_OPENSEARCH_VERSION


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

