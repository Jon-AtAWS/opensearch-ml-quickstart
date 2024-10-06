# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

from typing import Dict
from opensearchpy import OpenSearch

from configs import validate_configs, get_config, MINIMUM_OPENSEARCH_VERSION


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


def get_client_configs(host_type: str) -> Dict[str, str]:
    required_args = ["host_url", "username", "password"]
    if host_type == "os":
        configs = {
            "port": get_config("OS_PORT"),
            "host_url": get_config("OS_HOST_URL"),
            "username": get_config("OS_USERNAME"),
            "password": get_config("OS_PASSWORD"),
        }
        validate_configs(configs, required_args)
        return configs
    elif host_type == "aos":
        configs = {
            "port": get_config("AOS_PORT"),
            "host_url": get_config("AOS_HOST_URL"),
            "username": get_config("AOS_USERNAME"),
            "password": get_config("AOS_PASSWORD"),
            "domain_name": get_config("AOS_DOMAIN_NAME"),
            "region": get_config("AOS_REGION"),
            "aws_user_name": get_config("AOS_AWS_USER_NAME"),
        }
        validate_configs(configs, required_args)
        return configs
    else:
        raise ValueError("host_type must either be os or aos")


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
