# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

import logging
from opensearchpy import OpenSearch
from configs import get_client_configs, MINIMUM_OPENSEARCH_VERSION


def get_client(host_type: str) -> OpenSearch:
    configs = get_client_configs(host_type)
    logging.info(f"Connecting to OpenSearch with configs\n{configs}")
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
    # AOS does not support local hosting of ML models, but local OpenSearch does
    # support it. So set the allow_registering_model_via_url and
    # only_run_on_ml_node settings to whether or not the host_type is "os"
    # (local OpenSearch).
    settings = {
        "plugins.ml_commons.memory_feature_enabled": True,
        "plugins.ml_commons.rag_pipeline_feature_enabled": True,
        "plugins.ml_commons.trusted_connector_endpoints_regex": [
            "^https://bedrock-runtime\\..*[a-z0-9-]\\.amazonaws\\.com/.*$"
        ]
    }
    if host_type == "os":
        settings.update({
            "plugins.ml_commons.allow_registering_model_via_url": True,
            "plugins.ml_commons.only_run_on_ml_node": True,
        })
    try:
        logging.info(f"Setting cluster settings: {{'persistent': {settings}}}")
        client.cluster.put_settings(body={"persistent": settings})
    except Exception as e:
        logging.error(f"Failed to set cluster settings: {e}")
        raise e
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

