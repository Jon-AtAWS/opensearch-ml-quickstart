# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

import logging
from opensearchpy import OpenSearch
from configs.configuration_manager import get_opensearch_config, get_minimum_opensearch_version


def get_client(host_type: str) -> OpenSearch:
    """
    Create and configure an OpenSearch client for the specified host type.
    
    Parameters:
        host_type (str): Either "os" (self-managed) or "aos" (Amazon OpenSearch Service)
    
    Returns:
        OpenSearch: Configured OpenSearch client
    """
    opensearch_config = get_opensearch_config(host_type)
    
    logging.info(f"Connecting to OpenSearch: {opensearch_config.host_url}:{opensearch_config.port}")
    
    hosts = [{"host": opensearch_config.host_url, "port": opensearch_config.port}] if opensearch_config.port else opensearch_config.host_url
    client = OpenSearch(
        hosts=hosts,
        http_auth=(opensearch_config.username, opensearch_config.password),
        use_ssl=True,
        verify_certs=False,
        ssl_assert_hostname=False,
        ssl_show_warn=False,
        timeout=300
    )
    check_client_version(client)
    
    # AOS does not support local hosting of ML models, but local OpenSearch does
    # support it. So set the allow_registering_model_via_url and
    # only_run_on_ml_node settings to whether or not the host_type is "os"
    # (local OpenSearch).
    settings = {
        "plugins.ml_commons.memory_feature_enabled": True,
        "plugins.ml_commons.agent_framework_enabled": True,
        "plugins.ml_commons.rag_pipeline_feature_enabled": True,
        "plugins.ml_commons.trusted_connector_endpoints_regex": [
            # Bedrock runtime endpoints
            "^https://bedrock-runtime\\..*[a-z0-9-]\\.amazonaws\\.com/.*$",
            # SageMaker runtime endpoints  
            "^https://runtime\\.sagemaker\\..*[a-z0-9-]\\.amazonaws\\.com/.*$"
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
    Checks if the given version is at least the minimum version
    """
    info = client.info()
    version = info["version"]["number"]
    minimum_version = get_minimum_opensearch_version()
    if parse_version(version) < parse_version(minimum_version):
        raise ValueError(
            f"The minimum required version for opensearch cluster is {minimum_version}"
        )

