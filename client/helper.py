# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

import boto3
import logging
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.configuration_manager import (
    get_opensearch_config,
    get_minimum_opensearch_version,
    get_raw_config_value,
)


def get_client(host_type: str, use_request_signing=False) -> OpenSearch:
    """
    Create and configure an OpenSearch client for the specified host type.
    
    Parameters:
        host_type (str): Either "os" (self-managed) or "aos" (Amazon OpenSearch Service)
        use_request_signing (bool): Whether to use AWS Sig V4 request signing (default: False)
    
    Returns:
        OpenSearch: Configured OpenSearch client
    """
    opensearch_config = get_opensearch_config(host_type)
    
    logging.info(f"Connecting to OpenSearch: {opensearch_config.host_url}:{opensearch_config.port}")

    hosts = opensearch_config.host_url
    if opensearch_config.port:
        hosts = [{
            "host": opensearch_config.host_url,
            "port": opensearch_config.port
        }]

    if use_request_signing:
        logging.info("Using AWS Sig V4 request signing for authentication\n\n")
        region = get_raw_config_value("AWS_REGION")
        access_key = get_raw_config_value("AWS_ACCESS_KEY_ID")
        secret_key = get_raw_config_value("AWS_SECRET_ACCESS_KEY")
        
        # Create credentials from config values instead of using boto3 default chain
        from botocore.credentials import Credentials
        credentials = Credentials(access_key, secret_key)
        logging.info(f"Using AWS credentials: {credentials}")
        http_auth = AWSV4SignerAuth(credentials, region)
        client = OpenSearch(
            hosts=hosts,
            http_auth=http_auth,
            use_ssl=True,
            verify_certs=False,
            ssl_assert_hostname=False,
            ssl_show_warn=False,
            connection_class=RequestsHttpConnection,
            timeout=300
        )
        try:
            sts_client = boto3.client('sts')
            caller_identity = sts_client.get_caller_identity()
            iam_user_arn = caller_identity['Arn']
            verify_iam_user_permissions(client, iam_user_arn)
        except Exception as e:
            logging.warning(f"Could not verify IAM user permissions: {e}")

    else:
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


def verify_iam_user_permissions(client: OpenSearch, iam_user_arn: str):
    """
    Verify that the IAM user has proper OpenSearch permissions.
    First checks if user can access security API, then validates index permissions.
    """
    try:
        # Test security API access
        response = client.transport.perform_request("GET", "/_plugins/_security/api/rolesmapping")
    except Exception as e:
        if "403" in str(e) or "Forbidden" in str(e):
            raise ValueError(f"IAM user {iam_user_arn} cannot access OpenSearch security API. "
                             f"User is likely not mapped to any roles or lacks security permissions.")
        raise ValueError(f"Failed to access OpenSearch security API: {e}")
    
    # Find user's roles
    user_roles = []
    for role_name, role_data in response.items():
        users = role_data.get("users", [])
        if iam_user_arn in users:
            user_roles.append(role_name)
    
    if not user_roles:
        raise ValueError(f"IAM user {iam_user_arn} is not mapped to any OpenSearch roles")
    
    # Check role permissions
    try:
        roles_response = client.transport.perform_request("GET", "/_plugins/_security/api/roles")
        
        has_index_permissions = False
        for role_name in user_roles:
            role_data = roles_response.get(role_name, {})
            index_permissions = role_data.get("index_permissions", [])
            
            for perm in index_permissions:
                allowed_actions = perm.get("allowed_actions", [])
                if any(action in allowed_actions for action in ["*", "indices:*", "indices:admin/create", "indices:data/write/index"]):
                    has_index_permissions = True
                    break
        
        if not has_index_permissions:
            raise ValueError(f"IAM user {iam_user_arn} lacks required index permissions (create, read, write)")
        
        logging.info(f"IAM user {iam_user_arn} has proper permissions via roles: {user_roles}")
        
    except Exception as e:
        if "lacks required index permissions" in str(e):
            raise
        logging.warning(f"Could not verify detailed permissions: {e}")


def parse_version(version: str):
    return tuple(map(int, version.split(".")))


def check_client_version(client: OpenSearch):
    """
    Checks if the given version is at least the minimum version
    """
    try:
        info = client.info(request_timeout=5)  # Add 5 second timeout
    except Exception as e:
        # Check for specific backend role authentication issue
        error_text = f"{str(e)} {getattr(e, 'info', '')}"
        if "backend_roles=[]" in error_text and "no permissions for [cluster:monitor/main]" in error_text:
            raise ValueError(
                f"Authentication failed: IAM user has empty backend_roles. "
                f"To fix this issue, add the IAM user ARN directly as a user in OpenSearch Security, "
                f"or configure the IAM user to assume a role that is mapped as a backend role. "
                f"Original error: {e}"
            )
        logging.error(f"Failed to check OpenSearch version: {e}")
        raise
    
    version = info["version"]["number"]
    minimum_version = get_minimum_opensearch_version()
    if parse_version(version) < parse_version(minimum_version):
        raise ValueError(
            f"The minimum required version for opensearch cluster is {minimum_version}"
        )
