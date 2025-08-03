# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

"""
Connector helper functions for retrieving remote connector configurations and creating outbound connectors.
"""

import time
import json
import boto3
import logging
import requests
from typing import Dict, Any, Optional
from requests_aws4auth import AWS4Auth
from requests.auth import HTTPBasicAuth
from configs.configuration_manager import get_raw_config_value, get_ml_base_uri


def get_remote_connector_configs(connector_type: str, host_type: str) -> Dict[str, str]:
    """Get remote connector configurations for the specified connector and host type."""
    if connector_type not in {"sagemaker", "bedrock"}:
        raise ValueError(f"connector_type must be either sagemaker or bedrock")
    if host_type not in {"os", "aos"}:
        raise ValueError(f"host_type must either be os or aos")

    if connector_type == "sagemaker" and host_type == "os":
        configs = {
            "access_key": get_raw_config_value("AWS_ACCESS_KEY_ID"),
            "secret_key": get_raw_config_value("AWS_SECRET_ACCESS_KEY"),
            "region": get_raw_config_value("AWS_REGION"),
            "connector_version": get_raw_config_value("SAGEMAKER_CONNECTOR_VERSION"),
            "sparse_url": get_raw_config_value("SAGEMAKER_SPARSE_URL"),
            "dense_url": get_raw_config_value("SAGEMAKER_DENSE_URL"),
            "model_dimensions": get_raw_config_value("SAGEMAKER_DENSE_MODEL_DIMENSION"),
        }
        # Validate that all required configs are present
        missing = [k for k, v in configs.items() if v is None or v == ""]
        if missing:
            raise ValueError(f"Missing required OS Sagemaker configurations: {missing}")
        return configs
    elif connector_type == "sagemaker" and host_type == "aos":
        configs = {
            "dense_arn": get_raw_config_value("SAGEMAKER_DENSE_ARN"),
            "sparse_arn": get_raw_config_value("SAGEMAKER_SPARSE_ARN"),
            "connector_role_name": get_raw_config_value("SAGEMAKER_CONNECTOR_ROLE_NAME"),
            "create_connector_role_name": get_raw_config_value("SAGEMAKER_CREATE_CONNECTOR_ROLE_NAME"),
            "region": get_raw_config_value("AWS_REGION"),
            "connector_version": get_raw_config_value("SAGEMAKER_CONNECTOR_VERSION"),
            "sparse_url": get_raw_config_value("SAGEMAKER_SPARSE_URL"),
            "dense_url": get_raw_config_value("SAGEMAKER_DENSE_URL"),
            "model_dimensions": get_raw_config_value("SAGEMAKER_DENSE_MODEL_DIMENSION"),
        }
        # Validate that all required configs are present
        missing = [k for k, v in configs.items() if v is None or v == ""]
        if missing:
            raise ValueError(f"Missing required AOS Sagemaker configurations: {missing}")
        return configs
    elif connector_type == "bedrock" and host_type == "os":
        configs = {
            "access_key": get_raw_config_value("AWS_ACCESS_KEY_ID"),
            "secret_key": get_raw_config_value("AWS_SECRET_ACCESS_KEY"),
            "region": get_raw_config_value("AWS_REGION"),
            "connector_version": get_raw_config_value("BEDROCK_CONNECTOR_VERSION"),
            "dense_url": get_raw_config_value("BEDROCK_EMBEDDING_URL"),
            "model_dimensions": get_raw_config_value("BEDROCK_MODEL_DIMENSION"),
        }
        # Validate that all required configs are present
        missing = [k for k, v in configs.items() if v is None or v == ""]
        if missing:
            raise ValueError(f"Missing required OS Bedrock configurations: {missing}")
        return configs
    else:  # bedrock and aos
        configs = {
            "dense_arn": get_raw_config_value("BEDROCK_ARN"),
            "connector_role_name": get_raw_config_value("BEDROCK_CONNECTOR_ROLE_NAME"),
            "create_connector_role_name": get_raw_config_value("BEDROCK_CREATE_CONNECTOR_ROLE_NAME"),
            "region": get_raw_config_value("AWS_REGION"),
            "connector_version": get_raw_config_value("BEDROCK_CONNECTOR_VERSION"),
            "model_dimensions": get_raw_config_value("BEDROCK_MODEL_DIMENSION"),
            "dense_url": get_raw_config_value("BEDROCK_EMBEDDING_URL"),
        }
        # Validate that all required configs are present
        missing = [k for k, v in configs.items() if v is None or v == ""]
        if missing:
            raise ValueError(f"Missing required AOS Bedrock configurations: {missing}")
        return configs


# =============================================================================
# Connector Creation Functions
# =============================================================================

def create_connector_with_iam_roles(
    opensearch_domain_url: str,
    opensearch_domain_arn: str,
    opensearch_username: str,
    opensearch_password: str,
    aws_user_name: str,
    region: str,
    connector_role_inline_policy: Dict[str, Any],
    connector_role_name: str,
    create_connector_role_name: str,
    connector_payload: Dict[str, Any],
    sleep_time_in_seconds: int = 10,
) -> str:
    """
    Create a connector using IAM roles for both OpenSearch access and outbound service access.
    
    This is used for AOS deployments where IAM roles are required to:
    1. Authenticate to OpenSearch to create the connector
    2. Allow the connector to authenticate to outbound services (Bedrock/SageMaker)
    
    Args:
        opensearch_domain_url: OpenSearch domain URL
        opensearch_domain_arn: OpenSearch domain ARN
        opensearch_username: OpenSearch username
        opensearch_password: OpenSearch password
        aws_user_name: AWS IAM user name
        region: AWS region
        connector_role_inline_policy: IAM policy for outbound service access
        connector_role_name: Name of IAM role for connector
        create_connector_role_name: Name of IAM role for creating connector
        connector_payload: Connector configuration payload
        sleep_time_in_seconds: Time to wait for IAM propagation
        
    Returns:
        str: Created connector ID
    """
    # Initialize AWS clients
    opensearch_client = boto3.client("es", region_name=region)
    iam_client = boto3.client("iam")
    sts_client = boto3.client("sts")
    
    # Step 1: Create IAM role configured in connector
    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": "es.amazonaws.com"},
                "Action": "sts:AssumeRole",
            }
        ],
    }

    logging.info("Step 1: Create IAM role configured in connector")
    if not _role_exists(iam_client, connector_role_name):
        connector_role_arn = _create_iam_role(
            iam_client, connector_role_name, trust_policy, connector_role_inline_policy
        )
    else:
        logging.info("Connector role exists, skip creating")
        connector_role_arn = _get_role_arn(iam_client, connector_role_name)

    # Step 2: Configure IAM role in OpenSearch
    # 2.1 Create IAM role for signing create connector request
    user_arn = _get_user_arn(iam_client, aws_user_name)
    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"AWS": user_arn},
                "Action": "sts:AssumeRole",
            }
        ],
    }

    inline_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": "iam:PassRole",
                "Resource": connector_role_arn,
            },
            {
                "Effect": "Allow",
                "Action": "es:ESHttpPost",
                "Resource": opensearch_domain_arn,
            },
        ],
    }

    logging.info("Step 2: Configure IAM role in OpenSearch")
    logging.info("Step 2.1: Create IAM role for signing create connector request")
    if not _role_exists(iam_client, create_connector_role_name):
        create_connector_role_arn = _create_iam_role(
            iam_client, create_connector_role_name, trust_policy, inline_policy
        )
    else:
        logging.info("Create connector role exists, skip creating")
        create_connector_role_arn = _get_role_arn(iam_client, create_connector_role_name)

    # 2.2 Map backend role
    logging.info(
        f"Step 2.2: Map IAM role {create_connector_role_name} to OpenSearch permission role"
    )
    _map_iam_role_to_backend_role(
        opensearch_domain_url, opensearch_username, opensearch_password, create_connector_role_arn
    )

    # Step 3: Create connector
    logging.info("Step 3: Create connector in OpenSearch")
    time.sleep(sleep_time_in_seconds)
    connector_payload["credential"] = {"roleArn": connector_role_arn}
    connector_id = _create_connector_with_role_auth(
        opensearch_domain_url, region, sts_client, create_connector_role_arn, connector_payload
    )
    
    return connector_id


def create_connector_with_basic_auth(
    os_client,
    connector_payload: Dict[str, Any],
) -> str:
    """
    Create a connector using basic auth for OpenSearch access.
    
    This is used for local OpenSearch deployments where:
    1. Basic auth (username/password) is used to authenticate to OpenSearch
    2. AWS credentials in the payload are used for outbound service access
    
    Args:
        os_client: OpenSearch client instance
        connector_payload: Connector configuration payload (should include credentials)
        
    Returns:
        str: Created connector ID
    """
    logging.info("Creating connector with basic auth to OpenSearch")
    logging.info(f"Connector payload: {json.dumps(connector_payload, indent=2)}")
    
    response = os_client.http.post(
        url=f"{get_ml_base_uri()}/connectors/_create",
        body=connector_payload,
    )
    
    connector_id = response["connector_id"]
    logging.info(f"Created connector with ID: {connector_id}")
    return connector_id


# =============================================================================
# Helper Functions (moved from AosConnectorHelper)
# =============================================================================

def _role_exists(iam_client, role_name: str) -> bool:
    """Check if an IAM role exists."""
    try:
        iam_client.get_role(RoleName=role_name)
        return True
    except iam_client.exceptions.NoSuchEntityException:
        return False


def _create_iam_role(iam_client, role_name: str, trust_policy_json: Dict, inline_policy_json: Dict) -> str:
    """Create an IAM role with trust and inline policies."""
    try:
        # Create the role with the trust policy
        create_role_response = iam_client.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(trust_policy_json),
            Description="Role with custom trust and inline policies",
        )

        # Get the ARN of the newly created role
        role_arn = create_role_response["Role"]["Arn"]

        # Attach the inline policy to the role
        iam_client.put_role_policy(
            RoleName=role_name,
            PolicyName="InlinePolicy",
            PolicyDocument=json.dumps(inline_policy_json),
        )

        logging.info(f"Created role: {role_name}")
        return role_arn

    except Exception as e:
        logging.error(f"Error creating the role: {e}")
        raise


def _get_role_arn(iam_client, role_name: str) -> str:
    """Get the ARN of an IAM role."""
    try:
        response = iam_client.get_role(RoleName=role_name)
        return response["Role"]["Arn"]
    except iam_client.exceptions.NoSuchEntityException:
        logging.error(f"The requested role {role_name} does not exist")
        raise
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise


def _get_user_arn(iam_client, username: str) -> str:
    """Get the ARN of an IAM user."""
    try:
        response = iam_client.get_user(UserName=username)
        user_arn = response["User"]["Arn"]
        return user_arn
    except iam_client.exceptions.NoSuchEntityException:
        logging.error(f"IAM user '{username}' not found.")
        raise


def _map_iam_role_to_backend_role(
    opensearch_domain_url: str,
    opensearch_username: str, 
    opensearch_password: str,
    role_arn: str,
    os_security_role: str = "ml_full_access"
) -> None:
    """Map an IAM role to an OpenSearch backend role."""
    url = f"{opensearch_domain_url}/_plugins/_security/api/rolesmapping/{os_security_role}"
    r = requests.get(
        url,
        auth=HTTPBasicAuth(opensearch_username, opensearch_password),
    )
    role_mapping = json.loads(r.text)
    headers = {"Content-Type": "application/json"}
    
    if "status" in role_mapping and role_mapping["status"] == "NOT_FOUND":
        data = {"backend_roles": [role_arn]}
        response = requests.put(
            url,
            headers=headers,
            data=json.dumps(data),
            auth=HTTPBasicAuth(opensearch_username, opensearch_password),
        )
    else:
        role_mapping = role_mapping[os_security_role]
        role_mapping["backend_roles"].append(role_arn)
        data = [
            {
                "op": "replace",
                "path": "/backend_roles",
                "value": list(set(role_mapping["backend_roles"])),
            }
        ]
        response = requests.patch(
            url,
            headers=headers,
            data=json.dumps(data),
            auth=HTTPBasicAuth(opensearch_username, opensearch_password),
        )


def _create_connector_with_role_auth(
    opensearch_domain_url: str,
    region: str,
    sts_client,
    create_connector_role_arn: str,
    payload: Dict[str, Any],
    role_session_name: str = "connector_creation_session"
) -> str:
    """Create a connector using assumed role authentication."""
    # Assume role to get temporary credentials
    assumed_role_object = sts_client.assume_role(
        RoleArn=create_connector_role_arn,
        RoleSessionName=role_session_name,
    )
    temp_credentials = assumed_role_object["Credentials"]
    
    # Create AWS4Auth with temporary credentials
    awsauth = AWS4Auth(
        temp_credentials["AccessKeyId"],
        temp_credentials["SecretAccessKey"],
        region,
        "es",
        session_token=temp_credentials["SessionToken"],
    )

    path = "/_plugins/_ml/connectors/_create"
    url = opensearch_domain_url + path
    headers = {"Content-Type": "application/json"}

    logging.info(f"Creating connector with payload: {json.dumps(payload, indent=2)}")
    r = requests.post(url, auth=awsauth, json=payload, headers=headers)
    
    # Debug: Log the response
    logging.info(f"Connector creation response status: {r.status_code}")
    logging.info(f"Connector creation response text: {r.text}")
    
    try:
        response_json = json.loads(r.text)
        logging.info(f"Connector creation response JSON keys: {list(response_json.keys())}")
        
        if "connector_id" in response_json:
            connector_id = response_json["connector_id"]
            logging.info(f"Successfully extracted connector_id: {connector_id}")
            return connector_id
        else:
            logging.error(f"connector_id not found in response. Available keys: {list(response_json.keys())}")
            logging.error(f"Full response: {response_json}")
            raise KeyError("connector_id not found in response")
            
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse JSON response: {e}")
        logging.error(f"Raw response: {r.text}")
        raise
