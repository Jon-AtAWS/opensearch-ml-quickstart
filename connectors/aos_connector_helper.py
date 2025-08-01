# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

import time
import json
import boto3
import logging
import requests
from requests_aws4auth import AWS4Auth
from requests.auth import HTTPBasicAuth


# This Python code is compatible with AWS OpenSearch versions 2.13 and higher.
class AosConnectorHelper:
    def __init__(
        self,
        region,
        opensearch_domain_name,
        opensearch_domain_username,
        opensearch_domain_password,
        aws_user_name,
    ):
        # singleton boto3 clients for opensearch, iam and sts
        self.opensearch_client = boto3.client("es", region_name=region)
        self.iam_client = boto3.client("iam")
        self.sts_client = boto3.client("sts")
        self.region = region
        (
            self.opensearch_domain_url,
            self.opensearch_domain_arn,
        ) = self.get_opensearch_domain_info(opensearch_domain_name)
        self.opensearch_domain_username = opensearch_domain_username
        self.opensearch_domain_opensearch_domain_password = opensearch_domain_password
        self.aws_user_name = aws_user_name

    def get_opensearch_domain_info(self, domain_name):
        try:
            # Describe the domain to get its information
            response = self.opensearch_client.describe_elasticsearch_domain(
                DomainName=domain_name
            )
            domain_info = response["DomainStatus"]

            # Extract the domain URL and ARN
            domain_url = domain_info["Endpoint"]
            domain_arn = domain_info["ARN"]

            return f"https://{domain_url}", domain_arn

        except self.opensearch_client.exceptions.ResourceNotFoundException:
            logging.error(f"Domain '{domain_name}' not found.")
            return None, None

    def get_user_arn(self, username):
        try:
            # Get information about the IAM user
            response = self.iam_client.get_user(UserName=username)
            user_arn = response["User"]["Arn"]
            return user_arn
        except self.iam_client.exceptions.NoSuchEntityException:
            logging.error(f"IAM user '{username}' not found.")
            return None

    def role_exists(self, role_name):
        try:
            self.iam_client.get_role(RoleName=role_name)
            return True
        except self.iam_client.exceptions.NoSuchEntityException:
            return False

    def create_iam_role(self, role_name, trust_policy_json, inline_policy_json):
        try:
            # Create the role with the trust policy
            create_role_response = self.iam_client.create_role(
                RoleName=role_name,
                AssumeRolePolicyDocument=json.dumps(trust_policy_json),
                Description="Role with custom trust and inline policies",
            )

            # Get the ARN of the newly created role
            role_arn = create_role_response["Role"]["Arn"]

            # Attach the inline policy to the role
            self.iam_client.put_role_policy(
                RoleName=role_name,
                PolicyName="InlinePolicy",  # you can replace this with your preferred policy name
                PolicyDocument=json.dumps(inline_policy_json),
            )

            logging.info(f"Created role: {role_name}")
            return role_arn

        except Exception as e:
            logging.error(f"Error creating the role: {e}")
            return None

    def get_role_arn(self, role_name):
        try:
            response = self.iam_client.get_role(RoleName=role_name)
            # Return ARN of the role
            return response["Role"]["Arn"]
        except self.iam_client.exceptions.NoSuchEntityException:
            logging.error(f"The requested role {role_name} does not exist")
            return None
        except Exception as e:
            logging.error(f"An error occurred: {e}")
            return None

    def map_iam_role_to_backend_role(self, role_arn, os_security_role="ml_full_access"):
        url = f"{self.opensearch_domain_url}/_plugins/_security/api/rolesmapping/{os_security_role}"
        r = requests.get(
            url,
            auth=HTTPBasicAuth(
                self.opensearch_domain_username,
                self.opensearch_domain_opensearch_domain_password,
            ),
        )
        role_mapping = json.loads(r.text)
        headers = {"Content-Type": "application/json"}
        if "status" in role_mapping and role_mapping["status"] == "NOT_FOUND":
            data = {"backend_roles": [role_arn]}
            response = requests.put(
                url,
                headers=headers,
                data=json.dumps(data),
                auth=HTTPBasicAuth(
                    self.opensearch_domain_username,
                    self.opensearch_domain_opensearch_domain_password,
                ),
            )
            # print(response.text)
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
                auth=HTTPBasicAuth(
                    self.opensearch_domain_username,
                    self.opensearch_domain_opensearch_domain_password,
                ),
            )
            # print(response.text)

    def assume_role(
        self, create_connector_role_arn, role_session_name="your_session_name"
    ):
        # role_arn = f"arn:aws:iam::{aws_account_id}:role/{role_name}"
        assumed_role_object = self.sts_client.assume_role(
            RoleArn=create_connector_role_arn,
            RoleSessionName=role_session_name,
        )

        # Obtain the temporary credentials from the assumed role
        temp_credentials = assumed_role_object["Credentials"]

        return temp_credentials

    def create_connector(self, create_connector_role_name, payload):
        create_connector_role_arn = self.get_role_arn(create_connector_role_name)
        temp_credentials = self.assume_role(create_connector_role_arn)
        awsauth = AWS4Auth(
            temp_credentials["AccessKeyId"],
            temp_credentials["SecretAccessKey"],
            self.region,
            "es",
            session_token=temp_credentials["SessionToken"],
        )

        path = "/_plugins/_ml/connectors/_create"
        url = self.opensearch_domain_url + path

        headers = {"Content-Type": "application/json"}

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

    def create_connector_with_role(
        self,
        connector_role_inline_policy,
        connector_role_name,
        create_connector_role_name,
        create_connector_payload,
        sleep_time_in_seconds=10,
    ):
        # Step1: Create IAM role configued in connector
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

        logging.info("Step1: Create IAM role configued in connector")
        if not self.role_exists(connector_role_name):
            connector_role_arn = self.create_iam_role(
                connector_role_name, trust_policy, connector_role_inline_policy
            )
        else:
            logging.info("role exists, skip creating")
            connector_role_arn = self.get_role_arn(connector_role_name)
        # print(connector_role_arn)
        logging.info("----------")

        # Step 2: Configure IAM role in OpenSearch
        # 2.1 Create IAM role for Signing create connector request
        user_arn = self.get_user_arn(self.aws_user_name)
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
                    "Resource": self.opensearch_domain_arn,
                },
            ],
        }

        logging.info("Step 2: Configure IAM role in OpenSearch")
        logging.info("Step 2.1: Create IAM role for Signing create connector request")
        if not self.role_exists(create_connector_role_name):
            create_connector_role_arn = self.create_iam_role(
                create_connector_role_name, trust_policy, inline_policy
            )
        else:
            logging.info("role exists, skip creating")
            create_connector_role_arn = self.get_role_arn(create_connector_role_name)
        # print(create_connector_role_arn)
        logging.info("----------")

        # 2.2 Map backend role
        logging.info(
            f"Step 2.2: Map IAM role {create_connector_role_name} to OpenSearch permission role"
        )
        self.map_iam_role_to_backend_role(create_connector_role_arn)
        logging.info("----------")

        # 3. Create connector
        logging.info("Step 3: Create connector in OpenSearch")
        # When you create an IAM role, it can take some time for the changes to propagate across AWS systems.
        # During this time, some services might not immediately recognize the new role or its permissions.
        # So we wait for some time before creating connector.
        # If you see such error: ClientError: An error occurred (AccessDenied) when calling the AssumeRole operation
        # you can rerun this function.

        # Wait for some time
        time.sleep(sleep_time_in_seconds)
        payload = create_connector_payload
        payload["credential"] = {"roleArn": connector_role_arn}
        connector_id = self.create_connector(create_connector_role_name, payload)
        # print(connector_id)
        logging.info("----------")
        return connector_id
