import boto3
import json
import logging
import os
import sys
import time

import cmd_line_interface

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from client import OsMlClientWrapper, get_client, index_utils
from configs.configuration_manager import (
    get_raw_config_value,
    get_pipeline_field_map,
    get_config_for,
    get_base_mapping_path,
    get_qanda_file_reader_path,
)
from connectors.helper import get_remote_connector_configs
from mapping.helper import get_base_mapping
from models import get_ml_model
from data_process.amazon_pqa_dataset import AmazonPQADataset


logging.basicConfig(
    format="%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)


def create_combined_bedrock_role(role_name):
    """Create a combined IAM role with both create connector and invoke model permissions."""
    iam_client = boto3.client('iam', region_name=get_raw_config_value("AWS_REGION"))
    
    # Check if role already exists
    try:
        response = iam_client.get_role(RoleName=role_name)
        logging.info(f"Combined role {role_name} already exists")
        return response['Role']['Arn']
    except iam_client.exceptions.NoSuchEntityException:
        pass
    
    # Trust policy for both OpenSearch service and current user to assume the role
    current_user_arn = iam_client.get_user()['User']['Arn']
    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {
                    "Service": "es.amazonaws.com"
                },
                "Action": "sts:AssumeRole"
            },
            {
                "Effect": "Allow",
                "Principal": {
                    "AWS": current_user_arn
                },
                "Action": "sts:AssumeRole"
            }
        ]
    }
    
    # Combined policy with both create connector and invoke model permissions
    combined_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [
                    "bedrock:InvokeModel"
                ],
                "Resource": [
                    "arn:aws:bedrock:*::foundation-model/amazon.titan-embed-text-v1",
                    f"arn:aws:bedrock:*::foundation-model/{get_raw_config_value('BEDROCK_LLM_MODEL_NAME')}"
                ]
            }
        ]
    }
    
    # Create the role
    iam_client.create_role(
        RoleName=role_name,
        AssumeRolePolicyDocument=json.dumps(trust_policy),
        Description="Combined role for Bedrock connector creation and model invocation"
    )
    
    # Attach the inline policy
    iam_client.put_role_policy(
        RoleName=role_name,
        PolicyName="CombinedBedrockPolicy",
        PolicyDocument=json.dumps(combined_policy)
    )
    
    # Get the role ARN
    response = iam_client.get_role(RoleName=role_name)
    role_arn = response['Role']['Arn']
    
    logging.info(f"Created combined Bedrock role: {role_arn}")
    return role_arn



def fix_connector_region(client, connector_id, target_region, credential):
    """Fix connector URL to match the target region."""
    try:
        # Get current connector
        connector = client.os_client.transport.perform_request(
            "GET", f"/_plugins/_ml/connectors/{connector_id}"
        )
        
        # Check if URL needs fixing
        current_url = connector['actions'][0]['url']
        if target_region not in current_url:
            # Update URL to correct region
            new_url = f"https://bedrock-runtime.{target_region}.amazonaws.com/model/amazon.titan-embed-text-v1/invoke"


                # "credential": credential,
                # "actions": [{
                #     "action_type": "PREDICT",
                #     "method": "POST",
                #     "url": new_url,
                #     "headers": connector['actions'][0]['headers'],
                #     "request_body": connector['actions'][0]['request_body'],
                #     "pre_process_function": connector['actions'][0]['pre_process_function'],
                #     "post_process_function": connector['actions'][0]['post_process_function']
                # }]
            # Create update body with only the actions field
            
            logging.info(f'Credential for fixing connector: {credential}')
            update_body = {
                "credential": credential,
                "description": "updated connector"
            }
            
            # Update the connector
            client.os_client.transport.perform_request(
                "PUT", f"/_plugins/_ml/connectors/{connector_id}",
                body=update_body
            )
            logging.info(f"Fixed connector {connector_id} URL to use region {target_region}")
        else:
            logging.info(f"Connector {connector_id} already uses correct region {target_region}")
            
    except Exception as e:
        logging.warning(f"Could not fix connector region: {e}")


def get_role_arn(role_name):
    """Retrieve the ARN for a named IAM role."""
    iam_client = boto3.client('iam')
    role_response = iam_client.get_role(RoleName=role_name)
    return role_response['Role']['Arn']


def execute_workflow_and_monitor(client, workflow_name, body):
    """Execute workflow and monitor until completion."""
    logging.info("Creating workflow with body: %s", body)

    # Start workflow
    result = client.os_client.transport.perform_request(
        "POST", f"/_plugins/_flow_framework/workflow?use_case={workflow_name}&provision=true",
        body=body
    )

    workflow_id = result.get("workflow_id")
    if not workflow_id:
        raise ValueError("No workflow_id returned from workflow creation")
    
    logging.info(f"Workflow created with ID: {workflow_id}")

    # Monitor workflow status
    while True:
        status_result = client.os_client.transport.perform_request(
            "GET", f"/_plugins/_flow_framework/workflow/{workflow_id}/_status"
        )
        
        logging.info(80*"=")
        logging.info(f"Workflow status response: {status_result}")
        logging.info(80*"=")

        state = status_result.get("state", "UNKNOWN")
        logging.info(f"Workflow {workflow_id} state: {state}")

        if state == "COMPLETED":
            logging.info("Workflow completed successfully")
            
            # Extract all resource IDs by type from resources_created
            resources = status_result.get("resources_created", [])
            result_dict = {"workflow_id": workflow_id,
                           'state': status_result.get("state", "UNKNOWN")
                        }
            
            for resource in resources:
                resource_type = resource.get("resource_type")
                resource_id = resource.get("resource_id")
                if resource_type and resource_id:
                    result_dict[resource_type] = resource_id
            
            return result_dict
        elif state in ["FAILED", "TIMEOUT"]:
            raise RuntimeError(f"Workflow failed with state: {status_result}")

        time.sleep(1)


def build_workflow_query(query_text, model_id=None, **kwargs):
    """
    Build neural search query for dense vector search in workflow example.

    Parameters:
        query_text (str): The search query text
        model_id (str): ML model ID for generating embeddings
        **kwargs: Additional parameters (unused)

    Returns:
        dict: OpenSearch query dictionary
    """
    if not model_id:
        raise ValueError("Model ID must be provided for workflow search.")
    return {
        "size": 5,
        "query": {
            "neural": {
                "chunk_embedding": {
                    "query_text": query_text,
                    "model_id": model_id,
                    "k": 10,
                }
            }
        },
    }


def main():

    host_type = "aos"
    model_host = "bedrock"
    index_name = "semantic_search_workflow"
    embedding_type = "dense"
    pipeline_name = "semantic-search-workflow-pipeline"

    args = cmd_line_interface.get_command_line_args()

    if args.opensearch_type != "aos":
        logging.error(
            "This example is designed for Amazon OpenSearch Service (AOS) only."
        )
        sys.exit(1)

    config = get_config_for(host_type, provider=model_host, model_type='embedding')
    config.update({
        "with_knn": True,
        "pipeline_field_map": get_pipeline_field_map(),
        "index_name": index_name,
        "pipeline_name": pipeline_name,
        "embedding_type": embedding_type,
        "categories": args.categories,
        "delete_existing_index": args.delete_existing_index,
        "bulk_send_chunk_size": args.bulk_send_chunk_size,
    })

    client = OsMlClientWrapper(get_client(host_type, use_request_signing=True))
    
    if args.delete_existing_index and args.no_load:
        raise ValueError("Cannot use -d (delete) and --no_load for this example.")

    if args.no_load:
        model_name = f"{host_type}_{model_host}"
        model_config = get_remote_connector_configs(
            host_type=host_type, connector_type=model_host
        )
        model_config["model_name"] = model_name
        model_config["embedding_type"] = embedding_type
        config.update(model_config)

        ml_model = get_ml_model(
            host_type=host_type,
            model_host=model_host,
            model_config=model_config,
            os_client=client.os_client,
            ml_commons_client=client.ml_commons_client,
            model_group_id=client.ml_model_group.model_group_id(),
        )
        model_id = ml_model.model_id()
    else:
        logging.info("Skipping data loading, per command line argument")

        # Create a combined role with both create connector and invoke model permissions
        combined_role_name = "os_ml_qs_bedrock_combined_role"
        combined_role_arn = create_combined_bedrock_role(combined_role_name)
        logging.info(f"Using combined role ARN: {combined_role_arn}")
        
        region = get_raw_config_value("AWS_REGION")
        
        body = {
            "create_connector.credential.roleArn": combined_role_arn,
            "create_connector.region": region
        }
        
        result = execute_workflow_and_monitor(client, "bedrock_titan_embedding_model_deploy", body)
        
        logging.info(f'Deploy model workflow completed. {json.dumps(result, indent=2)}')
        
        # Fix connector region if needed
        connector_id = result.get("connector_id")
        credential = boto3.Session(region_name=region).get_credentials()
        credential = {
            "access_key": credential.access_key,
            "secret_key": credential.secret_key,
            "session_token": credential.token
        }
        if connector_id:
            fix_connector_region(client, connector_id, region, credential)
        connector_id = result.get("connector_id")
        model_id = result.get("model_id")
        if not connector_id or not model_id:
            raise ValueError("Failed to retrieve connector_id or model_id from workflow result.")

        logging.info(f'client: {client}, os_client: {client.os_client}')
        if client.os_client.indices.exists(index=index_name):
            if not args.delete_existing_index:
                raise ValueError(f"Index {index_name} already exists. Please use -d to delete it or choose a different name.")
            else:
                logging.info(f"Deleting existing index {index_name}")
                client.os_client.indices.delete(index=index_name)

        pipeline_field_map = get_pipeline_field_map()
        body = {
            "create_ingest_pipeline.pipeline_id": pipeline_name,
            "create_ingest_pipeline.description": "A text embedding pipeline",
            "create_ingest_pipeline.model_id": model_id,
            "text_embedding.field_map.input": list(pipeline_field_map.keys())[0],
            "text_embedding.field_map.output": list(pipeline_field_map.values())[0],
            "create_index.name": index_name,
            "create_index.settings.number_of_shards": "2",
            "create_index.mappings.method.engine": "lucene",
            "create_index.mappings.method.space_type": "l2",
            "create_index.mappings.method.name": "hnsw",
            "text_embedding.field_map.output.dimension": config["model"]["model_dimension"],    
        }
        result = execute_workflow_and_monitor(client, "semantic_search", body=body)
        logging.info(f'Semantic search index setup workflow completed. {json.dumps(result, indent=2)}')

        mapping = get_base_mapping(get_base_mapping_path())['mappings']
        try:
            client.os_client.indices.put_mapping(index=index_name, body=mapping)
            logging.info(f"Mapping updated for index {index_name}")
        except Exception as e:
            logging.error(f"Failed to update mapping for index {index_name}: {e}")
            raise

    # Load data using dataset abstraction
    if not getattr(args, "no_load", False):
        dataset = AmazonPQADataset(max_number_of_docs=args.number_of_docs_per_category)
        total_docs = dataset.load_data(
            os_client=client.os_client,
            index_name=config["index_name"],
            filter_criteria=args.categories,
            bulk_chunk_size=args.bulk_send_chunk_size
        )
        print(f"Loaded {total_docs} documents")

    logging.info("Setup complete! Starting interactive search interface...")

    # Start interactive search loop using the generic function
    cmd_line_interface.interactive_search_loop(
        client=client,
        index_name=index_name,
        model_info=model_id,
        model_id=model_id,
        query_builder_func=build_workflow_query,
        question=args.question,
    )


if __name__ == "__main__":
    main()
