import boto3
import json
from opensearchpy import OpenSearch
from opensearchpy.helpers import bulk
import os
import time
import uuid


OPENSEARCH_HOST = os.environ.get("OPENSEARCH_HOST", "localhost")
OPENSEARCH_PORT = os.environ.get("OPENSEARCH_PORT", 9200)
OPENSEARCH_AUTH = (
    os.environ.get("OPENSEARCH_ADMIN_USER", "admin"),
    os.environ.get("OPENSEARCH_ADMIN_PASSWORD", ""),
)
REGION = os.environ.get("AWS_REGION", "us-west-2")


# Step 0 set up
os_client = OpenSearch(
    hosts=[{"host": OPENSEARCH_HOST, "port": OPENSEARCH_PORT}],
    http_auth=OPENSEARCH_AUTH,
    use_ssl=True,
    verify_certs=False,
    ssl_assert_hostname=False,
    ssl_show_warn=False,
)
os_client.cluster.put_settings(
    body={
        "persistent": {
            "plugins.ml_commons.memory_feature_enabled": True,
            "plugins.ml_commons.rag_pipeline_feature_enabled": True,
            "plugins.ml_commons.trusted_connector_endpoints_regex": [
                "^https://bedrock-runtime\\..*[a-z0-9-]\\.amazonaws\\.com/.*$"
            ],
        }
    }
)
session = boto3.client("sts", REGION).get_session_token()


## Step 1: Create a connector to a model
response = os_client.transport.perform_request(
    "POST",
    "/_plugins/_ml/connectors/_create",
    body={
        "name": "Amazon Bedrock",
        "description": "Test connector for Amazon Bedrock",
        "version": 1,
        "protocol": "aws_sigv4",
        "credential": {
            "access_key": session["Credentials"]["AccessKeyId"],
            "secret_key": session["Credentials"]["SecretAccessKey"],
            "session_token": session["Credentials"]["SessionToken"],
        },
        "parameters": {
            "region": f"{REGION}",
            "service_name": "bedrock",
            "model": "anthropic.claude-v2",
        },
        "actions": [
            {
                "action_type": "predict",
                "method": "POST",
                "headers": {"content-type": "application/json"},
                "url": "https://bedrock-runtime.${parameters.region}.amazonaws.com/model/${parameters.model}/invoke",
                "request_body": '{"prompt":"\\n\\nHuman: ${parameters.inputs}\\n\\nAssistant:","max_tokens_to_sample":300,"temperature":0.5,"top_k":250,"top_p":1,"stop_sequences":["\\\\n\\\\nHuman:"]}',
            }
        ],
    },
)
connector_id = response["connector_id"]


## Step 2: Register and deploy the model
response = os_client.transport.perform_request(
    "POST",
    "/_plugins/_ml/models/_register",
    body={
        "name": "Amazon Bedrock",
        "function_name": "remote",
        "description": "bedrock",
        "connector_id": f"{connector_id}",
    },
)
task_id = response["task_id"]
status = response["status"]
while status != "COMPLETED":
    response = os_client.transport.perform_request(
        "GET", f"/_plugins/_ml/tasks/{task_id}"
    )
    status = response["state"]
    print(f"Task status: {status}")
model_id = response["model_id"]
print(f"Model ID: {model_id}")


## Deploy the model
response = os_client.transport.perform_request(
    "POST", f"/_plugins/_ml/models/{model_id}/_deploy", body=""
)


## Step 3: Create a search pipeline
response = os_client.transport.perform_request(
    "PUT",
    f"/_search/pipeline/rag_pipeline",
    body={
        "response_processors": [
            {
                "retrieval_augmented_generation": {
                    "tag": "conversation demo",
                    "description": "Demo pipeline Using Bedrock Connector",
                    "model_id": f"{model_id}",
                    "context_field_list": ["aggregated_answers"],
                    "system_prompt": "You are a helpful assistant",
                    "user_instructions": "Generate a concise and informative answer in less than 100 words for the given question",
                }
            }
        ]
    },
)


## Step 4: Ingest RAG data into an index
os_client.indices.delete(index="converse", ignore=[400, 404])
os_client.indices.create(
    index="converse",
    body={
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 1,
            "index.search.default_pipeline": "rag_pipeline",
        },
        "mappings": {
            "properties": {
                "question_id": {"type": "keyword"},
                "title": {"type": "text"},
                "question_text": {"type": "text"},
                "asin": {"type": "keyword"},
                "bullet_point1": {"type": "text"},
                "bullet_point2": {"type": "text"},
                "bullet_point3": {"type": "text"},
                "bullet_point4": {"type": "text"},
                "bullet_point5": {"type": "text"},
                "product_description": {"type": "text"},
                "brand_name": {"type": "keyword"},
                "item_name": {"type": "text"},
                "question_type": {"type": "keyword"},
                "answer_aggregated": {"type": "keyword"},
                "answers": {"properties": {"answer_text": {"type": "text"}}},
            }
        },
    },
)


rag_sources = [
    "/Users/handler/datasets/amazon_pqa/amazon_pqa_earbud_headphones.json",
    "/Users/handler/datasets/amazon_pqa/amazon_pqa_headsets.json",
    "/Users/handler/datasets/amazon_pqa/amazon_pqa_diffusers.json",
    "/Users/handler/datasets/amazon_pqa/amazon_pqa_mattresses.json",
    "/Users/handler/datasets/amazon_pqa/amazon_pqa_mp3_&_mp4_players.json",
    "/Users/handler/datasets/amazon_pqa/amazon_pqa_sheet_&_pillowcase_sets.json",
    "/Users/handler/datasets/amazon_pqa/amazon_pqa_batteries.json",
    "/Users/handler/datasets/amazon_pqa/amazon_pqa_casual.json",
    "/Users/handler/datasets/amazon_pqa/amazon_pqa_costumes.json",
]

## Load data into OpenSearch
for rag_source in rag_sources:
    print(f"Processing {rag_source}")
    with open(rag_source, "r") as f:
        nline = 0
        buffer = []
        for line in f:
            data = json.loads(line)
            data["aggregated_answers"] = ""
            for answer in data["answers"]:
                data["aggregated_answers"] += answer["answer_text"] + "\n"
            buffer.append({"_op_type": "create", "_index": "converse", "_source": data})
            nline += 1
            if nline % 5000 == 0:
                print(nline, " lines processed")
                bulk(os_client, buffer)
                buffer = []
            if nline > 20000:
                break


## RAG pipeline

## Step 5: Create a conversation memory
time.sleep(1)
conversation_name = f"conversation-{str(uuid.uuid1())[:8]}"
response = os_client.transport.perform_request(
    "POST", "/_plugins/_ml/memory/", body={"name": conversation_name}
)
memory_id = response["memory_id"]
print(f"Memory ID: {memory_id}")

##Step 6: Use the pipeline for RAG
time.sleep(1)
while True:
    question = input("Enter your question (or 'q' to quit): ")
    question = question.strip()
    if question.lower() == "q":
        break
    response = os_client.search(
        index="converse",
        body={
            "query": {
                "simple_query_string": {"query": question, "fields": ["question_text"]}
            },
            "ext": {
                "generative_qa_parameters": {
                    "llm_model": "bedrock/claude",
                    "llm_question": question,
                    "memory_id": f"{memory_id}",
                    "context_size": 5,
                    "message_size": 5,
                    "timeout": 30,
                }
            },
        },
    )

    print(json.dumps(response, indent=4))
    print()
    print(f"These are the questions retrieved by the lexical query: {question}")
    for hit in response["hits"]["hits"]:
        print(f"{hit['_source']['item_name']}:\n\t{hit['_source']['question_text']}")
    print()
    print()
    print(response["ext"]["retrieval_augmented_generation"]["answer"])
