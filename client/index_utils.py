import logging
from typing import Dict
from opensearchpy import helpers, OpenSearch
from data_process import QAndAFileReader


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


def load_category(client: OpenSearch, pqa_reader: QAndAFileReader, category, config):
    SPACE_SEPARATOR = " "
    logging.info(f'Loading category "{category}"')
    docs = []
    number_of_docs = 0
    for doc in pqa_reader.questions_for_category(
        pqa_reader.amazon_pqa_category_name_to_constant(category), enriched=True
    ):
        doc["_index"] = config["index_name"]
        doc["_id"] = doc["question_id"]
        doc["chunk"] = SPACE_SEPARATOR.join(
            [doc["product_description"], doc["brand_name"], doc["item_name"]]
        )
        # limit the document token count to 500 tokens from embedding models
        doc["chunk"] = SPACE_SEPARATOR.join(doc["chunk"].split()[:500])
        # documents less than 4 words are meaningless
        if len(doc["chunk"]) <= 4:
            logging.info(f"Empty chunk for {doc}")
            continue
        docs.append(doc)
        number_of_docs += 1
        if number_of_docs % 2000 == 0:
            logging.info(f"Sending {number_of_docs} docs")
            send_bulk_ignore_exceptions(client, config, docs)
            docs = []
    if len(docs) > 0:
        logging.info(f'Category "{category}" complete. Sending {number_of_docs} docs')
        send_bulk_ignore_exceptions(client, config, docs)


def get_index_size(client: OpenSearch, index_name, unit="mb"):
    """Get the index size from the opensearch client"""
    if not client.indices.exists(index=index_name):
        return 0
    return int(
        client.cat.indices(
            index=index_name, params={"bytes": f"{unit}", "h": "pri.store.size"}
        )
    )


def delete_index(client: OpenSearch, index_name: str):
    """Delete the index if it exists"""
    if client.indices.exists(index=index_name):
        logging.info(f"Deleting index {index_name}")
        client.indices.delete(index=index_name)
    else:
        logging.info(f"Index {index_name} does not exist. Skipping deletion.")


def idempotent_create_index(os_client: OpenSearch, index_name="", settings=None):
    """
    Create the index with settings.
    """
    if not index_name:
        raise ValueError("idempotent_create_index: index name must be specified")
    if not settings:
        raise ValueError("idempotent_create_index: settings must be specified")
    try:
        response = os_client.indices.create(index_name, body=settings)
        logging.info(
            f"idempotent_create_index response: {response}",
        )
    except Exception as e:
        logging.error(f"Error creating index {index_name} due to exception: {e}")
