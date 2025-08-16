import logging
from typing import Dict
from opensearchpy import helpers, OpenSearch
from data_process import QAndAFileReader


SPACE_SEPARATOR = " "


def handle_index_creation(
    os_client: OpenSearch,
    config: Dict[str, str],
    delete_existing: bool = False,
):
    # First check if the index exists and delete it if the command line
    # parameter delete_existing is set. If it doesn't exist, log that and
    # skip deletion.
    index_name = config["index_name"]
    logging.info(f"Handling index creation for {index_name}")

    index_settings = config["index_settings"]
    import json

    logging.info(f"Index settings: {json.dumps(index_settings, indent=2)}")

    index_exists = os_client.indices.exists(index=index_name)
    logging.info(f"Index {index_name} exists: {index_exists}")
    logging.info(f"Delete existing index: {delete_existing}")

    if delete_existing:
        if index_exists:
            logging.info(f"Deleting index {index_name}")
            os_client.indices.delete(index=index_name)
        else:
            logging.info(f"Index {index_name} does not exist. Skipping deletion.")

    if os_client.indices.exists(index=index_name):
        logging.info(f'index "{index_name}" exists. Skipping index creation.')
        return

    try:
        response = os_client.indices.create(index_name, body=index_settings)
        logging.info(f"Create index response: {response}")
    except Exception as e:
        logging.error(f"Error creating index {index_name} due to exception: {e}")


def handle_data_loading(
    os_client: OpenSearch,
    pqa_reader: QAndAFileReader,
    config: Dict[str, str],
    no_load: bool = False,
    enriched: bool = True,
):
    """
    Handle data loading into OpenSearch index.

    Parameters:
        client (OpenSearch): OpenSearch client instance
        pqa_reader (QAndAFileReader): Reader for Amazon PQA dataset
        config (Dict[str, str]): Configuration dictionary containing index settings and categories

    Returns:
        None
    """
    if no_load:
        logging.info("Skipping data loading, per command line argument")
        return

    for category in config["categories"]:
        load_category(
            os_client=os_client,
            pqa_reader=pqa_reader,
            category=category,
            config=config,
            enriched=enriched,
        )


def load_category(os_client: OpenSearch, pqa_reader: QAndAFileReader, category, config, enriched=True):
    logging.info(f'Loading category "{category}"')
    docs = []
    number_of_docs = 0
    for doc in pqa_reader.questions_for_category(
        pqa_reader.amazon_pqa_category_name_to_constant(category),
        enriched=enriched
    ):
        doc["_index"] = config["index_name"]
        doc["_id"] = doc["question_id"]
        # TODO: Work on a much better way to handle the chunking
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
            send_bulk_ignore_exceptions(os_client, config, docs)
            docs = []
    if len(docs) > 0:
        logging.info(f'Finished reading "{category}" going to send remaining docs')
        send_bulk_ignore_exceptions(os_client, config, docs)


def send_bulk_ignore_exceptions(client: OpenSearch, config: Dict[str, str], docs):
    logging.info(f"Sending {config['bulk_send_chunk_size']} docs over the wire")
    try:
        status = helpers.bulk(
            client,
            docs,
            chunk_size=config["bulk_send_chunk_size"],
            request_timeout=300,
            max_retries=10,
            raise_on_error=False,
        )
        return status
    except Exception as e:
        logging.error(f"Error sending bulk: {e}")


def get_index_size(client: OpenSearch, index_name, unit="mb"):
    """Get the index size from the opensearch client"""
    if not client.indices.exists(index=index_name):
        return 0

    index_size = 0
    try:
        index_size = int(
            client.cat.indices(
                index=index_name, params={"bytes": f"{unit}", "h": "pri.store.size"}
            )
        )
    except Exception as e:
        logging.error(f"Error getting index size for {index_name}: {e}")
    return index_size
