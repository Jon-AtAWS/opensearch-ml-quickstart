import logging
from typing import Dict
from opensearchpy import helpers, OpenSearch
from data_process.base_dataset import BaseDataset


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
    logging.info(10*"=" + f"Handling index creation for {index_name}")

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
        response = os_client.indices.create(index=index_name, body=index_settings)
        logging.info(f"Create index response: {response}")
    except Exception as e:
        logging.error(f"Error creating index {index_name} due to exception: {e}")
    logging.info(10*"=" + f"Index {index_name} created successfully.")


def handle_data_loading(
    os_client: OpenSearch,
    dataset: BaseDataset,
    config: Dict[str, str],
    no_load: bool = False,
):
    """
    Handle data loading coordination using dataset's load_data method.

    Parameters:
        os_client (OpenSearch): OpenSearch client instance
        dataset (BaseDataset): Dataset reader instance
        config (Dict[str, str]): Configuration dictionary containing index settings

    Returns:
        None
    """
    if no_load:
        logging.info("Skipping data loading, per command line argument")
        return

    filter_criteria = config.get("categories")  # Will be None for datasets without categories
    bulk_chunk_size = config.get("bulk_send_chunk_size", 100)
    index_name = config["index_name"]
    
    total_docs = dataset.load_data(
        os_client=os_client,
        index_name=index_name,
        filter_criteria=filter_criteria,
        bulk_chunk_size=bulk_chunk_size
    )
    
    logging.info(f"Successfully loaded {total_docs} documents into index {index_name}")


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
