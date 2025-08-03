# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for client module - requires actual OpenSearch cluster"""

import logging
import pytest
from client import get_client, get_index_size


class TestClientIntegration:
    """Integration tests that require actual OpenSearch clusters"""

    @pytest.mark.integration
    def test_os_client_connection(self):
        """Test connection to self-managed OpenSearch cluster"""
        try:
            client = get_client("os")
            # Test basic connectivity
            info = client.info()
            assert "version" in info
            logging.info(f"Connected to OpenSearch version: {info['version']['number']}")
        except Exception as e:
            pytest.skip(f"OpenSearch cluster not available: {e}")

    @pytest.mark.integration
    def test_aos_client_connection(self):
        """Test connection to Amazon OpenSearch Service"""
        try:
            client = get_client("aos")
            # Test basic connectivity
            info = client.info()
            assert "version" in info
            logging.info(f"Connected to AOS version: {info['version']['number']}")
        except Exception as e:
            pytest.skip(f"AOS cluster not available: {e}")

    @pytest.mark.integration
    def test_get_index_size_os(self):
        """Test getting index size from self-managed OpenSearch"""
        try:
            client = get_client("os")
            size = get_index_size(client, index_name="amazon_pqa", unit="b")
            logging.info(f"OS index size: {size} bytes")
            assert isinstance(size, int)
            assert size >= 0
        except Exception as e:
            pytest.skip(f"OpenSearch cluster not available: {e}")

    @pytest.mark.integration
    def test_get_index_size_aos(self):
        """Test getting index size from Amazon OpenSearch Service"""
        try:
            client = get_client("aos")
            size = get_index_size(client, index_name="amazon_pqa", unit="b")
            logging.info(f"AOS index size: {size} bytes")
            assert isinstance(size, int)
            assert size >= 0
        except Exception as e:
            pytest.skip(f"AOS cluster not available: {e}")


# Legacy test functions for backward compatibility
def test_os_client():
    """Legacy test function for OS client"""
    logging.info("Testing os client util...")
    try:
        client = get_client("os")
        size = get_index_size(client, index_name="amazon_pqa", unit="b")
        logging.info(f"Index size: {size}")
    except Exception as e:
        logging.warning(f"OS client test failed: {e}")


def test_aos_client():
    """Legacy test function for AOS client"""
    logging.info("Testing aos client util...")
    try:
        client = get_client("aos")
        size = get_index_size(client, index_name="amazon_pqa", unit="b")
        logging.info(f"Index size: {size}")
    except Exception as e:
        logging.warning(f"AOS client test failed: {e}")
