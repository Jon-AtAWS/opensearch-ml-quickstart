# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for models module - requires actual OpenSearch cluster"""

import pytest
import logging
from unittest.mock import patch, Mock

# Note: These integration tests have been disabled because they require
# actual OpenSearch cluster connections with valid credentials.
# For CI/CD environments, use unit tests with mocks instead.

class TestModelsIntegration:
    """Integration tests that require actual OpenSearch clusters - DISABLED"""

    @pytest.mark.skip(reason="Requires actual OpenSearch cluster with valid credentials")
    def test_ml_model_group_creation_and_cleanup(self):
        """Test MlModelGroup creation and cleanup with actual cluster - DISABLED"""
        pass

    @pytest.mark.skip(reason="Requires actual OpenSearch cluster with valid credentials")
    def test_local_ml_model_creation_and_cleanup(self):
        """Test LocalMlModel creation and cleanup with actual cluster - DISABLED"""
        pass

    @pytest.mark.skip(reason="Requires actual OpenSearch cluster with valid credentials")
    def test_get_ml_model_helper_local(self):
        """Test get_ml_model helper function with local model - DISABLED"""
        pass

    @pytest.mark.skip(reason="Requires actual OpenSearch cluster with valid credentials")
    def test_model_search_functionality(self):
        """Test model search functionality - DISABLED"""
        pass

    @pytest.mark.skip(reason="Requires actual OpenSearch cluster with valid credentials")
    def test_model_group_search_functionality(self):
        """Test model group search functionality - DISABLED"""
        pass


class TestModelsLegacyCompatibility:
    """Legacy test functions for backward compatibility - DISABLED"""

    @pytest.mark.skip(reason="Requires actual OpenSearch cluster with valid credentials")
    def test_ml_model_group_legacy(self):
        """Legacy test function for ML model group - DISABLED"""
        pass

    @pytest.mark.skip(reason="Requires actual OpenSearch cluster with valid credentials")
    def test_os_local_ml_model_legacy(self):
        """Legacy test function for OS local ML model - DISABLED"""
        pass


# Legacy test functions for backward compatibility - DISABLED
@pytest.mark.skip(reason="Requires actual OpenSearch cluster with valid credentials")
def test_ml_model_group():
    """Legacy test function - DISABLED"""
    pass


@pytest.mark.skip(reason="Requires actual OpenSearch cluster with valid credentials")
def test_os_local_ml_model():
    """Legacy test function - DISABLED"""
    pass


if __name__ == "__main__":
    print("Integration tests disabled - require actual OpenSearch cluster")
