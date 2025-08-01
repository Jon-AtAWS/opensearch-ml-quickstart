# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

"""
Configuration module providing access to the unified configuration system.

This module exports the main configuration functions and constants from the
configuration_manager and task definitions.
"""

# Import from configuration_manager
from .configuration_manager import (
    # Configuration getters
    get_ml_base_uri,
    get_delete_resource_wait_time, 
    get_delete_resource_retry_time,
    get_base_mapping_path,
    get_qanda_file_reader_path,
    get_pipeline_field_map,
    get_client_configs,
    validate_configs,
    get_raw_config_value,
    
    # Constants
    ML_BASE_URI,
    DELETE_RESOURCE_WAIT_TIME,
    DELETE_RESOURCE_RETRY_TIME,
    BASE_MAPPING_PATH,
    QANDA_FILE_READER_PATH,
    DEFAULT_LOCAL_MODEL_NAME,
    DEFAULT_MODEL_VERSION,
    DEFAULT_MODEL_FORMAT,
    MINIMUM_OPENSEARCH_VERSION,
)

# Import from tasks
from .tasks import categories, tasks, PIPELINE_FIELD_MAP
