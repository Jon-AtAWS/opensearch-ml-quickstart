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
    get_opensearch_config,
    get_model_config,
    get_embedding_config,
    get_llm_config,
    get_ml_base_uri,
    get_delete_resource_wait_time, 
    get_delete_resource_retry_time,
    get_base_mapping_path,
    get_qanda_file_reader_path,
    get_minimum_opensearch_version,
    get_pipeline_field_map,
    get_client_configs,
    validate_configs,
    get_raw_config_value,
    get_available_combinations,
    validate_all_configs,
    get_config_info,
    get_project_root,
    get_local_dense_embedding_model_name,
    get_local_dense_embedding_model_version,
    get_local_dense_embedding_model_format,
    get_local_dense_embedding_model_dimension,
    
    # Configuration override context manager
    config_override,
    
    # Configuration classes
    OpenSearchConfig,
    ModelConfig,
    OpenSearchType,
    ModelProvider,
    ModelType,
    
    # Functions
    get_local_dense_embedding_model_dimension,
)
