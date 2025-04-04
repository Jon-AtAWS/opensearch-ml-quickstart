# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

import os

# Env
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_ENV_PATH = os.path.join(PROJECT_ROOT, "configs", ".env")

BASE_MAPPING_PATH = os.path.join(PROJECT_ROOT, "mapping", "base_mapping.json")

QANDA_FILE_READER_PATH= os.path.join(PROJECT_ROOT, "datasets", "amazon_pqa")

# Open Search
MINIMUM_OPENSEARCH_VERSION = "2.13.0"

# ML connectors, models and model_groups
ML_BASE_URI = "/_plugins/_ml"
DELETE_RESOURCE_WAIT_TIME = 5
DELETE_RESOURCE_RETRY_TIME = 5

DEFAULT_LOCAL_MODEL_NAME = (
    "huggingface/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
DEFAULT_MODEL_VERSION = "1.0.1"
DEFAULT_MODEL_FORMAT = "TORCH_SCRIPT"
