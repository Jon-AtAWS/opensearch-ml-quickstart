# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

# Env
DEFAULT_ENV_PATH = "./configs/.env"

# Open Search
MINIMUM_OPENSEARCH_VERSION = "2.13.0"

# ML models
ML_BASE_URI = "/_plugins/_ml"
DELETE_MODEL_WAIT_TIME = 5
DELETE_MODEL_RETRY_TIME = 5

DEFAULT_LOCAL_MODEL_NAME = (
    "huggingface/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
DEFAULT_MODEL_VERSION = "1.0.1"
DEFAULT_MODEL_FORMAT = "TORCH_SCRIPT"
