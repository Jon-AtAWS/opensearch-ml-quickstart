# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

categories = [
    "sheet and pillowcase sets",
    # "unlocked cell phones",
    # "external hard drives",
    # "computer cases",
    # "car",
    # "bed frames",
    # "jeans",
]


ml_models = [
    ("huggingface/sentence-transformers/all-distilroberta-v1", 768),
    ("huggingface/sentence-transformers/all-MiniLM-L", 384),
    ("huggingface/sentence-transformers/all-MiniLM-L12-v2", 384),
    ("huggingface/sentence-transformers/all-mpnet-base-v2", 768),
    ("huggingface/sentence-transformers/msmarco-distilbert-base-tas-b", 768),
    ("huggingface/sentence-transformers/multi-qa-MiniLM-L6-cos-v1", 384),
    ("huggingface/sentence-transformers/multi-qa-mpnet-base-dot-v1", 384),
    ("huggingface/sentence-transformers/paraphrase-MiniLM-L3-v2", 384),
    ("huggingface/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", 384),
    ("huggingface/sentence-transformers/paraphrase-mpnet-base-v2", 768),
    ("huggingface/sentence-transformers/distiluse-base-multilingual-cased-v1", 512),
]


PIPELINE_FIELD_MAP = {"chunk": "chunk_embedding"}

# please refer to supported model versions in https://opensearch.org/docs/latest/ml-commons-plugin/pretrained-models/

tasks = {
    "best_compression": {
        "with_knn": False,
        "categories": categories,
        "max_cat_docs": -1,
        "cleanup": False,
        "compression": "best_compression",
        "pipeline_field_map": None,
        "model_name": None,
        "model_dimensions": None,
    },
    "zstd": {
        "with_knn": False,
        "categories": categories,
        "max_cat_docs": -1,
        "cleanup": False,
        "compression": "zstd",
        "pipeline_field_map": None,
        "model_name": None,
        "model_dimensions": None,
    },
    "zstd_no_dict": {
        "with_knn": False,
        "categories": categories,
        "max_cat_docs": -1,
        "cleanup": False,
        "compression": "zstd_no_dict",
        "pipeline_field_map": None,
        "model_name": None,
        "model_dimensions": None,
    },
    "knn_384": {
        "with_knn": True,
        "categories": categories,
        "max_cat_docs": -1,
        "cleanup": True,
        "compression": "best_compression",
        "pipeline_field_map": PIPELINE_FIELD_MAP,
        "model_name": "huggingface/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "model_dimensions": 384,
    },
    "knn_512": {
        "with_knn": True,
        "categories": categories,
        "max_cat_docs": -1,
        "cleanup": True,
        "compression": "best_compression",
        "pipeline_field_map": PIPELINE_FIELD_MAP,
        "model_name": "huggingface/sentence-transformers/distiluse-base-multilingual-cased-v1",
        "model_dimensions": 512,
    },
    "knn_768": {
        "with_knn": True,
        "categories": categories,
        "max_cat_docs": -1,
        "cleanup": True,
        "compression": "best_compression",
        "pipeline_field_map": PIPELINE_FIELD_MAP,
        "model_name": "huggingface/sentence-transformers/msmarco-distilbert-base-tas-b",
        "model_version": "1.0.2",
        "model_dimensions": 768,
    },
    "knn_768_no_cleanup": {
        "with_knn": True,
        "categories": categories,
        "max_cat_docs": -1,
        "cleanup": False,
        "compression": "best_compression",
        "pipeline_field_map": PIPELINE_FIELD_MAP,
        "model_name": "huggingface/sentence-transformers/msmarco-distilbert-base-tas-b",
        "model_version": "1.0.2",
        "model_dimensions": 768,
    },
}
