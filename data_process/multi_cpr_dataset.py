# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

import os
import csv
import logging
from typing import Dict, Any, List, Iterator, Tuple, Optional

from .base_dataset import BaseDataset


class MultiCPRDataset(BaseDataset):
    """Multi-CPR: A Multi Domain Chinese Dataset for Passage Retrieval.

    Supports three domains from the Alibaba-NLP/Multi-CPR dataset:
    - ecom: E-commerce product queries
    - video: Entertainment video queries
    - medical: Medical domain queries

    Data format per domain:
        corpus.tsv      - pid\\tpassage_content
        train.query.txt - qid\\tquery_content
        dev.query.txt   - qid\\tquery_content
        qrels.train.tsv - qid\\t0\\tpid\\t1
        qrels.dev.tsv   - qid\\t0\\tpid\\t1

    See: https://github.com/Alibaba-NLP/Multi-CPR
    """

    DATASET_PATH = "~/datasets/multi_cpr"

    DOMAIN_MAP = {
        "ecom": "ecom",
        "video": "video",
        "medical": "medical",
    }

    def __init__(self, directory: Optional[str] = None, max_number_of_docs: int = -1):
        expanded = os.path.expanduser(directory if directory is not None else self.DATASET_PATH)
        super().__init__(expanded, max_number_of_docs)

    # ── helpers ──────────────────────────────────────────────────────────

    def _domain_dir(self, domain: str) -> str:
        return os.path.join(self.directory, domain)

    def _read_corpus(self, domain: str) -> Iterator[Dict[str, Any]]:
        """Yield {pid, passage, domain} from corpus.tsv."""
        path = os.path.join(self._domain_dir(domain), "corpus.tsv")
        count = 0
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                if len(row) < 2:
                    continue
                yield {
                    "pid": row[0],
                    "passage": row[1],
                    "domain": domain,
                    "chunk_text": row[1],
                }
                count += 1
                if 0 < self.max_number_of_docs <= count:
                    break

    def _read_queries(self, domain: str, split: str = "dev") -> Dict[str, str]:
        """Return {qid: query_text} from train.query.txt or dev.query.txt."""
        path = os.path.join(self._domain_dir(domain), f"{split}.query.txt")
        queries = {}
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                if len(row) >= 2:
                    queries[row[0]] = row[1]
        return queries

    def _read_qrels(self, domain: str, split: str = "dev") -> Dict[str, str]:
        """Return {qid: pid} from qrels.{split}.tsv."""
        path = os.path.join(self._domain_dir(domain), f"qrels.{split}.tsv")
        qrels = {}
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                if len(row) >= 4:
                    qrels[row[0]] = row[2]
        return qrels

    # ── BaseDataset: Data Processing ─────────────────────────────────────

    def get_batches(self, filter_criteria: Optional[List[str]] = None) -> Iterator[Tuple[List[Dict[str, Any]], int]]:
        domains = filter_criteria or list(self.DOMAIN_MAP.keys())
        for domain in domains:
            docs = list(self._read_corpus(domain))
            yield docs, len(docs)

    def get_available_filters(self) -> List[str]:
        return list(self.DOMAIN_MAP.keys())

    # ── BaseDataset: Preprocessing Lifecycle ──────────────────────────────

    def requires_preprocessing(self) -> bool:
        return False

    def is_preprocessed(self) -> bool:
        return True

    def preprocess(self, os_client, model_id, **kwargs) -> None:
        pass

    def get_preprocessing_status(self) -> Dict[str, Any]:
        return {"status": "complete", "message": "No preprocessing required"}

    def get_preprocessing_requirements(self) -> Dict[str, Any]:
        return {}

    def estimate_preprocessing_time(self, source_size: int) -> str:
        return "0 seconds"

    def validate_preprocessing_inputs(self, inputs: Dict[str, Any]) -> bool:
        return True

    def get_source_data_pattern(self) -> str:
        return "*/corpus.tsv"

    def get_processed_data_pattern(self) -> str:
        return "*/corpus.tsv"

    # ── BaseDataset: Index Configuration ──────────────────────────────────

    def requires_ingest_pipeline(self) -> bool:
        return True

    def get_index_mapping(self) -> Dict[str, Any]:
        """Index mapping for Multi-CPR passages.

        The 'passage' field uses the default text analyzer.  If you have the
        IK Analysis plugin installed you can switch to ``ik_max_word`` /
        ``ik_smart`` for better Chinese tokenisation.
        """
        return {
            "properties": {
                "pid": {"type": "keyword"},
                "domain": {"type": "keyword"},
                "passage": {"type": "text"},
                "chunk_text": {"type": "text"},
            }
        }

    def get_pipeline_config(self) -> Optional[Dict[str, Any]]:
        return {
            "description": "Multi-CPR text embedding pipeline",
            "processors": [
                {
                    "text_embedding": {
                        "model_id": "{model_id}",
                        "field_map": {"chunk_text": "chunk_vector"},
                    }
                }
            ],
        }

    def get_index_name_prefix(self) -> str:
        return "multi_cpr"

    def get_bulk_chunk_size(self) -> int:
        return 200

    # ── BaseDataset: Display / UI ─────────────────────────────────────────

    def format_search_result(self, document: Dict[str, Any], score: float) -> str:
        source = document.get("_source", document)
        return (
            f"Score: {score:.4f}\n"
            f"Domain: {source.get('domain', 'N/A')}\n"
            f"PID: {source.get('pid', 'N/A')}\n"
            f"Passage: {source.get('passage', 'N/A')[:300]}\n"
            f"---"
        )

    def get_result_summary_fields(self) -> List[str]:
        return ["domain", "pid", "passage"]

    def get_searchable_text_preview(self, document: Dict[str, Any]) -> str:
        return document.get("passage", "")[:100] + "..."

    # ── BaseDataset: Metadata / Configuration ─────────────────────────────

    def get_dataset_info(self) -> Dict[str, str]:
        return {
            "name": "Multi-CPR",
            "description": "Multi-domain Chinese dataset for passage retrieval (E-commerce, Video, Medical)",
            "version": "1.0",
        }

    def get_sample_queries(self) -> List[str]:
        return [
            "尼康z62",
            "海神妈祖",
            "大人能把手放在睡觉婴儿胸口吗",
            "无线蓝牙耳机推荐",
        ]

    def estimate_index_size(self, document_count: int) -> str:
        kb_per_doc = 1
        total_mb = (document_count * kb_per_doc) / 1024
        if total_mb < 1024:
            return f"{total_mb:.0f} MB"
        return f"{total_mb / 1024:.1f} GB"

    def validate_search_params(self, params: Dict[str, Any]) -> bool:
        return True

    def handle_search_error(self, error: Exception, query: Dict[str, Any]) -> str:
        return f"Multi-CPR search error: {str(error)}"

    def load_data(
        self,
        os_client,
        index_name: str,
        filter_criteria: Optional[List[str]] = None,
        bulk_chunk_size: int = 200,
    ) -> int:
        """Load Multi-CPR corpus passages into an OpenSearch index."""
        from opensearchpy import helpers

        total = 0
        domains = filter_criteria or list(self.DOMAIN_MAP.keys())

        for domain in domains:
            logging.info(f"Loading Multi-CPR domain: {domain}")
            buf: List[Dict[str, Any]] = []
            domain_count = 0

            for doc in self._read_corpus(domain):
                passage = doc.get("passage", "").strip()
                if not passage:
                    continue

                doc["_index"] = index_name
                doc["_id"] = f"{domain}_{doc['pid']}"
                # Truncate to ~500 tokens for embedding
                doc["chunk_text"] = " ".join(passage.split()[:500])
                buf.append(doc)
                domain_count += 1

                if len(buf) >= bulk_chunk_size:
                    helpers.bulk(os_client, buf, chunk_size=bulk_chunk_size)
                    buf = []

            if buf:
                helpers.bulk(os_client, buf, chunk_size=bulk_chunk_size)

            total += domain_count
            logging.info(f"Loaded {domain_count} passages for domain '{domain}'")

        return total
