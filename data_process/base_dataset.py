# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Iterator, Tuple, Optional, Union
import logging


class BaseDataset(ABC):
    """Abstract base class for dataset readers with preprocessing, indexing, and search capabilities."""
    
    def __init__(self, directory: str, max_number_of_docs: int = -1):
        self.directory = directory
        self.max_number_of_docs = max_number_of_docs
    
    # Data Processing
    @abstractmethod
    def get_batches(self, filter_criteria: Optional[List[str]] = None) -> Iterator[Tuple[List[Dict[str, Any]], int]]:
        """Returns document batches with counts."""
        pass
    
    @abstractmethod
    def get_available_filters(self) -> List[str]:
        """List available filter keys (categories, etc.)."""
        pass
    
    # Preprocessing Lifecycle
    @abstractmethod
    def requires_preprocessing(self) -> bool:
        """Boolean indicating if dataset needs preprocessing."""
        pass
    
    @abstractmethod
    def is_preprocessed(self) -> bool:
        """Check if preprocessing has been completed."""
        pass
    
    @abstractmethod
    def preprocess(self, os_client, model_id, **kwargs) -> None:
        """Execute preprocessing step."""
        pass
    
    @abstractmethod
    def get_preprocessing_status(self) -> Dict[str, Any]:
        """Return preprocessing progress/status."""
        pass
    
    @abstractmethod
    def get_preprocessing_requirements(self) -> Dict[str, Any]:
        """Required resources (models, clients, etc.)."""
        pass
    
    @abstractmethod
    def estimate_preprocessing_time(self, source_size: int) -> str:
        """Time estimate for preprocessing."""
        pass
    
    @abstractmethod
    def validate_preprocessing_inputs(self, inputs: Dict[str, Any]) -> bool:
        """Validate preprocessing parameters."""
        pass
    
    @abstractmethod
    def get_source_data_pattern(self) -> str:
        """Pattern for raw source files."""
        pass
    
    @abstractmethod
    def get_processed_data_pattern(self) -> str:
        """Pattern for processed files."""
        pass
    
    # Index Configuration
    @abstractmethod
    def requires_ingest_pipeline(self) -> bool:
        """Boolean for embedding creation strategy."""
        pass
    
    @abstractmethod
    def get_index_mapping(self) -> Dict[str, Any]:
        """Index mapping for this dataset."""
        pass
    
    @abstractmethod
    def get_pipeline_config(self) -> Optional[Dict[str, Any]]:
        """Ingest pipeline configuration."""
        pass
    
    @abstractmethod
    def get_index_name_prefix(self) -> str:
        """Dataset-specific index naming."""
        pass
    
    @abstractmethod
    def get_bulk_chunk_size(self) -> int:
        """Optimal bulk indexing size."""
        pass
    
    # Display/UI
    @abstractmethod
    def format_search_result(self, document: Dict[str, Any], score: float) -> str:
        """Format single search result."""
        pass
    
    @abstractmethod
    def get_result_summary_fields(self) -> List[str]:
        """Key fields for result summaries."""
        pass
    
    @abstractmethod
    def get_searchable_text_preview(self, document: Dict[str, Any]) -> str:
        """Text snippet for display."""
        pass
    
    # Metadata/Configuration
    @abstractmethod
    def get_dataset_info(self) -> Dict[str, str]:
        """Name, description, version."""
        pass
    
    @abstractmethod
    def get_sample_queries(self) -> List[str]:
        """Example queries for users."""
        pass
    
    @abstractmethod
    def estimate_index_size(self, document_count: int) -> str:
        """Storage estimates."""
        pass
    
    @abstractmethod
    def validate_search_params(self, params: Dict[str, Any]) -> bool:
        """Parameter validation."""
        pass
    
    @abstractmethod
    def handle_search_error(self, error: Exception, query: Dict[str, Any]) -> str:
        """Dataset-specific error handling."""
        pass
    
    @abstractmethod
    def load_data(self, os_client, index_name: str, filter_criteria: Optional[List[str]] = None, bulk_chunk_size: int = 100) -> int:
        """Load dataset into OpenSearch index. Returns number of documents loaded."""
        pass
    
    def create_index(self, os_client, index_name: str, delete_existing: bool = False, index_settings: Optional[Dict[str, Any]] = None) -> bool:
        """Create OpenSearch index with appropriate mapping. Returns True if created/exists."""
        logging.info(f"Handling index creation for {index_name}")
        
        index_exists = os_client.indices.exists(index=index_name)
        logging.info(f"Index {index_name} exists: {index_exists}")
        
        if delete_existing and index_exists:
            logging.info(f"Deleting existing index {index_name}")
            os_client.indices.delete(index=index_name)
            index_exists = False
        
        if index_exists:
            logging.info(f'Index "{index_name}" exists. Skipping creation.')
            return True
        
        try:
            if index_settings is None:
                index_settings = {"mappings": self.get_index_mapping()}
            response = os_client.indices.create(index=index_name, body=index_settings)
            logging.info(f"Created index {index_name}: {response}")
            return True
        except Exception as e:
            logging.error(f"Error creating index {index_name}: {e}")
            return False
    
    def update_mapping(self, base_mapping: Dict[str, Any], updates: Dict[str, Any]) -> None:
        """Update mapping with additional fields (e.g., vector fields)."""
        for key, value in updates.items():
            if (
                key in base_mapping
                and isinstance(base_mapping[key], dict)
                and isinstance(value, dict)
            ):
                self.update_mapping(base_mapping[key], value)
            else:
                base_mapping[key] = value
    
    def create_index(self, os_client, index_name: str, delete_existing: bool = False, index_settings: Optional[Dict[str, Any]] = None) -> bool:
        """Create OpenSearch index with appropriate mapping. Returns True if created/exists."""
        logging.info(f"Handling index creation for {index_name}")
        
        index_exists = os_client.indices.exists(index=index_name)
        logging.info(f"Index {index_name} exists: {index_exists}")
        
        if delete_existing and index_exists:
            logging.info(f"Deleting existing index {index_name}")
            os_client.indices.delete(index=index_name)
            index_exists = False
        
        if index_exists:
            logging.info(f'Index "{index_name}" exists. Skipping creation.')
            return True
        
        try:
            if index_settings is None:
                index_settings = {"mappings": self.get_index_mapping()}
            response = os_client.indices.create(index=index_name, body=index_settings)
            logging.info(f"Created index {index_name}: {response}")
            return True
        except Exception as e:
            logging.error(f"Error creating index {index_name}: {e}")
            return False
