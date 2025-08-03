# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for mapping module with real-world scenarios"""

import json
import os
import tempfile
import pytest

from mapping import get_base_mapping, mapping_update


class TestMappingIntegrationScenarios:
    """Integration tests for real-world mapping scenarios"""

    def test_opensearch_knn_vector_integration(self):
        """Test integrating k-NN vector search capabilities into base mapping"""
        # Create a base mapping similar to what might be used in production
        base_mapping = {
            "settings": {
                "index": {
                    "number_of_shards": 1,
                    "number_of_replicas": 0,
                    "refresh_interval": "1s"
                }
            },
            "mappings": {
                "properties": {
                    "title": {
                        "type": "text",
                        "analyzer": "standard"
                    },
                    "content": {
                        "type": "text",
                        "analyzer": "standard"
                    },
                    "category": {
                        "type": "keyword"
                    },
                    "timestamp": {
                        "type": "date"
                    }
                }
            }
        }
        
        # Add k-NN vector search capabilities
        knn_settings = {
            "settings": {
                "index": {
                    "knn": True,
                    "knn.algo_param.ef_search": 100,
                    "knn.algo_param.ef_construction": 128,
                    "knn.space_type": "l2"
                }
            },
            "mappings": {
                "properties": {
                    "title_vector": {
                        "type": "knn_vector",
                        "dimension": 768,
                        "method": {
                            "name": "hnsw",
                            "space_type": "l2",
                            "engine": "nmslib",
                            "parameters": {
                                "ef_construction": 128,
                                "m": 24
                            }
                        }
                    },
                    "content_vector": {
                        "type": "knn_vector",
                        "dimension": 384,
                        "method": {
                            "name": "hnsw",
                            "space_type": "cosinesimil",
                            "engine": "faiss"
                        }
                    }
                }
            }
        }
        
        mapping_update(base_mapping, knn_settings)
        
        # Verify original settings are preserved
        assert base_mapping["settings"]["index"]["number_of_shards"] == 1
        assert base_mapping["settings"]["index"]["refresh_interval"] == "1s"
        
        # Verify k-NN settings are added
        assert base_mapping["settings"]["index"]["knn"] is True
        assert base_mapping["settings"]["index"]["knn.algo_param.ef_search"] == 100
        
        # Verify original mappings are preserved
        assert base_mapping["mappings"]["properties"]["title"]["type"] == "text"
        assert base_mapping["mappings"]["properties"]["category"]["type"] == "keyword"
        
        # Verify vector mappings are added
        assert base_mapping["mappings"]["properties"]["title_vector"]["type"] == "knn_vector"
        assert base_mapping["mappings"]["properties"]["title_vector"]["dimension"] == 768
        assert base_mapping["mappings"]["properties"]["content_vector"]["dimension"] == 384

    def test_multi_language_analysis_integration(self):
        """Test integrating multi-language analysis settings"""
        base_mapping = {
            "settings": {
                "index": {
                    "number_of_shards": 2
                }
            },
            "mappings": {
                "properties": {
                    "content": {"type": "text"}
                }
            }
        }
        
        # Add multi-language analysis
        analysis_settings = {
            "settings": {
                "analysis": {
                    "analyzer": {
                        "multilang_analyzer": {
                            "type": "custom",
                            "tokenizer": "standard",
                            "filter": [
                                "lowercase",
                                "stop",
                                "snowball"
                            ]
                        },
                        "english_analyzer": {
                            "type": "custom",
                            "tokenizer": "standard",
                            "filter": [
                                "lowercase",
                                "english_stop",
                                "english_stemmer"
                            ]
                        }
                    },
                    "filter": {
                        "english_stop": {
                            "type": "stop",
                            "stopwords": "_english_"
                        },
                        "english_stemmer": {
                            "type": "stemmer",
                            "language": "english"
                        }
                    }
                }
            },
            "mappings": {
                "properties": {
                    "content_en": {
                        "type": "text",
                        "analyzer": "english_analyzer"
                    },
                    "content_multilang": {
                        "type": "text",
                        "analyzer": "multilang_analyzer"
                    }
                }
            }
        }
        
        mapping_update(base_mapping, analysis_settings)
        
        # Verify analysis settings are added
        assert "analysis" in base_mapping["settings"]
        assert "multilang_analyzer" in base_mapping["settings"]["analysis"]["analyzer"]
        assert "english_analyzer" in base_mapping["settings"]["analysis"]["analyzer"]
        
        # Verify new mappings are added
        assert base_mapping["mappings"]["properties"]["content_en"]["analyzer"] == "english_analyzer"
        assert base_mapping["mappings"]["properties"]["content_multilang"]["analyzer"] == "multilang_analyzer"

    def test_security_and_performance_settings_integration(self):
        """Test integrating security and performance settings"""
        base_mapping = {
            "settings": {
                "index": {
                    "number_of_shards": 1
                }
            },
            "mappings": {
                "properties": {
                    "data": {"type": "text"}
                }
            }
        }
        
        # Add security and performance settings
        advanced_settings = {
            "settings": {
                "index": {
                    "max_result_window": 50000,
                    "max_rescore_window": 10000,
                    "blocks": {
                        "read_only": False,
                        "write": False
                    },
                    "routing": {
                        "allocation": {
                            "enable": "all"
                        }
                    },
                    "translog": {
                        "flush_threshold_size": "512mb",
                        "sync_interval": "5s"
                    }
                }
            },
            "mappings": {
                "_source": {
                    "enabled": True,
                    "excludes": ["sensitive_field"]
                },
                "properties": {
                    "sensitive_field": {
                        "type": "text",
                        "index": False,
                        "store": False
                    },
                    "searchable_field": {
                        "type": "text",
                        "index": True,
                        "store": True
                    }
                }
            }
        }
        
        mapping_update(base_mapping, advanced_settings)
        
        # Verify performance settings
        assert base_mapping["settings"]["index"]["max_result_window"] == 50000
        assert base_mapping["settings"]["index"]["translog"]["flush_threshold_size"] == "512mb"
        
        # Verify security settings
        assert base_mapping["mappings"]["_source"]["excludes"] == ["sensitive_field"]
        assert base_mapping["mappings"]["properties"]["sensitive_field"]["index"] is False

    def test_dynamic_template_integration(self):
        """Test integrating dynamic templates"""
        base_mapping = {
            "mappings": {
                "properties": {
                    "static_field": {"type": "keyword"}
                }
            }
        }
        
        # Add dynamic templates
        dynamic_settings = {
            "mappings": {
                "dynamic_templates": [
                    {
                        "strings_as_keywords": {
                            "match_mapping_type": "string",
                            "match": "*_keyword",
                            "mapping": {
                                "type": "keyword",
                                "ignore_above": 256
                            }
                        }
                    },
                    {
                        "strings_as_text": {
                            "match_mapping_type": "string",
                            "match": "*_text",
                            "mapping": {
                                "type": "text",
                                "analyzer": "standard"
                            }
                        }
                    }
                ]
            }
        }
        
        mapping_update(base_mapping, dynamic_settings)
        
        # Verify dynamic templates are added
        assert "dynamic_templates" in base_mapping["mappings"]
        assert len(base_mapping["mappings"]["dynamic_templates"]) == 2
        assert "strings_as_keywords" in base_mapping["mappings"]["dynamic_templates"][0]

    def test_file_based_mapping_workflow(self):
        """Test complete workflow: load from file, update, and verify"""
        # Create a temporary base mapping file
        initial_mapping = {
            "settings": {
                "index": {
                    "number_of_shards": 1,
                    "number_of_replicas": 0
                }
            },
            "mappings": {
                "properties": {
                    "title": {"type": "text"},
                    "description": {"type": "text"}
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(initial_mapping, f)
            temp_path = f.name
        
        try:
            # Step 1: Load base mapping
            loaded_mapping = get_base_mapping(temp_path)
            
            # Step 2: Add search enhancements
            search_enhancements = {
                "settings": {
                    "index": {
                        "knn": True,
                        "number_of_replicas": 1  # Update existing setting
                    },
                    "analysis": {
                        "analyzer": {
                            "custom_search_analyzer": {
                                "type": "custom",
                                "tokenizer": "standard",
                                "filter": ["lowercase", "stop"]
                            }
                        }
                    }
                },
                "mappings": {
                    "properties": {
                        "title": {
                            "type": "text",
                            "analyzer": "custom_search_analyzer",  # Update existing field
                            "fields": {
                                "keyword": {"type": "keyword"}
                            }
                        },
                        "embedding": {  # Add new field
                            "type": "knn_vector",
                            "dimension": 512
                        },
                        "tags": {  # Add new field
                            "type": "keyword"
                        }
                    }
                }
            }
            
            mapping_update(loaded_mapping, search_enhancements)
            
            # Step 3: Verify the complete result
            # Original settings preserved/updated
            assert loaded_mapping["settings"]["index"]["number_of_shards"] == 1
            assert loaded_mapping["settings"]["index"]["number_of_replicas"] == 1  # Updated
            
            # New settings added
            assert loaded_mapping["settings"]["index"]["knn"] is True
            assert "analysis" in loaded_mapping["settings"]
            
            # Original mappings updated
            assert loaded_mapping["mappings"]["properties"]["title"]["type"] == "text"
            assert loaded_mapping["mappings"]["properties"]["title"]["analyzer"] == "custom_search_analyzer"
            assert "keyword" in loaded_mapping["mappings"]["properties"]["title"]["fields"]
            
            # Original mappings preserved
            assert loaded_mapping["mappings"]["properties"]["description"]["type"] == "text"
            
            # New mappings added
            assert loaded_mapping["mappings"]["properties"]["embedding"]["type"] == "knn_vector"
            assert loaded_mapping["mappings"]["properties"]["tags"]["type"] == "keyword"
            
        finally:
            os.unlink(temp_path)

    def test_real_world_amazon_pqa_mapping_scenario(self):
        """Test scenario similar to Amazon PQA dataset mapping"""
        # Base mapping for Q&A data
        base_mapping = {
            "settings": {
                "index": {
                    "number_of_shards": 2,
                    "number_of_replicas": 1
                }
            },
            "mappings": {
                "properties": {
                    "question_text": {"type": "text"},
                    "answer_text": {"type": "text"},
                    "product_id": {"type": "keyword"},
                    "category": {"type": "keyword"}
                }
            }
        }
        
        # Add ML search capabilities for Q&A matching
        ml_enhancements = {
            "settings": {
                "index": {
                    "knn": True,
                    "knn.algo_param.ef_search": 100
                },
                "analysis": {
                    "analyzer": {
                        "qa_analyzer": {
                            "type": "custom",
                            "tokenizer": "standard",
                            "filter": [
                                "lowercase",
                                "stop",
                                "stemmer"
                            ]
                        }
                    }
                }
            },
            "mappings": {
                "properties": {
                    "question_text": {
                        "type": "text",
                        "analyzer": "qa_analyzer",
                        "fields": {
                            "keyword": {"type": "keyword"},
                            "suggest": {
                                "type": "completion",
                                "analyzer": "simple"
                            }
                        }
                    },
                    "question_embedding": {
                        "type": "knn_vector",
                        "dimension": 768,
                        "method": {
                            "name": "hnsw",
                            "space_type": "cosinesimil",
                            "engine": "nmslib"
                        }
                    },
                    "answer_embedding": {
                        "type": "knn_vector",
                        "dimension": 768,
                        "method": {
                            "name": "hnsw",
                            "space_type": "cosinesimil",
                            "engine": "nmslib"
                        }
                    },
                    "semantic_similarity_score": {
                        "type": "float"
                    },
                    "user_metadata": {
                        "type": "nested",
                        "properties": {
                            "user_id": {"type": "keyword"},
                            "rating": {"type": "integer"},
                            "helpful_votes": {"type": "integer"}
                        }
                    }
                }
            }
        }
        
        mapping_update(base_mapping, ml_enhancements)
        
        # Verify Q&A specific enhancements
        assert base_mapping["settings"]["index"]["knn"] is True
        assert "qa_analyzer" in base_mapping["settings"]["analysis"]["analyzer"]
        
        # Verify enhanced question field
        question_field = base_mapping["mappings"]["properties"]["question_text"]
        assert question_field["analyzer"] == "qa_analyzer"
        assert "suggest" in question_field["fields"]
        assert question_field["fields"]["suggest"]["type"] == "completion"
        
        # Verify vector fields for semantic search
        assert base_mapping["mappings"]["properties"]["question_embedding"]["type"] == "knn_vector"
        assert base_mapping["mappings"]["properties"]["answer_embedding"]["dimension"] == 768
        
        # Verify nested user metadata
        user_metadata = base_mapping["mappings"]["properties"]["user_metadata"]
        assert user_metadata["type"] == "nested"
        assert user_metadata["properties"]["rating"]["type"] == "integer"

    def test_error_recovery_and_validation(self):
        """Test error recovery scenarios in integration workflows"""
        base_mapping = {"settings": {"index": {"number_of_shards": 1}}}
        
        # Test with various problematic updates that should still work
        problematic_updates = [
            # None values
            {"settings": {"index": {"some_setting": None}}},
            # Empty nested structures
            {"mappings": {"properties": {}}},
            # Mixed types
            {"settings": {"mixed": {"string": "value", "number": 42, "boolean": True}}}
        ]
        
        for update in problematic_updates:
            # Should not raise exceptions
            mapping_update(base_mapping, update)
        
        # Verify final state is reasonable
        assert base_mapping["settings"]["index"]["number_of_shards"] == 1
        assert base_mapping["settings"]["index"]["some_setting"] is None
        assert "mappings" in base_mapping
        assert base_mapping["settings"]["mixed"]["string"] == "value"
        assert base_mapping["settings"]["mixed"]["number"] == 42
        assert base_mapping["settings"]["mixed"]["boolean"] is True
