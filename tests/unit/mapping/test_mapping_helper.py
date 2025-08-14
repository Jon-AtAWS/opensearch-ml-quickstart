# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for mapping helper functions"""

import json
import os
import tempfile
import pytest
from unittest.mock import patch, mock_open

from mapping.helper import get_base_mapping, mapping_update


class TestGetBaseMapping:
    """Test cases for get_base_mapping function"""

    def test_get_base_mapping_valid_file(self):
        """Test loading a valid JSON mapping file"""
        test_mapping = {
            "settings": {"index": {"number_of_shards": 1}},
            "mappings": {"properties": {"field1": {"type": "text"}}}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_mapping, f)
            temp_path = f.name
        
        try:
            result = get_base_mapping(temp_path)
            assert result == test_mapping
        finally:
            os.unlink(temp_path)

    def test_get_base_mapping_complex_structure(self):
        """Test loading a complex mapping structure"""
        complex_mapping = {
            "settings": {
                "index": {
                    "number_of_shards": 2,
                    "number_of_replicas": 1,
                    "analysis": {
                        "analyzer": {
                            "custom_analyzer": {
                                "type": "custom",
                                "tokenizer": "standard"
                            }
                        }
                    }
                }
            },
            "mappings": {
                "properties": {
                    "title": {
                        "type": "text",
                        "fields": {
                            "keyword": {"type": "keyword"}
                        }
                    },
                    "nested_field": {
                        "type": "nested",
                        "properties": {
                            "sub_field": {"type": "keyword"}
                        }
                    }
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(complex_mapping, f)
            temp_path = f.name
        
        try:
            result = get_base_mapping(temp_path)
            assert result == complex_mapping
            assert result["settings"]["index"]["analysis"]["analyzer"]["custom_analyzer"]["type"] == "custom"
        finally:
            os.unlink(temp_path)

    def test_get_base_mapping_empty_file(self):
        """Test loading an empty JSON file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({}, f)
            temp_path = f.name
        
        try:
            result = get_base_mapping(temp_path)
            assert result == {}
        finally:
            os.unlink(temp_path)

    def test_get_base_mapping_file_not_found(self):
        """Test handling of non-existent file"""
        with pytest.raises(FileNotFoundError):
            get_base_mapping("/nonexistent/path/mapping.json")

    def test_get_base_mapping_invalid_json(self):
        """Test handling of invalid JSON content"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("{ invalid json content")
            temp_path = f.name
        
        try:
            with pytest.raises(json.JSONDecodeError):
                get_base_mapping(temp_path)
        finally:
            os.unlink(temp_path)

    def test_get_base_mapping_permission_error(self):
        """Test handling of permission errors"""
        with patch("builtins.open", mock_open()) as mock_file:
            mock_file.side_effect = PermissionError("Permission denied")
            
            with pytest.raises(PermissionError):
                get_base_mapping("/restricted/path/mapping.json")

    def test_get_base_mapping_with_unicode(self):
        """Test loading JSON with unicode characters"""
        unicode_mapping = {
            "mappings": {
                "properties": {
                    "title": {"type": "text"},
                    "description": {"type": "text"}
                }
            },
            "metadata": {
                "author": "Test User 测试",
                "description": "Mapping with unicode: café, naïve, résumé"
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
            json.dump(unicode_mapping, f, ensure_ascii=False)
            temp_path = f.name
        
        try:
            result = get_base_mapping(temp_path)
            assert result == unicode_mapping
            assert "测试" in result["metadata"]["author"]
            assert "café" in result["metadata"]["description"]
        finally:
            os.unlink(temp_path)


class TestMappingUpdate:
    """Test cases for mapping_update function"""

    def test_mapping_update_simple_override(self):
        """Test simple key-value override"""
        base_mapping = {"key1": "value1", "key2": "value2"}
        settings = {"key2": "new_value2", "key3": "value3"}
        
        mapping_update(base_mapping, settings)
        
        assert base_mapping["key1"] == "value1"
        assert base_mapping["key2"] == "new_value2"
        assert base_mapping["key3"] == "value3"

    def test_mapping_update_nested_dict_merge(self):
        """Test nested dictionary merging"""
        base_mapping = {
            "settings": {
                "index": {
                    "number_of_shards": 1,
                    "number_of_replicas": 0
                }
            },
            "mappings": {
                "properties": {
                    "field1": {"type": "text"}
                }
            }
        }
        
        settings = {
            "settings": {
                "index": {
                    "number_of_replicas": 2,
                    "refresh_interval": "1s"
                }
            },
            "mappings": {
                "properties": {
                    "field2": {"type": "keyword"}
                }
            }
        }
        
        mapping_update(base_mapping, settings)
        
        # Check that nested values are properly merged
        assert base_mapping["settings"]["index"]["number_of_shards"] == 1  # preserved
        assert base_mapping["settings"]["index"]["number_of_replicas"] == 2  # updated
        assert base_mapping["settings"]["index"]["refresh_interval"] == "1s"  # added
        assert base_mapping["mappings"]["properties"]["field1"]["type"] == "text"  # preserved
        assert base_mapping["mappings"]["properties"]["field2"]["type"] == "keyword"  # added

    def test_mapping_update_deep_nesting(self):
        """Test deeply nested dictionary updates"""
        base_mapping = {
            "level1": {
                "level2": {
                    "level3": {
                        "level4": {
                            "existing_key": "existing_value",
                            "to_be_updated": "old_value"
                        }
                    }
                }
            }
        }
        
        settings = {
            "level1": {
                "level2": {
                    "level3": {
                        "level4": {
                            "to_be_updated": "new_value",
                            "new_key": "new_value"
                        }
                    }
                }
            }
        }
        
        mapping_update(base_mapping, settings)
        
        assert base_mapping["level1"]["level2"]["level3"]["level4"]["existing_key"] == "existing_value"
        assert base_mapping["level1"]["level2"]["level3"]["level4"]["to_be_updated"] == "new_value"
        assert base_mapping["level1"]["level2"]["level3"]["level4"]["new_key"] == "new_value"

    def test_mapping_update_type_mismatch_override(self):
        """Test behavior when types don't match (should override)"""
        base_mapping = {
            "key1": {"nested": "dict"},
            "key2": "string_value",
            "key3": 123
        }
        
        settings = {
            "key1": "now_a_string",  # dict -> string
            "key2": {"now": "a_dict"},  # string -> dict
            "key3": ["now", "a", "list"]  # int -> list
        }
        
        mapping_update(base_mapping, settings)
        
        assert base_mapping["key1"] == "now_a_string"
        assert base_mapping["key2"] == {"now": "a_dict"}
        assert base_mapping["key3"] == ["now", "a", "list"]

    def test_mapping_update_empty_dicts(self):
        """Test updating with empty dictionaries"""
        base_mapping = {"existing": "value"}
        settings = {}
        
        mapping_update(base_mapping, settings)
        assert base_mapping == {"existing": "value"}
        
        # Test empty base mapping
        base_mapping = {}
        settings = {"new": "value"}
        
        mapping_update(base_mapping, settings)
        assert base_mapping == {"new": "value"}

    def test_mapping_update_none_values(self):
        """Test handling of None values"""
        base_mapping = {
            "key1": "value1",
            "key2": None,
            "nested": {"inner": "value"}
        }
        
        settings = {
            "key1": None,
            "key2": "now_has_value",
            "key3": None,
            "nested": {"inner": None, "new_inner": "value"}
        }
        
        mapping_update(base_mapping, settings)
        
        assert base_mapping["key1"] is None
        assert base_mapping["key2"] == "now_has_value"
        assert base_mapping["key3"] is None
        assert base_mapping["nested"]["inner"] is None
        assert base_mapping["nested"]["new_inner"] == "value"

    def test_mapping_update_list_values(self):
        """Test handling of list values (should override, not merge)"""
        base_mapping = {
            "list_field": ["item1", "item2"],
            "nested": {
                "inner_list": [1, 2, 3]
            }
        }
        
        settings = {
            "list_field": ["new_item1", "new_item2", "new_item3"],
            "nested": {
                "inner_list": [4, 5]
            }
        }
        
        mapping_update(base_mapping, settings)
        
        assert base_mapping["list_field"] == ["new_item1", "new_item2", "new_item3"]
        assert base_mapping["nested"]["inner_list"] == [4, 5]

    def test_mapping_update_complex_opensearch_mapping(self):
        """Test with realistic OpenSearch mapping structure"""
        base_mapping = {
            "settings": {
                "index": {
                    "number_of_shards": 1,
                    "number_of_replicas": 0
                }
            },
            "mappings": {
                "properties": {
                    "title": {
                        "type": "text",
                        "analyzer": "standard"
                    },
                    "content": {
                        "type": "text"
                    }
                }
            }
        }
        
        # Add vector search capabilities
        vector_settings = {
            "settings": {
                "index": {
                    "knn": True,
                    "knn.algo_param.ef_search": 100
                }
            },
            "mappings": {
                "properties": {
                    "content_vector": {
                        "type": "knn_vector",
                        "dimension": 768,
                        "method": {
                            "name": "hnsw",
                            "space_type": "l2",
                            "engine": "nmslib"
                        }
                    }
                }
            }
        }
        
        mapping_update(base_mapping, vector_settings)
        
        # Verify original settings preserved
        assert base_mapping["settings"]["index"]["number_of_shards"] == 1
        assert base_mapping["settings"]["index"]["number_of_replicas"] == 0
        
        # Verify new settings added
        assert base_mapping["settings"]["index"]["knn"] is True
        assert base_mapping["settings"]["index"]["knn.algo_param.ef_search"] == 100
        
        # Verify original mappings preserved
        assert base_mapping["mappings"]["properties"]["title"]["type"] == "text"
        assert base_mapping["mappings"]["properties"]["content"]["type"] == "text"
        
        # Verify new mapping added
        assert base_mapping["mappings"]["properties"]["content_vector"]["type"] == "knn_vector"
        assert base_mapping["mappings"]["properties"]["content_vector"]["dimension"] == 768

    def test_mapping_update_modifies_original(self):
        """Test that mapping_update modifies the original dictionary in place"""
        original = {"key": "value"}
        original_id = id(original)
        
        settings = {"new_key": "new_value"}
        mapping_update(original, settings)
        
        # Should be the same object
        assert id(original) == original_id
        assert original == {"key": "value", "new_key": "new_value"}

    def test_mapping_update_recursive_behavior(self):
        """Test the recursive nature of the function"""
        base_mapping = {
            "a": {
                "b": {
                    "c": {
                        "d": "original"
                    }
                }
            }
        }
        
        settings = {
            "a": {
                "b": {
                    "c": {
                        "d": "updated",
                        "e": "new"
                    },
                    "f": "also_new"
                }
            }
        }
        
        mapping_update(base_mapping, settings)
        
        assert base_mapping["a"]["b"]["c"]["d"] == "updated"
        assert base_mapping["a"]["b"]["c"]["e"] == "new"
        assert base_mapping["a"]["b"]["f"] == "also_new"


class TestMappingIntegration:
    """Integration tests combining both functions"""

    def test_load_and_update_base_mapping(self):
        """Test loading base mapping and updating it"""
        # Create a base mapping file
        base_mapping = {
            "settings": {
                "index": {
                    "number_of_shards": 1,
                    "number_of_replicas": 0
                }
            },
            "mappings": {
                "properties": {
                    "title": {"type": "text"},
                    "content": {"type": "text"}
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(base_mapping, f)
            temp_path = f.name
        
        try:
            # Load the base mapping
            loaded_mapping = get_base_mapping(temp_path)
            
            # Update it with vector search settings
            vector_updates = {
                "settings": {
                    "index": {
                        "knn": True
                    }
                },
                "mappings": {
                    "properties": {
                        "vector_field": {
                            "type": "knn_vector",
                            "dimension": 384
                        }
                    }
                }
            }
            
            mapping_update(loaded_mapping, vector_updates)
            
            # Verify the result
            assert loaded_mapping["settings"]["index"]["number_of_shards"] == 1
            assert loaded_mapping["settings"]["index"]["knn"] is True
            assert loaded_mapping["mappings"]["properties"]["title"]["type"] == "text"
            assert loaded_mapping["mappings"]["properties"]["vector_field"]["type"] == "knn_vector"
            
        finally:
            os.unlink(temp_path)

    def test_real_base_mapping_file(self):
        """Test with the actual base_mapping.json file from the project"""
        # Get the path to the actual base mapping file
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        base_mapping_path = os.path.join(project_root, "mapping", "base_mapping.json")
        
        if os.path.exists(base_mapping_path):
            # Load the actual base mapping
            mapping = get_base_mapping(base_mapping_path)
            
            # Verify it has expected structure
            assert "settings" in mapping
            assert "mappings" in mapping
            assert "properties" in mapping["mappings"]
            
            # Test updating it
            updates = {
                "settings": {
                    "index": {
                        "knn": True
                    }
                }
            }
            
            original_shards = mapping["settings"]["index"]["number_of_shards"]
            mapping_update(mapping, updates)
            
            # Verify update worked and original values preserved
            assert mapping["settings"]["index"]["knn"] is True
            assert mapping["settings"]["index"]["number_of_shards"] == original_shards
        else:
            pytest.skip("base_mapping.json not found in expected location")


class TestMappingEdgeCases:
    """Test edge cases and error conditions"""

    def test_mapping_update_circular_reference_prevention(self):
        """Test that we don't create circular references"""
        base_mapping = {"key": "value"}
        
        # This should not create circular references
        mapping_update(base_mapping, base_mapping)
        
        # Should still be accessible
        assert base_mapping["key"] == "value"

    def test_mapping_update_with_special_characters(self):
        """Test mapping update with special characters in keys"""
        base_mapping = {
            "normal_key": "value",
            "key.with.dots": "dot_value",
            "key-with-dashes": "dash_value"
        }
        
        settings = {
            "key.with.dots": "updated_dot_value",
            "key@with@symbols": "symbol_value",
            "key with spaces": "space_value"
        }
        
        mapping_update(base_mapping, settings)
        
        assert base_mapping["key.with.dots"] == "updated_dot_value"
        assert base_mapping["key@with@symbols"] == "symbol_value"
        assert base_mapping["key with spaces"] == "space_value"

    def test_get_base_mapping_large_file(self):
        """Test loading a large mapping file"""
        # Create a large mapping structure
        large_mapping = {
            "settings": {"index": {"number_of_shards": 1}},
            "mappings": {
                "properties": {}
            }
        }
        
        # Add many properties
        for i in range(1000):
            large_mapping["mappings"]["properties"][f"field_{i}"] = {
                "type": "text" if i % 2 == 0 else "keyword",
                "index": True
            }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(large_mapping, f)
            temp_path = f.name
        
        try:
            result = get_base_mapping(temp_path)
            assert len(result["mappings"]["properties"]) == 1000
            assert result["mappings"]["properties"]["field_0"]["type"] == "text"
            assert result["mappings"]["properties"]["field_1"]["type"] == "keyword"
        finally:
            os.unlink(temp_path)
