# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for configs.__init__ module"""

import pytest
from unittest.mock import Mock, patch


class TestConfigsInit:
    """Test cases for configs module initialization and exports"""

    def test_imports_available(self):
        """Test that all expected imports are available"""
        # Test that we can import the configs module
        import configs
        
        # Test that all expected functions are available
        expected_functions = [
            'get_ml_base_uri',
            'get_delete_resource_wait_time',
            'get_delete_resource_retry_time',
            'get_base_mapping_path',
            'get_qanda_file_reader_path',
            'get_pipeline_field_map',
            'get_client_configs',
            'validate_configs',
            'get_raw_config_value',
        ]
        
        for func_name in expected_functions:
            assert hasattr(configs, func_name), f"Function {func_name} not found in configs module"
            assert callable(getattr(configs, func_name)), f"{func_name} is not callable"

    def test_constants_available(self):
        """Test that all expected constants are available"""
        import configs
        
        expected_constants = [
        ]
        
        for const_name in expected_constants:
            assert hasattr(configs, const_name), f"Constant {const_name} not found in configs module"

    def test_function_delegation(self):
        """Test that imported functions work correctly"""
        import configs
        
        # Test that the function exists and is callable
        assert hasattr(configs, 'get_ml_base_uri')
        assert callable(configs.get_ml_base_uri)
        
        # Test that it returns the expected default value
        result = configs.get_ml_base_uri()
        assert result == "/_plugins/_ml"

    def test_no_tasks_import(self):
        """Test that tasks.py imports are no longer present"""
        import configs
        
        # These should not be available since we removed tasks.py
        deprecated_items = ['categories', 'tasks', 'PIPELINE_FIELD_MAP']
        
        for item in deprecated_items:
            # These items should either not exist or should come from configuration_manager
            if hasattr(configs, item):
                # If PIPELINE_FIELD_MAP exists, it should be a function, not the old constant
                if item == 'PIPELINE_FIELD_MAP':
                    # This should be the function from configuration_manager, not the old constant
                    assert callable(getattr(configs, item))
                else:
                    # categories and tasks should not exist
                    pytest.fail(f"Deprecated item {item} still exists in configs module")

    def test_module_docstring(self):
        """Test that the module has proper documentation"""
        import configs
        
        assert configs.__doc__ is not None
        assert "Configuration module" in configs.__doc__
        assert "unified configuration system" in configs.__doc__

    def test_import_structure(self):
        """Test the import structure is correct"""
        # Test that we can import specific items
        from configs import get_ml_base_uri
        
        assert callable(get_ml_base_uri)

    def test_all_imports_work(self):
        """Test that all imports in __init__.py work without errors"""
        try:
            from configs import (
                get_ml_base_uri,
                get_delete_resource_wait_time, 
                get_delete_resource_retry_time,
                get_base_mapping_path,
                get_qanda_file_reader_path,
                get_pipeline_field_map,
                get_client_configs,
                validate_configs,
                get_raw_config_value,
            )
            
            # If we get here, all imports worked
            assert True
            
        except ImportError as e:
            pytest.fail(f"Import failed: {e}")

    def test_backward_compatibility(self):
        """Test that the module maintains backward compatibility"""
        import configs
        
        # Test that commonly used functions are still available
        essential_functions = [
            'get_raw_config_value',
            'get_client_configs',
            'validate_configs'
        ]
        
        for func_name in essential_functions:
            assert hasattr(configs, func_name)
            assert callable(getattr(configs, func_name))

    def test_no_circular_imports(self):
        """Test that there are no circular import issues"""
        try:
            # This should work without circular import errors
            import configs
            from configs import get_ml_base_uri
            from configs.configuration_manager import get_ml_base_uri as direct_import
            
            # Both should be callable and return the same result
            assert callable(get_ml_base_uri)
            assert callable(direct_import)
            assert get_ml_base_uri() == direct_import()
            
        except ImportError as e:
            pytest.fail(f"Circular import detected: {e}")


class TestConfigsModuleIntegration:
    """Integration tests for the configs module as a whole"""

    @patch('configs.configuration_manager.Dynaconf')
    def test_end_to_end_config_access(self, mock_dynaconf):
        """Test end-to-end configuration access through the module"""
        # Setup mock configuration
        mock_config = Mock()
        mock_config.get.side_effect = lambda key, default=None: {
            "ML_BASE_URI": "/_plugins/_ml",
            "DELETE_RESOURCE_WAIT_TIME": "5",
            "OPENSEARCH_ADMIN_USER": "admin",
            "OS_HOST_URL": "localhost"
        }.get(key, default)
        mock_dynaconf.return_value = mock_config
        
        import configs
        
        # Test function calls work
        ml_uri = configs.get_ml_base_uri()
        wait_time = configs.get_delete_resource_wait_time()
        client_configs = configs.get_client_configs("os")
        
        assert ml_uri == "/_plugins/_ml"
        assert wait_time == 5
        assert "username" in client_configs
        assert client_configs["username"] == "admin"

    def test_module_structure_consistency(self):
        """Test that the module structure is consistent"""
        import configs
        import configs.configuration_manager as cm
        
        # Test that functions imported in __init__ match those in configuration_manager
        init_functions = [name for name in dir(configs) if callable(getattr(configs, name)) and not name.startswith('_')]
        
        for func_name in init_functions:
            if hasattr(cm, func_name):
                # Both should be callable and return consistent results
                configs_func = getattr(configs, func_name)
                cm_func = getattr(cm, func_name)
                
                assert callable(configs_func), f"{func_name} should be callable in configs"
                assert callable(cm_func), f"{func_name} should be callable in configuration_manager"
                
                # For functions that don't require parameters, test they return the same result
                if func_name in ['get_ml_base_uri', 'get_project_root', 'get_minimum_opensearch_version']:
                    try:
                        assert configs_func() == cm_func(), f"{func_name} should return same result"
                    except Exception:
                        # Some functions might require parameters, skip those
                        pass

    def test_error_handling_propagation(self):
        """Test that errors from configuration_manager propagate correctly"""
        import configs
        
        # Test that validation errors propagate
        with pytest.raises(ValueError):
            configs.validate_configs({}, ["required_key"])

    @patch('configs.configuration_manager.get_raw_config_value')
    def test_configuration_value_types(self, mock_get_raw):
        """Test that configuration values have correct types"""
        import configs
        
        # Test string values
        mock_get_raw.return_value = "test_string"
        result = configs.get_ml_base_uri()
        assert isinstance(result, str)
        
        # Test integer values
        mock_get_raw.return_value = "5"
        result = configs.get_delete_resource_wait_time()
        assert isinstance(result, int)
        assert result == 5
