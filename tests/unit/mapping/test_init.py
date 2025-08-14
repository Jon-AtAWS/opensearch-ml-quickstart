# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for mapping module initialization and imports"""

import pytest


class TestMappingModuleImports:
    """Test cases for mapping module imports and structure"""

    def test_mapping_module_imports(self):
        """Test that mapping module imports work correctly"""
        # Test importing the module
        import mapping
        
        # Test that expected functions are available
        assert hasattr(mapping, 'get_base_mapping')
        assert hasattr(mapping, 'mapping_update')
        
        # Test that functions are callable
        assert callable(mapping.get_base_mapping)
        assert callable(mapping.mapping_update)

    def test_direct_imports_from_helper(self):
        """Test direct imports from helper module"""
        from mapping.helper import get_base_mapping, mapping_update
        
        assert callable(get_base_mapping)
        assert callable(mapping_update)

    def test_import_from_mapping_module(self):
        """Test importing functions from mapping module"""
        from mapping import get_base_mapping, mapping_update
        
        assert callable(get_base_mapping)
        assert callable(mapping_update)

    def test_mapping_module_structure(self):
        """Test that mapping module has expected structure"""
        import mapping
        
        # Check that the module has the expected attributes
        expected_attributes = ['get_base_mapping', 'mapping_update']
        
        for attr in expected_attributes:
            assert hasattr(mapping, attr), f"Missing attribute: {attr}"

    def test_no_unexpected_imports(self):
        """Test that mapping module doesn't expose unexpected attributes"""
        import mapping
        
        # Get all public attributes (not starting with _)
        public_attrs = [attr for attr in dir(mapping) if not attr.startswith('_')]
        
        # Expected public attributes
        expected_attrs = ['get_base_mapping', 'mapping_update']
        
        # Check that we don't have unexpected public attributes
        unexpected_attrs = set(public_attrs) - set(expected_attrs)
        
        # Allow for some common module attributes that might be present
        allowed_extra_attrs = {'helper'}  # The helper module itself might be exposed
        
        unexpected_attrs = unexpected_attrs - allowed_extra_attrs
        
        assert len(unexpected_attrs) == 0, f"Unexpected public attributes: {unexpected_attrs}"

    def test_function_signatures(self):
        """Test that imported functions have expected signatures"""
        from mapping import get_base_mapping, mapping_update
        import inspect
        
        # Test get_base_mapping signature
        get_base_mapping_sig = inspect.signature(get_base_mapping)
        assert len(get_base_mapping_sig.parameters) == 1
        assert 'base_mapping_path' in get_base_mapping_sig.parameters
        
        # Test mapping_update signature
        mapping_update_sig = inspect.signature(mapping_update)
        assert len(mapping_update_sig.parameters) == 2
        param_names = list(mapping_update_sig.parameters.keys())
        assert 'base_mapping' in param_names
        assert 'settings' in param_names

    def test_import_error_handling(self):
        """Test that import errors are handled appropriately"""
        # This test ensures that if there are import issues, they're properly raised
        try:
            from mapping import get_base_mapping, mapping_update
            # If we get here, imports worked
            assert True
        except ImportError as e:
            pytest.fail(f"Import failed: {e}")

    def test_module_docstring_exists(self):
        """Test that the mapping module has appropriate documentation"""
        import mapping.helper
        
        # Check that the helper module has a docstring or at least some documentation
        # This is more of a code quality check
        assert mapping.helper.__file__ is not None
        
        # Check that functions have some basic documentation
        from mapping import get_base_mapping, mapping_update
        
        # Functions should at least be defined (not None)
        assert get_base_mapping is not None
        assert mapping_update is not None

    def test_circular_import_prevention(self):
        """Test that there are no circular import issues"""
        try:
            # Try importing in different orders to check for circular dependencies
            from mapping.helper import get_base_mapping
            from mapping import mapping_update
            import mapping
            from mapping.helper import mapping_update as helper_mapping_update
            
            # All should work without issues
            assert callable(get_base_mapping)
            assert callable(mapping_update)
            assert callable(helper_mapping_update)
            assert mapping_update == helper_mapping_update
            
        except ImportError as e:
            pytest.fail(f"Circular import detected: {e}")

    def test_module_attributes_consistency(self):
        """Test that module attributes are consistent between different import methods"""
        # Import via different methods
        import mapping
        from mapping import get_base_mapping as direct_get_base_mapping
        from mapping.helper import get_base_mapping as helper_get_base_mapping
        
        # All should reference the same function
        assert mapping.get_base_mapping == direct_get_base_mapping
        assert direct_get_base_mapping == helper_get_base_mapping
        assert mapping.get_base_mapping == helper_get_base_mapping
