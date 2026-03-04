# Fix for Issue #5: Duplicated Code

## Problem

Multiple instances of duplicated code across the codebase:

### 1. Duplicate Function Definition
`get_pipeline_field_map` was defined twice in `configs/configuration_manager.py`:

```python
# Line 657-660
def get_pipeline_field_map() -> Dict[str, str]:
    """Get the pipeline field mapping."""
    return get_raw_config_value("PIPELINE_FIELD_MAP", {"chunk": "chunk_embedding"})

# Line 661-664 - DUPLICATE!
def get_pipeline_field_map() -> Dict[str, str]:
    """Get the pipeline field mapping."""
    return get_raw_config_value("PIPELINE_FIELD_MAP", {"chunk": "chunk_embedding"})
```

### 2. Duplicated Logic
`mapping_update` in `mapping/helper.py` was duplicated as `update_mapping` in `data_process/base_dataset.py`:

```python
# mapping/helper.py
def mapping_update(base_mapping, settings):
    for key, value in settings.items():
        if (
            key in base_mapping
            and isinstance(base_mapping[key], dict)
            and isinstance(value, dict)
        ):
            mapping_update(base_mapping[key], value)
        else:
            base_mapping[key] = value

# data_process/base_dataset.py - DUPLICATE LOGIC!
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
```

## Solution

### 1. Remove Duplicate Function

**File: configs/configuration_manager.py**

Remove the second definition of `get_pipeline_field_map()` (lines 661-664).

Keep only:
```python
def get_pipeline_field_map() -> Dict[str, str]:
    """Get the pipeline field mapping."""
    return get_raw_config_value("PIPELINE_FIELD_MAP", {"chunk": "chunk_embedding"})
```

### 2. Consolidate Mapping Logic

**File: data_process/base_dataset.py**

Add import at the top:
```python
from mapping.helper import mapping_update
```

Replace the `update_mapping` method implementation:
```python
def update_mapping(self, base_mapping: Dict[str, Any], updates: Dict[str, Any]) -> None:
    """Update mapping with additional fields (e.g., vector fields).
    
    This is a convenience wrapper around mapping.helper.mapping_update.
    """
    mapping_update(base_mapping, updates)
```

## Benefits

- ✅ Single source of truth for each function
- ✅ Easier maintenance (changes only need to be made once)
- ✅ Reduced code size
- ✅ Less chance of implementations diverging
- ✅ Clearer code organization
- ✅ Follows DRY (Don't Repeat Yourself) principle

## Testing

Verify the changes work:

```python
# Test get_pipeline_field_map is only defined once
from configs import configuration_manager
import inspect

source = inspect.getsource(configuration_manager)
count = source.count("def get_pipeline_field_map")
assert count == 1, f"Expected 1 definition, found {count}"

# Test that BaseDataset.update_mapping still works
from data_process import base_dataset
from mapping import helper

assert hasattr(base_dataset.BaseDataset, 'update_mapping')
assert hasattr(helper, 'mapping_update')
```

## Files Modified

1. **configs/configuration_manager.py**
   - Removed duplicate `get_pipeline_field_map()` function

2. **data_process/base_dataset.py**
   - Added import: `from mapping.helper import mapping_update`
   - Changed `update_mapping()` to delegate to `mapping_update()`
