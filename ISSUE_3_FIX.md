# Fix for Issue #3: Interactive input() Calls in Library Code

## Problem

Library code in `MlModel`, `MlModelGroup`, `MlConnector`, and `OsMlClientWrapper` contained interactive `input()` prompts. This made the code:
- **Untestable**: Tests would hang waiting for user input
- **Unusable in automation**: Scripts would block waiting for confirmation
- **Surprising**: Library code shouldn't prompt users directly

Example from before:
```python
# models/ml_model.py
def _undeploy_and_delete_model(self, model_id):
    user_input = (
        input(f"Do you want to undeploy and delete the model {model_id}? (y/n): ")
        .strip()
        .lower()
    )
    if user_input != "y":
        logging.info("Undeploy and delete model canceled.")
        return
    # ... deletion code ...
```

## Solution

Added an optional `confirm` parameter (default `True`) to all cleanup/delete methods:

### Files Modified

1. **models/ml_model.py**
   - `_undeploy_and_delete_model(model_id, confirm=True)`

2. **models/ml_model_group.py**
   - `_delete_model_group(model_group_id, confirm=True)`

3. **connectors/ml_connector.py**
   - `_delete_connector(connector_id, confirm=True)`

4. **client/os_ml_client_wrapper.py**
   - `cleanup_kNN(ml_model=None, index_name=None, pipeline_name=None, confirm=True)`

## After

```python
# models/ml_model.py
def _undeploy_and_delete_model(self, model_id, confirm=True):
    if confirm:
        user_input = (
            input(f"Do you want to undeploy and delete the model {model_id}? (y/n): ")
            .strip()
            .lower()
        )
        if user_input != "y":
            logging.info("Undeploy and delete model canceled.")
            return
    
    # ... deletion code runs without prompting if confirm=False ...
```

## Usage Examples

### Interactive mode (backward compatible):
```python
# Still works exactly as before
model._undeploy_and_delete_model(model_id)  # Will prompt
wrapper.cleanup_kNN(model, index, pipeline)  # Will prompt
```

### Automated mode (new capability):
```python
# For tests and automation - no prompts!
model._undeploy_and_delete_model(model_id, confirm=False)
wrapper.cleanup_kNN(model, index, pipeline, confirm=False)
```

### In test suites:
```python
def test_cleanup():
    # No hanging, no prompts
    wrapper.cleanup_kNN(
        ml_model=model,
        index_name="test_index",
        pipeline_name="test_pipeline",
        confirm=False
    )
```

### In CI/CD pipelines:
```python
# cleanup_script.py
import os

if os.environ.get('CI'):
    # Non-interactive in CI
    wrapper.cleanup_kNN(model, index, pipeline, confirm=False)
else:
    # Interactive locally
    wrapper.cleanup_kNN(model, index, pipeline)
```

## Benefits

- ✅ Library code is now testable
- ✅ Can be used in automation scripts
- ✅ Backward compatible (defaults to interactive mode)
- ✅ Clear separation between library and CLI concerns
- ✅ No surprises for library users
- ✅ Follows best practices for library design

## Testing

Test that the parameter works:
```python
# Should not prompt
wrapper.cleanup_kNN(model, index, pipeline, confirm=False)

# Should prompt (default behavior)
wrapper.cleanup_kNN(model, index, pipeline)
```
