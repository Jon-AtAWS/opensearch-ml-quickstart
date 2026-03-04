# Fix for Issue #2: sys.path.append Everywhere

## Problem

Almost every file in the project used `sys.path.append(os.path.dirname(...))` to manipulate the Python path. This was messy, hard to maintain, and indicated poor project structure.

Example from before:
```python
# examples/dense_hnsw_search.py
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from client import OsMlClientWrapper, get_client
```

## Solution

Removed all `sys.path.append` calls from 17 files across the project:

### Library Code
- `client/helper.py`
- `client/os_ml_client_wrapper.py`

### Example Files
- `examples/cmd_line_interface.py`
- `examples/conversational_agent.py`
- `examples/conversational_search.py`
- `examples/dense_exact_search.py`
- `examples/dense_hnsw_search.py`
- `examples/hybrid_local_search.py`
- `examples/hybrid_search.py`
- `examples/lexical_search.py`
- `examples/mcp/launch.py`
- `examples/mcp/mcp_server.py`
- `examples/semantic_search_workflow.py`
- `examples/sparse_search.py`
- `examples/strands_search.py`
- `examples/workflow_example.py`
- `examples/workflow_with_template.py`

## After

Files are now clean:
```python
# examples/dense_hnsw_search.py
import os
import sys

from client import OsMlClientWrapper, get_client
```

Library files use proper relative imports:
```python
# client/os_ml_client_wrapper.py
from . import index_utils
```

## Setup Required

Users need to set `PYTHONPATH` once to the project root:

```bash
# Option 1: In shell session
export PYTHONPATH="${PWD}:${PYTHONPATH}"

# Option 2: In shell profile (~/.bashrc, ~/.zshrc)
export PYTHONPATH="/full/path/to/opensearch-ml-quickstart:${PYTHONPATH}"

# Option 3: When running scripts
PYTHONPATH=/path/to/opensearch-ml-quickstart python examples/dense_hnsw_search.py
```

## Benefits

- ✅ No more path manipulation hacks in every file
- ✅ Cleaner, more maintainable code
- ✅ Better IDE support and autocomplete
- ✅ Simple one-time PYTHONPATH setup
- ✅ Follows Python best practices
- ✅ Easier to understand for new contributors

## Testing

All files compile successfully:
```bash
python -m py_compile client/helper.py examples/dense_hnsw_search.py
```

Verify no sys.path.append remains:
```bash
grep -r "sys.path.append" --include="*.py" client/ examples/ models/ connectors/
# Should return 0 results
```

## For IDE Users

**VS Code:** Add to `.vscode/settings.json`:
```json
{
  "python.analysis.extraPaths": ["${workspaceFolder}"]
}
```

**PyCharm:** Right-click project root → "Mark Directory as" → "Sources Root"
