# Client Module Unit Tests

This directory contains comprehensive unit tests for the `client` module of OpenSearch ML Quickstart.

## Test Structure

### Unit Tests (No External Dependencies)
- **`test_helper.py`** - Tests for `client.helper` module
  - `TestGetClient` - Tests OpenSearch client creation and configuration
  - `TestParseVersion` - Tests version string parsing
  - `TestCheckClientVersion` - Tests version validation

- **`test_index_utils.py`** - Tests for `client.index_utils` module
  - `TestHandleIndexCreation` - Tests index creation and deletion
  - `TestHandleDataLoading` - Tests data loading orchestration
  - `TestLoadCategory` - Tests category-specific data loading
  - `TestSendBulkIgnoreExceptions` - Tests bulk document sending
  - `TestGetIndexSize` - Tests index size retrieval

- **`test_os_ml_client_wrapper.py`** - Tests for `client.os_ml_client_wrapper` module
  - `TestOsMlClientWrapper` - Tests the main client wrapper class
    - Initialization and setup
    - Pipeline configuration (dense/sparse)
    - k-NN setup and configuration
    - Resource cleanup

### Integration Tests (Require OpenSearch Cluster)
- **`test_integration.py`** - Integration tests requiring actual OpenSearch clusters
  - `TestClientIntegration` - Tests real cluster connectivity
  - Legacy test functions for backward compatibility

### Legacy Tests
- **`aos_helper_test.py`** - Legacy AOS helper test (maintained for compatibility)
- **`os_helper_test.py`** - Legacy OS helper test (maintained for compatibility)

## Running Tests

### Using pytest (Recommended)

```bash
# Run all unit tests (excludes integration tests by default)
pytest tests/unit/client/

# Run specific test file
pytest tests/unit/client/test_helper.py

# Run specific test class
pytest tests/unit/client/test_helper.py::TestGetClient

# Run specific test method
pytest tests/unit/client/test_helper.py::TestGetClient::test_get_client_os_with_port

# Run with verbose output
pytest -v tests/unit/client/

# Run integration tests (requires OpenSearch cluster)
pytest -m integration tests/unit/client/

# Run all tests including integration
pytest -m "unit or integration" tests/unit/client/
```

### Using the Custom Test Runner

```bash
# Run unit tests only
python tests/run_client_tests.py

# Run unit tests and integration tests
python tests/run_client_tests.py --integration
```

### Using Legacy Test Functions

```bash
# Run legacy OS test
python -c "from tests.unit.client.os_helper_test import test; test()"

# Run legacy AOS test  
python -c "from tests.unit.client.aos_helper_test import test; test()"
```

## Test Features

### Mocking Strategy
- **OpenSearch Client**: Mocked using `unittest.mock.Mock` with `spec=OpenSearch`
- **ML Commons Client**: Mocked to avoid ML model dependencies
- **Configuration**: Mocked to avoid file system dependencies
- **External APIs**: All external calls are mocked for isolation

### Test Coverage
- **Client Creation**: Tests both OS and AOS client creation with different configurations
- **Error Handling**: Tests exception handling and error conditions
- **Configuration Validation**: Tests version checking and configuration validation
- **Index Operations**: Tests index creation, deletion, and size retrieval
- **Data Loading**: Tests bulk data loading with various scenarios
- **Pipeline Management**: Tests ML pipeline creation for dense and sparse embeddings
- **Resource Cleanup**: Tests proper cleanup of models, pipelines, and indices

### Test Data
- Uses mock data and configurations to avoid dependencies on actual datasets
- Parameterized tests for different scenarios and edge cases
- Realistic mock responses that match OpenSearch API behavior

## Dependencies

### Required for Unit Tests
- `pytest` - Test framework
- `unittest.mock` - Mocking framework (built-in)
- `opensearchpy` - OpenSearch Python client (for type hints)

### Required for Integration Tests
- Running OpenSearch cluster (self-managed or AOS)
- Proper configuration in `configs/osmlqs.yaml`
- Network connectivity to OpenSearch cluster

### Optional
- `pytest-cov` - For test coverage reports
- `pytest-xdist` - For parallel test execution

## Configuration

Tests use the pytest configuration in `tests/pytest.ini`:
- Excludes integration tests by default
- Configures logging and output format
- Sets up test markers for categorization

## Best Practices

### Writing New Tests
1. **Use descriptive test names** that explain what is being tested
2. **Mock external dependencies** to ensure test isolation
3. **Test both success and failure cases** for comprehensive coverage
4. **Use appropriate assertions** that provide clear failure messages
5. **Group related tests** in test classes for organization

### Test Organization
- **Unit tests** should not require external resources
- **Integration tests** should be marked with `@pytest.mark.integration`
- **Slow tests** should be marked with `@pytest.mark.slow`
- **Mock setup** should be done in `setup_method()` when shared across tests

### Maintenance
- Keep tests updated when the client module changes
- Ensure mocks match the actual API behavior
- Add tests for new functionality
- Remove or update tests for deprecated functionality

## Troubleshooting

### Common Issues

**Import Errors**:
```bash
# Ensure PYTHONPATH includes project root
export PYTHONPATH=/path/to/opensearch-ml-quickstart:$PYTHONPATH
```

**Mock Failures**:
- Verify mock specifications match actual classes
- Check that mock return values match expected types
- Ensure all required methods are mocked

**Integration Test Failures**:
- Verify OpenSearch cluster is running and accessible
- Check configuration in `configs/osmlqs.yaml`
- Ensure proper authentication credentials

**Dependency Issues**:
```bash
# Install test dependencies
pip install pytest pytest-cov

# Install project dependencies
pip install -r requirements.txt
```

## Contributing

When adding new tests:
1. Follow the existing naming conventions
2. Add appropriate docstrings and comments
3. Include both positive and negative test cases
4. Update this README if adding new test categories
5. Ensure tests pass in isolation and as part of the full suite
