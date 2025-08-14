# Models Module Unit Tests

This directory contains comprehensive unit tests for the `models` module of OpenSearch ML Quickstart.

## Test Structure

### Unit Tests (No External Dependencies)

#### **`test_ml_model.py`**
**Tests for the abstract `MlModel` base class:**

- **`TestMlModel`**: Tests for the abstract base class
  - Initialization with existing and new models
  - Model registration workflow
  - Model search functionality (`find_models`)
  - Model cleanup and deletion
  - String representation and model ID access
  - Error handling and retry logic
  - Abstract method enforcement

#### **`test_local_ml_model.py`**
**Tests for `LocalMlModel` class:**

- **`TestLocalMlModel`**: Tests for local model implementation
  - Initialization with default and custom model names
  - Configuration manager integration
  - Model registration with HuggingFace models
  - Version and format configuration handling
  - Inheritance from `MlModel` base class
  - Error handling during registration

#### **`test_remote_ml_model.py`**
**Tests for `RemoteMlModel` class:**

- **`TestRemoteMlModel`**: Tests for remote model implementation
  - Initialization for dense and sparse embeddings
  - Model deployment workflow
  - Connector integration
  - Task status monitoring
  - Cleanup including connector cleanup
  - Error handling during deployment

#### **`test_ml_model_group.py`**
**Tests for `MlModelGroup` class:**

- **`TestMlModelGroup`**: Tests for model group management
  - Model group creation and registration
  - Model group search and discovery
  - Model group deletion and cleanup
  - HTTP API interaction
  - Error handling and retry logic

#### **`test_helper.py`**
**Tests for helper functions:**

- **`TestGetMlModelGroup`**: Tests for `get_ml_model_group` function
- **`TestGetMlModel`**: Tests for `get_ml_model` function
  - Local model creation
  - Remote model creation (OS + Bedrock/SageMaker)
  - AOS model creation with domain configuration
  - Connector name generation
  - Configuration parameter handling
  - All supported host/model type combinations

### Integration Tests (Require OpenSearch Cluster)

#### **`test_integration.py`**
**Integration tests requiring actual OpenSearch clusters:**

- **`TestModelsIntegration`**: Real cluster testing
  - Model group creation and cleanup
  - Local model creation and deployment
  - Model search functionality
  - Helper function integration
  - End-to-end workflows

- **`TestModelsLegacyCompatibility`**: Legacy test compatibility
  - Backward compatibility with existing test functions
  - Legacy test function wrappers

### Legacy Tests (Maintained for Compatibility)

#### **`ml_model_group_test.py`**
**Legacy ML model group test (updated for compatibility)**

#### **`os_local_ml_model_test.py`**
**Legacy OS local ML model test (updated for compatibility)**

## Running Tests

### Using pytest (Recommended)

```bash
# Run all unit tests (excludes integration tests by default)
pytest tests/unit/models/

# Run specific test file
pytest tests/unit/models/test_ml_model.py

# Run specific test class
pytest tests/unit/models/test_ml_model.py::TestMlModel

# Run specific test method
pytest tests/unit/models/test_ml_model.py::TestMlModel::test_ml_model_init_with_defaults

# Run with verbose output
pytest -v tests/unit/models/

# Run integration tests (requires OpenSearch cluster)
pytest -m integration tests/unit/models/

# Run all tests including integration
pytest -m "unit or integration" tests/unit/models/
```

### Using the Custom Test Runner

```bash
# Run unit tests only
python tests/run_models_tests.py

# Run unit tests and integration tests
python tests/run_models_tests.py --integration
```

### Using Legacy Test Functions

```bash
# Run legacy ML model group test
python -c "from tests.unit.ml_models.ml_model_group_test import test; test()"

# Run legacy OS local ML model test
python -c "from tests.unit.ml_models.os_local_ml_model_test import test; test()"
```

## Test Features

### Comprehensive Mocking Strategy

#### **OpenSearch and ML Commons Clients**
```python
# Mock OpenSearch client
mock_os_client = Mock(spec=OpenSearch)
mock_os_client.http.post.return_value = {"task_id": "test_task"}
mock_os_client.http.get.return_value = {"state": "COMPLETED"}

# Mock ML Commons client
mock_ml_commons_client = Mock(spec=MLCommonClient)
mock_ml_commons_client.register_pretrained_model.return_value = None
mock_ml_commons_client.search_model.return_value = search_result
```

#### **Configuration Manager Integration**
```python
@patch('models.local_ml_model.get_local_embedding_model_name')
@patch('models.local_ml_model.get_local_embedding_model_version')
def test_configuration_integration(self, mock_get_version, mock_get_name):
    mock_get_name.return_value = "test/model"
    mock_get_version.return_value = "1.0.1"
    # Test configuration manager integration
```

#### **Connector Mocking**
```python
# Mock ML connector for remote models
mock_ml_connector = Mock(spec=MlConnector)
mock_ml_connector.connector_id.return_value = "test_connector_id"
mock_ml_connector.clean_up.return_value = None
```

### Test Coverage Areas

#### **Model Lifecycle Management**
- Model registration and deployment
- Model search and discovery
- Model cleanup and deletion
- Model group management
- Error handling and retry logic

#### **Configuration Integration**
- Configuration manager integration
- Default value handling
- Custom configuration override
- Type conversion and validation

#### **API Integration**
- OpenSearch ML Commons API calls
- HTTP request/response handling
- Task status monitoring
- Error response handling

#### **Inheritance and Polymorphism**
- Abstract base class enforcement
- Method overriding validation
- Polymorphic behavior testing
- Interface compliance

### Test Data and Scenarios

#### **Mock Model Data**
```python
# Model search response
search_result = {
    "hits": {
        "hits": [
            {
                "_id": "model1",
                "_source": {
                    "name": "Test Model",
                    "model_id": "actual_model_id"
                }
            }
        ]
    }
}

# Model group data
model_groups = [
    {
        "_id": "group1",
        "_source": {"name": "Test Group"}
    }
]
```

#### **Configuration Scenarios**
- Default configuration values
- Custom configuration overrides
- Missing configuration handling
- Invalid configuration handling
- Configuration type conversion

#### **Error Scenarios**
- Network failures
- API errors
- Invalid responses
- Timeout conditions
- Resource conflicts

## Dependencies

### Required for Unit Tests
- `pytest` - Test framework
- `unittest.mock` - Mocking framework (built-in)
- `opensearchpy` - OpenSearch Python client (for type hints)
- `opensearch_py_ml` - ML Commons client (for type hints)

### Required for Integration Tests
- Running OpenSearch cluster (self-managed)
- Proper configuration in `configs/osmlqs.yaml`
- Network connectivity to OpenSearch cluster
- ML Commons plugin enabled

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
1. **Use descriptive test names** that explain the scenario being tested
2. **Mock external dependencies** to ensure test isolation
3. **Test both success and failure cases** for comprehensive coverage
4. **Use realistic mock data** that matches actual API responses
5. **Group related tests** in test classes for organization

### Test Organization
- **Unit tests** should not require external OpenSearch clusters
- **Integration tests** should be marked with `@pytest.mark.integration`
- **Mock setup** should be done in `setup_method()` when shared across tests
- **Test data** should be realistic and representative

### Maintenance
- Keep tests updated when model classes change
- Ensure mocks match actual API behavior
- Add tests for new model types and functionality
- Update integration tests when OpenSearch APIs change

## Troubleshooting

### Common Issues

**Import Errors**:
```bash
# Ensure PYTHONPATH includes project root
export PYTHONPATH=/path/to/opensearch-ml-quickstart:$PYTHONPATH
```

**Mock Failures**:
- Verify mock return values match expected types
- Check that all required methods are mocked
- Ensure mock side effects handle all test scenarios

**Integration Test Failures**:
- Verify OpenSearch cluster is running and accessible
- Check that ML Commons plugin is enabled
- Ensure proper authentication credentials
- Verify network connectivity

**Dependency Issues**:
```bash
# Install test dependencies
pip install pytest pytest-cov

# Install project dependencies
pip install -r requirements.txt
```

### Model-Specific Issues

**Local Model Tests**:
- Ensure configuration manager functions are properly mocked
- Check that model registration parameters are correct
- Verify HuggingFace model names are valid

**Remote Model Tests**:
- Mock connector creation and cleanup
- Verify task status monitoring logic
- Check deployment payload structure

**Model Group Tests**:
- Mock HTTP API responses correctly
- Verify model group search logic
- Check cleanup confirmation handling

## Contributing

When adding new tests:
1. Follow the existing naming conventions and structure
2. Add appropriate docstrings and comments
3. Include both positive and negative test cases
4. Mock external dependencies appropriately
5. Update this README if adding new test categories
6. Ensure tests pass in isolation and as part of the full suite

### Test Categories to Consider
- **New Model Types**: Add tests when new model implementations are added
- **New Connectors**: Test new connector integrations
- **New Configuration Options**: Test new configuration parameters
- **Performance Tests**: Add tests for model performance characteristics
- **Security Tests**: Test authentication and authorization scenarios

## Architecture Integration

The models module tests integrate with other parts of the system:

### **Configuration System Integration**
- Tests verify proper configuration manager usage
- Mock configuration values for different scenarios
- Test default value handling and overrides

### **Client System Integration**
- Mock OpenSearch and ML Commons clients
- Test proper client method invocation
- Verify error handling from client operations

### **Connector System Integration**
- Mock connector creation and management
- Test connector lifecycle integration
- Verify proper connector cleanup

## Test Statistics

- **5 main test modules** with comprehensive coverage
- **15+ test classes** organized by functionality
- **100+ individual test methods** covering all scenarios
- **Integration tests** for real-world validation
- **Legacy compatibility** maintained
- **Full mock isolation** for unit tests
- **Realistic test data** matching production scenarios

This comprehensive test suite ensures the models module is thoroughly tested, maintainable, and reliable for production use, covering all aspects of ML model lifecycle management in OpenSearch.
