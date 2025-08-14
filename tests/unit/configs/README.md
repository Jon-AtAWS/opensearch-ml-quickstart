# Configs Module Unit Tests

This directory contains comprehensive unit tests for the `configs` module of OpenSearch ML Quickstart.

## Test Structure

### Unit Tests (No External Dependencies)

#### **`test_configuration_manager.py`**
**Comprehensive tests for `configs.configuration_manager` module:**

- **`TestEnums`**: Tests for configuration enums
  - `OpenSearchType` enum (OS, AOS)
  - `ModelProvider` enum (LOCAL, BEDROCK, SAGEMAKER)
  - `ModelType` enum (EMBEDDING, LLM)

- **`TestDataClasses`**: Tests for configuration dataclasses
  - `OpenSearchConfig` creation and defaults
  - `ModelConfig` creation and defaults
  - Field validation and type checking

- **`TestConfigurationManager`**: Tests for the main ConfigurationManager class
  - Initialization and setup
  - Configuration value retrieval
  - Safe type conversions (int, float)
  - Configuration building and validation
  - OpenSearch and model configuration generation

- **`TestModuleFunctions`**: Tests for module-level functions
  - `get_opensearch_config()`, `get_model_config()`
  - `get_embedding_config()`, `get_llm_config()`
  - `get_raw_config_value()` and utility functions
  - Constants and default value handling

#### **`test_configs_init.py`**
**Tests for `configs.__init__` module:**

- **`TestConfigsInit`**: Tests for module initialization
  - Import availability and correctness
  - Function and constant delegation
  - Removal of deprecated tasks.py imports
  - Module documentation and structure

- **`TestConfigsModuleIntegration`**: Integration tests for the module
  - End-to-end configuration access
  - Module structure consistency
  - Error handling propagation
  - Configuration value type validation

#### **`test_config_validation.py`**
**Tests for configuration validation and edge cases:**

- **`TestConfigurationValidation`**: Configuration validation logic
  - Valid configuration combinations
  - Missing required field handling
  - Invalid combination detection
  - Configuration info and key listing

- **`TestConfigurationEdgeCases`**: Edge cases and error conditions
  - Empty and partial configurations
  - Invalid numeric values
  - None and empty string handling
  - Unsupported types and providers

- **`TestConfigurationFileHandling`**: Configuration file handling
  - File path construction and validation
  - File existence checking
  - Path handling edge cases

- **`TestConfigurationConstants`**: Configuration constants testing
  - Default value validation
  - Constant value reasonableness
  - Type checking for all constants

### Integration Tests (Require Configuration Files)

#### **`test_integration.py`**
**Integration tests requiring actual configuration files:**

- **`TestConfigurationIntegration`**: Real configuration file tests
  - Actual config file loading
  - Raw config value retrieval
  - Client configuration generation
  - Available combinations validation
  - Model configuration testing

- **`TestConfigurationConsistency`**: Configuration consistency tests
  - Bedrock URL consistency across services
  - AWS region consistency
  - OpenSearch configuration validation
  - Cross-service configuration alignment

## Running Tests

### Using pytest (Recommended)

```bash
# Run all unit tests (excludes integration tests by default)
pytest tests/unit/configs/

# Run specific test file
pytest tests/unit/configs/test_configuration_manager.py

# Run specific test class
pytest tests/unit/configs/test_configuration_manager.py::TestConfigurationManager

# Run specific test method
pytest tests/unit/configs/test_configuration_manager.py::TestConfigurationManager::test_configuration_manager_init

# Run with verbose output
pytest -v tests/unit/configs/

# Run integration tests (requires actual config files)
pytest -m integration tests/unit/configs/

# Run all tests including integration
pytest -m "unit or integration" tests/unit/configs/
```

### Using the Custom Test Runner

```bash
# Run unit tests only
python tests/run_configs_tests.py

# Run unit tests and integration tests
python tests/run_configs_tests.py --integration
```

### Using Legacy Test Functions

```bash
# Run legacy integration tests
python -c "from tests.unit.configs.test_integration import test_config_loading; test_config_loading()"

# Run model config tests
python -c "from tests.unit.configs.test_integration import test_model_configs; test_model_configs()"
```

## Test Features

### Comprehensive Mocking Strategy
- **Dynaconf**: Mocked to avoid file system dependencies
- **Configuration Values**: Mocked with realistic test data
- **File System**: Mocked for path and file operations
- **External Dependencies**: All external calls mocked for isolation

### Test Coverage Areas

#### **Configuration Management**
- Configuration loading and parsing
- Type conversion and validation
- Default value handling
- Error condition management

#### **Data Structures**
- Enum value validation
- Dataclass creation and defaults
- Field type checking
- Serialization/deserialization

#### **Module Integration**
- Import structure validation
- Function delegation testing
- Constant availability checking
- Backward compatibility verification

#### **Validation Logic**
- Required field validation
- Configuration combination checking
- Value format validation
- Consistency checking across configurations

#### **Edge Cases**
- Empty and None value handling
- Invalid type conversion
- Missing configuration files
- Malformed configuration data

### Test Data and Scenarios

#### **Mock Configuration Data**
```python
{
    "OPENSEARCH_ADMIN_USER": "admin",
    "OPENSEARCH_ADMIN_PASSWORD": "password",
    "OS_HOST_URL": "localhost",
    "OS_PORT": "9200",
    "AOS_HOST_URL": "https://test.aos.amazonaws.com",
    "AWS_REGION": "us-west-2",
    "BEDROCK_LLM_MODEL_NAME": "claude-3",
    "BEDROCK_LLM_MAX_TOKENS": "8000",
    "BEDROCK_LLM_TEMPERATURE": "0.1",
    # ... and more
}
```

#### **Test Scenarios**
- **Valid Configurations**: Complete, properly formatted configurations
- **Invalid Configurations**: Missing required fields, invalid formats
- **Edge Cases**: Empty values, None values, type conversion failures
- **Integration**: Real configuration file loading and validation

## Dependencies

### Required for Unit Tests
- `pytest` - Test framework
- `unittest.mock` - Mocking framework (built-in)
- `dataclasses` - For dataclass testing (built-in)
- `typing` - For type hint testing (built-in)

### Required for Integration Tests
- `dynaconf` - Configuration management library
- Actual `osmlqs.yaml` configuration file
- Valid configuration values in the file

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
4. **Use realistic test data** that matches actual configuration formats
5. **Group related tests** in test classes for organization

### Test Organization
- **Unit tests** should not require external files or services
- **Integration tests** should be marked with `@pytest.mark.integration`
- **Mock setup** should be done in `setup_method()` when shared across tests
- **Test data** should be realistic and representative

### Maintenance
- Keep tests updated when configuration structure changes
- Ensure mocks match actual configuration behavior
- Add tests for new configuration options
- Update integration tests when config file format changes

## Troubleshooting

### Common Issues

**Import Errors**:
```bash
# Ensure PYTHONPATH includes project root
export PYTHONPATH=/path/to/opensearch-ml-quickstart:$PYTHONPATH
```

**Mock Failures**:
- Verify mock return values match expected types
- Check that all required configuration keys are mocked
- Ensure mock side_effect functions handle all test cases

**Integration Test Failures**:
- Verify `osmlqs.yaml` file exists and is properly formatted
- Check that required configuration values are present
- Ensure dynaconf is installed and working

**Dependency Issues**:
```bash
# Install test dependencies
pip install pytest pytest-cov

# Install project dependencies (including dynaconf)
pip install -r requirements.txt
```

### Configuration File Issues

**Missing Configuration File**:
- Ensure `configs/osmlqs.yaml` exists
- Check file permissions and accessibility
- Verify file format is valid YAML

**Invalid Configuration Values**:
- Check that required keys are present
- Verify value formats match expected types
- Ensure consistency across related configuration values

**Dynaconf Issues**:
- Install dynaconf: `pip install dynaconf`
- Check dynaconf version compatibility
- Verify configuration file path is correct

## Contributing

When adding new tests:
1. Follow the existing naming conventions and structure
2. Add appropriate docstrings and comments
3. Include both positive and negative test cases
4. Mock external dependencies appropriately
5. Update this README if adding new test categories
6. Ensure tests pass in isolation and as part of the full suite

### Test Categories to Consider
- **New Configuration Options**: Add tests when new config keys are added
- **New Validation Logic**: Test new validation rules and constraints
- **New Data Structures**: Test new dataclasses or enums
- **New Integration Points**: Test new external service integrations
- **Performance Tests**: Add tests for configuration loading performance

## Test Statistics

- **4 main test modules** with comprehensive coverage
- **12+ test classes** organized by functionality
- **80+ individual test methods** covering all scenarios
- **Integration tests** for real-world validation
- **Edge case coverage** for robust error handling

This comprehensive test suite ensures the configs module is thoroughly tested, maintainable, and reliable for production use.
