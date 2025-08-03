# Data Process Module Unit Tests

This directory contains comprehensive unit tests for the `data_process` module of OpenSearch ML Quickstart.

## Test Structure

### Unit Tests (No External Dependencies)

#### **`test_qanda_file_reader.py`**
**Comprehensive tests for `data_process.qanda_file_reader` module:**

- **`TestQAndAFileReader`**: Tests for the main QAndAFileReader class
  - **Initialization**: Tests constructor with default and custom parameters
  - **Category Management**: Tests category name/constant/filename conversions
  - **File Operations**: Tests file size retrieval and error handling
  - **Data Reading**: Tests question reading with and without enrichment
  - **Data Enrichment**: Tests question enrichment with fake data generation
  - **Utility Methods**: Tests helper methods and internal functions
  - **Error Handling**: Tests various error conditions and edge cases

### Integration Tests (Require Amazon PQA Data Files)

#### **`test_integration.py`**
**Integration tests requiring actual Amazon PQA data files:**

- **`TestDataProcessIntegration`**: Real data file testing
  - Actual data file reading and processing
  - File size validation with real files
  - Question reading and enrichment with real data
  - Data consistency validation
  - Cross-category data processing

- **`TestDataProcessLegacyCompatibility`**: Legacy test compatibility
  - Backward compatibility with existing test functions
  - Legacy test function wrappers

### Legacy Tests (Maintained for Compatibility)

#### **`qanda_file_reader_test.py`**
**Legacy QAndA file reader test (updated for compatibility)**

## Running Tests

### Using pytest (Recommended)

```bash
# Run all unit tests (excludes integration tests by default)
pytest tests/unit/data_process/

# Run specific test file
pytest tests/unit/data_process/test_qanda_file_reader.py

# Run specific test class
pytest tests/unit/data_process/test_qanda_file_reader.py::TestQAndAFileReader

# Run specific test method
pytest tests/unit/data_process/test_qanda_file_reader.py::TestQAndAFileReader::test_qanda_file_reader_init_default

# Run with verbose output
pytest -v tests/unit/data_process/

# Run integration tests (requires Amazon PQA data files)
pytest -m integration tests/unit/data_process/

# Run all tests including integration
pytest -m "unit or integration" tests/unit/data_process/
```

### Using the Custom Test Runner

```bash
# Run unit tests only
python tests/run_data_process_tests.py

# Run unit tests and integration tests
python tests/run_data_process_tests.py --integration
```

### Using Legacy Test Functions

```bash
# Run legacy QAndA file reader test
python -c "from tests.unit.data_process.qanda_file_reader_test import test; test()"
```

## Test Features

### Comprehensive Mocking Strategy

#### **File System Operations**
```python
# Mock file operations
@patch('builtins.open', new_callable=mock_open)
@patch('os.path.getsize')
def test_file_operations(self, mock_getsize, mock_file):
    mock_getsize.return_value = 1024
    mock_file.return_value.__iter__ = Mock(return_value=iter(test_data))
    # Test file operations without actual files
```

#### **Random Data Generation**
```python
# Mock random and faker for consistent testing
@patch('random.random')
def test_random_generation(self, mock_random):
    mock_random.return_value = 0.5  # Consistent test results
    reader.fake.name = Mock(return_value="Test Name")
    # Test data enrichment with predictable results
```

#### **Configuration Integration**
```python
# Mock configuration manager
@patch('data_process.qanda_file_reader.get_qanda_file_reader_path')
def test_configuration_integration(self, mock_get_path):
    mock_get_path.return_value = "/test/data/path"
    # Test configuration integration
```

### Test Coverage Areas

#### **Data File Management**
- File path construction and validation
- File size retrieval and error handling
- File existence checking
- Directory structure validation

#### **Category Management**
- Category name to constant conversion
- Constant to filename mapping
- Reverse lookups and validation
- Category enumeration and iteration

#### **Data Reading and Processing**
- JSON file parsing and iteration
- Document limiting and pagination
- Error handling for malformed data
- Memory-efficient streaming processing

#### **Data Enrichment**
- Fake data generation for testing
- Answer enrichment with user data
- Bullet point concatenation
- Category name injection

#### **Utility Functions**
- String transformation utilities
- Mapping generation functions
- Validation and consistency checks
- Debug and development helpers

### Test Data and Scenarios

#### **Mock Question Data**
```python
sample_question = {
    "question_id": "test_id_1",
    "asin": "B001TEST",
    "question": "Test question?",
    "answers": [
        {"answer": "Test answer 1", "answer_time": "2023-01-01"},
        {"answer": "Test answer 2", "answer_time": "2023-01-02"}
    ],
    "bullet_point1": "Feature 1",
    "bullet_point2": "Feature 2",
    "bullet_point3": "Feature 3",
    "bullet_point4": "Feature 4",
    "bullet_point5": "Feature 5",
    "product_description": "Test product",
    "brand_name": "Test Brand",
    "item_name": "Test Item"
}
```

#### **Test Scenarios**
- **Valid Data**: Well-formed JSON with all required fields
- **Invalid Data**: Malformed JSON, missing fields, empty files
- **Edge Cases**: Empty answers, missing bullet points, boundary conditions
- **Large Data**: Testing with document limits and pagination
- **Error Conditions**: File not found, permission errors, parsing failures

#### **Enrichment Testing**
- Gender distribution testing (male/female/other)
- Age range validation (0-100)
- Product rating validation (1-5)
- Geographic coordinate validation
- Fake name generation consistency

## Dependencies

### Required for Unit Tests
- `pytest` - Test framework
- `unittest.mock` - Mocking framework (built-in)
- `faker` - Fake data generation library
- `json` - JSON processing (built-in)

### Required for Integration Tests
- Amazon PQA dataset files in `datasets/amazon_pqa/` directory
- Proper configuration in `configs/osmlqs.yaml`
- File system access to data directory

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
2. **Mock file system operations** to avoid dependencies on actual files
3. **Test both success and failure cases** for comprehensive coverage
4. **Use realistic test data** that matches actual Amazon PQA format
5. **Group related tests** in test classes for organization

### Test Organization
- **Unit tests** should not require actual Amazon PQA data files
- **Integration tests** should be marked with `@pytest.mark.integration`
- **Mock setup** should be done in `setup_method()` when shared across tests
- **Test data** should be representative of real Amazon PQA structure

### Maintenance
- Keep tests updated when QAndAFileReader class changes
- Ensure mocks match actual file formats and API behavior
- Add tests for new data processing functionality
- Update integration tests when Amazon PQA data format changes

## Troubleshooting

### Common Issues

**Import Errors**:
```bash
# Ensure PYTHONPATH includes project root
export PYTHONPATH=/path/to/opensearch-ml-quickstart:$PYTHONPATH
```

**Mock Failures**:
- Verify mock return values match expected data types
- Check that file content mocks return proper JSON format
- Ensure random mocks provide consistent test results

**Integration Test Failures**:
- Verify Amazon PQA data files are present in datasets directory
- Check that data files are properly formatted JSON
- Ensure configuration points to correct data directory

**Data Format Issues**:
- Verify test data matches actual Amazon PQA format
- Check that required fields are present in mock data
- Ensure JSON parsing handles actual data structure

### Data-Specific Issues

**File Operations**:
- Mock file operations to avoid file system dependencies
- Test both successful and failed file operations
- Verify proper error handling for missing files

**JSON Processing**:
- Test with both valid and invalid JSON data
- Verify proper error handling for malformed JSON
- Check memory efficiency with large data files

**Data Enrichment**:
- Test faker integration and consistency
- Verify random data generation within expected ranges
- Check that enrichment doesn't modify original data structure

## Contributing

When adding new tests:
1. Follow the existing naming conventions and structure
2. Add appropriate docstrings and comments
3. Include both positive and negative test cases
4. Mock external dependencies appropriately
5. Update this README if adding new test categories
6. Ensure tests pass in isolation and as part of the full suite

### Test Categories to Consider
- **New Data Sources**: Add tests when supporting new data formats
- **New Enrichment Features**: Test new data enrichment capabilities
- **Performance Tests**: Add tests for large data file processing
- **Memory Tests**: Test memory efficiency with large datasets
- **Concurrency Tests**: Test thread safety if applicable

## Amazon PQA Dataset Integration

The data_process module is specifically designed to work with the Amazon PQA (Product Question Answering) dataset:

### **Dataset Structure**
- **Categories**: 100+ product categories (jeans, monitors, headphones, etc.)
- **File Format**: JSON Lines format (one JSON object per line)
- **Question Structure**: Product questions with multiple answers
- **Metadata**: Product information, bullet points, descriptions

### **Data Processing Pipeline**
1. **Category Selection**: Choose specific product categories to process
2. **File Reading**: Stream JSON data from category-specific files
3. **Data Enrichment**: Add fake user data for testing/demo purposes
4. **Document Limiting**: Control memory usage with document limits
5. **Format Transformation**: Convert to OpenSearch-compatible format

### **Testing Strategy**
- **Unit Tests**: Mock all file operations and data structures
- **Integration Tests**: Use actual Amazon PQA data files when available
- **Performance Tests**: Validate memory efficiency with large files
- **Consistency Tests**: Ensure data integrity across processing pipeline

## Architecture Integration

The data_process module integrates with other parts of the system:

### **Configuration System Integration**
- Uses configuration manager for data directory paths
- Respects configuration-driven data limits
- Integrates with environment-specific settings

### **Search System Integration**
- Provides data in format expected by search indices
- Supports both dense and sparse search scenarios
- Enables category-based data loading

### **ML Pipeline Integration**
- Formats data for embedding generation
- Supports chunking for ML model input limits
- Provides enriched data for training/testing

## Test Statistics

- **1 main test module** with comprehensive coverage
- **1 test class** with 25+ test methods
- **50+ individual test scenarios** covering all functionality
- **Integration tests** for real-world validation
- **Legacy compatibility** maintained
- **Full mock isolation** for unit tests
- **Realistic test data** matching Amazon PQA format

This comprehensive test suite ensures the data_process module is thoroughly tested, maintainable, and reliable for production use, covering all aspects of Amazon PQA data processing and enrichment.
