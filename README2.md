# OpenSearch ML Quickstart

A comprehensive toolkit for building AI-powered search applications with OpenSearch, featuring semantic search, sparse search, hybrid search, and conversational AI capabilities. This quickstart provides production-ready examples and abstractions for working with OpenSearch's ML Commons plugin, vector embeddings, and large language models.

## 🚀 Features

- **Multiple Search Types**: Dense vector, sparse vector, hybrid, lexical, and conversational search
- **Flexible Model Support**: Local models, Amazon Bedrock, Amazon SageMaker, and Hugging Face
- **OpenSearch Deployment Options**: Self-managed clusters and Amazon OpenSearch Service
- **Production-Ready Examples**: 8+ complete search implementations with interactive interfaces
- **Workflow Automation**: OpenSearch Flow Framework integration for automated setup
- **Comprehensive ML Model Abstraction**: Unified interface for different model hosting options

## 📁 Project Structure

```
opensearch-ml-quickstart/
├── examples/                    # Interactive search examples
│   ├── cmd_line_interface.py   # Unified CLI and interface utilities
│   ├── dense_exact_search.py   # Dense vector search (exact k-NN)
│   ├── dense_hnsw_search.py    # Dense vector search (HNSW)
│   ├── sparse_search.py        # Neural sparse search
│   ├── hybrid_search.py        # Hybrid dense + sparse search
│   ├── lexical_search.py       # Traditional keyword search
│   ├── conversational_search.py # RAG-powered conversational AI
│   ├── workflow_example.py     # Custom workflow templates
│   └── workflow_with_template.py # Built-in workflow templates
├── ml_models/                  # ML model abstraction layer
├── client/                     # OpenSearch client wrappers
├── configs/                    # Configuration management
├── data_process/              # Data processing utilities
├── mapping/                   # Index mapping utilities
└── datasets/                  # Sample data storage
```

## 🏗️ Architecture Overview

### ML Model Hierarchy

The `ml_models/` directory provides a comprehensive abstraction layer for different ML model hosting scenarios:

```
MlModel (Abstract Base Class)
├── LocalMlModel                    # Models deployed within OpenSearch cluster
│   └── Supports: Hugging Face models, ONNX models
└── RemoteMlModel                   # Models hosted externally via connectors
    ├── OsBedrockMlModel           # Self-managed OS → Amazon Bedrock
    ├── OsSagemakerMlModel         # Self-managed OS → Amazon SageMaker  
    ├── AosBedrockMlModel          # Amazon OpenSearch Service → Bedrock
    └── AosSagemakerMlModel        # Amazon OpenSearch Service → SageMaker
```

#### Key Classes and Their Purpose:

**`MlModel` (Base Class)**
- Provides unified interface for all model types
- Handles model lifecycle: registration, deployment, deletion
- Manages model groups and versioning
- Abstract methods: `model_id()`, `deploy()`, `delete()`

**`LocalMlModel`**
- Deploys models directly within OpenSearch cluster nodes
- Supports Hugging Face transformers and ONNX models
- Ideal for self-managed clusters with sufficient resources
- **Not supported** on Amazon OpenSearch Service

**`RemoteMlModel`**
- Base class for externally hosted models
- Uses OpenSearch ML connectors for communication
- Handles connector creation and management
- Supports both dense and sparse embeddings

**Cloud-Specific Implementations:**
- **`AosBedrockMlModel`**: Amazon OpenSearch Service with Bedrock models
- **`AosSagemakerMlModel`**: Amazon OpenSearch Service with SageMaker endpoints
- **`OsBedrockMlModel`**: Self-managed OpenSearch with Bedrock models  
- **`OsSagemakerMlModel`**: Self-managed OpenSearch with SageMaker endpoints

### Client Architecture

The `client/` directory provides OpenSearch client abstractions and utilities:

#### Core Components:

**`OsMlClientWrapper`**
- Main client wrapper combining OpenSearch and ML Commons clients
- Provides high-level methods for index management, pipeline setup
- Handles k-NN configuration and neural search setup
- Manages model groups and connector lifecycle

**`get_client()` Factory Function**
- Creates appropriate client instances based on deployment type
- Supports both self-managed (`os`) and managed service (`aos`) configurations
- Handles authentication and connection management

**`index_utils` Module**
- Index creation and management utilities
- Data loading and bulk operations
- Mapping application and validation
- Category-based data processing

**Key Methods:**
```python
# Client initialization
client = OsMlClientWrapper(get_client("aos"))  # or "os"

# Index and pipeline setup
client.setup_for_kNN(ml_model, index_name, pipeline_name, field_map, embedding_type)

# Model group management
model_group_id = client.ml_model_group.model_group_id()
```

## 🎯 Search Examples

The `examples/` directory contains 8 production-ready search implementations:

| Example | Search Type | Use Case |
|---------|-------------|----------|
| `dense_exact_search.py` | Dense Vector (Exact k-NN) | High accuracy semantic search |
| `dense_hnsw_search.py` | Dense Vector (HNSW) | Fast approximate semantic search |
| `sparse_search.py` | Neural Sparse | Keyword-aware semantic search |
| `hybrid_search.py` | Dense + Sparse Hybrid | Best of both worlds |
| `lexical_search.py` | Traditional Keyword | Classic text search |
| `conversational_search.py` | RAG + LLM | Conversational AI with context |
| `workflow_example.py` | Custom Workflow | Automated setup with custom templates |
| `workflow_with_template.py` | Built-in Workflow | Automated setup with OpenSearch templates |

### Unified Command-Line Interface

All examples use the consolidated `cmd_line_interface.py` module providing:
- **Argument parsing**: Consistent CLI options across all examples
- **Interactive search**: Generic search loop with customizable query builders
- **User experience**: Colorized output, error handling, multiple quit options
- **Extensibility**: Callback pattern for custom search logic

## 📋 Prerequisites

### System Requirements
- **Python 3.10+** (required for pandas 2.0.3 compatibility)
- **OpenSearch 2.13.0+** (tested through 2.16.0)
- **Docker Desktop** (for local OpenSearch deployment)

### Data Requirements
- [Amazon PQA Dataset](https://registry.opendata.aws/amazon-pqa/) (download and extract to `datasets/` directory)

### OpenSearch Deployment Options

#### Option 1: Self-Managed OpenSearch
- Local Docker deployment (provided `docker-compose.yml`)
- Custom cluster deployment
- Supports both local and remote models

#### Option 2: Amazon OpenSearch Service
- Managed OpenSearch domain
- **Public access** configuration required
- **Fine-grained access control** with master user
- Remote models only (Bedrock/SageMaker)

### Model Hosting Options

#### Local Models (Self-Managed Only)
- Hugging Face transformers
- ONNX models
- Deployed within OpenSearch cluster

#### Remote Models (Both Deployments)
- **Amazon Bedrock**: Foundation models (Titan, Claude, etc.)
- **Amazon SageMaker**: Custom model endpoints
- Requires AWS IAM permissions

## 🛠️ Setup and Installation

### 1. Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd opensearch-ml-quickstart

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip3 install -r requirements.txt
```

### 2. Download Sample Data

```bash
# Download Amazon PQA dataset
# Visit: https://registry.opendata.aws/amazon-pqa/
# Extract files to: datasets/amazon-pqa/
```

### 3. Configuration

All configuration is managed through files in the `configs/` directory:

- **`.env`**: Environment variables and credentials
- **`config.py`**: Application configuration
- **`tasks.py`**: Test task definitions

## ⚙️ Configuration Guide

### Local OpenSearch + Local Model

**Minimal setup for local development:**

```bash
# configs/.env
OS_USERNAME=admin
OS_PASSWORD=YourSecurePassword123!
OS_HOST_URL=localhost
OS_PORT=9200
```

**Start local OpenSearch:**
```bash
export OPENSEARCH_INITIAL_ADMIN_PASSWORD="YourSecurePassword123!"
docker compose up
```

**Run example:**
```bash
python examples/dense_exact_search.py -c "earbud headphones" -n 100 -d
```

### Local OpenSearch + Remote Model (Bedrock)

**Configuration:**
```bash
# configs/.env - OpenSearch
OS_USERNAME=admin
OS_PASSWORD=YourSecurePassword123!
OS_HOST_URL=localhost
OS_PORT=9200

# Bedrock Configuration
OS_BEDROCK_ACCESS_KEY=your-aws-access-key
OS_BEDROCK_SECRET_KEY=your-aws-secret-key
OS_BEDROCK_REGION=us-west-2
OS_BEDROCK_URL=https://bedrock-runtime.us-west-2.amazonaws.com/model/amazon.titan-embed-text-v1/invoke
OS_BEDROCK_MODEL_DIMENSION=1536
```

**Run example:**
```bash
python examples/dense_exact_search.py -o os -c "earbud headphones" -n 100 -d
```

### Amazon OpenSearch Service + Bedrock

**Configuration:**
```bash
# configs/.env - OpenSearch Service
AOS_USERNAME=master-user
AOS_PASSWORD=YourMasterPassword123!
AOS_DOMAIN_NAME=your-domain-name
AOS_REGION=us-west-2
AOS_HOST_URL=https://your-domain.us-west-2.es.amazonaws.com
AOS_AWS_USER_NAME=your-iam-username

# Bedrock Configuration
AOS_BEDROCK_REGION=us-west-2
AOS_BEDROCK_CONNECTOR_ROLE_NAME=opensearch-bedrock-role
AOS_BEDROCK_CREATE_CONNECTOR_ROLE_NAME=opensearch-create-connector-role
AOS_BEDROCK_URL=https://bedrock-runtime.us-west-2.amazonaws.com/model/amazon.titan-embed-text-v1/invoke
AOS_BEDROCK_MODEL_DIMENSION=1536
```

**Run example:**
```bash
python examples/dense_exact_search.py -o aos -c "earbud headphones" -n 100 -d
```

### Amazon OpenSearch Service + SageMaker

**Configuration:**
```bash
# configs/.env - OpenSearch Service  
AOS_USERNAME=master-user
AOS_PASSWORD=YourMasterPassword123!
AOS_DOMAIN_NAME=your-domain-name
AOS_REGION=us-west-2
AOS_HOST_URL=https://your-domain.us-west-2.es.amazonaws.com
AOS_AWS_USER_NAME=your-iam-username

# SageMaker Configuration
AOS_SAGEMAKER_REGION=us-west-2
AOS_SAGEMAKER_CONNECTOR_ROLE_NAME=opensearch-sagemaker-role
AOS_SAGEMAKER_CREATE_CONNECTOR_ROLE_NAME=opensearch-create-sagemaker-role
AOS_SAGEMAKER_ENDPOINT_NAME=your-sagemaker-endpoint
## 🚀 Usage Examples

### Quick Start - Local Development

```bash
# 1. Start local OpenSearch
export OPENSEARCH_INITIAL_ADMIN_PASSWORD="YourSecurePassword123!"
docker compose up -d

# 2. Run dense vector search example
python examples/dense_exact_search.py \
  --categories "earbud headphones" \
  --number-of-docs-per-category 100 \
  --delete-existing-index

# 3. Interactive search interface will start
# Enter queries like: "wireless bluetooth headphones"
# Type 'quit' to exit
```

### Production Examples

#### Hybrid Search (Dense + Sparse)
```bash
python examples/hybrid_search.py \
  --opensearch-type aos \
  --categories "headphones" "speakers" \
  --number-of-docs-per-category 1000 \
  --delete-existing-index
```

#### Conversational AI Search
```bash
python examples/conversational_search.py \
  --opensearch-type aos \
  --categories "electronics" \
  --number-of-docs-per-category 500 \
  --delete-existing-index
```

#### Workflow Automation
```bash
python examples/workflow_example.py \
  --opensearch-type aos \
  --categories "all" \
  --number-of-docs-per-category 2000 \
  --delete-existing-index
```

### Command Line Options

All examples support these common options:

```bash
-o, --opensearch-type {os,aos}     # OpenSearch deployment type
-c, --categories [CATEGORIES ...]  # Data categories to load
-n, --number-of-docs-per-category  # Documents per category
-d, --delete-existing-index        # Delete existing index
-s, --bulk-send-chunk-size         # Bulk operation batch size
--no-load                          # Skip data loading
```

## 🔧 Advanced Configuration

### Custom Model Integration

```python
# Example: Custom Bedrock model
from ml_models import AosBedrockMlModel

model = AosBedrockMlModel(
    os_client=client.os_client,
    ml_commons_client=client.ml_commons_client,
    ml_connector=bedrock_connector,
    model_group_id=model_group_id,
    model_name="custom-claude-model"
)
```

### Custom Search Implementation

```python
# Example: Custom query builder
def build_custom_query(query_text, **kwargs):
    return {
        "size": 10,
        "query": {
            "bool": {
                "must": [
                    {"match": {"title": query_text}},
                    {"neural": {
                        "embedding": {
                            "query_text": query_text,
                            "model_id": kwargs["model_id"]
                        }
                    }}
                ]
            }
        }
    }

# Use with generic search loop
cmd_line_interface.interactive_search_loop(
    client=client,
    index_name="my_index",
    model_info="Custom Model",
    query_builder_func=build_custom_query,
    model_id=model.model_id()
)
```

### Workflow Templates

```python
# Custom workflow template
workflow_template = {
    "name": "custom_semantic_search",
    "description": "Custom semantic search workflow",
    "use_case": "SEMANTIC_SEARCH",
    "workflows": {
        "provision": {
            "nodes": [
                {
                    "id": "create_index_node",
                    "type": "create_index",
                    "user_inputs": {
                        "index_name": "custom_index",
                        "configurations": index_settings
                    }
                }
            ]
        }
    }
}
```

## 🧪 Testing

### Unit Tests
```bash
pytest tests/unit/
```

### Integration Tests
```bash
# Run specific test file
pytest tests/integration/main_test.py

# Run all integration tests (takes longer)
pytest tests/integration/
```

**Note**: Comment out model types or host types in test files if you haven't configured all options.

## 🔍 Troubleshooting

### Common Issues

#### `is_datetime_or_timedelta` Error
- **Cause**: Pandas version compatibility
- **Solution**: Use Python 3.10+ and pandas 2.0.3
- **Alternative**: Use Anaconda to manage Python versions

#### Connection Timeouts
- **Local OpenSearch**: Check Docker container status
- **Amazon OpenSearch Service**: Verify security group and network access

#### Model Deployment Failures
- **Local models**: Ensure sufficient cluster resources
- **Remote models**: Verify AWS credentials and permissions

#### Index Creation Errors
- **Solution**: Use `--delete-existing-index` flag
- **Check**: Index naming conflicts and mapping compatibility

### Debug Mode

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Performance Optimization

#### For Large Datasets:
```bash
# Increase bulk chunk size
python examples/dense_exact_search.py -s 500

# Process specific categories only
python examples/dense_exact_search.py -c "electronics" "books"

# Limit documents per category
python examples/dense_exact_search.py -n 1000
```

#### For Production:
- Use HNSW for faster approximate search
- Implement hybrid search for better relevance
- Configure appropriate index settings for your use case

## 📚 Additional Resources

- [OpenSearch Documentation](https://opensearch.org/docs/)
- [OpenSearch ML Commons Plugin](https://opensearch.org/docs/latest/ml-commons-plugin/)
- [Amazon OpenSearch Service](https://docs.aws.amazon.com/opensearch-service/)
- [Amazon Bedrock](https://docs.aws.amazon.com/bedrock/)
- [Amazon SageMaker](https://docs.aws.amazon.com/sagemaker/)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

