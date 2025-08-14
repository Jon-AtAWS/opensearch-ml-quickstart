# OpenSearch ML Quickstart

The OpenSearch ML Quickstart is a comprehensive toolkit designed to simplify the process of building AI-powered search applications with OpenSearch. This project provides a unified framework for implementing various search capabilities including semantic search, sparse search, hybrid search, conversational AI, and agentic workflows. By offering production-ready examples and abstractions for OpenSearch's ML Commons plugin, vector embeddings, and large language models, developers can quickly implement advanced search functionality without deep expertise in each technology.

Whether you're building a product search engine, a knowledge base, or a conversational AI system, this toolkit provides the building blocks you need to get started quickly while maintaining the flexibility to customize for your specific use case.

## 🚀 Features

OpenSearch ML Quickstart offers a rich set of capabilities designed to address different search scenarios and requirements. The toolkit is built around the principle of providing multiple search approaches that can be used independently or combined for optimal results.

### Search Types

The toolkit supports six distinct search approaches, each optimized for different use cases and data characteristics:

**Dense Vector Search** leverages semantic understanding through dense embeddings, offering both exact k-NN for maximum accuracy and HNSW (Hierarchical Navigable Small World) for fast approximate results. This approach excels at understanding the meaning behind queries, making it ideal for scenarios where users might phrase questions differently but seek similar information.

**Sparse Vector Search** provides keyword-aware semantic search using neural sparse models. Unlike dense vectors, sparse vectors maintain interpretability while still capturing semantic relationships, making them particularly effective for domains where specific terminology matters.

**Hybrid Search** combines the strengths of both dense and sparse approaches, using sophisticated ranking algorithms to merge results for optimal relevance. This approach often provides the best overall search experience by balancing semantic understanding with keyword precision.

**Traditional Lexical Search** implements classic keyword-based text search using BM25 and other established algorithms. While not semantic, this approach remains valuable for exact phrase matching and scenarios where traditional search behavior is expected.

**Conversational Search** implements Retrieval-Augmented Generation (RAG) patterns, combining search results with large language models to provide contextual, conversational responses. This approach transforms search from a list of results into an interactive dialogue.

**Agentic Search** represents the cutting edge of search technology, employing AI agents that can reason about complex queries, plan multi-step search strategies, and execute sophisticated information retrieval tasks autonomously.

### Model Hosting Flexibility

The toolkit supports multiple model hosting strategies to accommodate different infrastructure preferences and requirements:

**Local Models** can be deployed directly within your OpenSearch cluster, supporting both Hugging Face transformers and ONNX models. This approach provides maximum control and eliminates external dependencies, though it requires sufficient cluster resources.

**Amazon Bedrock** integration provides access to foundation models like Titan and Claude without managing model infrastructure. This managed approach offers enterprise-grade reliability and automatic scaling.

**Amazon SageMaker** support enables the use of custom model endpoints, allowing you to deploy specialized models while benefiting from SageMaker's managed infrastructure.

**Hugging Face** direct integration allows you to leverage the vast ecosystem of open-source models with minimal configuration overhead.

### Deployment Flexibility

The toolkit accommodates different deployment preferences through two primary approaches:

**Self-managed OpenSearch** deployments provide complete control over your search infrastructure, whether running locally for development or in production environments. This approach supports all model hosting options and provides maximum customization flexibility.

**Amazon OpenSearch Service** offers a fully managed experience with automatic scaling, security, and maintenance. While this approach is limited to remote model hosting, it significantly reduces operational overhead.

## 📁 Project Structure

The OpenSearch ML Quickstart is organized into several key directories, each serving a specific purpose in the overall architecture. This modular structure allows developers to understand and extend individual components without needing to grasp the entire codebase.

The **examples** directory contains ready-to-run demonstrations of different search capabilities. Each example is a complete, working implementation that showcases a specific search approach, from basic dense vector search to advanced agentic workflows. The `cmd_line_interface.py` module provides a unified command-line experience across all examples, ensuring consistent user interaction patterns.

The **models** directory implements a sophisticated abstraction layer for machine learning models. The design follows object-oriented principles with a clear inheritance hierarchy, allowing the same code to work seamlessly with local models deployed within OpenSearch or remote models hosted on external services like Amazon Bedrock or SageMaker.

The **client** directory provides high-level wrappers around OpenSearch's native client libraries. These wrappers abstract away the complexity of ML Commons operations, index management, and data loading, presenting a simplified interface for common tasks.

The **configs** directory houses a comprehensive configuration management system that handles the complexity of different deployment scenarios, model types, and hosting options. The system provides type-safe configuration with automatic validation and environment-aware loading.

```
opensearch-ml-quickstart/
├── examples/                          # Interactive search examples
│   ├── cmd_line_interface.py         # Unified CLI and interface utilities
│   ├── dense_exact_search.py         # Dense vector search (exact k-NN)
│   ├── dense_hnsw_search.py          # Dense vector search (HNSW)
│   ├── sparse_search.py              # Neural sparse search
│   ├── hybrid_search.py              # Hybrid dense + sparse search
│   ├── lexical_search.py             # Traditional keyword search
│   ├── conversational_search.py      # RAG-powered conversational AI
│   ├── agentic_search.py             # AI agents for complex search tasks
│   ├── workflow_example.py           # Custom workflow templates
│   └── workflow_with_template.py     # Built-in workflow templates
├── models/                           # ML model abstraction layer
│   ├── ml_model.py                   # Abstract base class for all models
│   ├── local_ml_model.py             # Local model implementations
│   ├── remote_ml_model.py            # Remote model base class
│   ├── ml_model_group.py             # Model group management
│   └── helper.py                     # Model factory functions
├── connectors/                       # ML connector implementations
│   ├── ml_connector.py               # Abstract base connector class
│   ├── embedding_connector.py        # Embedding model connectors
│   ├── llm_connector.py              # LLM model connectors
│   ├── helper.py                     # Connector configuration utilities
│   └── connector_payloads/           # Connector payload templates
├── client/                           # OpenSearch client wrappers
│   ├── helper.py                     # Client factory and utilities
│   ├── os_ml_client_wrapper.py       # ML Commons client wrapper
│   └── index_utils.py                # Index management utilities
├── configs/                          # Configuration management system
│   ├── __init__.py                   # Configuration module interface
│   ├── configuration_manager.py      # Core configuration logic
│   ├── osmlqs.yaml                   # Main configuration file
│   └── tasks.py                      # Task definitions
├── data_process/                     # Data processing utilities
│   └── qanda_file_reader.py          # Amazon PQA dataset reader
├── mapping/                          # Index mapping utilities
│   └── base_mapping.json             # Base index mapping template
└── datasets/                         # Sample data storage
    └── amazon_pqa/                   # Amazon PQA dataset
```

## 🏗️ Architecture Overview

The OpenSearch ML Quickstart follows a layered architecture that separates concerns and provides clear abstractions for different aspects of search functionality. This design enables developers to work at the appropriate level of abstraction for their needs while maintaining the flexibility to customize lower-level components when necessary.

### Configuration Management System

At the foundation of the toolkit is a sophisticated configuration management system that handles the complexity of different deployment scenarios, model types, and hosting options. The system provides type-safe configuration with automatic validation and environment-aware loading, ensuring that your application is properly configured regardless of the deployment environment.

```python
from configs import get_opensearch_config, get_model_config

# Get OpenSearch configuration for your deployment type
os_config = get_opensearch_config("os")  # Self-managed OpenSearch
# or
os_config = get_opensearch_config("aos")  # Amazon OpenSearch Service

# Get model configuration for your specific setup
model_config = get_model_config("aos", "bedrock", "embedding")
```

The configuration system offers several key advantages that make it easy to manage complex deployments. **Type-safe configuration** uses enum-based validation to ensure that only valid combinations of deployment types and model providers are used, preventing common configuration errors at startup time. **Environment-aware loading** automatically detects configuration sources from .env files, YAML configuration files, and environment variables, providing flexibility in how you manage secrets and settings across different environments. **Built-in validation** checks for required fields and validates configuration combinations, giving you immediate feedback when something is misconfigured. **Flexible deployment support** accommodates multiple OpenSearch and model hosting scenarios without requiring code changes.

The configuration system uses a **flat YAML structure** in `osmlqs.yaml` that is processed by the `configuration_manager` to create structured, type-safe configuration objects. The flat structure makes it easy to manage environment variables and override specific settings:

```yaml
# configs/osmlqs.yaml - Flat structure with descriptive variable names
# OpenSearch configurations
OS_HOST_URL: localhost
OS_PORT: 9200
OPENSEARCH_ADMIN_USER: admin
OPENSEARCH_ADMIN_PASSWORD: admin

AOS_DOMAIN_NAME: my-domain
AOS_HOST_URL: https://my-domain.us-west-2.es.amazonaws.com
AWS_REGION: us-west-2

# Model configurations
BEDROCK_EMBEDDING_URL: https://bedrock-runtime.us-west-2.amazonaws.com/model/amazon.titan-embed-text-v1/invoke
BEDROCK_MODEL_DIMENSION: 1536
BEDROCK_LLM_MODEL_NAME: anthropic.claude-v2
BEDROCK_LLM_MAX_TOKENS: 1000

# SageMaker configurations
SAGEMAKER_DENSE_ARN: arn:aws:sagemaker:us-west-2:123456789:endpoint/my-endpoint
SAGEMAKER_DENSE_MODEL_DIMENSION: 384
```

The `configuration_manager` transforms this flat structure into a nested, type-safe configuration hierarchy by:

1. **Loading** the flat YAML using Dynaconf for environment variable support
2. **Parsing** variable names to determine their logical grouping (OS vs AOS, Bedrock vs SageMaker, etc.)
3. **Building** structured configuration objects with proper type conversion and validation
4. **Providing** a clean API that abstracts the flat structure complexity

This approach combines the simplicity of flat configuration files with the benefits of structured, validated configuration objects in your application code.

### ML Model Hierarchy

The model system represents one of the most sophisticated aspects of the toolkit, implementing a well-designed class hierarchy that abstracts different model types and hosting scenarios. This design allows the same application code to work seamlessly with local models deployed within your OpenSearch cluster or remote models hosted on external services like Amazon Bedrock or SageMaker.

The hierarchy follows object-oriented design principles with a clear inheritance structure:

```
MlModel (Abstract Base Class)
├── LocalMlModel                      # Models deployed within OpenSearch cluster
│   └── Supports: Hugging Face models, ONNX models
└── RemoteMlModel                     # Models hosted externally via connectors
    └── Uses connector framework for external model integration
```

The model factory pattern simplifies the creation of model instances by abstracting away the complexity of choosing the right implementation class:

```python
from models.helper import get_ml_model

# Create a Bedrock embedding model for Amazon OpenSearch Service
embedding_model = get_ml_model(
    host_type="aos",                  # Amazon OpenSearch Service
    model_type="bedrock",             # Amazon Bedrock models
    model_config=embedding_config,    # Configuration from config system
    embedding_type="dense"            # Dense vector embeddings
)

# Create a SageMaker LLM model for self-managed OpenSearch
llm_model = get_ml_model(
    host_type="os",                   # Self-managed OpenSearch
    model_type="sagemaker",           # Amazon SageMaker endpoints
    model_config=llm_config,          # LLM-specific configuration
    embedding_type="llm"              # Large language model
)
```

This architecture provides several key benefits that make it easy to work with different model hosting scenarios. The **unified interface** ensures that all models implement the same `MlModel` interface, so your application code doesn't need to change when you switch between different model types or hosting options. **Easy migration** between hosting options requires minimal code changes, allowing you to start with local development and move to production hosting without rewriting your application. **Type safety** provides automatic validation of model and hosting combinations, preventing invalid configurations at runtime. **Lifecycle management** includes built-in model registration, deployment, and cleanup functionality, handling the complex orchestration required to manage ML models in OpenSearch.

### Connector Architecture

The connector system provides a sophisticated abstraction layer for integrating with external ML services. Connectors handle the complex details of communicating with different model providers, managing authentication, and translating between OpenSearch's ML Commons API and external service protocols.

The connector hierarchy follows a similar pattern to the model system:

```
MlConnector (Abstract Base Class)
├── EmbeddingConnector               # Specialized for embedding models
│   ├── Bedrock embedding connectors
│   ├── SageMaker embedding connectors
│   └── Hugging Face embedding connectors
└── LlmConnector                     # Specialized for large language models
    ├── Bedrock LLM connectors
    ├── SageMaker LLM connectors
    └── Custom LLM endpoint connectors
```

The connector system provides several key capabilities:

**Service Integration** handles the specifics of different ML service APIs, including authentication, request formatting, and response parsing. **Protocol Translation** converts between OpenSearch's internal model representation and the specific formats required by external services. **Error Handling** provides robust error handling and retry logic for network issues and service failures. **Configuration Management** automatically generates the correct connector configurations based on your deployment type and model provider.

```python
from connectors.helper import get_remote_connector_configs

# Get connector configuration for Bedrock embedding model
connector_config = get_remote_connector_configs(
    host_type="aos",                  # Amazon OpenSearch Service
    model_type="bedrock",             # Amazon Bedrock
    embedding_type="embedding"        # Embedding model type
)

# The connector handles all the complex integration details
# including authentication, request formatting, and response parsing
```

### Client Architecture

The client system provides high-level abstractions for OpenSearch operations, wrapping the native OpenSearch client libraries with ML-aware functionality. This approach simplifies common tasks like setting up k-NN search, managing ingest pipelines, and handling bulk data loading.

```python
from client.helper import get_client
from client.os_ml_client_wrapper import OsMlClientWrapper

# Get an OpenSearch client configured for your deployment
os_client = get_client("aos")  # Amazon OpenSearch Service
# or
os_client = get_client("os")   # Self-managed OpenSearch

# Create an ML-enabled wrapper that adds ML Commons functionality
ml_client = OsMlClientWrapper(os_client)

# Setup for k-NN search with automatic pipeline and index configuration
ml_client.setup_for_kNN(
    ml_model=embedding_model,         # Your configured embedding model
    index_name="my_index",            # Target index name
    pipeline_name="my_pipeline",      # Ingest pipeline name
    embedding_type="dense"            # Type of embeddings (dense/sparse)
)
```

The client wrapper handles many of the complex details involved in setting up ML-powered search, including creating ingest pipelines that automatically generate embeddings for new documents, configuring index mappings with the appropriate vector field types and dimensions, and managing the lifecycle of ML models and their associated resources.

## 🎯 Search Examples

The OpenSearch ML Quickstart provides a comprehensive set of examples that demonstrate different search approaches and their practical applications. Each example is designed to be both educational and production-ready, showing how to implement specific search patterns while following best practices for configuration, error handling, and user interaction.

### Core Search Types

The following table outlines the primary search examples included in the toolkit, each targeting specific use cases and requirements:

| Example | Search Type | Use Case | Model Requirements |
|---------|-------------|----------|-------------------|
| `dense_exact_search.py` | Dense Vector (Exact k-NN) | High accuracy semantic search | Embedding model |
| `dense_hnsw_search.py` | Dense Vector (HNSW) | Fast approximate semantic search | Embedding model |
| `sparse_search.py` | Neural Sparse | Keyword-aware semantic search | Sparse embedding model |
| `hybrid_search.py` | Dense + Sparse Hybrid | Best of both worlds | Dense + Sparse models |
| `lexical_search.py` | Traditional Keyword | Classic text search | None |
| `conversational_search.py` | RAG + LLM | Conversational AI with context | Embedding + LLM models |
| `conversational_agent.py` | AI Agents | Complex reasoning and planning | LLM model with agent capabilities |

**Dense Vector Search** examples demonstrate how to implement semantic search using vector embeddings. The exact k-NN approach provides the highest accuracy by computing exact distances between query and document vectors, making it ideal for applications where precision is critical. The HNSW (Hierarchical Navigable Small World) variant trades some accuracy for significantly improved performance, making it suitable for large-scale applications where sub-second response times are required.

**Sparse Vector Search** showcases neural sparse models that combine the interpretability of traditional keyword search with the semantic understanding of neural networks. This approach is particularly valuable in domains where specific terminology and exact keyword matches remain important, such as legal documents or technical specifications.

**Hybrid Search** demonstrates how to combine multiple search approaches to achieve optimal results. By merging dense semantic search with sparse keyword-aware search, this approach often provides the best overall user experience, capturing both the semantic intent and specific terminology in user queries.

**Conversational Search** implements Retrieval-Augmented Generation (RAG) patterns, showing how to build interactive AI assistants that can answer questions based on your document corpus. This example demonstrates the integration of search results with large language models to provide contextual, conversational responses.

**Agentic Search** represents the most advanced example, implementing AI agents capable of complex reasoning and multi-step planning. These agents can break down complex queries, execute multiple search strategies, and synthesize results to provide comprehensive answers to sophisticated questions.

### Advanced Examples

Beyond the core search types, the toolkit includes advanced examples that demonstrate workflow automation and template-based setup:

| Example | Type | Description |
|---------|------|-------------|
| `workflow_example.py` | Custom Workflow | Automated setup with custom templates |
| `workflow_with_template.py` | Built-in Workflow | Automated setup with OpenSearch templates |

These workflow examples show how to automate the entire setup process for search applications, from index creation and model deployment to pipeline configuration and data loading. The custom workflow example demonstrates how to define your own setup templates, while the built-in workflow example uses OpenSearch's native workflow templates for standardized deployments.

### Agentic Search (New!)

The agentic search example represents a breakthrough in search technology, demonstrating how to build AI agents that can handle complex, multi-faceted queries through sophisticated reasoning and planning capabilities.

These AI agents can **reason** about complex queries by analyzing the user's intent and breaking down multi-part questions into manageable components. They can **plan** execution strategies by determining the optimal sequence of search operations and data analysis steps needed to provide comprehensive answers. The agents **execute** these plans by performing multiple searches, analyzing intermediate results, and refining their approach based on what they discover. Finally, they can **learn** from feedback and results, adapting their strategies for similar future queries.

Agentic search requires OpenSearch version 3.1, not currently available on Amazon OpenSearch Service.

```python
# Example: Agentic search for complex research tasks
python examples/conversational_agent.py --host-type os --categories "electronics,books" --max-docs 1000

# The agent can handle queries like:
# "Find the best wireless headphones under $200 and compare them with similar books about audio technology"
```

This example demonstrates how agentic search can handle queries that would require multiple traditional searches and manual analysis, providing a single interface for complex information retrieval tasks.

### Unified Command-Line Interface

All examples utilize the consolidated `cmd_line_interface.py` module, which provides a consistent and user-friendly experience across different search types. This unified approach ensures that users can easily switch between different search examples without learning new command-line interfaces or interaction patterns.

The interface provides several common features across all examples, including interactive search loops that allow users to perform multiple queries in a single session, consistent error handling and user feedback, standardized configuration loading and validation, and common data loading and index management operations.

```bash
# Dense vector search with specific categories and document limits
python examples/dense_exact_search.py --host-type os --categories "electronics" --max-docs 500

# Hybrid search with index recreation for clean testing
python examples/hybrid_search.py --host-type aos --categories "books,electronics" --delete-existing-index

# Conversational search for technology-related queries
python examples/conversational_search.py --host-type aos --categories "technology"

# Agentic search with extended document corpus for complex reasoning
python examples/agentic_search.py --host-type os --categories "research" --max-docs 1000
```

The command-line interface supports several common options that work across all examples. The `--host-type` parameter specifies whether you're using self-managed OpenSearch (`os`) or Amazon OpenSearch Service (`aos`). The `--categories` option allows you to specify which data categories to load from the Amazon PQA dataset. The `--max-docs` parameter limits the number of documents loaded per category, useful for testing or resource-constrained environments. The `--delete-existing-index` flag forces deletion of existing indices, ensuring a clean setup for testing. The `--no-load` option skips data loading entirely, useful when working with pre-existing indices.

## 📋 Prerequisites

Before getting started with the OpenSearch ML Quickstart, you'll need to ensure your environment meets certain requirements and that you have access to the necessary data and services. The toolkit is designed to work in various deployment scenarios, from local development to production environments.

### System Requirements

The toolkit requires **Python 3.10 or later** due to its dependency on pandas 2.0.3, which introduced important performance improvements and API changes that the data processing components rely on. You'll also need **OpenSearch 2.13.0 or later** (tested through 2.16.0) to ensure compatibility with the ML Commons plugin features used throughout the examples. For local development, **Docker Desktop** is essential as it provides the easiest way to run a complete OpenSearch cluster with all necessary plugins pre-configured.

### Data Requirements

The examples in this toolkit use the **Amazon Product Question Answering (PQA) Dataset**, which provides a rich corpus of product-related questions and answers across multiple categories. This dataset is particularly well-suited for demonstrating search capabilities because it contains natural language queries with corresponding answers and contextual information. You'll need to download this dataset from the [AWS Open Data Registry](https://registry.opendata.aws/amazon-pqa/) and extract it to the `datasets/amazon_pqa/` directory in your project root.

### Deployment Options

The toolkit supports two primary deployment approaches, each with its own advantages and considerations:

**Self-managed OpenSearch Cluster** deployments give you complete control over your search infrastructure. This approach works well for local development using Docker Compose, custom cluster deployments in your own infrastructure, or when you need specific cluster configurations. The key advantage is that self-managed clusters support both local models (deployed within the cluster) and remote models (hosted on external services), providing maximum flexibility in your model hosting strategy.

**Amazon OpenSearch Service** offers a fully managed experience where AWS handles cluster provisioning, scaling, security, and maintenance. This approach requires configuring public access to your domain and setting up fine-grained access control with a master user. While this deployment option is limited to remote models (Amazon Bedrock or SageMaker endpoints), it significantly reduces operational overhead and provides enterprise-grade reliability.

### Model Hosting Requirements

The choice of model hosting approach depends on your deployment strategy and performance requirements:

**Local Models** can only be used with self-managed OpenSearch clusters and require sufficient cluster resources to host the models. The toolkit supports both Hugging Face transformers and ONNX models, allowing you to choose between ease of use and optimized performance. Local models provide the lowest latency and eliminate external dependencies, but require careful resource planning.

**Remote Models** work with both deployment options and leverage external services for model hosting. **Amazon Bedrock** provides access to foundation models like Titan and Claude without requiring you to manage model infrastructure, offering automatic scaling and enterprise-grade reliability. **Amazon SageMaker** enables the use of custom model endpoints, allowing you to deploy specialized models while benefiting from SageMaker's managed infrastructure. Both remote model options require proper **AWS IAM permissions** to access the respective services.

## 🛠️ Setup and Installation

Getting started with the OpenSearch ML Quickstart involves setting up your development environment, downloading the necessary data, and configuring the system for your specific deployment scenario. The process is designed to be straightforward while accommodating different infrastructure preferences.

### 1. Environment Setup

Begin by cloning the repository and setting up a Python virtual environment to isolate the project dependencies:

```bash
# Clone the repository
git clone <repository-url>
cd opensearch-ml-quickstart

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

The virtual environment ensures that the project's dependencies don't conflict with other Python projects on your system. The requirements.txt file includes all necessary packages for running the examples and working with different model hosting options.

### 2. Download Sample Data

The toolkit uses the Amazon Product Question Answering (PQA) dataset for demonstrations. This dataset provides realistic product-related queries and answers that showcase the different search capabilities effectively:

```bash
# Download Amazon PQA dataset from AWS Open Data Registry
# Extract the downloaded files to datasets/amazon_pqa/ directory
```

The dataset is organized with separate files for different product categories (electronics, books, etc.), allowing you to test search functionality with domain-specific content.

### 3. Configuration Setup

Edit the configuration file at `configs/osmlqs.yaml` to specify your OpenSearch deployment and model hosting preferences. The configuration system supports multiple deployment scenarios and automatically validates your settings. The configuration file uses a flat structure with descriptive variable names, making it easy to manage environment variables and override specific settings. The configuration_manager processes this flat structure to create type-safe, structured configuration objects for your application.

### 4. Environment Variables (Optional)

For enhanced security, you can store sensitive configuration values in environment variables or a .env file:

```bash
# Create .env file for sensitive configuration
echo "AWS_ACCESS_KEY_ID=your-key" >> .env
echo "AWS_SECRET_ACCESS_KEY=your-secret" >> .env
echo "OPENSEARCH_ADMIN_PASSWORD=your-password" >> .env
```

The configuration system automatically detects and uses environment variables, allowing you to keep sensitive credentials out of your configuration files while maintaining the flexibility to override settings for different environments.

## 🚀 Usage Examples

The OpenSearch ML Quickstart provides flexible deployment options to accommodate different development and production scenarios. Whether you're experimenting locally or deploying to production, the examples demonstrate how to leverage the toolkit's capabilities effectively.

### Local Development with Self-managed OpenSearch

For local development and testing, you can quickly spin up a complete OpenSearch environment using Docker. This approach provides full control over your cluster configuration and supports all model hosting options:

```bash
# Start local OpenSearch cluster with all necessary plugins
docker-compose up -d

# Run dense vector search with specific data categories and document limits
python examples/dense_exact_search.py \
  --host-type os \
  --categories "electronics,books" \
  --max-docs 500

# Run hybrid search with index recreation for clean testing environment
python examples/hybrid_search.py \
  --host-type os \
  --categories "technology" \
  --delete-existing-index
```

The local development setup is ideal for prototyping, testing different search approaches, and understanding how the various components work together before moving to production deployments.

### Production with Amazon OpenSearch Service

For production deployments, Amazon OpenSearch Service provides a managed solution that handles scaling, security, and maintenance automatically. This approach is particularly well-suited for applications that need enterprise-grade reliability:

```bash
# Run conversational search with extended document corpus for comprehensive testing
python examples/conversational_search.py \
  --host-type aos \
  --categories "research,technology" \
  --max-docs 1000

# Run agentic search with large dataset for complex reasoning scenarios
python examples/agentic_search.py \
  --host-type aos \
  --categories "science,technology" \
  --max-docs 2000
```

The production examples demonstrate how to work with larger datasets and more sophisticated search patterns that are typical in real-world applications.

### Automated Workflow Setup

The toolkit includes workflow automation examples that demonstrate how to streamline the setup process for search applications. These workflows handle the complex orchestration of index creation, model deployment, pipeline configuration, and data loading:

```bash
# Use built-in workflow templates for standardized deployments
python examples/workflow_with_template.py \
  --host-type aos \
  --categories "electronics"

# Use custom workflow for specialized deployment requirements
python examples/workflow_example.py \
  --host-type os \
  --categories "books,technology"
```

The workflow examples are particularly valuable for production deployments where you need consistent, repeatable setup processes that can be automated and version-controlled.

## 🔧 Advanced Configuration

The OpenSearch ML Quickstart provides extensive customization options for developers who need to adapt the toolkit to specific requirements or integrate with existing systems.

### Custom Model Integration

The model factory system makes it straightforward to integrate custom models or modify existing model configurations. This example shows how to programmatically create and deploy a model instance:

```python
from models.helper import get_ml_model
from configs import get_model_config

# Get configuration for your model
model_config = get_model_config("aos", "bedrock", "embedding")

# Create model instance
model = get_ml_model(
    host_type="aos",
    model_type="bedrock", 
    model_config=model_config,
    embedding_type="dense"
)

# Deploy and use
model.deploy()
model_id = model.model_id()
```

This approach allows you to create models dynamically based on runtime conditions, user preferences, or configuration changes without modifying the core toolkit code.

### Custom Search Implementation

For applications that need specialized search behavior, you can implement custom query builders while still leveraging the toolkit's infrastructure and user interface components:

```python
from client.helper import get_client
from client.os_ml_client_wrapper import OsMlClientWrapper

def custom_search_query(query_text, model_id):
    return {
        "query": {
            "neural": {
                "chunk_vector": {
                    "query_text": query_text,
                    "model_id": model_id,
                    "k": 10
                }
            }
        }
    }

# Use with the CLI framework
from examples.cmd_line_interface import interactive_search_loop

interactive_search_loop(
    client=os_client,
    index_name="my_index",
    query_builder=lambda q: custom_search_query(q, model_id)
)
```

This pattern allows you to implement domain-specific search logic while maintaining compatibility with the toolkit's command-line interface and user interaction patterns.

### Workflow Templates

For organizations that need to standardize deployment processes, the toolkit supports custom workflow templates that define repeatable setup procedures:

```python
from configs import get_opensearch_config, get_model_config

# Define custom workflow template
workflow_template = {
    "name": "custom-search-workflow",
    "description": "Custom search setup workflow",
    "use_case": "SEMANTIC_SEARCH",
    "version": {
        "template": "1.0.0",
        "compatibility": ["2.12.0", "3.0.0"]
    },
    "workflows": {
        "provision": {
            "nodes": [
                {
                    "id": "create_index",
                    "type": "create_index_step",
                    "user_inputs": {
                        "index_name": "custom_index",
                        "settings": {"number_of_shards": 2}
                    }
                }
            ]
        }
    }
}
```

Custom workflow templates enable teams to codify best practices and ensure consistent deployments across different environments and team members.

## 🧪 Testing

The project includes comprehensive test coverage designed to ensure reliability and maintainability across all components. The testing strategy encompasses multiple levels of validation, from individual unit tests to integration scenarios that verify component interactions.

### Running Tests

The test suite is organized to support different testing scenarios and development workflows:

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/unit/models/          # Model tests
pytest tests/unit/client/          # Client tests  
pytest tests/unit/configs/         # Configuration tests

# Run with coverage
pytest --cov=. --cov-report=html

# Run integration tests (requires OpenSearch)
pytest tests/integration/ -m integration
```

The testing framework uses pytest with comprehensive fixtures and mocking to ensure tests run quickly and reliably without external dependencies.

### Test Categories

The test suite is organized into several distinct categories, each focusing on specific aspects of the system:

**Unit Tests** form the foundation of the testing strategy, testing individual components in isolation to ensure they behave correctly under various conditions. These tests use extensive mocking to eliminate external dependencies and focus on the logic within each component.

**Integration Tests** verify that different components work together correctly, testing the interactions between the configuration system, model factories, client wrappers, and other major subsystems. Some integration tests require actual OpenSearch connections and are automatically skipped when the necessary infrastructure is not available.

**Configuration Tests** specifically validate the configuration management system, ensuring that all supported deployment combinations work correctly and that invalid configurations are properly rejected with helpful error messages.

**Model Tests** focus on the ML model hierarchy and factory patterns, verifying that the abstraction layer correctly handles different model types and hosting scenarios while maintaining a consistent interface.

**Client Tests** validate the OpenSearch client wrappers and utilities, ensuring that the high-level abstractions correctly translate to appropriate OpenSearch API calls and handle error conditions gracefully.

## 🔍 Troubleshooting

When working with the OpenSearch ML Quickstart, you may encounter various issues related to configuration, connectivity, model deployment, or performance. This section provides guidance for diagnosing and resolving common problems.

### Common Issues

#### Configuration Issues

Configuration problems are often the first hurdle when setting up the toolkit. The configuration system provides built-in validation tools to help identify and resolve these issues:

```bash
# Validate configuration
python -c "from configs import validate_all_configs; print(validate_all_configs())"

# Check available combinations
python -c "from configs import get_available_combinations; print(get_available_combinations())"
```

These commands will help you verify that your configuration file is properly structured and that you're using valid combinations of deployment types and model providers.

#### Connection Issues

Connection problems can occur at multiple levels, depending on your deployment approach:

**Local OpenSearch** issues typically involve Docker container problems. Check that your containers are running properly and that the necessary ports are accessible. Verify that the OpenSearch cluster is healthy and that all required plugins are installed.

**Amazon OpenSearch Service** connection issues often relate to network access and security configurations. Ensure that your security groups allow access from your client location, verify that your domain is publicly accessible if required, and check that your VPC configuration permits the necessary network traffic.

**Authentication** problems can occur with both deployment types. Verify that your credentials are correctly configured in your configuration file or environment variables, ensure that IAM roles and policies are properly set up for AWS services, and confirm that OpenSearch user permissions are adequate for the operations you're attempting.

#### Model Deployment Issues

Model deployment can fail for various reasons depending on your hosting approach:

**Local Models** require sufficient cluster resources and proper model compatibility. Verify that your OpenSearch cluster has adequate memory and CPU resources to host the models you're trying to deploy. Check that the model format (Hugging Face or ONNX) is supported by your OpenSearch version and that all necessary ML Commons plugins are installed and configured.

**Remote Models** depend on external service connectivity and permissions. Ensure that your AWS credentials have the necessary permissions to access Bedrock or SageMaker services. Verify that the model endpoints are accessible from your OpenSearch cluster and that network connectivity allows communication with the external services.

**Model Registration** issues often involve missing model groups or inadequate permissions. Ensure that the required model group exists in your OpenSearch cluster and that your user account has the necessary permissions to register and deploy models.

#### Performance Optimization

Performance issues can manifest in various ways, from slow search responses to high memory usage:

**Large Datasets** may require optimization of bulk loading operations. Consider increasing the bulk chunk size in your data loading configuration, optimize your index settings for better write performance, and ensure that your cluster has sufficient resources to handle the data volume.

**Search Performance** can be improved through various techniques. Use HNSW indexing for faster approximate search when exact results aren't required, optimize your vector dimensions and search parameters, and consider implementing result caching for frequently accessed queries.

**Memory Usage** optimization is particularly important when working with large models or datasets. Consider using sparse models instead of dense models in memory-constrained environments, optimize your model configurations to reduce memory footprint, and implement proper resource cleanup in your applications.

### Debug Mode

When troubleshooting complex issues, enabling debug logging can provide valuable insights into what's happening behind the scenes:

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python examples/dense_exact_search.py --host-type os
```

Debug mode will provide detailed information about configuration loading, model deployment, search operations, and error conditions, helping you identify the root cause of problems more quickly.

## 📚 Additional Resources

### Documentation
- [OpenSearch Documentation](https://opensearch.org/docs/)
- [OpenSearch ML Commons Plugin](https://opensearch.org/docs/latest/ml-commons-plugin/)
- [Amazon OpenSearch Service](https://docs.aws.amazon.com/opensearch-service/)
- [Amazon Bedrock](https://docs.aws.amazon.com/bedrock/)
- [Amazon SageMaker](https://docs.aws.amazon.com/sagemaker/)

### Model Resources
- [Hugging Face Models](https://huggingface.co/models)
- [OpenSearch Pre-trained Models](https://opensearch.org/docs/latest/ml-commons-plugin/pretrained-models/)
- [Amazon Bedrock Model IDs](https://docs.aws.amazon.com/bedrock/latest/userguide/model-ids.html)

### Community
- [OpenSearch Forum](https://forum.opensearch.org/)
- [GitHub Issues](https://github.com/opensearch-project/opensearch-ml-quickstart/issues)
- [AWS re:Post](https://repost.aws/tags/TA4IvCeWI1TE66q4jEj4Z9zg/amazon-open-search-service)

## 🤝 Contributing

We welcome contributions to OpenSearch ML Quickstart! Here's how to get started:

### Development Setup
```bash
# Fork and clone the repository
git clone <your-fork-url>
cd opensearch-ml-quickstart

# Create development environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Run tests to ensure everything works
pytest
```

### Contribution Guidelines
1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Add tests** for new functionality
4. **Ensure** all tests pass (`pytest`)
5. **Update** documentation as needed
6. **Submit** a pull request

### Code Standards
- Follow PEP 8 style guidelines
- Add type hints for new functions
- Include docstrings for public methods
- Maintain test coverage above 80%
- Update configuration schema for new options

## 📄 License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

---

## Quick Start Summary

1. **Install**: `pip install -r requirements.txt`
2. **Configure**: Edit `configs/osmlqs.yaml` with your settings
3. **Data**: Download Amazon PQA dataset to `datasets/amazon_pqa/`
4. **Run**: `python examples/dense_exact_search.py --host-type os`
5. **Explore**: Try different search types and model configurations

For detailed setup instructions, see the [Setup and Installation](#-setup-and-installation) section above.
