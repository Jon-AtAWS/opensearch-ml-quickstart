# OpenSearch ML Quickstart

## Overview

OpenSearch ML Quickstart provides practical code examples for developers who want to implement AI-powered search applications. The toolkit bridges the gap between OpenSearch's powerful ML capabilities and real-world applications by offering production-ready abstractions and comprehensive examples that demonstrate semantic search, conversational AI, and agentic workflows.

The library supports both open-source OpenSearch deployments and Amazon OpenSearch Service managed clusters, adapting its approach based on your infrastructure choice. For open-source deployments, you can deploy models locally within your OpenSearch cluster, providing maximum control and eliminating external dependencies. Both deployment types support remote models hosted on services like Amazon Bedrock and SageMaker, enabling you to leverage powerful foundation models without managing the underlying infrastructure.

At its core, the toolkit provides three main abstraction layers that simplify complex ML operations. The MLModel classes create a unified interface for working with different types of machine learning models, whether they're embedding models for semantic search or large language models for conversational AI. These classes handle the complexity of model registration, deployment, and lifecycle management across different hosting environments. The Connector classes manage the integration between OpenSearch and external ML services, abstracting away the details of authentication, request formatting, and response parsing. The Configuration classes provide a sophisticated system for managing the numerous parameters required for different deployment scenarios, with built-in validation and environment-aware loading.

The examples demonstrate practical implementations of these abstractions, ranging from basic semantic search to advanced agentic workflows. Each example is designed to be both educational and production-ready, showing how to combine the toolkit's components to solve real-world search challenges. The workflow examples demonstrate automated setup processes, while the agent examples showcase cutting-edge AI capabilities that can reason about complex queries and execute multi-step search strategies.

## 📋 Prerequisites

Before getting started with the OpenSearch ML Quickstart, you'll need to ensure your environment meets certain requirements and that you have access to the necessary data and services. The toolkit is designed to work in various deployment scenarios, from local development to production environments.

### System Requirements

**Python 3.10 or later**. The system has been tested through 3.13. See the [Python downloads page](https://www.python.org/downloads/) for instructions.  

### The Amazon PQA Dataset

The toolkit uses the Amazon Product Question and Answers (PQA) dataset as its primary demonstration corpus, providing a rich collection of real-world product questions and answers that showcase the various search capabilities effectively. This dataset contains over 3 million questions and answers across multiple product categories, making it ideal for testing semantic search, conversational AI, and other advanced search features.

Each document in the dataset represents a product with associated questions and answers from real customers. A typical source document contains structured information including the product's ASIN (Amazon Standard Identification Number), category, brand name, item description, and multiple question-answer pairs with metadata about the respondents. For example, a gaming product might include questions about compatibility, performance specifications, and user experiences, along with detailed answers from verified purchasers.

The QandAFileReader class provides sophisticated data processing capabilities that go beyond simple file parsing. It enriches the raw dataset by creating searchable text chunks that combine product information with question-answer pairs, generating embeddings-ready content that preserves the semantic relationships between products and their associated discussions. The reader handles multiple product categories simultaneously, supports filtering by document count for testing purposes, and creates structured documents that work seamlessly with the toolkit's indexing and search capabilities. This enrichment process transforms the raw product data into a format optimized for vector search, ensuring that semantic queries can effectively match user intent with relevant product information and community knowledge.

You can find the data set in the [AWS Open Data Registry](https://registry.opendata.aws/amazon-pqa/) 

`cd <path/to/your/opensearch-ml-quickstart>/datasets`  
`aws s3 cp --no-sign-request s3://amazon-pqa/amazon-pqa.tar.gz .`  
`tar -xzf amazon-pqa.tar.gz`  

### Set up Model Access

If you plan to use Amazon Bedrock either for embedding models or for LLM models, you'll need to enable model access for your account. The Bedrock documentation has instructions for [getting started with Bedrock and enabling model access](https://docs.aws.amazon.com/bedrock/latest/userguide/model-access-modify.html)

## Setup

Getting started with OpenSearch ML Quickstart requires setting up your development environment, configuring your deployment preferences, and understanding the authentication mechanisms. 

### Create your environment

Begin by cloning the repository and setting up a Python virtual environment to isolate dependencies. 

`python -m venv .venv`  
`source .venv/bin/activate` (or `.venv\Scripts\activate` on Windows)  
`pip install -r requirements.txt`  

### Set up OpenSearch

You'll also need **OpenSearch 2.19.0 or later** (tested through 3.2) to ensure compatibility with the ML Commons plugin features used throughout the examples. The agentic examples require **OpenSearch 3.2**. For local development, Docker Desktop is provides the easiest way to run a complete OpenSearch cluster with all necessary plugins pre-configured.

Download and install [Docker Desktop](https://docs.docker.com/desktop/) if that's your preferred method. You can find a Docker Compose file in the opensearch-ml-quickstart folder. OpenSearch requires an `OPENSEARCH_INITIAL_ADMIN_PASSWORD` set as an environment variable before you start the system. You use this password with the `admin` user to bootstrap and use the system. Best practice is to create a new user and password once you've logged in for the first time. Execute these commands

`cd <path/to/your/opensearch-ml-quickstart>`  
`export OPENSEARCH_INITIAL_ADMIN_PASSWORD=ExamplePassword`  
`docker compose up`  

For Amazon OpenSearch Service, create a new domain in the AWS Management Console. See the [Getting Started Guide](https://docs.aws.amazon.com/opensearch-service/latest/developerguide/gsg.html) for instructions. The quickstart will work with a single-node deployment (m7g.large), though we recommend two nodes to avoid a yellow cluster. Ensure that you enable fine-grained access control and set up an admin user with a secure password. Note the domain endpoint URL, as you'll need it for configuration.

## Configure the system

Use your favorite text editor to open `opensearch-ml-quickstart/configs/osmlqs.yaml`. 

* If plan to run with Amazon OpenSearch Service, set your `AOS_DOMAIN_NAME`, `AOS_HOST_URL`, and `AOS_PORT`.
* Set your `OPENSEARCH_ADMIN_USER`, and `OPENSEARCH_ADMIN_PASSWORD`.
* Set your `AWS_REGION`, `AWS_USER_NAME`, `AWS_ACCESS_KEY_ID`, and `AWS_SECRET_ACCESS_KEY` if you are using AWS (for OpenSearch Service, Bedrock, or SageMaker, e.g.)
* Set the 2 role names for Bedrock, and 2 for SageMaker. The quickstart creates AWS Identity and Access Management (IAM) roles as part of the model deployment when you are using Bedrock or SageMaker. The `_CONNECTOR_ROLE` roles enable the code to connect to Bedrock or SageMaker. The `_CREATE_CONNECTOR_ROLE` roles enable the code to create connectors in OpenSearch Service.

The rest of the config file lets you set up your model parameters. Set values in the sections that you plan to use.

* For local embedding models, see the documentation for [the supported model names, versions, and parameters](https://docs.opensearch.org/latest/ml-commons-plugin/pretrained-models/).
* For using SageMaker sparse models with OpenSearch Service, OpenSearch Service provides [CloudFormation integrations](https://docs.aws.amazon.com/opensearch-service/latest/developerguide/cfn-template.html) that make deploying sparse models one-click. If you use one of these integrations, change the values of `SPARSE/DENSE ARN` and `SPARSE/DENSE URL` to the SageMaker endpoints.

**Environment variable support** provides enhanced security by allowing sensitive configuration values to be stored outside of configuration files. The configuration system automatically detects environment variables and `.env` files, enabling you to override specific settings for different environments while keeping credentials secure. 

## 🚀 Usage Examples

The OpenSearch ML Quickstart provides flexible deployment options to accommodate different development and production scenarios. Whether you're experimenting locally or deploying to production, the examples demonstrate how to leverage the toolkit's capabilities effectively.

The scripts in the `opensearch-ml-quickstart/examples` folder follow a similar command-line pattern.  

* You control the OpenSearch deployment type with the `-o, --opensearch-type` parameter, `os` for self-managed and local deployments, `aos` for Amazon OpenSearch Service.  
* You can specify one or many `-c, --categories` of PQA data you want to load, as well as the `-n, --number-of-documents` to load from each of those categories.  
* The data set is large! We recommend getting started with just one category, like `jeans`, and a small document set of 100. Once you confirm that data is loading properly and the examples are working, you can load the full data set by removing these command-line options.
* Each example creates its own index, and the quickstart reuses these indices on subsequent runs (default). You can force the examples to delete and reload the data with the  `-d, --delete-existing-index` flags, or specify `--no-load`.
* By default, the examples use an interactive loop for you to send queries, and see results. If you want to run non-interactive, you can specify a `-q, --question` on the command line and the script will execute once.

Summary of command-line parameters.

```
options:
  -h, --help            show this help message and exit
  -d, --delete-existing-index
                        Delete the index if it already exists
  -c CATEGORIES [CATEGORIES ...], --categories CATEGORIES [CATEGORIES ...]
                        List of categories to load into the index
  -s BULK_SEND_CHUNK_SIZE, --bulk-send-chunk-size BULK_SEND_CHUNK_SIZE
                        Chunk size for bulk sending documents to OpenSearch
  -n NUMBER_OF_DOCS_PER_CATEGORY, --number-of-docs-per-category NUMBER_OF_DOCS_PER_CATEGORY
                        Number of documents to load per category
  -o {os,aos}, --opensearch-type {os,aos}
                        Type of OpenSearch instance to connect to: local=os or remote=aos
  --no-load             Skip loading data into the index
  -q QUESTION, --question QUESTION
                        Execute search with this question and exit (instead of interactive
                        loop)
```

### Command-line Examples

Run dense vector search with specific data categories and document limits  

```
python examples/dense_exact_search.py \
  --host-type os \
  --categories "electronics,books" \
  -n 500
```

Run hybrid search with index recreation for clean testing environment  

```
python examples/hybrid_search.py \
  --host-type os \
  --categories "technology" \
  --delete-existing-index
```

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

## ML Models and Connectors

The OpenSearch ML Quickstart distinguishes between two primary types of machine learning models, each serving different purposes in the search pipeline. Embedding models transform text into vector representations that capture semantic meaning, enabling similarity-based search operations. These models can produce either dense vectors, which are compact numerical representations suitable for semantic similarity, or sparse vectors, which maintain interpretability while capturing semantic relationships. Large Language Models (LLMs) generate human-like text responses and can engage in conversational interactions, making them essential for conversational search and agentic workflows.

For LLM models, the toolkit supports two distinct connector strategies that correspond to different Amazon Bedrock API approaches. The predict strategy uses Bedrock's traditional invoke API with the legacy message format, providing compatibility with older implementations and specific use cases that require the original API structure. The converse strategy leverages Bedrock's newer converse API, which offers a more standardized interface for conversational interactions and improved support for multi-turn conversations and tool usage. The converse API is recommended for new implementations, particularly for agentic workflows and complex conversational scenarios, while the predict API remains available for backward compatibility and specific integration requirements.

Connector setup varies significantly based on your deployment architecture and model hosting preferences. For self-managed OpenSearch deployments, connectors can integrate with local models deployed within the cluster or remote models hosted on external services. The connector configuration includes authentication credentials, endpoint URLs, and request/response formatting specifications. Amazon OpenSearch Service deployments require additional IAM role configuration, where the toolkit automatically creates connector roles with appropriate permissions to invoke external services like Bedrock or SageMaker.

Model deployment follows a structured process that begins with connector creation and registration. For remote models, the system first establishes the connector with the external service, validates connectivity and authentication, then registers the model within OpenSearch's ML Commons framework. Local model deployment involves uploading model artifacts to the cluster, allocating computational resources, and configuring the model for inference. The toolkit abstracts these complexities through the MLModel classes, which handle the orchestration of connector setup, model registration, and deployment verification.

The deployment process includes automatic validation of model compatibility, resource allocation, and performance optimization. For embedding models, the system configures appropriate vector dimensions and indexing parameters. For LLMs, it establishes conversation memory management and response formatting. The toolkit monitors deployment status and provides detailed feedback on any configuration issues, ensuring that models are properly integrated and ready for use in search applications.

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
    os_client=os_client,
    ml_commons_client=ml_commons_client,
    model_group_id=model_group_id
)

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
    model_info="Custom Model",
    query_builder_func=custom_search_query
)
```

This pattern allows you to implement domain-specific search logic while maintaining compatibility with the toolkit's command-line interface and user interaction patterns.

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

**Hugging Face** direct integration allows you to leverage open-source models with minimal configuration overhead.

### Deployment Flexibility

The toolkit accommodates different deployment preferences through two primary approaches:

**Self-managed OpenSearch** deployments provide complete control over your search infrastructure, whether running locally for development or in production environments. This approach supports all model hosting options and provides maximum customization flexibility.

**Amazon OpenSearch Service** offers a fully managed experience with automatic scaling, security, and maintenance. While this approach is limited to remote model hosting, it significantly reduces operational overhead.

## 📁 Project Structure

The OpenSearch ML Quickstart is organized into several key directories, each serving a specific purpose in the overall architecture. This modular structure allows developers to understand and extend individual components without needing to grasp the entire codebase.

The **examples** directory contains ready-to-run demonstrations of different search capabilities. Each example is a complete, working implementation that showcases a specific search approach, from basic dense vector search to advanced agentic workflows. The `cmd_line_interface.py` module provides a unified command-line experience across all examples, ensuring consistent user interaction patterns.

The **models** directory implements an abstraction layer for machine learning models. The design follows object-oriented principles with a clear inheritance hierarchy, allowing the same code to work seamlessly with local models deployed within OpenSearch or remote models hosted on external services like Amazon Bedrock or SageMaker.

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
│   ├── conversational_agent.py       # AI agents for complex search tasks
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

At the foundation of the toolkit is a configuration management system that handles the complexity of different deployment scenarios, model types, and hosting options. The system provides type-safe configuration with automatic validation and environment-aware loading, ensuring that your application is properly configured regardless of the deployment environment.

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

The model system implements a class hierarchy that abstracts different model types and hosting scenarios. This design allows the same application code to work seamlessly with local models deployed within your OpenSearch cluster or remote models hosted on external services like Amazon Bedrock or SageMaker.

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
    os_client=os_client,              # OpenSearch client
    ml_commons_client=ml_commons_client,  # ML Commons client
    model_group_id=model_group_id     # Model group ID
)

# Create a SageMaker LLM model for self-managed OpenSearch
llm_model = get_ml_model(
    host_type="os",                   # Self-managed OpenSearch
    model_type="sagemaker",           # Amazon SageMaker endpoints
    model_config=llm_config,          # LLM-specific configuration
    os_client=os_client,              # OpenSearch client
    ml_commons_client=ml_commons_client,  # ML Commons client
    model_group_id=model_group_id     # Model group ID
)
```

This architecture provides several key benefits that make it easy to work with different model hosting scenarios. The **unified interface** ensures that all models implement the same `MlModel` interface, so your application code doesn't need to change when you switch between different model types or hosting options. **Easy migration** between hosting options requires minimal code changes, allowing you to start with local development and move to production hosting without rewriting your application. **Type safety** provides automatic validation of model and hosting combinations, preventing invalid configurations at runtime. **Lifecycle management** includes built-in model registration, deployment, and cleanup functionality, handling the complex orchestration required to manage ML models in OpenSearch.

### Connector Architecture

The connector system provides an abstraction layer for integrating with external ML services. Connectors handle the complex details of communicating with different model providers, managing authentication, and translating between OpenSearch's ML Commons API and external service protocols.

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
    connector_type="bedrock",         # Amazon Bedrock
    host_type="aos"                   # Amazon OpenSearch Service
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

## Details of the Examples

### Example Summary

| Example | OpenSearch Type | Model Hosting | Model Host | Model Type |
|---------|----------------|---------------|------------|------------|
| `dense_exact_search.py` | aos | remote | sagemaker | embedding |
| `dense_hnsw_search.py` | aos | remote | bedrock | embedding |
| `sparse_search.py` | aos | remote | sagemaker | embedding |
| `hybrid_search.py` | aos | remote | sagemaker | embedding |
| `hybrid_local_search.py` | os | local | n/a | embedding |
| `lexical_search.py` | os/aos | n/a | n/a | n/a |
| `conversational_search.py` | aos | remote | sagemaker | embedding + llm |
| `conversational_agent.py` | os | local | n/a | embedding + llm |
| `workflow_example.py` | aos | remote | bedrock | embedding |
| `workflow_with_template.py` | aos | remote | sagemaker | embedding |

The OpenSearch ML Quickstart provides a comprehensive set of examples that demonstrate different search approaches and their practical applications. Each example is designed to showcase how the toolkit's classes work together to solve real-world search challenges, from basic semantic search to advanced agentic workflows.

The dense vector search examples (`dense_exact_search.py` and `dense_hnsw_search.py`) demonstrate the fundamental building blocks of semantic search. These examples utilize the MLModel classes to deploy embedding models that transform text into vector representations, while the Configuration classes handle the complexity of different model hosting scenarios. The examples show how the OsMlClientWrapper simplifies the process of setting up k-NN search indices and ingest pipelines, automatically configuring the appropriate vector field mappings and dimensions based on the selected embedding model. The exact k-NN variant prioritizes accuracy by computing precise distances between vectors, while the HNSW implementation trades some precision for significantly improved performance on large datasets.

The sparse vector search example (`sparse_search.py`) showcases neural sparse models that maintain the interpretability of traditional keyword search while incorporating semantic understanding. This example demonstrates how the same MLModel abstraction can work with different types of embedding models, whether dense or sparse, while the underlying connector and configuration systems adapt automatically to the model requirements. The sparse approach is particularly valuable in domains where specific terminology matters, as it preserves the ability to understand which terms contributed to a match.

The hybrid search example (`hybrid_search.py`) illustrates the power of combining multiple search approaches through OpenSearch's native ranking capabilities. This example shows how to deploy both dense and sparse embedding models simultaneously, configure multiple ingest pipelines, and use sophisticated query structures that merge results from different search strategies. The Configuration classes handle the complexity of managing multiple model configurations, while the client wrapper orchestrates the setup of multiple vector fields and processing pipelines.

The conversational search example (`conversational_search.py`) demonstrates the integration of search with large language models to create interactive AI assistants. This example showcases how the Connector classes manage the integration between OpenSearch and external LLM services like Amazon Bedrock, handling authentication, request formatting, and response parsing. The example shows how to combine traditional search results with LLM-generated responses, creating a conversational interface that can answer questions based on your document corpus while maintaining context across multiple interactions.

The conversational agent example (`conversational_agent.py`) represents the most sophisticated implementation, showcasing OpenSearch's agent framework capabilities. This example demonstrates how to create AI agents that can reason about complex queries, plan multi-step search strategies, and execute sophisticated information retrieval tasks. The agent utilizes both embedding models for semantic search and LLM models for reasoning and response generation, showing how the toolkit's abstractions enable complex workflows while maintaining clean, manageable code. The agent can break down complex queries into sub-questions, execute multiple searches, and synthesize results into comprehensive answers.

The workflow examples (`workflow_example.py` and `workflow_with_template.py`) demonstrate automation and standardization of the setup process. These examples show how to use OpenSearch's workflow templates to automate the entire deployment pipeline, from index creation and model deployment to pipeline configuration and data loading. The workflow examples illustrate how the Configuration classes can be used to define repeatable deployment procedures, while the MLModel and Connector classes handle the actual resource provisioning and configuration. The custom workflow example shows how to define organization-specific templates, while the built-in workflow example leverages OpenSearch's native templates for standardized deployments.

### Unified Command-Line Interface

All examples utilize the consolidated `cmd_line_interface.py` module, which provides a consistent and user-friendly experience across different search types. This unified approach ensures that users can easily switch between different search examples without learning new command-line interfaces or interaction patterns.

The interface provides several common features across all examples, including interactive search loops that allow users to perform multiple queries in a single session, consistent error handling and user feedback, standardized configuration loading and validation, and common data loading and index management operations.

```bash
# Dense vector search with specific categories and document limits
python examples/dense_exact_search.py --host-type os --categories "electronics" --n 500

# Hybrid search with index recreation for clean testing
python examples/hybrid_search.py --host-type aos --categories "books,electronics" --delete-existing-index

# Conversational search for technology-related queries
python examples/conversational_search.py --host-type aos --categories "technology"

# Agentic search with extended document corpus for complex reasoning
python examples/agentic_search.py --host-type os --categories "research" --n 1000
```

The command-line interface supports several common options that work across all examples. The `--host-type` parameter specifies whether you're using self-managed OpenSearch (`os`) or Amazon OpenSearch Service (`aos`). The `--categories` option allows you to specify which data categories to load from the Amazon PQA dataset. The `-n` parameter limits the number of documents loaded per category, useful for testing or resource-constrained environments. The `--delete-existing-index` flag forces deletion of existing indices, ensuring a clean setup for testing. The `--no-load` option skips data loading entirely, useful when working with pre-existing indices.


## 🔍 Troubleshooting

When working with the OpenSearch ML Quickstart, you may encounter various issues related to configuration, connectivity, model deployment, or performance. This section provides guidance for diagnosing and resolving common problems.

### Common Issues

#### Configuration Issues

Configuration problems are often the first hurdle when setting up the toolkit. The configuration system provides built-in validation tools to help identify and resolve these issues:

```bash
# Check available combinations
python -c "from configs import get_available_combinations; print(get_available_combinations())"

# Validate configuration
python -c "from configs import validate_all_configs; print(validate_all_configs())"
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

## Appendix: Testing

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

The test suite is organized into several distinct categories, each focusing on specific aspects of the system. Unit Tests form the foundation of the testing strategy, testing individual components in isolation to ensure they behave correctly under various conditions. These tests use extensive mocking to eliminate external dependencies and focus on the logic within each component. Integration Tests verify that different components work together correctly, testing the interactions between the configuration system, model factories, client wrappers, and other major subsystems. Some integration tests require actual OpenSearch connections and are automatically skipped when the necessary infrastructure is not available. Configuration Tests specifically validate the configuration management system, ensuring that all supported deployment combinations work correctly and that invalid configurations are properly rejected with helpful error messages. Model Tests focus on the ML model hierarchy and factory patterns, verifying that the abstraction layer correctly handles different model types and hosting scenarios while maintaining a consistent interface. Client Tests validate the OpenSearch client wrappers and utilities, ensuring that the high-level abstractions correctly translate to appropriate OpenSearch API calls and handle error conditions gracefully.

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
