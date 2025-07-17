# OpenSearch ML Quickstart

The OpenSearch ML Quickstart is a comprehensive toolkit designed to simplify the process of building AI-powered search applications with OpenSearch. This project provides a unified framework for implementing various search capabilities including semantic search, sparse search, hybrid search, and conversational AI. By offering production-ready examples and abstractions for OpenSearch's ML Commons plugin, vector embeddings, and large language models, developers can quickly implement advanced search functionality without deep expertise in each technology.

## 🚀 Features

OpenSearch ML Quickstart offers a rich set of capabilities to enhance your search applications. The toolkit supports multiple search types including dense vector search for semantic understanding, sparse vector search for keyword awareness, hybrid search combining both approaches, traditional lexical search, and conversational search powered by RAG. 

The framework is designed with flexibility in mind, supporting various model hosting options including local models, Amazon Bedrock, Amazon SageMaker, and Hugging Face. You can deploy OpenSearch either as a self-managed cluster or through Amazon OpenSearch Service, depending on your operational preferences.

For developers looking to get started quickly, the toolkit includes over eight complete search implementations with interactive interfaces. The OpenSearch Flow Framework integration enables automated setup processes, while the comprehensive ML model abstraction layer provides a unified interface for different model hosting options.

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

The architecture of OpenSearch ML Quickstart is centered around a well-designed abstraction layer for ML models. This design allows developers to work with different model types and hosting scenarios through a consistent interface, hiding the complexity of the underlying implementations.

The `ml_models/` directory contains a class hierarchy that abstracts away the differences between local and remote models:

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

The `MlModel` base class serves as the foundation of this hierarchy, providing a unified interface for all model types. It handles the complete model lifecycle including registration, deployment, and deletion, while also managing model groups and versioning. The abstract methods like `model_id()`, `deploy()`, and `delete()` ensure that all derived classes implement the necessary functionality.

For scenarios where models can be deployed directly within the OpenSearch cluster, the `LocalMlModel` class provides support for Hugging Face transformers and ONNX models. This approach is ideal for self-managed clusters with sufficient resources but is not supported on Amazon OpenSearch Service.

The `RemoteMlModel` class serves as the base for all externally hosted models, using OpenSearch ML connectors for communication. It handles connector creation and management and supports both dense and sparse embeddings. The derived classes like `AosBedrockMlModel`, `AosSagemakerMlModel`, `OsBedrockMlModel`, and `OsSagemakerMlModel` provide specific implementations for different combinations of OpenSearch deployments and model hosting services.

This hierarchical design allows developers to switch between different model hosting options with minimal code changes, making it easy to experiment with different approaches or migrate between deployment strategies.

### Client Architecture

The client architecture in OpenSearch ML Quickstart is designed to simplify interactions with OpenSearch and its ML capabilities. The `client/` directory provides abstractions that hide the complexity of direct API calls and configuration management.

At the core of this architecture is the `OsMlClientWrapper` class, which combines OpenSearch and ML Commons clients into a unified interface. This wrapper provides high-level methods for index management, pipeline setup, k-NN configuration, neural search setup, and model group management. By encapsulating these functionalities, the wrapper allows developers to focus on building search applications rather than dealing with low-level OpenSearch configurations.

The `get_client()` factory function creates appropriate client instances based on the deployment type, supporting both self-managed OpenSearch clusters and Amazon OpenSearch Service. It handles authentication and connection management, ensuring that the correct credentials and endpoints are used.

The `index_utils` module complements the client wrapper by providing utilities for index creation and management, data loading and bulk operations, mapping application and validation, and category-based data processing. These utilities streamline common tasks and ensure consistent behavior across different search implementations.

Here's an example of how these components work together:

```python
# Client initialization
client = OsMlClientWrapper(get_client("aos"))  # or "os"

# Index and pipeline setup
client.setup_for_kNN(ml_model, index_name, pipeline_name, field_map, embedding_type)

# Model group management
model_group_id = client.ml_model_group.model_group_id()
```

This architecture enables developers to work with OpenSearch and its ML capabilities through a clean, high-level interface, reducing the learning curve and accelerating development.

## 🎯 Search Examples

OpenSearch ML Quickstart provides a comprehensive set of search examples that demonstrate different approaches to implementing search functionality. These examples serve as both reference implementations and starting points for custom development.

The `examples/` directory contains eight production-ready search implementations, each focusing on a specific search approach:

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

Each example demonstrates a complete implementation of a specific search approach, including index setup, data loading, model deployment, and interactive search. By studying these examples, developers can understand the tradeoffs between different search approaches and choose the one that best fits their requirements.

### Unified Command-Line Interface

To ensure consistency across examples and simplify user interaction, all examples use the consolidated `cmd_line_interface.py` module. This module provides a unified command-line interface with consistent argument parsing, interactive search capabilities, and error handling.

The interface supports argument parsing with consistent CLI options across all examples, making it easy to switch between different search approaches. The interactive search loop provides a generic framework for executing searches with customizable query builders, allowing developers to focus on the search logic rather than the user interface.

The user experience is enhanced with colorized output, robust error handling, and multiple quit options. The interface is also designed to be extensible through a callback pattern for custom search logic, enabling developers to adapt it to their specific requirements.

This unified approach to user interaction ensures that developers can focus on understanding and implementing different search approaches without having to reinvent the wheel for each example.

## 📋 Prerequisites

Before getting started with OpenSearch ML Quickstart, you'll need to ensure that your environment meets certain requirements. The toolkit requires Python 3.10 or later, primarily due to compatibility with pandas 2.0.3. You'll also need OpenSearch 2.13.0 or later, with testing having been conducted through version 2.16.0. For local development, Docker Desktop is recommended for running a self-managed OpenSearch cluster.

In terms of data, the examples use the Amazon PQA Dataset, which should be downloaded and extracted to the `datasets/` directory. This dataset provides a realistic corpus for demonstrating different search approaches.

OpenSearch ML Quickstart supports two deployment options for OpenSearch. The first option is a self-managed OpenSearch cluster, which can be deployed locally using the provided Docker Compose configuration or as a custom cluster. This option supports both local and remote models. The second option is Amazon OpenSearch Service, which provides a managed OpenSearch domain. This option requires public access configuration and fine-grained access control with a master user, and only supports remote models through Bedrock or SageMaker.

For model hosting, the toolkit supports both local and remote options. Local models, which are only supported with self-managed OpenSearch clusters, include Hugging Face transformers and ONNX models deployed within the OpenSearch cluster. Remote models, which are supported with both deployment options, include Amazon Bedrock foundation models like Titan and Claude, as well as custom model endpoints on Amazon SageMaker. Using remote models requires appropriate AWS IAM permissions.

## 🛠️ Setup and Installation

Setting up OpenSearch ML Quickstart involves several steps, starting with environment setup. After cloning the repository, you'll need to create and activate a virtual environment, then install the required dependencies using pip. This ensures that your development environment has all the necessary packages.

Next, you'll need to download the sample data. The examples use the Amazon PQA dataset, which should be downloaded from the AWS Open Data Registry and extracted to the `datasets/amazon-pqa/` directory. This dataset provides a realistic corpus for demonstrating different search approaches.

Configuration is managed through files in the `configs/` directory. The `.env` file contains environment variables and credentials, the `config.py` file contains application configuration, and the `tasks.py` file contains test task definitions. These files allow you to customize the behavior of the toolkit to suit your specific requirements.

## ⚙️ Configuration Guide

OpenSearch ML Quickstart supports various deployment scenarios, each requiring specific configuration. For local development with a self-managed OpenSearch cluster and local models, you'll need to set up environment variables for OpenSearch credentials and connection details. You can then start the local OpenSearch cluster using Docker Compose and run one of the examples.

For using a self-managed OpenSearch cluster with remote models like Amazon Bedrock, you'll need to configure both OpenSearch credentials and Bedrock access details. This includes AWS access keys, region, model endpoint URL, and model dimension.

When working with Amazon OpenSearch Service and Bedrock, you'll need to configure credentials for both services, including domain name, region, host URL, and IAM roles for connector creation and access. Similar configuration is required for using Amazon OpenSearch Service with SageMaker endpoints.

These configuration options provide flexibility in how you deploy and use OpenSearch ML Quickstart, allowing you to choose the approach that best fits your requirements and constraints.

## 🚀 Usage Examples

OpenSearch ML Quickstart provides several usage examples to help you get started quickly. For local development, you can start a local OpenSearch cluster using Docker Compose, then run one of the example scripts like `dense_exact_search.py`. This will set up the necessary index, load sample data, and start an interactive search interface where you can enter queries and see the results.

For production scenarios, you might want to use more advanced search approaches like hybrid search, which combines dense and sparse vectors for improved relevance. The `hybrid_search.py` example demonstrates this approach, allowing you to specify categories of data to load and the number of documents per category.

If you're interested in conversational AI, the `conversational_search.py` example shows how to implement a RAG-powered conversational search interface. This approach combines retrieval-augmented generation with large language models to provide contextually relevant responses to user queries.

For automated setup, the `workflow_example.py` and `workflow_with_template.py` examples demonstrate how to use the OpenSearch Flow Framework to automate the process of setting up indices, loading data, and deploying models. This approach is particularly useful for production deployments where you want to ensure consistent configuration.

All examples support a common set of command-line options, including the OpenSearch deployment type, data categories to load, number of documents per category, and options for deleting existing indices and controlling bulk operations. This consistency makes it easy to switch between different examples and approaches.

## 🔧 Advanced Configuration

Beyond the basic usage examples, OpenSearch ML Quickstart supports advanced configuration for custom model integration and search implementation. For custom model integration, you can create instances of the appropriate model classes with specific parameters. For example, you can create an `AosBedrockMlModel` instance with custom configuration for using a specific Bedrock model with Amazon OpenSearch Service.

For custom search implementation, you can define your own query builder function that constructs the appropriate OpenSearch query based on the user's input. This function can then be used with the generic search loop provided by the `cmd_line_interface` module, allowing you to customize the search behavior while reusing the interactive interface.

The toolkit also supports workflow templates for automated setup. You can define custom workflow templates that specify the steps for provisioning resources, creating indices, and deploying models. These templates can be used with the OpenSearch Flow Framework to automate the setup process.

These advanced configuration options provide flexibility for adapting OpenSearch ML Quickstart to your specific requirements, whether you're using custom models, implementing specialized search logic, or automating the deployment process.

## 🧪 Testing

OpenSearch ML Quickstart includes a comprehensive test suite to ensure the reliability of its components. The test suite is divided into unit tests and integration tests, allowing you to verify both the individual components and their interactions.

Unit tests focus on testing the behavior of individual classes and functions in isolation, ensuring that they work as expected. These tests can be run quickly and are useful for verifying changes to specific components.

Integration tests verify the interactions between different components, ensuring that they work together correctly. These tests take longer to run but provide more comprehensive validation of the system's behavior. You can run specific integration test files or the entire suite, depending on your needs.

When running tests, you may need to comment out model types or host types in the test files if you haven't configured all options. This allows you to focus on testing the components that are relevant to your specific deployment scenario.

## 🔍 Troubleshooting

When working with OpenSearch ML Quickstart, you may encounter various issues that require troubleshooting. Common issues include compatibility problems with pandas versions, connection timeouts, model deployment failures, and index creation errors.

For pandas compatibility issues, the solution is to use Python 3.10 or later with pandas 2.0.3. Alternatively, you can use Anaconda to manage Python versions and ensure compatibility.

Connection timeouts can occur with both local OpenSearch clusters and Amazon OpenSearch Service. For local clusters, check the Docker container status to ensure that the cluster is running correctly. For Amazon OpenSearch Service, verify that the security group and network access settings allow connections from your environment.

Model deployment failures can occur with both local and remote models. For local models, ensure that the OpenSearch cluster has sufficient resources to host the model. For remote models, verify that your AWS credentials and permissions are correctly configured.

Index creation errors can occur if an index with the same name already exists. Use the `--delete-existing-index` flag to force the deletion of existing indices before creating new ones. Also check for naming conflicts and mapping compatibility issues.

For more detailed troubleshooting, you can enable debug mode by configuring the logging level. This will provide more information about what's happening behind the scenes, helping you identify and resolve issues.

When working with large datasets, you may need to optimize performance by increasing the bulk chunk size, processing specific categories only, or limiting the number of documents per category. For production deployments, consider using HNSW for faster approximate search, implementing hybrid search for better relevance, and configuring appropriate index settings for your specific use case.

## 📚 Additional Resources

To learn more about the technologies used in OpenSearch ML Quickstart, you can refer to the official documentation for OpenSearch, the OpenSearch ML Commons Plugin, Amazon OpenSearch Service, Amazon Bedrock, and Amazon SageMaker. These resources provide detailed information about the underlying technologies and can help you understand how they work together in the context of this toolkit.

## 🤝 Contributing

Contributions to OpenSearch ML Quickstart are welcome. To contribute, fork the repository, create a feature branch, add tests for new functionality, ensure that all tests pass, and submit a pull request. This process ensures that contributions maintain the quality and reliability of the toolkit.

## 📄 License

OpenSearch ML Quickstart is licensed under the Apache License 2.0. This permissive license allows you to use, modify, and distribute the toolkit for both personal and commercial purposes, subject to the terms of the license.

