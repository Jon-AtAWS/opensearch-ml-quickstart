# OpenSearch ML quickstart

This tool kit provides a quickstart for working with OpenSearch and ML models, especially LLMs for vector embeddings to power sementic and semantic sparse search. The code includes client classes for OpenSearch's `ml_commons` plugin that manage connections, and setup for `ml_models` and `ml_model_groups`. 

You can run the code against a self-managed OpenSearch cluster, or an Amazon OpenSearch Service domain. And, you can use either a local or remote-hosted model. All of the model management is in the `<root>/ml_models` directory.

We've provided code in the `<root>/data_process` directory that can read files from the open source [Amazon-PQA data set](https://registry.opendata.aws/amazon-pqa/). You'll need to navigate to the URL, download and unzip the data, storing all of the files in a local directory (we've provided the `datasets` directory for this purpose). The PQA data provides an excellent data set to explore lexical and semantic search. You can also find code in `qanda_file_reader.py` with generators for reading one or many categories, and code that enriches the documents with metadata and other product information. Use the data enrichment to experiment with hybrid lexical and semantic search.

We used the PQA data to provide a test bed for exploring different models, and different knn engine parameterizations. `<root>/main.py` runs through a set of tests, defined in `<root>/configs/tasks.py`. You can create tasks for local and remote models, loading some or all of the PQA data to test timing and search requests. See `<root>/client/os_ml_client_wrapper.py` for code that sets up OpenSearch's Neural plugin for ingest.

The `MlModel` class provides an abstract base class for using ML models, with 2 direct descendants, `LocalMlModel`, and `RemoteMlModel`. If you're running self-managed, you can load models into your local cluster with the `LocalMlModel` class. Amazon OpenSearch Service does not support local models. If you are running self-managed you can create a connector to Amazon Bedrock, or Amazon Sagemaker with the `OsBedrockMlModel` and the `OsSagemakerMLModel` classes. For OpenSearch Service, you can use the `AosBedrockMlModel` and `AosSagemakerMlModel` classes. You can find the source for these classes in the `<root>/ml_models` folder.

# Set up and run with the Amazon_PQA data set

## Prerequisites

1. Python 3.x (The code has been tested against Python version 3.12). You can find downloads and installation instructions here: https://www.python.org/downloads/

1. The [Amazon_PQA data set](https://registry.opendata.aws/amazon-pqa/).

1. Have an OpenSearch cluster or Amazon OpenSearch Service domain running. 

    - For instructions on how to set up an Amazon OpenSearch Service domain, see the documentation here: https://docs.aws.amazon.com/opensearch-service/latest/developerguide/gsgcreate-domain.html
    
    - For self-managed OpenSearch you can run with Docker, or deploy onto compute. We've provided a `docker-compose.yml` in the `<root>` folder if you want to run locally on Docker. 
    
    - The minium required OpenSearch version is **OpenSearch 2.13.0**. The code has been tested through version 2.16.0

2. If you are using a remote model, you need to configure an IAM user. The user should have permission to [access the Sagemaker](https://docs.aws.amazon.com/sagemaker/latest/dg/security-iam.html) or [Bedrock](https://docs.aws.amazon.com/bedrock/latest/userguide/security_iam_id-based-policy-examples.html) model. If you are using Bedrock, you need to [secure model access](https://docs.aws.amazon.com/bedrock/latest/userguide/model-access.html) for the model you want to use.

## Setup

1. We recommend setting up a Python virtual environment. Change directory to the `<root>` folder and then:

```
python3 -m venv .venv
source .venv/bin/activate
```

2. Install all the required packages with 

```
pip3 install -r requirements.txt
```

## Configure

Modify files in the `configs` directory with your own values

In .env you set up credentials for contacting the OpenSearch domain or cluster. If you're running locally, you don't need to set up credentials for an Amazon OpenSearch Service domain, or vice-versa. 

You also set up access for your model in `.env`. If you are using a local model, you don't need to set up credentials for SageMaker, or Bedrock, and vice versa.

3. Fill out the config files under src/configs: client_configs.py, data_configs.py and model_configs.py. You only need to fill in the configs for your model and host.


# Usage

## Local

The simplest way to use the toolkit is to load data into OpenSearch running locally, on Docker desktop, and with a hugging face model loaded into the local, ML node.

1. Start Docker Desktop
2. Bring up the OpenSearch cluster with docker compose. Be sure to replace the `ExamplePassword!` below with a real password! Make a note of it, you'll need it in a second.

   1. `cd <root>`
   2. `export OPENSEARCH_INITAL_ADMIN_PASSWORD="<ExamplePassword!>"`
   3. `docker compose up`

3. Open `<root>/configs/.env` and set `OS_PASSWORD` to the password you used starting Docker.
4. Run main.py

    ```
    python3 src/main.py -ht=os -mt=local
    ```

   If this is your first time running, it will take a minute for the model to load and deploy. If you're watching the log output, you'll see it busy waiting.

The a

You can get details for each parameters via:

```
python3 src/main.py -h
```

If you want to run different local models, you can add more configs in test_cases.py.

# Working with remote models

## Bedrock

You configure OS_BEDROCK_URL in .env. The URL must be of the form: https://bedrock-runtime.`<region>`.amazonaws.com/model/`<model name>`/invoke. For example, to use Titan text embeddings, specify `https://bedrock-runtime.us-west-2.amazonaws.com/model/amazon.titan-embed-text-v1/invoke`

# Testing

This repo has both unit tests and integration tests. For unit tests, you can test with:

```
pytest tests/unit/
```

For integration tests, you can test with:

```
pytest tests/integration
```

As it takes longer to run integration test, we strongly recommend you running each integration test file like:

```
pytest tests/integration/main_test.py
```

Please note that you need to comment out some model type or host type if you have not specified all the copnfigs under `src/config`.
