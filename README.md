# OpenSearch ML quickstart

This tool kit provides a quickstart for working with OpenSearch and ML models, especially LLMs for vector embeddings to power sementic and semantic sparse search. The code includes client classes for OpenSearch's `ml_commons` plugin that manage connections, and setup for `ml_models` and `ml_model_groups`. 

You can run the code against a self-managed OpenSearch cluster, or an Amazon OpenSearch Service domain. And, you can use either a local or remote-hosted model. All of the model management is in the `<root>/ml_models` directory.

We've provided code in the `<root>/data_process` directory that can read files from the open source [Amazon-PQA data set](https://registry.opendata.aws/amazon-pqa/). You'll need to navigate to the URL, download and unzip the data, storing all of the files in a local directory (we've provided the `datasets` directory for this purpose, see the README there for further instructions). The PQA data provides an excellent data set to explore lexical and semantic search. You can also find code in `qanda_file_reader.py` with generators for reading one or many categories, and code that enriches the documents with metadata and other product information. Use the data enrichment to experiment with hybrid lexical and semantic search.

We used the PQA data to provide a test bed for exploring different models, and different knn engine parameterizations. `<root>/main.py` runs through a set of tests, defined in `<root>/configs/tasks.py`. You can create tasks for local and remote models, loading some or all of the PQA data to test timing and search requests. See `<root>/client/os_ml_client_wrapper.py` for code that sets up OpenSearch's Neural plugin for ingest.

The `MlModel` class provides an abstract base class for using ML models, with 2 direct descendants, `LocalMlModel`, and `RemoteMlModel`. If you're running self-managed, you can load models into your local cluster with the `LocalMlModel` class. Amazon OpenSearch Service does not support local models. Use a connector to Amazon Bedrock, or Amazon Sagemaker with the `OsBedrockMlModel` and the `OsSagemakerMLModel` classes. For OpenSearch Service, you can use the `AosBedrockMlModel` and `AosSagemakerMlModel` classes. You can find the source for these classes in the `<root>/ml_models` folder.

# Set up and run with the Amazon_PQA data set

## Prerequisites

1. Python 3.10 is required. Currently there's a dependency for pandas 2.0.3 that requires Python 3.10. You can find downloads and installation instructions here: https://www.python.org/downloads/.

1. The [Amazon_PQA data set](https://registry.opendata.aws/amazon-pqa/).

1. Have an OpenSearch cluster or Amazon OpenSearch Service domain running. 

    - For instructions on how to set up an Amazon OpenSearch Service domain, see the documentation here: https://docs.aws.amazon.com/opensearch-service/latest/developerguide/gsgcreate-domain.html
    
    - For self-managed OpenSearch you can run with Docker, or deploy onto compute. We've provided a `docker-compose.yml` in the `<root>` folder if you want to run locally on Docker. 
    
    - The minium required OpenSearch version is **OpenSearch 2.13.0**. The code has been tested through version 2.13.0, 2.16.0 and Amazon OpenSearch Service 2.13.0.

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

Modify files in the `configs` directory with your own values. See below for the different cases of local/managed service, and local/remote models. Be sure to set a correct path for `QANDA_FILE_READER_PATH`

# Usage

## Local OpenSearch, local model

The simplest way to use the toolkit is to load data into OpenSearch running locally, on Docker desktop, and with a hugging face model loaded into the local, ML node.

1. Start Docker Desktop
2. Bring up the OpenSearch cluster with docker compose. Be sure to replace the `ExamplePassword!` below with a real password! Make a note of it, you'll need it in a second.

   1. `cd <root>`
   2. `export OPENSEARCH_INITAL_ADMIN_PASSWORD="<ExamplePassword!>"`
   3. `docker compose up`

3. Make sure that `OS_PASSWORD` in `<root>/configs/.env` is the same as your `OPENSEARCH_INITIAL_ADMIN_PASSWORD` when you started docker.
4. To run main.py, be sure your virtual environment is active (`source .venv/bin/activate`), then

    ```
    cd <root>
    python main.py --model_type local --host_type os -c adapters --delete_existing --number_of_docs 10   
    ```

   NOTE: If this is your first time running, it will take a minute for the model to load and deploy. If you're watching the log output, you'll see it busy waiting.

This will load 10 items from the "adapters" category, using the default index, and ingest pipeline names (both: `amazon_pqa`). You can set `number_of_docs` to `-1` if you want to load the whole category. You can omit the `-c adapters` to load the whole data set (or include a comma-separated list for more than 1 category).

Since the above command line does not specify `--cleanup`, the toolkit leaves the index, model, and model group intact in the cluster. You can go to the Dev Tools tab in OpenSearch Dashboards and see the index is there with `GET /_cat/indices`

# Working with remote models

To work with a remote model, you'll need an AWS account, with sufficient permissions to access Amazon Bedrock (Bedrock) or Amazon SageMaker (SageMaker), and to add access for Bedrock models. You can use remote models either with OpenSearch running locally, or Amazon OpenSearch Service (OpenSearch Service) managed clusters. See below for instructions on setting up and running with an OpenSearch Service domain.

## Bedrock

First, set up model access for Amazon Bedrock in the AWS console. From the Bedrock console, scroll down to the **Bedrock Configurations** section in the left navigation panel. Click **Model Access**. Add access for your account (or verify that you have access) to the embeddings model of your choice.

You configure OS_BEDROCK_URL in `<root>/conifgs/.env`.

Set the access and secret key for the account connecting to Bedrock<br>
`OS_BEDROCK_ACCESS_KEY=<Your AWS Access Key>`
`OS_BEDROCK_SECRET_KEY=<Your AWS Secret Access Key>`

Set the destination region<br>
`OS_BEDROCK_REGION=<Destination bedrock region>`

Set the URL where the quickstart will access Bedrock. The URL must be of the form: https://bedrock-runtime.`<region>`.amazonaws.com/model/`<model name>`/invoke. For example, to use Titan text embeddings, specify `https://bedrock-runtime.us-west-2.amazonaws.com/model/amazon.titan-embed-text-v1/invoke`.<br>
`OS_BEDROCK_URL=<Bedrock URL>`

Depending on your model, set the number of vector dimensions for the generated embeddings. E.g., for Amazon Titan text embeddings, use 1536 dimensions.<br>
`OS_BEDROCK_MODEL_DIMENSION=1536`

# Working with Amazon OpenSearch Service

You can use opensearch-ml-quickstart to connect to an Amazon OpenSearch Service domain. Opensearch-ml-quickstart does not work with Amazon OpenSearch Serverless collections, you must deploy a managed cluster.

## Create a domain

Follow [the developer guide's instructions](https://docs.aws.amazon.com/opensearch-service/latest/developerguide/gsgcreate-domain.html) to create a domain, with these changes:

1. Under **Network** choose **Public Access**
2. Under **Fine-grained access control** choose **Create master user**
   - Set a master user name and password. Record these for later use

## Set up variables in `<root>/configs/.env` for the domain

Set the master user's login credentials<br>
`AOS_USERNAME=<Your domain's 'Master user' user name>`<br>
`AOS_PASSWORD=<Your domain's 'Master user' password>`

Set the domain name<br>
`AOS_DOMAIN_NAME=<Your domain's name>`<br>
`AOS_REGION=<Your domain's region>`

Set the domain's Domain Endpoint. You can find this on the OpenSearch Service console in the AWS console. Click on **Managed clusters > Domains > <your domain name>** in the left nav to get to your domain's dashboard. Locate the domain endpoint and copy-paste it<br>
`AOS_HOST_URL=<Your domain's endpoint>`

Set the user name from the IAM user that created the domain and will connect to Bedrock.<br>
`AOS_AWS_USER_NAME=<Your IAM user name>`

## Set up variables in `<root>/configs/.env` to connect with the model 

Set the region where you are connecting to Bedrock.<br>
`AOS_BEDROCK_REGION=<AWS region like us-west-2>`

Opensearch-ml-quickstart creates two roles for the connector. The first role the `AOS_BEDROCK_CREATE_CONNECTOR_ROLE` enables the quickstart to write to OpenSearch Service to create the connector, and to set up fine-grained access control to allow writes. The second role, `AOS_BEDROCK_CONNECTOR_ROLE`, enables OpenSearch Service to call Bedrock to create embeddings.<br>
`AOS_BEDROCK_CONNECTOR_ROLE_NAME=<Name for the role created role for OpenSearch->Bedrock access>`<br>
`AOS_BEDROCK_CREATE_CONNECTOR_ROLE_NAME=<Name for the role to create the connector>`

Set the URL and model dimensions for bedrock<br>
`AOS_BEDROCK_URL=<Set as above: https://bedrock-runtime.us-west-2.amazonaws.com/model/amazon.titan-embed-text-v1/invoke>`<br>
`AOS_BEDROCK_MODEL_DIMENSION=<Integer: the number of dimensions in the embeddings>`

Run the code

```
cd <root>
python main.py --model_type bedrock --host_type aos -c adapters --delete_existing --number_of_docs 10   
```

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

# Troubleshooting

## is_datetime_or_timedelta

See also (this GitHub issue)[https://github.com/opensearch-project/opensearch-py-ml/issues/263]

Opensearch-py-ml requires pandas, which in turn requires this function. OpenSearch-ml-quickstart requires pandas version 2.0.3, and Python 3.10 (lower versions might also work). You can use (Anaconda)[https://www.anaconda.com/] to create a Python 3.10 environment, and then use that to create your virtual environment.

## Documents not appearing in your index

If you are using Amazon Bedrock to generate embeddings, you may be hitting limits on the number of calls that you can make. 