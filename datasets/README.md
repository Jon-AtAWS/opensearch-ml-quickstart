# Datasets

If you choose to work with the Amazon PQA data:
1. Create an AWS account https://aws.amazon.com/resources/create-account/ if you don't already have one
1. Follow the instructions to [install the AWS Command Line Interface (CLI)](https://aws.amazon.com/cli/) 
2. Navigate to https://registry.opendata.aws/amazon-pqa/ to read the license and terms.
3. CD to the root directory, where you cloned this repo, and cd to the datasets directory
4. Use the `aws s3` command at your favorite command line to copy the PQA dataset to your local datasets directory

   `aws s3 cp s3://amazon_pqa .`

5. In `<root>/data_process/qanda_file_reader.py`, have a look at the `enrich_question` method to get an idea on whether you want to use enriched meteadata.

