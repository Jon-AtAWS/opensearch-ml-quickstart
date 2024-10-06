# Datasets

If you choose to work with the Amazon PQA data:

1. Create an AWS account https://aws.amazon.com/resources/create-account/ if you don't already have one
1. Follow the instructions to [install the AWS Command Line Interface (CLI)](https://aws.amazon.com/cli/) 
2. Navigate to https://registry.opendata.aws/amazon-pqa/ to read the license and terms.
3. cd `<root>/datasets` (root is the directory you cloned from the repo)
4. Use the `aws s3` command at your favorite command line to copy the PQA dataset to your local datasets directory

   `aws s3 cp s3://amazon-pqa/amazon-pqa.tar.gz .`

5. If you want to keep the Amazon PQA data in a separate directory, create one and change to that directory

   ```
   mkdir amazon_pqa
   mv amazon-pqa.tar.gz amazon_pqa`
   cd amazon_pqa
   ```

5. Unpack the contents of the archive (you can delete the `.tar.gz` after you extract the files)

   `tar -xzf amazon-pqa.tar.gz`

5. If unpacked your data in a folder other than `<root>/datasets/amazon_pqa`, go to `<root>/configs/.env` and set `QANDA_FILE_READER_PATH` to point to the directory that contains the data.
5. In `<root>/data_process/qanda_file_reader.py`, have a look at the `enrich_question` method to get an idea on whether you want to use enriched meteadata.

