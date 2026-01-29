# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0

# Source data: Twitter Customer Service dataset

import csv
import json
import uuid
import logging
import os
from collections import defaultdict
from datetime import datetime
import sys

# Add the parent directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from client import OsMlClientWrapper, get_client
from models import get_ml_model
from connectors.helper import get_remote_connector_configs

class TwitterCaseProcessor:
    def __init__(self, csv_file_path, output_dir="datasets"):
        self.csv_file_path = csv_file_path
        self.output_dir = output_dir
        self.conversations = defaultdict(list)
        self.tweets_by_id = {}
        
    def _is_company_account(self, author_id):
        """Determine if author is a company (not @number format)"""
        return not (author_id.startswith('@') and author_id[1:].isdigit())
    
    def _extract_company_name(self, author_id):
        """Extract company name from author_id"""
        if self._is_company_account(author_id):
            return author_id.replace('@', '') if author_id.startswith('@') else author_id
        return None
    
    def _parse_date_to_iso(self, date_str):
        """Convert Twitter date format to ISO format"""
        try:
            # Parse Twitter format: "Tue Oct 31 22:10:47 +0000 2017"
            dt = datetime.strptime(date_str, "%a %b %d %H:%M:%S %z %Y")
            return dt.isoformat()
        except ValueError:
            return date_str  # Return original if parsing fails
    
    def _parse_tweet_ids(self, tweet_ids_str):
        """Parse comma-separated tweet IDs"""
        if not tweet_ids_str or tweet_ids_str.strip() == '':
            return []
        return [tid.strip() for tid in tweet_ids_str.split(',')]
    
    def _build_conversation_groups(self):
        """Group tweets into conversations based on response relationships"""
        # First pass: load all tweets
        with open(self.csv_file_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                tweet_id = row['tweet_id']
                self.tweets_by_id[tweet_id] = row
        
        # Second pass: build conversation groups
        processed = set()
        
        for tweet_id, tweet in self.tweets_by_id.items():
            if tweet_id in processed:
                continue
                
            # Find root of conversation
            conversation_tweets = []
            self._collect_conversation_tweets(tweet_id, conversation_tweets, processed)
            
            if conversation_tweets:
                # Determine company name from company tweets in conversation
                company_name = None
                for t in conversation_tweets:
                    if self._is_company_account(t['author_id']):
                        company_name = self._extract_company_name(t['author_id'])
                        break
                
                if company_name:
                    self.conversations[len(self.conversations)] = {
                        'company_name': company_name,
                        'tweets': conversation_tweets
                    }
    
    def _collect_conversation_tweets(self, tweet_id, conversation_tweets, processed):
        """Recursively collect all tweets in a conversation thread"""
        if tweet_id in processed or tweet_id not in self.tweets_by_id:
            return
            
        tweet = self.tweets_by_id[tweet_id]
        processed.add(tweet_id)
        conversation_tweets.append(tweet)
        
        # Find tweets that respond to this one
        response_ids = self._parse_tweet_ids(tweet.get('response_tweet_id', ''))
        for response_id in response_ids:
            self._collect_conversation_tweets(response_id, conversation_tweets, processed)
        
        # Find tweet this responds to
        in_response_to = tweet.get('in_response_to_tweet_id', '')
        if in_response_to and in_response_to.strip():
            self._collect_conversation_tweets(in_response_to, conversation_tweets, processed)
    
    def _create_document(self, conversation_id, conversation_data, os_client, model_id):
        """Create OpenSearch document from conversation"""
        tweets = conversation_data['tweets']
        company_name = conversation_data['company_name']
        
        # Find inbound author (user, not company)
        inbound_author_id = None
        for tweet in tweets:
            if not self._is_company_account(tweet['author_id']):
                inbound_author_id = tweet['author_id']
                break
        
        # Build nested tweet documents
        nested_tweets = []
        for tweet in tweets:
            # Generate embedding for tweet text

            logging.getLogger('opensearch').setLevel(logging.CRITICAL + 1)
            logging.getLogger('urllib3').setLevel(logging.CRITICAL + 1)
            response = os_client.transport.perform_request(
                "POST", 
                f"/_plugins/_ml/models/{model_id}/_predict",
                body= {
                    "parameters": {
                        "input": [tweet['text']]
                    }
                }
            )
            inference_outputs = response["inference_results"][0]["output"]
            embedding = None
            for output in inference_outputs:
                if output["name"] == "sentence_embedding":
                    embedding = output["data"]
                    break
            if embedding is None:
                logging.error(f"Failed to get embedding. Response: {response}")
                sys.exit(1)
            logging.getLogger('opensearch').setLevel(logging.INFO)
            logging.getLogger('urllib3').setLevel(logging.INFO)

            nested_tweets.append({
                "tweet_id": tweet['tweet_id'],
                "author_id": tweet['author_id'],
                "inbound": tweet['inbound'] == 'True',
                "created_at": self._parse_date_to_iso(tweet['created_at']),
                "text": tweet['text'],
                "tweet_embedding": embedding,
                "response_tweet_id": tweet.get('response_tweet_id', ''),
                "in_response_to_tweet_id": tweet.get('in_response_to_tweet_id', ''),
                "is_company": self._is_company_account(tweet['author_id'])
            })
        
        # Sort tweets by timestamp
        nested_tweets.sort(key=lambda x: x['created_at'])
        
        return {
            "id": str(uuid.uuid4()),
            "conversation_id": conversation_id,
            "company_name": company_name,
            "inbound_author_id": inbound_author_id,
            "tweet_count": len(nested_tweets),
            "tweets": nested_tweets,
            "first_contact": min(self._parse_date_to_iso(t['created_at']) for t in tweets),
            "last_contact": max(self._parse_date_to_iso(t['created_at']) for t in tweets)
        }
    
    def process_and_write_files(self, os_client, model_id):
        """Process tweets and write to NDJSON files"""
        logging.info("Building conversation groups...")
        self._build_conversation_groups()
        
        logging.info(f"Found {len(self.conversations)} conversations")
        
        file_counter = 1
        doc_counter = 0
        current_file = None
        
        for conv_id, conv_data in self.conversations.items():
            if doc_counter % 100000 == 0:
                if current_file:
                    current_file.close()
                
                filename = f"{self.output_dir}/twcs{file_counter:05d}.json"
                current_file = open(filename, 'w', encoding='utf-8')
                logging.info(f"Writing to {filename}")
                file_counter += 1
            
            doc = self._create_document(conv_id, conv_data, os_client, model_id)
            current_file.write(json.dumps(doc) + '\n')
            doc_counter += 1
            if doc_counter % 1000 == 0:
                logging.info(f"Processed {doc_counter} conversations...")
        
        if current_file:
            current_file.close()
        
        logging.info(f"Processed {doc_counter} conversations into {file_counter-1} files")


def main():
    logging.basicConfig(level=logging.INFO)
    reader = TwitterCaseProcessor("datasets/twcs.csv", "datasets")
        # Hard-coded to use local OpenSearch deployment for local model support

    model_host = "sagemaker"
    host_type = "aos"
    embedding_type = "dense"

    # Initialize OpenSearch client
    logging.info(f"Getting client, OpenSearch type: {host_type}")
    os_client_wrapper = OsMlClientWrapper(get_client(host_type=host_type))
    os_client = os_client_wrapper.os_client

    config = { "embedding_type": embedding_type, }
    model_name = f"{host_type}_{model_host}"
    model_config = get_remote_connector_configs(
        host_type=host_type, connector_type=model_host
    )
    model_config["model_name"] = model_name
    config.update(model_config)
    
    embedding_model = get_ml_model(
        host_type=host_type,
        model_host=model_host,
        model_config=model_config,
        os_client=os_client,
        ml_commons_client=os_client_wrapper.ml_commons_client,
        model_group_id=os_client_wrapper.ml_model_group.model_group_id(),
    )

    # Get the model ID
    model_id = embedding_model.model_id()
    logging.info(f"Using model ID: {model_id}")

    reader.process_and_write_files(os_client_wrapper.os_client, model_id)


if __name__ == "__main__":
    main()
