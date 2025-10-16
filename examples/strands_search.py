#!/usr/bin/env python3
"""
Strands Search Agent

A conversational search agent built with strands-agents that can:
1. Perform semantic search using embeddings
2. Execute lexical search queries
3. Search through Q&A content
4. Categorize and filter search results

Similar to OpenSearch's conversational agent but using the strands-agents framework.
"""

import argparse
import json
import logging
import os
import sys
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# Add the opensearch-ml-quickstart path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strands import Agent, tool
from client import OsMlClientWrapper, get_client, index_utils
from configs.configuration_manager import (
    get_client_configs,
    get_base_mapping_path,
    get_local_dense_embedding_model_name,
    get_local_dense_embedding_model_version,
    get_local_dense_embedding_model_format,
    get_local_dense_embedding_model_dimension,
)
from data_process.amazon_pqa_dataset import AmazonPQADataset
from mapping import get_base_mapping, mapping_update
from models import get_ml_model
from data_process.amazon_pqa_dataset import AmazonPQADataset
from mapping import get_base_mapping, mapping_update
from models import get_ml_model

# Configure logging
logging.basicConfig(
    format="%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)

def create_index_settings(base_mapping_path, index_config):
    """
    Create OpenSearch index settings for strands search.

    Parameters:
        base_mapping_path (str): Path to base mapping configuration
        index_config (dict): Configuration containing pipeline and model settings

    Returns:
        dict: Updated index settings with dense vector configuration
    """
    settings = get_base_mapping(base_mapping_path)
    pipeline_name = index_config["pipeline_name"]
    model_dimension = index_config["model_dimensions"]
    knn_settings = {
        "settings": {"index": {"knn": True}, "default_pipeline": pipeline_name},
        "mappings": {
            "properties": {
                "chunk": {"type": "text", "index": False},
                "chunk_embedding": {
                    "type": "knn_vector",
                    "dimension": model_dimension,
                },
            }
        },
    }
    mapping_update(settings, knn_settings)
    return settings


@dataclass
class SearchResult:
    """Represents a search result with metadata"""
    content: str
    score: float
    metadata: Dict[str, Any]
    source: str

class SearchTools:
    """Collection of search tools for the agent"""
    
    def __init__(self, client: OsMlClientWrapper, index_name: str, embedding_model_id: str):
        self.client = client
        self.index_name = index_name
        self.embedding_model_id = embedding_model_id
        
    @tool
    def semantic_search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """
        Perform semantic search using embeddings
        
        Args:
            query: Natural language search query
            top_k: Number of results to return
            
        Returns:
            List of search results
        """
        logging.info(f"Performing semantic search for: {query}")
        
        search_query = {
            "query": {
                "neural": {
                    "chunk_embedding": {
                        "query_text": query,
                        "model_id": self.embedding_model_id
                    }
                }
            },
            "size": top_k,
            "_source": ["chunk", "item_name"]
        }
        
        try:
            response = self.client.os_client.search(
                index=self.index_name,
                body=search_query
            )
            
            results = []
            for hit in response["hits"]["hits"]:
                result = SearchResult(
                    content=hit["_source"].get("chunk", ""),
                    score=hit["_score"],
                    metadata={
                        "item_name": hit["_source"].get("item_name", ""),
                        "id": hit["_id"]
                    },
                    source="semantic_search"
                )
                results.append(result)
                
            return results
            
        except Exception as e:
            logging.error(f"Semantic search failed: {e}")
            return []
    
    @tool
    def lexical_search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """
        Perform lexical/keyword search
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of search results
        """
        logging.info(f"Performing lexical search for: {query}")
        
        search_query = {
            "query": {
                "simple_query_string": {
                    "query": query,
                    "fields": ["item_name^2", "product_description", "brand_name"]
                }
            },
            "size": top_k,
            "_source": ["item_name", "chunk"]
        }
        
        try:
            response = self.client.os_client.search(
                index=self.index_name,
                body=search_query
            )
            
            results = []
            for hit in response["hits"]["hits"]:
                result = SearchResult(
                    content=hit["_source"].get("chunk", ""),
                    score=hit["_score"],
                    metadata={
                        "item_name": hit["_source"].get("item_name", ""),
                        "id": hit["_id"]
                    },
                    source="lexical_search"
                )
                results.append(result)
                
            return results
            
        except Exception as e:
            logging.error(f"Lexical search failed: {e}")
            return []
    
    @tool
    def qna_search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """
        Search through Q&A content
        
        Args:
            query: Question to search for
            top_k: Number of results to return
            
        Returns:
            List of Q&A search results
        """
        logging.info(f"Performing Q&A search for: {query}")
        
        search_query = {
            "size": top_k,
            "_source": ["question_text", "answers", "item_name", "product_description", "brand_name"],
            "query": {
                "simple_query_string": {
                    "query": query,
                    "fields": ["question_text^4", "item_name", "product_description", "brand_name"]
                }
            }
        }
        
        try:
            response = self.client.os_client.search(
                index=self.index_name,
                body=search_query
            )
            
            results = []
            for hit in response["hits"]["hits"]:
                # Format Q&A content
                question = hit["_source"].get("question_text", "")
                answers = hit["_source"].get("answers", [])
                answer_text = ""
                if answers:
                    answer_text = answers[0].get("answer_text", "") if isinstance(answers[0], dict) else str(answers[0])
                
                content = f"Q: {question}\nA: {answer_text}"
                
                result = SearchResult(
                    content=content,
                    score=hit["_score"],
                    metadata={
                        "question_text": question,
                        "answers": answers,
                        "item_name": hit["_source"].get("item_name", ""),
                        "product_description": hit["_source"].get("product_description", ""),
                        "brand_name": hit["_source"].get("brand_name", ""),
                        "id": hit["_id"]
                    },
                    source="qna_search"
                )
                results.append(result)
                
            return results
            
        except Exception as e:
            logging.error(f"Q&A search failed: {e}")
            return []
    
    @tool
    def category_search(self, query: str) -> Dict[str, int]:
        """
        Get category information for the query
        
        Args:
            query: Search query
            
        Returns:
            Dictionary of categories and their counts
        """
        logging.info(f"Performing category search for: {query}")
        
        search_query = {
            "query": {
                "simple_query_string": {
                    "query": query,
                    "fields": ["item_name^2", "question", "product_description", "brand_name"]
                }
            },
            "aggs": {
                "categories": {
                    "terms": {
                        "field": "category_name",
                        "size": 10
                    }
                }
            },
            "size": 0
        }
        
        try:
            response = self.client.os_client.search(
                index=self.index_name,
                body=search_query
            )
            
            categories = {}
            if "aggregations" in response and "categories" in response["aggregations"]:
                for bucket in response["aggregations"]["categories"]["buckets"]:
                    categories[bucket["key"]] = bucket["doc_count"]
                    
            return categories
            
        except Exception as e:
            logging.error(f"Category search failed: {e}")
            return {}


class StrandsSearchAgent:
    """
    Main search agent class using strands-agents framework
    """
    
    def __init__(self, client: OsMlClientWrapper, index_name: str, embedding_model_id: str):
        self.client = client
        self.index_name = index_name
        self.embedding_model_id = embedding_model_id
        self.search_tools = SearchTools(client, index_name, embedding_model_id)
        
        # Create tools for the strands agent
        tools = [
            self.search_tools.semantic_search,
            self.search_tools.lexical_search,
            self.search_tools.qna_search,
            self.search_tools.category_search
        ]
        
        # Agent system prompt similar to OpenSearch agent
        system_prompt = """
You are a helpful assistant that can answer questions about products in your knowledge base.

The knowledge base contains user questions and answers, with one search document per user question. 
Each search document also contains product information such as item name, product description, and brand name.

You have tools that search based on matching the user question to the question in the search result, 
as well as lexical and semantic search against the product information.

Because the knowledge base is organized by user questions, you may not get a broadly diverse range 
of product information in the search results, so try variants of the user question to get a wider range of products.

First evaluate whether the user is asking a broad question about products, or a specific question about a product. 
If the question is broad, you will use the category, lexical, and semantic search tools to find products that 
are similar to the user's query. If it seems that the user question is about product features or use, 
you will use the Q&A search tool to find questions users have asked about products.

In summarizing the search results include whether you approached the question as a broad product search 
or a specific product question.

When summarizing search results from any of the tools, if there are search results, but none of them 
are relevant to the user question, summarize the results, and include why none of them was relevant.
"""
        
        # Create the strands Agent
        self.agent = Agent(tools=tools, system_prompt=system_prompt)
    
    def process_query(self, query: str) -> str:
        """
        Process a user query and return a response
        
        Args:
            query: User question/query
            
        Returns:
            Agent response
        """
        logging.info(f"Processing query: {query}")
        return self.agent(query)
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get the conversation history"""
        return getattr(self.agent, 'conversation_history', [])
    
    def clear_history(self):
        """Clear conversation history"""
        if hasattr(self.agent, 'clear_history'):
            self.agent.clear_history()

def get_command_line_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Run Strands Search Agent with OpenSearch backend"
    )
    parser.add_argument(
        "-o",
        "--opensearch-type",
        choices=["os", "aos"],
        default="os",
        help="Type of OpenSearch instance to connect to: local=os or remote=aos",
    )
    parser.add_argument(
        "-i",
        "--index-name",
        default="strands_search_knowledge_base",
        help="Name of the OpenSearch index to search",
    )
    parser.add_argument(
        "-d",
        "--delete-existing-index",
        default=False,
        action="store_true",
        help="Delete the index if it already exists",
    )
    parser.add_argument(
        "-c",
        "--categories",
        nargs="+",
        default=None,
        help="List of categories to load into the index",
    )
    parser.add_argument(
        "-s",
        "--bulk-send-chunk-size",
        type=int,
        default=100,
        help="Chunk size for bulk sending documents to OpenSearch",
    )
    parser.add_argument(
        "-n",
        "--number-of-docs-per-category",
        type=int,
        default=-1,
        help="Number of documents to load per category",
    )
    parser.add_argument(
        "--no-load",
        action="store_true",
        default=False,
        help="Skip loading data into the index",
    )
    parser.add_argument(
        "-q",
        "--question",
        type=str,
        default=None,
        help="Execute search with this question and exit (instead of interactive loop)",
    )
    
    return parser.parse_args()

def interactive_mode(agent: StrandsSearchAgent):
    """Run the agent in interactive mode"""
    print("Strands Search Agent")
    print("=" * 50)
    print("Ask questions about products in the knowledge base.")
    print("Type 'quit' or 'exit' to stop, 'clear' to clear history.\n")
    
    while True:
        try:
            query = input("You: ").strip()
            
            if query.lower() in ['quit', 'exit']:
                print("Goodbye!")
                break
            elif query.lower() == 'clear':
                agent.clear_history()
                print("Conversation history cleared.")
                continue
            elif not query:
                continue
            
            response = agent.process_query(query)
            print(f"\nAgent: {response}\n")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            logging.error(f"Error processing query: {e}")
            print(f"Error: {e}")

def main():
    """Main function"""
    args = get_command_line_args()
    
    host_type = args.opensearch_type
    model_host = "local" if host_type == "os" else "sagemaker"
    index_name = args.index_name
    embedding_type = "dense"
    ingest_pipeline_name = "strands-search-ingest-pipeline"
    
    # Create OpenSearch client
    try:
        client = OsMlClientWrapper(get_client(host_type))
        logging.info(f"Connected to OpenSearch ({host_type})")
    except Exception as e:
        logging.error(f"Failed to connect to OpenSearch: {e}")
        sys.exit(1)

    config = {
        "with_knn": True,
        "pipeline_field_map": {"chunk_text": "chunk_embedding"},
        "categories": args.categories,
        "index_name": index_name,
        "pipeline_name": ingest_pipeline_name,
        "embedding_type": embedding_type,
        "delete_existing_index": args.delete_existing_index,
        "bulk_send_chunk_size": args.bulk_send_chunk_size,
    }

    model_name = get_local_dense_embedding_model_name()
    model_config = {
        "model_name": model_name,
        "model_version": get_local_dense_embedding_model_version(),
        "model_dimensions": get_local_dense_embedding_model_dimension(),
        "embedding_type": embedding_type,
        "model_format": get_local_dense_embedding_model_format(),
    }
    embedding_ml_model = get_ml_model(
        host_type=host_type,
        model_host=model_host,
        model_config=model_config,
        os_client=client.os_client,
        ml_commons_client=client.ml_commons_client,
        model_group_id=client.ml_model_group.model_group_id(),
    )

    config.update(model_config)
    config["index_settings"] = create_index_settings(
        base_mapping_path=get_base_mapping_path(),
        index_config=config,
    )

    # Set up knowledge base
    from client import index_utils
    index_utils.handle_index_creation(
        os_client=client.os_client,
        config=config,
        delete_existing=config["delete_existing_index"],
    )

    logging.info("Setting up knowledge base with embeddings...")
    client.setup_for_kNN(
        ml_model=embedding_ml_model,
        index_name=config["index_name"],
        pipeline_name=ingest_pipeline_name,
        pipeline_field_map=config["pipeline_field_map"],
        embedding_type=config["embedding_type"],
    )

    # Load data into knowledge base using dataset abstraction
    if not args.no_load:
        dataset = AmazonPQADataset(max_number_of_docs=args.number_of_docs_per_category)
        total_docs = dataset.load_data(
            os_client=client.os_client,
            index_name=config["index_name"],
            filter_criteria=args.categories,
            bulk_chunk_size=args.bulk_send_chunk_size
        )
        logging.info(f"Loaded {total_docs} documents")

    # Create the search agent
    agent = StrandsSearchAgent(client, index_name, embedding_ml_model.model_id())
    
    if args.question:
        # Process single query from command line
        response = agent.process_query(args.question)
        print(response)
    else:
        # Run in interactive mode
        interactive_mode(agent)

if __name__ == "__main__":
    main()
