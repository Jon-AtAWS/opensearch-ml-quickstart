#!/usr/bin/env python3
"""
MCP Server exposing OpenSearch tools via Model Context Protocol
"""

import asyncio
import json
import logging
import os
import sys
from typing import Any, Dict

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolRequest,
    CallToolResult,
    ListToolsRequest,
    ListToolsResult,
    Tool,
    TextContent,
)

from client.helper import get_client
from client import OsMlClientWrapper, index_utils
from configs.configuration_manager import (
    get_base_mapping_path,
    get_local_dense_embedding_model_name,
    get_local_dense_embedding_model_version,
    get_local_dense_embedding_model_format,
    get_local_dense_embedding_model_dimension,
)
from data_process.amazon_pqa_dataset import AmazonPQADataset
from mapping import get_base_mapping, mapping_update
from models import get_ml_model
from connectors.helper import get_remote_connector_configs
from connectors import LlmConnector
from models import RemoteMlModel
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OpenSearchMCPServer:
    def __init__(self, categories=None, num_docs=100, delete_existing=False, bulk_chunk_size=100, no_load=False):
        self.server = Server("opensearch-tools")
        self.os_client = None
        self.ml_client = None
        self.index_name = "mcp_server_knowledge_base"
        self.embedding_model_id = None
        self.embedding_model = None
        self.memory_container_id = None
        self.categories = categories or ["jeans"]
        self.num_docs = num_docs
        self.delete_existing = delete_existing
        self.bulk_chunk_size = bulk_chunk_size
        self.no_load = no_load
        
        # Register handlers
        self.server.list_tools = self.list_tools
        self.server.call_tool = self.call_tool
    
    async def initialize(self):
        """Initialize OpenSearch client, models, and index"""
        try:
            print("🔧 Initializing OpenSearch MCP Server...")
            # Initialize clients
            self.os_client = get_client("os")
            self.ml_client = OsMlClientWrapper(self.os_client)
            logger.info("OpenSearch client initialized")
            print("✅ OpenSearch client connected")
            
            # Setup embedding model
            print("🤖 Setting up embedding model...")
            await self._setup_embedding_model()
            
            # Load data only if not skipped
            if not self.no_load:
                print("📊 Setting up index...")
                await self._setup_index()
                print("📚 Loading data...")
                await self._load_data()
            else:
                print("⏭️  Skipping data loading")
            
            # Create memory container for search assistant
            print("🧠 Setting up memory container...")
            await self._create_memory_container()
            print("🎉 MCP Server initialization complete!")
            
        except Exception as e:
            logger.error(f"Failed to initialize OpenSearch: {e}")
            print(f"❌ Initialization failed: {e}")
            raise
    
    async def _setup_embedding_model(self):
        """Setup embedding model for search operations"""
        try:
            print("  📝 Creating embedding model configuration...")
            model_config = {
                "model_name": get_local_dense_embedding_model_name(),
                "model_version": get_local_dense_embedding_model_version(),
                "model_dimensions": get_local_dense_embedding_model_dimension(),
                "embedding_type": "dense",
                "model_format": get_local_dense_embedding_model_format(),
            }
            
            print("  🚀 Deploying embedding model...")
            embedding_model = get_ml_model(
                host_type="os",
                model_host="local",
                model_config=model_config,
                os_client=self.os_client,
                ml_commons_client=self.ml_client.ml_commons_client,
                model_group_id=self.ml_client.ml_model_group.model_group_id(),
            )
            
            self.embedding_model_id = embedding_model.model_id()
            self.embedding_model = embedding_model  # Store the model object
            logger.info(f"Embedding model setup complete: {self.embedding_model_id}")
            print(f"  ✅ Embedding model ready: {self.embedding_model_id}")
            
        except Exception as e:
            logger.error(f"Failed to setup embedding model: {e}")
            print(f"  ❌ Embedding model setup failed: {e}")
            raise
    
    async def _setup_index(self):
        """Setup OpenSearch index with k-NN configuration"""
        try:
            # Create index settings
            base_mapping = get_base_mapping(get_base_mapping_path())
            model_dimension = get_local_dense_embedding_model_dimension()
            pipeline_name = f"{self.index_name}-pipeline"
            
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
            mapping_update(base_mapping, knn_settings)
            
            # Create index if it doesn't exist or delete if requested
            if self.delete_existing and self.os_client.indices.exists(index=self.index_name):
                self.os_client.indices.delete(index=self.index_name)
                logger.info(f"Deleted existing index: {self.index_name}")
            
            if not self.os_client.indices.exists(index=self.index_name):
                self.os_client.indices.create(index=self.index_name, body=base_mapping)
                logger.info(f"Created index: {self.index_name}")
            
            # Setup k-NN pipeline using the stored model object
            self.ml_client.setup_for_kNN(
                ml_model=self.embedding_model,
                index_name=self.index_name,
                pipeline_name=pipeline_name,
                pipeline_field_map={"chunk_text": "chunk_embedding"},
                embedding_type="dense"
            )
            
            logger.info(f"Index and pipeline setup complete")
            
        except Exception as e:
            logger.error(f"Failed to setup index: {e}")
            raise
    
    async def _load_data(self):
        """Load data into the index"""
        try:
            dataset = AmazonPQADataset(max_number_of_docs=self.num_docs)
            total_docs = dataset.load_data(
                os_client=self.os_client,
                index_name=self.index_name,
                filter_criteria=self.categories,
                bulk_chunk_size=self.bulk_chunk_size
            )
            logger.info(f"Loaded {total_docs} documents")
            
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise
    
    async def _create_memory_container(self):
        """Create OpenSearch memory container with LLM and embedding integration"""
        try:
            # Setup both LLM and embedding models for memory operations
            llm_model_id = await self._setup_memory_llm()
            embedding_model_id = await self._setup_memory_embedding()
            
            memory_config = {
                "name": "mcp-search-assistant",
                "description": "Memory container for MCP search assistant with LLM and embedding capabilities",
                "configuration": {
                    "embedding_model_type": "TEXT_EMBEDDING",
                    "embedding_model_id": embedding_model_id,
                    "embedding_dimension": get_local_dense_embedding_model_dimension(),
                    "llm_id": llm_model_id,
                    "strategies": [
                        {
                            "type": "USER_PREFERENCE",
                            "namespace": ["user_id"]
                        },
                        {
                            "type": "SUMMARY", 
                            "namespace": ["user_id", "session_id"]
                        }
                    ]
                }
            }
            
            logging.info(f"Creating memory container..., memory_config\n{json.dumps(memory_config, indent=2)}")
            response = self.os_client.transport.perform_request(
                "POST", "/_plugins/_ml/memory_containers/_create",
                body=memory_config
            )
            
            self.memory_container_id = response.get("memory_container_id")
            logger.info(f"Created memory container with LLM and embedding: {self.memory_container_id}")
            
        except Exception as e:
            logger.warning(f"Failed to create memory container: {e}")
            # Continue without memory if creation fails
    
    async def _setup_memory_embedding(self):
        """Set up embedding model for memory operations using OpenSearch ML Commons"""
        try:
            logger.info("Setting up memory embedding model...")
            from configs.configuration_manager import (
                get_local_dense_embedding_model_name,
                get_local_dense_embedding_model_version,
                get_local_dense_embedding_model_format,
                get_local_dense_embedding_model_dimension,
            )
            from models import get_ml_model
            
            # Create local embedding model for memory operations
            model_config = {
                "model_name": get_local_dense_embedding_model_name(),
                "model_version": get_local_dense_embedding_model_version(),
                "model_dimensions": get_local_dense_embedding_model_dimension(),
                "embedding_type": "dense",
                "model_format": get_local_dense_embedding_model_format(),
            }
            
            embedding_model = get_ml_model(
                host_type="os",
                model_host="local",
                model_config=model_config,
                os_client=self.os_client,
                ml_commons_client=self.ml_client.ml_commons_client,
                model_group_id=self.ml_client.ml_model_group.model_group_id()
            )
            
            model_id = embedding_model.model_id()
            logger.info(f"Memory embedding model setup complete: {model_id}")
            return model_id
            
        except Exception as e:
            logger.warning(f"Failed to setup memory embedding model: {e}")
            return None
    
    async def _setup_memory_llm(self):
        """Set up LLM model for memory operations using OpenSearch ML Commons"""
        try:
            logger.info("Setting up memory LLM model...")
            # Use the same Bedrock connector approach as the conversational agent
            from connectors.helper import get_remote_connector_configs
            from connectors import LlmConnector
            from models import RemoteMlModel
            
            # Get Bedrock connector configs
            connector_configs = get_remote_connector_configs("bedrock", "os")
            logger.info("Retrieved Bedrock connector configs for memory LLM")
            
            # Create LLM connector for memory operations using memory type
            llm_connector = LlmConnector(
                os_client=self.os_client,
                os_type="os",
                connector_configs=connector_configs,
                llm_type="memory",
                model_name="us.anthropic.claude-3-7-sonnet-20250219-v1:0"
            )
            logger.info("Created LLM connector for memory operations")
            
            # Create remote ML model for memory LLM
            llm_model = RemoteMlModel(
                os_client=self.os_client,
                ml_commons_client=self.ml_client.ml_commons_client,
                ml_connector=llm_connector,
                model_group_id=self.ml_client.ml_model_group.model_group_id(),
                model_name="Memory LLM for MCP"
            )
            
            model_id = llm_model.model_id()
            logger.info(f"Memory LLM setup complete: {model_id}")
            return model_id
            
        except Exception as e:
            logger.warning(f"Failed to setup memory LLM: {e}")
            return None
    
    async def _get_memories(self, user_id: str, query: str, limit: int = 5):
        """Retrieve relevant memories for the user and query"""
        if not self.memory_container_id:
            return {}
        
        try:
            # Get working memory for the user session
            response = self.os_client.transport.perform_request(
                "GET", f"/_plugins/_ml/memory_containers/{self.memory_container_id}/memories/working/_search",
                body={
                    "query": {"match_all": {}}, 
                    "sort": [{
                        "created_time": {"order": "desc"}
                    }]
                }
            )
            logging.info(f"Retrieved memories: {response}")

            return response
            
        except Exception as e:
            logger.warning(f"Failed to retrieve memories: {e}")
            return {}
    
    async def _store_memory(self, user_id: str, session_id: str, query: str, results: dict):
        """Store interaction as memory"""
        if not self.memory_container_id:
            return
        
        try:
            memory_data = {
                "messages": [
                    {
                        "role": "user",
                        "content": [query]
                    },
                    {
                        "role": "assistant", 
                        "content": [self._summarize_results(results)]
                    }
                ],
                "payload_type": "conversational",
                "namespace": {
                    "user_id": user_id,
                    "session_id": session_id
                },
                "infer": True
            }
            
            logging.info(f"Storing memory for user_id={user_id}, session_id={session_id}")
            logging.info(f"{memory_data}")
            self.os_client.transport.perform_request(
                "POST", f"/_plugins/_ml/memory_containers/{self.memory_container_id}/memories",
                body=memory_data
            )
            
        except Exception as e:
            logger.warning(f"Failed to store memory: {e}")
    
    def _summarize_results(self, results: dict) -> str:
        """Create a summary of search results for memory storage"""
        total_hits = results.get("total_hits", 0)
        result_items = results.get("results", [])
        
        if not result_items:
            return f"No results found (0 hits)"
        
        # Extract key information from top results
        top_items = [item.get("item_name", "Unknown") for item in result_items[:3]]
        return f"Found {total_hits} results. Top items: {', '.join(top_items)}"
    
    def _enhance_query_with_memory(self, query: str, memory: dict) -> str:
        """Enhance search query with relevant memory context"""
        if not memory or not memory.get("messages"):
            return query
        
        # Extract relevant context from memory messages
        memory_context = []
        for message in memory.get("messages", []):
            if message.get("role") == "user":
                memory_context.append(f"Previous: {message.get('content_text', '')}")
        
        if memory_context:
            enhanced = f"{query} (Context: {'; '.join(memory_context[:2])})"
            return enhanced
        
        return query
    
    async def list_tools(self, request: ListToolsRequest) -> ListToolsResult:
        """List available tools"""
        return ListToolsResult(
            tools=[
                Tool(
                    name="semantic_search",
                    description="Perform semantic search using dense vector embeddings with memory context",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "question": {"type": "string", "description": "Search query"},
                            "user_id": {"type": "string", "description": "User identifier for memory", "default": "default"},
                            "session_id": {"type": "string", "description": "Session identifier for memory", "default": "default"}
                        },
                        "required": ["question"]
                    }
                ),
                Tool(
                    name="lexical_search", 
                    description="Perform lexical search using keyword matching with memory context",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "question": {"type": "string", "description": "Search query"},
                            "user_id": {"type": "string", "description": "User identifier for memory", "default": "default"},
                            "session_id": {"type": "string", "description": "Session identifier for memory", "default": "default"}
                        },
                        "required": ["question"]
                    }
                ),
                Tool(
                    name="qna_search",
                    description="Search user Q&A about products with memory context",
                    inputSchema={
                        "type": "object", 
                        "properties": {
                            "question": {"type": "string", "description": "Search query"},
                            "user_id": {"type": "string", "description": "User identifier for memory", "default": "default"},
                            "session_id": {"type": "string", "description": "Session identifier for memory", "default": "default"}
                        },
                        "required": ["question"]
                    }
                ),
                Tool(
                    name="category_search",
                    description="Get product category aggregations with memory context",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "question": {"type": "string", "description": "Search query"},
                            "user_id": {"type": "string", "description": "User identifier for memory", "default": "default"},
                            "session_id": {"type": "string", "description": "Session identifier for memory", "default": "default"}
                        },
                        "required": ["question"]
                    }
                )
            ]
        )
    
    async def call_tool(self, request: CallToolRequest) -> CallToolResult:
        """Execute tool calls"""
        tool_name = request.params.name
        args = request.params.arguments
        
        try:
            if tool_name == "semantic_search":
                return await self._semantic_search(args)
            elif tool_name == "lexical_search":
                return await self._lexical_search(args)
            elif tool_name == "qna_search":
                return await self._qna_search(args)
            elif tool_name == "category_search":
                return await self._category_search(args)
            else:
                raise ValueError(f"Unknown tool: {tool_name}")
        except Exception as e:
            return CallToolResult(
                content=[TextContent(type="text", text=f"Error: {str(e)}")],
                isError=True
            )
    
    async def _semantic_search(self, args: Dict[str, Any]) -> CallToolResult:
        """Semantic search using embeddings with memory integration"""
        question = args["question"]
        user_id = args.get("user_id", "default")
        session_id = args.get("session_id", "default")
        
        # Retrieve relevant memories
        memory = await self._get_memories(user_id, question)
        
        # Enhance query with memory context
        enhanced_question = self._enhance_query_with_memory(question, memory)
        
        query = {
            "query": {
                "neural": {
                    "chunk_embedding": {
                        "query_text": enhanced_question,
                        "model_id": self.embedding_model_id
                    }
                }
            },
            "size": 5,
            "_source": ["chunk", "item_name", "question_text", "answers", "product_description", "brand_name"]
        }
        
        response = self.os_client.search(index=self.index_name, body=query)
        results = self._format_search_results(response, question)
        
        # Store interaction as memory
        await self._store_memory(user_id, session_id, question, results)
        
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(results, indent=2))]
        )
    
    async def _lexical_search(self, args: Dict[str, Any]) -> CallToolResult:
        """Lexical search using keywords with memory integration"""
        question = args["question"]
        user_id = args.get("user_id", "default")
        session_id = args.get("session_id", "default")
        
        # Retrieve relevant memories
        memories = await self._get_memories(user_id, question)
        
        # Enhance query with memory context
        enhanced_question = self._enhance_query_with_memory(question, memories)
        
        query = {
            "query": {
                "simple_query_string": {
                    "query": enhanced_question,
                    "fields": ["item_name^2", "product_description", "brand_name"]
                }
            },
            "size": 5,
            "_source": ["item_name", "chunk", "question_text", "answers", "product_description", "brand_name"]
        }
        
        response = self.os_client.search(index=self.index_name, body=query)
        results = self._format_search_results(response, question)
        
        # Store interaction as memory
        await self._store_memory(user_id, session_id, question, results)
        
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(results, indent=2))]
        )
    
    async def _qna_search(self, args: Dict[str, Any]) -> CallToolResult:
        """Q&A search with memory integration"""
        question = args["question"]
        user_id = args.get("user_id", "default")
        session_id = args.get("session_id", "default")
        
        # Retrieve relevant memories
        memories = await self._get_memories(user_id, question)
        
        # Enhance query with memory context
        enhanced_question = self._enhance_query_with_memory(question, memories)
        
        query = {
            "size": 5,
            "_source": ["question_text", "answers", "item_name", "product_description", "brand_name"],
            "query": {
                "simple_query_string": {
                    "query": enhanced_question,
                    "fields": ["question_text^4", "item_name", "product_description", "brand_name"]
                }
            }
        }
        
        response = self.os_client.search(index=self.index_name, body=query)
        results = self._format_search_results(response, question)
        
        # Store interaction as memory
        await self._store_memory(user_id, session_id, question, results)
        
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(results, indent=2))]
        )
    
    async def _category_search(self, args: Dict[str, Any]) -> CallToolResult:
        """Category aggregation search with memory integration"""
        question = args["question"]
        user_id = args.get("user_id", "default")
        session_id = args.get("session_id", "default")
        
        # Retrieve relevant memories
        memories = await self._get_memories(user_id, question)
        
        # Enhance query with memory context
        enhanced_question = self._enhance_query_with_memory(question, memories)
        
        query = {
            "query": {
                "simple_query_string": {
                    "query": enhanced_question,
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
        
        response = self.os_client.search(index=self.index_name, body=query)
        
        categories = []
        for bucket in response.get("aggregations", {}).get("categories", {}).get("buckets", []):
            categories.append({
                "category": bucket.get("key"),
                "count": bucket.get("doc_count")
            })
        
        results = {
            "query": question,
            "total_hits": response.get("hits", {}).get("total", {}).get("value", 0),
            "categories": categories
        }
        
        # Store interaction as memory
        await self._store_memory(user_id, session_id, question, results)
        
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(results, indent=2))]
        )
    
    def _format_search_results(self, response, query_text=None):
        """Format OpenSearch response"""
        results = []
        for hit in response.get("hits", {}).get("hits", []):
            source = hit.get("_source", {})
            results.append({
                "score": hit.get("_score"),
                **source
            })
        
        return {
            "query": query_text,
            "total_hits": response.get("hits", {}).get("total", {}).get("value", 0),
            "results": results
        }

def get_command_line_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="MCP Server with OpenSearch")
    parser.add_argument(
        "-c", "--categories",
        nargs="+",
        default=["jeans"],
        help="Categories to load"
    )
    parser.add_argument(
        "-n", "--number-of-docs-per-category",
        type=int,
        default=100,
        help="Number of documents per category"
    )
    parser.add_argument(
        "-d", "--delete-existing-index",
        action="store_true",
        help="Delete existing index"
    )
    parser.add_argument(
        "-s", "--bulk-send-chunk-size",
        type=int,
        default=100,
        help="Bulk send chunk size"
    )
    parser.add_argument(
        "--no-load",
        action="store_true",
        help="Skip loading data"
    )
    return parser.parse_args()

async def main():
    """Main entry point"""
    args = get_command_line_args()
    
    server = OpenSearchMCPServer(
        categories=args.categories,
        num_docs=args.number_of_docs_per_category,
        delete_existing=args.delete_existing_index,
        bulk_chunk_size=args.bulk_send_chunk_size,
        no_load=args.no_load
    )
    await server.initialize()
    
    async with stdio_server() as (read_stream, write_stream):
        await server.server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="opensearch-tools",
                server_version="1.0.0",
                capabilities=server.server.get_capabilities()
            )
        )

if __name__ == "__main__":
    asyncio.run(main())
