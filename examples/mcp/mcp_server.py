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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OpenSearchMCPServer:
    def __init__(self):
        self.server = Server("opensearch-tools")
        self.os_client = None
        self.index_name = "mcp_client_knowledge_base"
        self.embedding_model_id = None  # Will be set when needed
        
        # Register handlers
        self.server.list_tools = self.list_tools
        self.server.call_tool = self.call_tool
    
    async def initialize(self):
        """Initialize OpenSearch client"""
        try:
            self.os_client = get_client("os")
            logger.info("OpenSearch client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize OpenSearch client: {e}")
            raise
    
    async def list_tools(self, request: ListToolsRequest) -> ListToolsResult:
        """List available tools"""
        return ListToolsResult(
            tools=[
                Tool(
                    name="semantic_search",
                    description="Perform semantic search using dense vector embeddings",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "question": {"type": "string", "description": "Search query"},
                            "embedding_model_id": {"type": "string", "description": "Embedding model ID"}
                        },
                        "required": ["question", "embedding_model_id"]
                    }
                ),
                Tool(
                    name="lexical_search", 
                    description="Perform lexical search using keyword matching",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "question": {"type": "string", "description": "Search query"}
                        },
                        "required": ["question"]
                    }
                ),
                Tool(
                    name="qna_search",
                    description="Search user Q&A about products",
                    inputSchema={
                        "type": "object", 
                        "properties": {
                            "question": {"type": "string", "description": "Search query"}
                        },
                        "required": ["question"]
                    }
                ),
                Tool(
                    name="category_search",
                    description="Get product category aggregations",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "question": {"type": "string", "description": "Search query"}
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
        """Semantic search using embeddings"""
        question = args["question"]
        model_id = args["embedding_model_id"]
        
        query = {
            "query": {
                "neural": {
                    "chunk_embedding": {
                        "query_text": question,
                        "model_id": model_id
                    }
                }
            },
            "size": 5,
            "_source": ["chunk", "item_name", "question_text", "answers", "product_description", "brand_name"]
        }
        
        response = self.os_client.search(index=self.index_name, body=query)
        results = self._format_search_results(response, question)
        
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(results, indent=2))]
        )
    
    async def _lexical_search(self, args: Dict[str, Any]) -> CallToolResult:
        """Lexical search using keywords"""
        question = args["question"]
        
        query = {
            "query": {
                "simple_query_string": {
                    "query": question,
                    "fields": ["item_name^2", "product_description", "brand_name"]
                }
            },
            "size": 5,
            "_source": ["item_name", "chunk", "question_text", "answers", "product_description", "brand_name"]
        }
        
        response = self.os_client.search(index=self.index_name, body=query)
        results = self._format_search_results(response, question)
        
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(results, indent=2))]
        )
    
    async def _qna_search(self, args: Dict[str, Any]) -> CallToolResult:
        """Q&A search"""
        question = args["question"]
        
        query = {
            "size": 5,
            "_source": ["question_text", "answers", "item_name", "product_description", "brand_name"],
            "query": {
                "simple_query_string": {
                    "query": question,
                    "fields": ["question_text^4", "item_name", "product_description", "brand_name"]
                }
            }
        }
        
        response = self.os_client.search(index=self.index_name, body=query)
        results = self._format_search_results(response, question)
        
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(results, indent=2))]
        )
    
    async def _category_search(self, args: Dict[str, Any]) -> CallToolResult:
        """Category aggregation search"""
        question = args["question"]
        
        query = {
            "query": {
                "simple_query_string": {
                    "query": question,
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
            "total_hits": response.get("hits", {}).get("total", {}).get("value", 0),
            "categories": categories
        }
        
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

async def main():
    """Main entry point"""
    server = OpenSearchMCPServer()
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
