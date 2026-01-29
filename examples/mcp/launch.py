#!/usr/bin/env python3
"""
Deploy MCP Server to Bedrock AgentCore
"""

import os
import sys
from pathlib import Path
import logging

sys.path.append(str(Path(__file__).parent.parent.parent))

from bedrock_agentcore import BedrockAgentCoreApp
from mcp_server import OpenSearchMCPServer
import asyncio

# Set up verbose logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = BedrockAgentCoreApp()

# Initialize MCP server once at startup
mcp_server = None

async def initialize_server():
    """Initialize the MCP server at startup"""
    global mcp_server
    import sys
    
    # Parse command line arguments for the server
    from mcp_server import get_command_line_args
    args = get_command_line_args()
    
    mcp_server = OpenSearchMCPServer(
        categories=args.categories,
        num_docs=args.number_of_docs_per_category,
        delete_existing=args.delete_existing_index,
        bulk_chunk_size=args.bulk_send_chunk_size,
        no_load=args.no_load
    )
    await mcp_server.initialize()

@app.route("/", methods=["POST"])
async def mcp_endpoint(request):
    """AgentCore POST endpoint for MCP server"""
    from starlette.responses import JSONResponse
    
    logger.info("Received POST request to MCP endpoint")
    
    try:
        # Get JSON body from request
        request_data = await request.json()
        logger.info(f"Request data: {request_data}")
        
        method = request_data.get("method")
        params = request_data.get("params", {})
        logger.info(f"Method: {method}, Params: {params}")
        
        if method == "tools/call":
            from mcp.types import CallToolRequest, CallToolRequestParams
            
            tool_name = params.get("name")
            arguments = params.get("arguments", {})
            logger.info(f"Calling tool: {tool_name} with arguments: {arguments}")
            
            call_request = CallToolRequest(
                params=CallToolRequestParams(
                    name=tool_name,
                    arguments=arguments
                )
            )
            
            logger.info("Executing MCP tool call...")
            result = await mcp_server.call_tool(call_request)
            logger.info(f"Tool call result: isError={result.isError}, content_count={len(result.content)}")
            
            response_data = {
                "content": [{"type": "text", "text": content.text} for content in result.content],
                "isError": result.isError
            }
            logger.info(f"Returning response: {response_data}")
        else:
            logger.error(f"Unknown method: {method}")
            response_data = {"error": f"Unknown method: {method}"}
        
        return JSONResponse(response_data)
            
    except Exception as e:
        logger.error(f"Exception in MCP endpoint: {e}", exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)

if __name__ == "__main__":
    print("🚀 Starting OpenSearch MCP Server with AgentCore...")
    print("Available tools: semantic_search, lexical_search, qna_search, category_search")
    print("📍 Local endpoint: http://localhost:8000")
    print("Use with: python mcp_client_agent.py --mcp-endpoint http://localhost:8000")
    
    # Initialize server before starting the app
    print("🔧 Initializing MCP server...")
    asyncio.run(initialize_server())
    print("✅ Server ready!")
    print("Press Ctrl+C to stop")
    
    app.run(host="localhost", port=8000, log_level="debug")
