#!/usr/bin/env python3
"""
MCP Client Agent - Uses MCP tools deployed on AgentCore
"""

import argparse
import logging
import json
import requests

from strands import Agent, tool
from strands.models import BedrockModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ANSI color codes
BLUE = '\033[94m'
PINK = '\033[95m'
YELLOW = '\033[93m'
RESET = '\033[0m'

def display_search_results(tool_name: str, result_text: str):
    """Display search results with colored formatting"""
    try:
        result_data = json.loads(result_text)
        results = result_data.get('results', [])
        query = result_data.get('query', 'N/A')
        
        print(f"\n{BLUE}=== {tool_name.upper()} RESULTS ==={RESET}")
        print(f"{PINK}Query: {query}{RESET}")
        print(f"Total hits: {result_data.get('total_hits', 0)}")
        
        for i, result in enumerate(results, 1):
            print(f"\n{BLUE}Result {i}: {result.get('item_name', 'N/A')}{RESET}")
            
            # Display question if available
            if 'question_text' in result:
                print(f"{PINK}Q: {result['question_text']}{RESET}")
                
                # Display answers if available
                answers = result.get('answers', [])
                if answers:
                    for j, answer in enumerate(answers, 1):
                        if isinstance(answer, dict):
                            answer_text = answer.get('answer_text', str(answer))
                        else:
                            answer_text = str(answer)
                        print(f"  {YELLOW}A{j}: {answer_text}{RESET}")
        
        # Display categories if this is category search
        if 'categories' in result_data:
            print(f"\n{BLUE}=== CATEGORIES ==={RESET}")
            for cat in result_data['categories']:
                print(f"  {PINK}{cat['category']}: {cat['count']} items{RESET}")
                
    except json.JSONDecodeError:
        print(f"\n{YELLOW}Raw result: {result_text}{RESET}")
    except Exception as e:
        print(f"\n{YELLOW}Error displaying results: {e}{RESET}")
        print(f"Raw result: {result_text}")

class MCPClientAgent:
    """Agent that uses MCP tools via client connection"""
    
    def __init__(self, mcp_server_endpoint: str):
        self.mcp_server_endpoint = mcp_server_endpoint
        
        # Initialize Bedrock model
        self.model = BedrockModel(
            model_id="us.anthropic.claude-3-7-sonnet-20250219-v1:0"
        )
        
        # Create agent with MCP tools
        self.agent = Agent(
            model=self.model,
            system_prompt=(
                "You are a helpful assistant that can search for product information. "
                "You have access to multiple search tools: semantic search for meaning-based queries, "
                "lexical search for keyword matching, Q&A search for user questions, and category "
                "search for product categorization. Use the most appropriate tool based on the user's question."
            ),
            tools=[
                self.semantic_search,
                self.lexical_search, 
                self.qna_search,
                self.category_search
            ]
        )
    
    def _call_mcp_tool(self, tool_name: str, arguments: dict) -> str:
        """Call MCP tool via HTTP request"""
        try:
            payload = {
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": arguments
                }
            }
            
            response = requests.post(
                self.mcp_server_endpoint,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            if result.get("isError"):
                return f"Error: {result.get('content', [{}])[0].get('text', 'Unknown error')}"
            else:
                return result.get("content", [{}])[0].get("text", "No results")
                
        except Exception as e:
            logger.error(f"MCP tool call failed: {e}")
            return f"Tool call failed: {str(e)}"
    
    @tool
    def semantic_search(self, question: str) -> str:
        """Perform semantic search using dense vector embeddings"""
        result = self._call_mcp_tool("semantic_search", {
            "question": question
        })
        display_search_results("semantic_search", result)
        return result
    
    @tool
    def lexical_search(self, question: str) -> str:
        """Perform lexical search using keyword matching"""
        result = self._call_mcp_tool("lexical_search", {
            "question": question
        })
        display_search_results("lexical_search", result)
        return result
    
    @tool
    def qna_search(self, question: str) -> str:
        """Search user questions and answers about products"""
        result = self._call_mcp_tool("qna_search", {
            "question": question
        })
        display_search_results("qna_search", result)
        return result
    
    @tool
    def category_search(self, question: str) -> str:
        """Get product category information and counts"""
        result = self._call_mcp_tool("category_search", {
            "question": question
        })
        display_search_results("category_search", result)
        return result
    
    def chat(self, message: str) -> str:
        """Send message to agent and get response"""
        import asyncio
        return asyncio.run(self.agent.invoke_async(message))

def get_command_line_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="MCP Client Agent")
    parser.add_argument(
        "--mcp-endpoint",
        type=str,
        default="http://localhost:8000",
        help="MCP server endpoint from AgentCore deployment"
    )
    parser.add_argument(
        "-q", "--question",
        type=str,
        help="Single question to ask"
    )
    return parser.parse_args()

def main():
    """Main function"""
    args = get_command_line_args()
    
    # Create MCP client agent
    agent = MCPClientAgent(args.mcp_endpoint)
    
    # Handle single question
    if args.question:
        response = agent.chat(args.question)
        print(f"\nQ: {args.question}")
        print(f"A: {response}")
        return
    
    # Interactive loop
    print(f"\nMCP Client Agent ready!")
    print(f"Connected to: {args.mcp_endpoint}")
    print("Type 'quit' to exit.\n")
    
    while True:
        try:
            question = input("Ask about products: ").strip()
            if question.lower() in ['quit', 'exit', 'q']:
                break
            if question:
                response = agent.chat(question)
                print(f"\nResponse: {response}\n")
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
