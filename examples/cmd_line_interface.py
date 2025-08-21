"""
Command Line Interface Module

This module provides a complete command-line interface for OpenSearch ML quickstart
examples, including argument parsing, user interaction, and search interface functionality.

Features:
- Command line argument parsing for configuration options
- Interactive search interface with colorized output
- Generic search loop supporting multiple search types
- User input handling and error management
"""

import argparse
import json
import os
import sys


# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_process import QAndAFileReader


# Default categories from Amazon PQA dataset for testing
DEFAULT_CATEGORIES = [
    "earbud headphones",
    "headsets",
    "diffusers",
    "mattresses",
    "mp3 and mp4 players",
    "sheet and pillowcase sets",
    "batteries",
    "casual",
    "costumes",
]


def get_command_line_args():
    """
    Parse command line arguments for OpenSearch ML quickstart examples.

    Returns:
        argparse.Namespace: Parsed command line arguments with the following
        attributes:
            - force_index_creation (bool): Force index creation
            - delete_existing_index (bool): Delete existing index before
              creating new one
            - categories (list): List of Amazon PQA categories to process
            - bulk_send_chunk_size (int): Batch size for bulk document
              operations
            - number_of_docs_per_category (int): Maximum documents per category
            - opensearch_type (str): Connect to a local (os) or remote (aos)
              OpenSearch instance
    """
    parser = argparse.ArgumentParser(
        description="Run AI-powered search examples with OpenSearch ML Quickstart"
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
        default=5000,
        help="Number of documents to load per category",
    )
    parser.add_argument(
        "-o",
        "--opensearch-type",
        choices=["os", "aos"],
        default="aos",
        help="Type of OpenSearch instance to connect to: local=os or remote=aos",
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

    args = parser.parse_args()

    # Set default categories if none specified
    if args.categories is None:
        args.categories = DEFAULT_CATEGORIES
    elif args.categories == "all":
        args.categories = QAndAFileReader.AMAZON_PQA_CATEGORY_MAP.keys()

    return args


# ANSI escape sequence constants with improved colors
BOLD = "\033[1m"
RESET = "\033[0m"

# Headers
LIGHT_RED_HEADER = "\033[1;31m"
LIGHT_GREEN_HEADER = "\033[1;32m"
LIGHT_BLUE_HEADER = "\033[1;34m"
LIGHT_YELLOW_HEADER = "\033[1;33m"
LIGHT_PURPLE_HEADER = "\033[1;35m"


def print_hit(hit_id, hit):
    if not hit:
        return
    print(
        "--------------------------------------------------------------------------------"
    )
    print()
    print(
        f'{LIGHT_PURPLE_HEADER}Item {hit_id + 1} category:{RESET} {hit["_source"]["category_name"]}'
    )
    print(
        f'{LIGHT_YELLOW_HEADER}Item {hit_id + 1} product name:{RESET} {hit["_source"]["item_name"]}'
    )
    print()
    if hit["_source"]["product_description"]:
        print(f"{LIGHT_BLUE_HEADER}Production description:{RESET}")
        print(hit["_source"]["product_description"])
        print()
    if "question_text" in hit["_source"]:
        print(f'{LIGHT_RED_HEADER}Question:{RESET} {hit["_source"]["question_text"]}')
    else:
        print(f"{LIGHT_RED_HEADER}Question: {RESET} No question text available")
    if "answers" in hit["_source"]:
        for answer_id, answer in enumerate(hit["_source"]["answers"]):
            print(
                f'{LIGHT_GREEN_HEADER}Answer {answer_id + 1}:{RESET} {answer["answer_text"]}'
            )
    else:
        print(f"{LIGHT_GREEN_HEADER}Answer: {RESET} No answers available")
    print()


def print_query(query):
    print(f"{LIGHT_GREEN_HEADER}Search query:{RESET}")
    print(json.dumps(query, indent=4))
    print(
        "--------------------------------------------------------------------------------"
    )
    print()


def print_answer(answer):
    print(
        "--------------------------------------------------------------------------------"
    )
    print()
    print(f"{LIGHT_YELLOW_HEADER}LLM Answer:{RESET}")
    print(answer)
    print()
    print(
        "--------------------------------------------------------------------------------"
    )


def print_search_interface_header(index_name, model_id):
    """
    Print the header for the interactive search interface.

    Parameters:
        index_name (str): Name of the search index
        model_id (str): ID of the ML model being used
    """
    print(f"\n{LIGHT_GREEN_HEADER}Interactive Search Interface{RESET}")
    print(f"Index: {index_name}")
    print(f"Model ID: {model_id}")
    print("Enter your search queries below. Type 'quit' to exit.\n")


def print_search_prompt():
    """
    Print the search query prompt.

    Returns:
        str: User input for the search query
    """
    return input(f"{LIGHT_BLUE_HEADER}Search Query:{RESET} ")


def print_executing_search():
    """Print message indicating search is being executed."""
    print(f"\n{LIGHT_YELLOW_HEADER}Executing Search Query:{RESET}")


def print_search_results(search_results):
    """
    Print the complete search results, including summary and individual hits.

    Parameters:
        search_results (dict): OpenSearch search response containing hits
    """
    hits = search_results["hits"]["hits"]
    total_hits = search_results["hits"]["total"]["value"]

    print(f"\n{LIGHT_GREEN_HEADER}Search Results:{RESET}")
    print(f"Found {total_hits} total matches, showing top {len(hits)} results:\n")

    if not hits:
        print("No results found for your query.")
    else:
        for hit_id, hit in enumerate(hits):
            print_hit(hit_id, hit)

    print("\n" + "=" * 80 + "\n")


def print_goodbye():
    """Print goodbye message when exiting."""
    print("Goodbye!")


def print_search_interrupted():
    """Print message when search is interrupted."""
    print("\nSearch interrupted. Goodbye!")


def print_search_error(error):
    """
    Print search error message.

    Parameters:
        error (Exception): The error that occurred during search
    """
    print(f"An error occurred during search: {error}")


def print_empty_query_warning():
    """Print warning for empty search queries."""
    print("Please enter a valid search query.")


def extract_agent_response_text(agent_response):
    """
    Extract all response entries from agent results.
    
    Parameters:
        agent_response (dict): Agent execution response
        
    Returns:
        list: List of response entries with their types
    """
    responses = []
    if "inference_results" in agent_response:
        for result in agent_response["inference_results"]:
            print(f'➡️ Processing result: {json.dumps(result, indent=2)}')
            if "output" in result:
                for output in result["output"]:
                    print(f'\t📕 Processing output: {str(output)[:255]}')
                    if output.get("name") == "response" and "result" in output:
                        response_text = output["result"]
                        print(f'\t\t👄 Response text: {response_text[:255]}')
                        if response_text:
                            # Try to parse as JSON first
                            try:
                                parsed_json = json.loads(response_text)
                                # print(f'    📊 Parsed JSON: {parsed_json}')
                                responses.append({"type": "json", "content": parsed_json})
                            except json.JSONDecodeError:
                                # Check if it looks like search results
                                if response_text.strip().startswith('{"_index"'):
                                    responses.append({"type": "search_results", "content": response_text})
                                else:
                                    responses.append({"type": "text", "content": response_text})
                        else:
                            print(f'\t\t❌ No response_text found in response {output}')
                    else:
                        print(f'\t❌ This output is not an LLM response: {output}')
            else:
                print(f'❌ No output found in result')
    return responses


def process_and_print_agent_results(search_results, **kwargs):
    """
    Process and display conversational agent results with proper formatting for different response types.
    
    Parameters:
        search_results (dict): Agent execution response
        **kwargs: Additional parameters (unused)
    """
    # print(f"Full agent results\n{json.dumps(search_results, indent=2)}")
    responses = extract_agent_response_text(search_results)
    
    if not responses:
        print(f"\n📄 No valid response found in agent results")
        return
    
    for i, response in enumerate(responses):
        if response["type"] == "json":
            # Handle JSON responses (LLM messages with tool use)
            content = response["content"]
            if "output" in content and "message" in content["output"]:
                message = content["output"]["message"]
                if "content" in message:
                    for msg_content in message["content"]:
                        if "text" in msg_content:
                            print(f"\n🤖 Agent: {msg_content['text']}")
                        elif "toolUse" in msg_content:
                            tool_use = msg_content["toolUse"]
                            tool_name = tool_use.get('name', 'Unknown')
                            tool_input = tool_use.get('input', {})
                            print(f"\n🔧 Tool: {tool_name}")
                            for key, value in tool_input.items():
                                print(f"   {key}: {value}")
        
        elif response["type"] == "search_results":
            # Handle search results
            print(f"\n📊 Search Results:")
            lines = response["content"].strip().split('\n')
            for line in lines:
                if line.strip():
                    try:
                        result = json.loads(line)
                        if "_source" in result and "chunk" in result["_source"]:
                            score = result.get("_score", "N/A")
                            print(f"  • {result['_source']['chunk']} (Score: {score})")
                    except json.JSONDecodeError:
                        print(f"  • {line}")
        
        elif response["type"] == "text":
            # Handle final text responses
            print(f"\n💬 Final Response: {response['content']}")
    
    print()


def print_agent_query(query_body):
    """
    Print the agent query in a formatted way.

    Parameters:
        query_body (dict): Agent query parameters
    """
    print(f"{LIGHT_GREEN_HEADER}Agent Query:{RESET}")
    print(json.dumps(query_body, indent=4))
    print(
        "--------------------------------------------------------------------------------"
    )


def interactive_agent_loop(client, agent_id, model_info, build_agent_query_func, agent_executor_func):
    """
    Interactive loop for conversational agent queries.

    Parameters:
        client (OsMlClientWrapper): OpenSearch ML client wrapper
        agent_id (str): ID of the conversational agent
        model_info (str): Model information for display
        build_agent_query_func (callable): Function that builds agent query
            from user input, takes (query_text, agent_id) and returns query dict
        agent_executor_func (callable): Function that executes agent queries
    """
    import logging

    print_search_interface_header("Conversational Agent", model_info)

    while True:
        try:
            query_text = print_search_prompt()

            if query_text.lower().strip() in ["quit", "exit", "q"]:
                print_goodbye()
                break

            if not query_text.strip():
                print_empty_query_warning()
                continue

            # Build agent query
            query_body = build_agent_query_func(query_text, agent_id=agent_id)

            print_executing_search()
            print_agent_query(query_body)

            # Execute agent query using the provided function
            agent_response = agent_executor_func(client, agent_id, query_body)

            # Process and display results
            process_and_print_agent_results(agent_response)

        except KeyboardInterrupt:
            print_search_interrupted()
            break
        except Exception as e:
            logging.error(f"Agent query error: {e}")
            print_search_error(e)


def interactive_search_loop(
    client,
    index_name,
    model_info,
    query_builder_func,
    result_processor_func=None,
    question=None,
    **kwargs,
):
    """
    Generic interactive search interface for user queries.

    Parameters:
        client (OsMlClientWrapper): OpenSearch ML client wrapper
        index_name (str): Name of the index to search
        model_info (str): Model information to display in header
        query_builder_func (callable): Function that takes (query_text, **kwargs) and returns search query dict
        result_processor_func (callable, optional): Function to process results before printing
        question (str, optional): If provided, execute once with this question instead of interactive loop
        **kwargs: Additional parameters passed to query_builder_func and result_processor_func
    """
    import logging

    # Extract model objects and convert to model IDs for query builders
    processed_kwargs = kwargs.copy()

    # Handle single ml_model parameter
    if "ml_model" in processed_kwargs:
        ml_model = processed_kwargs.pop("ml_model")
        if ml_model is None:
            raise ValueError("ML model cannot be None when required for search.")
        processed_kwargs["model_id"] = ml_model.model_id()

    # Handle dense_ml_model parameter (for hybrid search)
    if "dense_ml_model" in processed_kwargs:
        dense_ml_model = processed_kwargs.pop("dense_ml_model")
        if dense_ml_model is None:
            raise ValueError("Dense ML model cannot be None when required for search.")
        processed_kwargs["dense_model_id"] = dense_ml_model.model_id()

    # Handle sparse_ml_model parameter (for hybrid search)
    if "sparse_ml_model" in processed_kwargs:
        sparse_ml_model = processed_kwargs.pop("sparse_ml_model")
        if sparse_ml_model is None:
            raise ValueError("Sparse ML model cannot be None when required for search.")
        processed_kwargs["sparse_model_id"] = sparse_ml_model.model_id()

    print_search_interface_header(index_name, model_info)

    while True:
        try:
            if question:
                query_text = question
            else:
                query_text = print_search_prompt()

                if query_text.lower().strip() in ["quit", "exit", "q"]:
                    print_goodbye()
                    break

                if not query_text.strip():
                    print_empty_query_warning()
                    continue

            # Build search query using the provided function
            search_query = query_builder_func(query_text, **processed_kwargs)

            print_executing_search()
            print_query(search_query)

            # Execute search with any additional search parameters
            search_params = processed_kwargs.get("search_params", {})
            search_results = client.os_client.search(
                index=index_name, body=search_query, **search_params
            )

            # Process results if custom processor provided
            if result_processor_func:
                result_processor_func(search_results, **processed_kwargs)
            else:
                print_search_results(search_results)
                
            # Break out if question was provided (single execution)
            if question:
                break
                
        except KeyboardInterrupt:
            print_search_interrupted()
            break
        except Exception as e:
            logging.error(f"Search error: {e}")
            print_search_error(e)
            if question:
                break
