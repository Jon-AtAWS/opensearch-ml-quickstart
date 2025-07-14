import json


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
    print("--------------------------------------------------------------------------------")
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
    print(
        f'{LIGHT_RED_HEADER}Question:{RESET} {hit["_source"]["question_text"]}'
    )
    for answer_id, answer in enumerate(hit["_source"]["answers"]):
        print(
            f'{LIGHT_GREEN_HEADER}Answer {answer_id + 1}:{RESET} {answer["answer_text"]}'
        )
    print()


def print_query(query):
    print(f"{LIGHT_GREEN_HEADER}Search query:{RESET}")
    print(json.dumps(query, indent=4))
    print("--------------------------------------------------------------------------------")
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
    
    print("\n" + "="*80 + "\n")


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
