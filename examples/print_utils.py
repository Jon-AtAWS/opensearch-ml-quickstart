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
