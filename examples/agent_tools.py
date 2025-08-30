def list_index_tool(): 
    return {
        "type": "ListIndexTool",
        "name": "ListIndexTool",
        "include_output_in_agent_response": True
    }


def index_mapping_tool():
    return {
        "type": "IndexMappingTool",
        "name": "IndexMappingTool",
        "parameters": {
            "index": "${paramaters.index}",
            "input": "${parameters.question}",
        },
        "include_output_in_agent_response": True
    }


def get_products_tool_semantic(index_name, embedding_model_id):
    return {
        "type": "SearchIndexTool",
        "name": "SemanticSearchTool",
        "include_output_in_agent_response": True,
        "description":
            "This tool provides item catalog information by executing a dense "
            "vector search on the product catalog. The dense embedding is computed "
            "from the chunk field which contains the item name, and description. It "
            "returns the top 5 results with the chunk and product name. The tool is "
            "useful for finding items that are similar to the user's query ",
        "parameters": {
            "input": "{\"index\": \"${parameters.index}\", \"query\": ${parameters.query} }",
            "index": index_name,
            "query": {
                "query": {
                    "neural": {
                        "chunk_embedding": {
                            "query_text": "${parameters.question}",
                            "model_id": embedding_model_id
                        }
                    }
                },
                "size": 5,
                "_source": ["chunk", "item_name"]
            }
        },
        "attributes": {
            "input_schema": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "Natural language question"
                    }
                },
                "required": [ "question" ],
                "additionalProperties": False
            },
            "strict": False
        }
    }

def get_products_tool_lexical(index_name):
    return {
        "type": "SearchIndexTool",
        "name": "LexicalSearchTool",
        "include_output_in_agent_response": True,
        "description": 
            "This tool provides item catalog information by executing a "
            "lexical search on the product catalog. It uses a simple_query_string "
            "query against the item_name, product description, and brand name. "
            "results are sorted by lexical relevance. The tool returns the item name and chunk "
            "field for each result. ",
        "parameters": {
            "input": "{\"index\": \"${parameters.index}\", \"query\": ${parameters.query} }",
            "index": index_name,
            "query": {
                "query": {
                    "simple_query_string": {
                        "query": "${parameters.question}",
                        "fields": ["item_name^2", "product_description", "brand_name"]
                    }
                },
                "size": 5,
                "_source": ["item_name", "chunk"]
            }
        },
        "attributes": {
            "input_schema": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "Natural language question"
                    }
                },
                "required": [ "question" ],
                "additionalProperties": False
            },
            "strict": False
        }
    }


def get_products_qna_lexical(index_name):
    return {
        "type": "SearchIndexTool",
        "name": "QuestionSearchTool",
        "include_output_in_agent_response": True,
        "description": 
            "The catalog contains user questions and answers about the products "
            "in the catalog. This tool uses lexical search to find questions users "
            "have asked about products. The search results contain the item name, "
            "product description, brand name, question, and one or many answers "
            "to the question. The search results are selected based on matching "
            "the user query to the question in the search result, but might also "
            "match the item name, product description, or brand name. The answers "
            "in the search results are not generated, but are the actual answers "
            "provided by users. Each answer contains demographic information about "
            "the user who answered the question. ",
        "include_output_in_agent_response": True,
        "parameters": {
            "input": "{\"index\": \"${parameters.index}\", \"query\": ${parameters.query} }",
            "index": index_name,
            "query": {
                "size": 5,
                "_source": ["question_text", "answers", "item_name", "product_description", "brand_name"],
                "query": {
                    "simple_query_string": {
                        "query": "${parameters.question}",
                        "fields": ["question_text^4", "item_name", "product_description", "brand_name"]
                    }
                }
            }
        },
        "attributes": {
            "input_schema": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "Natural language question"
                    }
                },
                "required": [ "question" ],
                "additionalProperties": False
            },
            "strict": False
        }
    }


def get_categories_tool(index_name):
    return {
        "type": "SearchIndexTool",
        "name": "CategorySearchTool",
        "include_output_in_agent_response": True,
        "description": 
            "This tool provides category information about items in the catalog "
            "It compares the user question to products and then returns an aggregation "
            "of the matching categories with counts of how many products are in those "
            "categories. Use this information to add a category filter to the "
            "query to narrow down the search results.",
        "parameters": {
            "return_raw_response": True,
            "input": "{\"index\": \"${parameters.index}\", \"query\": ${parameters.query} }",
            "index": index_name,
            "query": {
                "query": {
                    "simple_query_string": {
                        "query": "${parameters.question}",
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
        },
        "attributes": {
            "input_schema": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "Natural language question"
                    }
                },
                "required": [ "question" ],
                "additionalProperties": False
            },
            "strict": False
        }
    }


