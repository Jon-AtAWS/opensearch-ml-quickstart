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
        "description": "This tool provides item catalog information by executing a dense "
        "vector search on the product catalog. It uses the OpenSearch ML semantic search to "
        "perform a vector search on the chunk field of the specified index, returning the top 5 "
        "most relevant results. The tool returns the product name, and description, for "
        "each result. The tool is useful for finding items that are similar to the user's query ",
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
                "_source": "chunk"
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
        "description": 
            "This tool provides item catalog information by executing a "
            "lexical search on the product catalog. It uses a simple_query_string "
            "query against the item_name, product description, and brand name. "
            "results are sorted by lexical relevance. The tool returns the chunk "
            "field for each result. ",
        "parameters": {
            "input": "{\"index\": \"${parameters.index}\", \"query\": ${parameters.query} }",
            "index": index_name,
            "query": {
                "query": {
                    "simple_query_string": {
                        "query": "${parameters.question}",
                        "fields": ["item_name^2", "product_description", "brand_name"],
                    }
                },
                "size": 5,
                "_source": "chunk"
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
