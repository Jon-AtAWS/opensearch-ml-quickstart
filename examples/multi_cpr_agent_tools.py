# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

"""Agent tools for Multi-CPR conversational agent.

Provides search tools tailored to the Multi-CPR Chinese passage retrieval dataset
with e-commerce, video, and medical domains.
"""

from agent_tools import list_index_tool, index_mapping_tool


def get_passages_tool_semantic(index_name, embedding_model_id):
    """Semantic search tool for Multi-CPR passages using dense embeddings."""
    return {
        "type": "SearchIndexTool",
        "name": "SemanticSearchTool",
        "include_output_in_agent_response": True,
        "description":
            "此工具通过密集向量搜索在Multi-CPR知识库中查找相关段落。"
            "知识库包含电商、视频和医疗三个领域的中文段落。"
            "搜索基于语义相似度匹配，返回最相关的段落内容。"
            "适用于需要理解查询语义含义的搜索场景。",
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
                "_source": ["passage", "domain", "pid"]
            }
        },
        "attributes": {
            "input_schema": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "用户的自然语言问题"
                    }
                },
                "required": ["question"],
                "additionalProperties": False
            },
            "strict": False
        }
    }


def get_passages_tool_lexical(index_name):
    """Lexical search tool for Multi-CPR passages using keyword matching."""
    return {
        "type": "SearchIndexTool",
        "name": "LexicalSearchTool",
        "include_output_in_agent_response": True,
        "description":
            "此工具通过关键词匹配在Multi-CPR知识库中搜索段落。"
            "使用simple_query_string查询对段落文本进行词汇匹配。"
            "适用于包含特定关键词或产品名称的精确搜索。",
        "parameters": {
            "input": "{\"index\": \"${parameters.index}\", \"query\": ${parameters.query} }",
            "index": index_name,
            "query": {
                "query": {
                    "simple_query_string": {
                        "query": "${parameters.question}",
                        "fields": ["passage^2", "chunk_text"]
                    }
                },
                "size": 5,
                "_source": ["passage", "domain", "pid"]
            }
        },
        "attributes": {
            "input_schema": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "用户的自然语言问题"
                    }
                },
                "required": ["question"],
                "additionalProperties": False
            },
            "strict": False
        }
    }


def get_domain_filter_tool(index_name):
    """Domain aggregation tool to discover which domains have relevant content."""
    return {
        "type": "SearchIndexTool",
        "name": "DomainFilterTool",
        "include_output_in_agent_response": True,
        "description":
            "此工具用于查看Multi-CPR知识库中各领域的内容分布。"
            "返回电商(ecom)、视频(video)、医疗(medical)三个领域的匹配数量。"
            "可用于了解哪个领域与用户问题最相关。",
        "parameters": {
            "return_raw_response": True,
            "input": "{\"index\": \"${parameters.index}\", \"query\": ${parameters.query} }",
            "index": index_name,
            "query": {
                "query": {
                    "simple_query_string": {
                        "query": "${parameters.question}",
                        "fields": ["passage", "chunk_text"]
                    }
                },
                "aggs": {
                    "domains": {
                        "terms": {
                            "field": "domain",
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
                        "description": "用户的自然语言问题"
                    }
                },
                "required": ["question"],
                "additionalProperties": False
            },
            "strict": False
        }
    }
