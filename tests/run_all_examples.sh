#!/bin/bash

# Script to run all OpenSearch ML Quickstart examples
# Uses consistent parameters: -d -c jeans -n 100 -q "comfortable jeans"

if [ ! -d "examples" ]; then
  echo "Current directory is not the project root."
  echo "Change to the project root directory and re-run the script."
  exit 1
fi

echo "Running all OpenSearch ML Quickstart examples..."
echo "Parameters: -d -c jeans -n 100 -q \"comfortable jeans\""
echo "=========================================="

# AOS (Amazon OpenSearch Service) examples
echo "1. Dense Exact Search (AOS + SageMaker)"
python examples/dense_exact_search.py -d -c jeans -n 100 -q "comfortable jeans"
sleep 1s

echo -e "\n2. Dense HNSW Search (AOS + SageMaker)" 
python examples/dense_hnsw_search.py -d -c jeans -n 100 -q "comfortable jeans"
sleep 1s

echo -e "\n3. Sparse Search (AOS + SageMaker)"
python examples/sparse_search.py -d -c jeans -n 100 -q "comfortable jeans"
sleep 1s

echo -e "\n4. Hybrid Search (AOS + SageMaker)"
python examples/hybrid_search.py -d -c jeans -n 100 -q "comfortable jeans"
sleep 1s

echo -e "\n5. Conversational Search (AOS + SageMaker + Bedrock)"
python examples/conversational_search.py -d -c jeans -n 100 -q "comfortable jeans"
sleep 1s

echo -e "\n6. Workflow Example (AOS + Bedrock)"
python examples/workflow_example.py -d -c jeans -n 100 -q "comfortable jeans"
sleep 1s

echo -e "\n7. Workflow with Template (AOS + SageMaker)"
python examples/workflow_with_template.py -d -c jeans -n 100 -q "comfortable jeans"
sleep 1s

echo -e "\n8. Semantic Search Workflow (AOS + Bedrock)"
python examples/semantic_search_workflow.py -d -c jeans -n 100 -q "comfortable jeans"
sleep 1s

# OS (Self-managed OpenSearch) examples
echo -e "\n9. Hybrid Local Search (OS + Local models)"
python examples/hybrid_local_search.py -o os -d -c jeans -n 100 -q "comfortable jeans"
sleep 1s

echo -e "\n10. Lexical Search (OS - no ML models)"
python examples/lexical_search.py -o os -d -c jeans -n 100 -q "comfortable jeans"
sleep 1s

echo -e "\n11. Conversational Agent (OS + Local models)"
python examples/conversational_agent.py -o os -d -c jeans -n 100 -q "comfortable jeans"
sleep 1s

echo -e "\nAll examples completed!"
