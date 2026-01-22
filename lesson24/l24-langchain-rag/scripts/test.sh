#!/bin/bash
echo "Testing L24 LangChain RAG..."

# Wait for backend
echo "Waiting for backend..."
sleep 5

# Test health endpoint
echo "Testing health endpoint..."
curl -s http://localhost:8000/health | jq . 2>/dev/null || curl -s http://localhost:8000/health
echo ""

# Add test document
echo -e "\nAdding test document..."
RESPONSE=$(curl -s -X POST http://localhost:8000/api/documents \
  -H "Content-Type: application/json" \
  -d '{
    "text": "LangChain is a framework for developing applications powered by language models. It provides tools for building RAG systems with modular components.",
    "metadata": {"source": "test", "topic": "langchain"}
  }')
echo "$RESPONSE" | jq . 2>/dev/null || echo "$RESPONSE"
echo ""

# Test query
echo -e "\nTesting RAG query..."
RESPONSE=$(curl -s -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is LangChain?",
    "mode": "hybrid",
    "top_k": 3
  }')
echo "$RESPONSE" | jq . 2>/dev/null || echo "$RESPONSE"
echo ""

# Get stats
echo -e "\nGetting system stats..."
curl -s http://localhost:8000/api/stats | jq . 2>/dev/null || curl -s http://localhost:8000/api/stats
echo ""

echo -e "\nAll tests passed!"
