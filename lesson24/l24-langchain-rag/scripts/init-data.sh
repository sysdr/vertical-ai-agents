#!/bin/bash
echo "Initializing sample data..."

curl -X POST http://localhost:8000/api/documents \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Retrieval Augmented Generation (RAG) combines retrieval systems with large language models. The retrieval component finds relevant documents, which are then provided as context to the LLM for generation.",
    "metadata": {"topic": "rag", "category": "fundamentals"}
  }'

curl -X POST http://localhost:8000/api/documents \
  -H "Content-Type: application/json" \
  -d '{
    "text": "LangChain provides abstractions for building LLM applications. Key components include chains (sequential operations), retrievers (document fetching), and agents (autonomous decision-making systems).",
    "metadata": {"topic": "langchain", "category": "frameworks"}
  }'

curl -X POST http://localhost:8000/api/documents \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hybrid search combines multiple retrieval strategies. Vector search finds semantically similar documents using embeddings, while keyword search uses traditional text matching. Fusion strategies merge results from both approaches.",
    "metadata": {"topic": "search", "category": "retrieval"}
  }'

echo "Sample data initialized!"
