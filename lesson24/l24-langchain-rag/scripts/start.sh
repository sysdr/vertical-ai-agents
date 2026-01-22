#!/bin/bash
echo "Starting L24 LangChain RAG system..."

# Start with Docker
docker-compose up -d

echo "System started!"
echo "Backend: http://localhost:8000"
echo "Frontend: http://localhost:3000"
echo "API Docs: http://localhost:8000/docs"
