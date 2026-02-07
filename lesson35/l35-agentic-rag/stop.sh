#!/bin/bash

echo "=== Stopping L35 Agentic RAG System ==="

if [ -f .backend.pid ]; then
  kill $(cat .backend.pid) 2>/dev/null || true
  rm .backend.pid
fi

if [ -f .frontend.pid ]; then
  kill $(cat .frontend.pid) 2>/dev/null || true
  rm .frontend.pid
fi

# Fallback: kill by port
lsof -ti:8000 | xargs kill -9 2>/dev/null || true
lsof -ti:3000 | xargs kill -9 2>/dev/null || true

echo "✓ System stopped"
