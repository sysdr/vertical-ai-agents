#!/bin/bash

echo "Stopping L27 RAG Evaluation System..."

pkill -f "uvicorn app.main:app"
pkill -f "vite"

echo "✅ System stopped"
