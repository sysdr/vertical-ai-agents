#!/bin/bash

# Start backend
uvicorn backend.main:app --host 0.0.0.0 --port 8000 &

# Serve frontend build
cd frontend
npx serve -s build -l 3000 &

wait
