#!/bin/bash

echo "Stopping all services..."

pkill -f "uvicorn main:app"
pkill -f "react-scripts start"

echo "All services stopped"
