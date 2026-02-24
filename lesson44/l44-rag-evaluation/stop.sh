#!/bin/bash
echo "🛑 Stopping L44 services..."
pkill -f "uvicorn app.main" 2>/dev/null && echo "✅ Backend stopped" || echo "Backend not running"
pkill -f "react-scripts start" 2>/dev/null && echo "✅ Frontend stopped" || echo "Frontend not running"
echo "Done"
