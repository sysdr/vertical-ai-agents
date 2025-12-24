#!/bin/bash
echo "🧹 Clearing old agent data..."

cd "$(dirname "$0")/.." || exit 1

# Clear decision log and memory files
rm -f data/agent_001_decisions.json
rm -f data/agent_001_memory.json

echo "✅ Old data cleared!"
echo "The agent will start fresh on the next interaction."
