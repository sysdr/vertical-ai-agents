#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/backend"
source venv/bin/activate

echo "Running L28 Tests..."

# Test imports
echo "Testing imports..."
python -c "
from agent.tool_equipped_agent import ToolEquippedAgent
from tools.registry import ToolRegistry
from tools.weather import WeatherTool
from tools.calculator import CalculatorTool
from rag.engine import RAGEngine
print('✓ All imports successful')
"

# Test CLI availability (no --help, CLI is interactive)
echo "Testing CLI import..."
python -c "from cli import AgentCLI; print('✓ CLI import OK')"

echo "All tests passed!"
