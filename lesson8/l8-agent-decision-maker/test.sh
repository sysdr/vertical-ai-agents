#!/bin/bash
set -e

echo "🧪 Running L8 Tests..."

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/backend"
source venv/bin/activate

echo "Running unit tests..."
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"
python -m pytest ../tests/ -v --tb=short

echo ""
echo "Testing API endpoints..."
python << PYTHON
import asyncio
import sys
sys.path.insert(0, '$SCRIPT_DIR')
from backend.main import DecisionMaker, ToolRegistry

async def test():
    print("Testing tool registry...")
    registry = ToolRegistry()
    assert len(registry.tools) == 4
    print("✓ Tool registry OK")
    
    print("Testing calculator tool...")
    calc = registry.get("calculator")
    result = await calc.execute({"operation": "factorial", "operands": [5]})
    assert result["success"] and result["result"] == 120
    print("✓ Calculator tool OK")
    
    print("Testing decision maker...")
    dm = DecisionMaker(registry)
    response = await dm.decision_maker(
        "Calculate 5 factorial",
        {},
        max_steps=5,
        timeout_seconds=10
    )
    assert response.success
    print("✓ Decision maker OK")
    
    print("\n✅ All tests passed!")

asyncio.run(test())
PYTHON

deactivate
