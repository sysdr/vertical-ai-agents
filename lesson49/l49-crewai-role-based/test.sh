#!/bin/bash
set -e
BACKEND="http://localhost:8049"

echo "=== L49 Tests ==="

# Health check
echo -n "Health endpoint... "
HEALTH=$(curl -sf "$BACKEND/health")
echo "$HEALTH" | python3 -c "import sys,json; d=json.load(sys.stdin); assert d['framework']=='CrewAI', 'Wrong framework'"
echo "✅"

# Run crew (short topic for speed)
echo -n "Crew execution... "
RESULT=$(curl -sf -X POST "$BACKEND/run-crew"   -H "Content-Type: application/json"   -d '{"topic":"transformer attention mechanism"}')
echo "$RESULT" | python3 -c "
import sys, json
d = json.load(sys.stdin)
assert d['agent_count'] == 3, f'Expected 3 agents, got {d[\"agent_count\"]}'
assert d['task_count'] == 3, f'Expected 3 tasks, got {d[\"task_count\"]}'
assert len(d['final_output']) > 100, 'Final output too short'
print(f'  Agents: {d[\"agent_count\"]} | Tasks: {d[\"task_count\"]} | Output: {len(d[\"final_output\"])} chars')
"
echo "✅"

echo ""
echo "=== All tests passed ==="
