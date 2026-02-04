#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

echo "Testing L32 ReAct Agent..."

cd backend
if [ ! -d "venv" ]; then
    python3 -m venv venv --without-pip 2>/dev/null || python3 -m venv venv
    if [ ! -f "venv/bin/pip" ]; then
        curl -sS https://bootstrap.pypa.io/get-pip.py | venv/bin/python3
    fi
    source venv/bin/activate
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

pytest test_agent.py -v

echo "✓ Tests complete"
