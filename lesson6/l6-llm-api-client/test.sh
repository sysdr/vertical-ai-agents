#!/bin/bash
set -e

echo "Running L6 Tests..."
source venv/bin/activate

if [ "$1" == "rate_limit" ]; then
    echo "Testing rate limiter..."
    python -m pytest tests/test_api_client.py::test_rate_limiter_basic -v
    python -m pytest tests/test_api_client.py::test_rate_limiter_throttling -v
elif [ "$1" == "circuit_breaker" ]; then
    echo "Testing circuit breaker..."
    python -m pytest tests/test_api_client.py::test_circuit_breaker_opens -v
elif [ "$1" == "cost_tracking" ]; then
    echo "Testing cost tracker..."
    python -m pytest tests/test_api_client.py::test_cost_tracker -v
else
    echo "Running all tests..."
    python -m pytest tests/test_api_client.py -v
fi

echo "✅ Tests complete!"
