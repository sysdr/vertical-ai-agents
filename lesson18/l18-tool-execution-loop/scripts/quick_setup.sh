#!/bin/bash

# Quick setup - reads API key from stdin or argument
# Usage examples:
#   echo "your_key" | ./scripts/quick_setup.sh
#   ./scripts/quick_setup.sh your_key
#   cat api_key.txt | ./scripts/quick_setup.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Get API key from argument, stdin, or environment
if [ -n "$1" ]; then
    API_KEY="$1"
elif [ -t 0 ]; then
    # No stdin and no argument - prompt
    echo "🔑 Quick API Key Setup"
    echo "======================"
    echo ""
    echo "Get your API key from: https://makersuite.google.com/app/apikey"
    echo ""
    read -p "Enter your API key: " API_KEY
else
    # Read from stdin
    read API_KEY
fi

if [ -z "$API_KEY" ]; then
    echo "❌ Error: API key cannot be empty!"
    exit 1
fi

# Use the set_api_key script
"$SCRIPT_DIR/set_api_key.sh" "$API_KEY"


