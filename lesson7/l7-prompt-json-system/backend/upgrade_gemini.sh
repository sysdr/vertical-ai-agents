#!/bin/bash
# Upgrade google-generativeai to latest version

cd "$(dirname "$0")"
source venv/bin/activate

echo "Upgrading google-generativeai library..."
pip install --upgrade "google-generativeai>=0.8.0" --no-cache-dir

echo ""
echo "Current version:"
pip show google-generativeai | grep Version

echo ""
echo "✅ Upgrade complete!"

