#!/bin/bash
# Script to clear the default collection for re-indexing

echo "🗑️  Clearing default collection..."
curl -X DELETE http://localhost:8000/collections/default

echo ""
echo "✅ Collection cleared. You can now re-index your documents with proper embeddings."
echo "   Use the Index Documents panel in the dashboard to add documents."


