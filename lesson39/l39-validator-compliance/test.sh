#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🧪 Testing L39 Validator Compliance System..."

API_URL="http://localhost:8000"
# Pretty-print JSON when jq is available
pp() { command -v jq &>/dev/null && jq "$@" || cat; }

# Test health endpoint
echo "1. Testing health endpoint..."
curl -s $API_URL/health | pp '.'

# Test validation with sample data
echo -e "\n2. Testing validation with sample healthcare document..."
curl -s -X POST $API_URL/validate \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [
      {
        "id": "doc1",
        "content": "Patient with SSN 123-45-6789 prescribed ibuprofen. No peer-reviewed citations provided.",
        "source": "Medical Record"
      }
    ],
    "query": "Treatment recommendations",
    "domain": "healthcare"
  }' | pp '.'

# Test statistics
echo -e "\n3. Testing statistics endpoint..."
curl -s $API_URL/stats | pp '.'

# Test audit trail
echo -e "\n4. Testing audit trail..."
curl -s "$API_URL/audit-trail?limit=5" | pp '.audit_trail | length'

echo -e "\n✅ Tests complete!"
