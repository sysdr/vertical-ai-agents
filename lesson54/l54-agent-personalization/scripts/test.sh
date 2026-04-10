#!/usr/bin/env bash
BASE="http://localhost:8054"
PASS=0; FAIL=0

check() {
  local label="$1"; local cmd="$2"; local expect="$3"
  result=$(eval "$cmd" 2>/dev/null)
  if echo "$result" | grep -q "$expect"; then
    echo -e "\033[0;32m✓\033[0m $label"
    ((PASS++))
  else
    echo -e "\033[0;31m✗\033[0m $label → expected '$expect'"
    echo "  Got: $(echo $result | head -c 120)"
    ((FAIL++))
  fi
}

echo "=== L54 Test Suite ==="

check "Health check" "curl -sf $BASE/health" '"ok"'
check "List profiles" "curl -sf $BASE/profiles" '"exec-001"'

check "Create test profile" \
  "curl -sf -X POST $BASE/profiles -H 'Content-Type: application/json' \
   -d '{\"user_id\":\"test-999\",\"display_name\":\"Test User\",\"consent_behavioral\":true}'" \
  '"test-999"'

check "Get profile" "curl -sf $BASE/profiles/test-999" '"display_name"'

check "Update preferences" \
  "curl -sf -X PATCH $BASE/profiles/test-999/preferences -H 'Content-Type: application/json' \
   -d '{\"preferences\":{\"formality\":0.8}}'" \
  '"formality"'

check "Get persona" "curl -sf $BASE/profiles/exec-001/persona" '"persona"'

check "Chat with personalization" \
  "curl -sf -X POST $BASE/chat -H 'Content-Type: application/json' \
   -H 'X-User-Id: exec-001' \
   -d '{\"message\":\"What is Kubernetes?\",\"budget_fraction\":1.0}'" \
  '"persona"'

check "Chat - constrained budget" \
  "curl -sf -X POST $BASE/chat -H 'Content-Type: application/json' \
   -H 'X-User-Id: dev-002' \
   -d '{\"message\":\"What is Redis?\",\"budget_fraction\":0.2}'" \
  '"MINIMAL"'

check "Analytics stats" "curl -sf $BASE/analytics/stats" '"total_profiles"'

check "ADK context export" "curl -sf $BASE/profiles/exec-001/adk-context" '"user_id"'

check "Delete profile" "curl -sf -X DELETE $BASE/profiles/test-999" '"deleted"'

echo ""
echo "Results: $PASS passed, $FAIL failed"
[ $FAIL -eq 0 ] && echo -e "\033[0;32mAll tests passed\033[0m" || echo -e "\033[0;31m$FAIL test(s) failed\033[0m"
