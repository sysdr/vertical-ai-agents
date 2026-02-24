#!/bin/bash
echo "🧪 L44 Test Suite"
echo "=================="
BASE="http://localhost:8044"

check() {
  local name=$1 cmd=$2 expect=$3
  result=$(eval "$cmd" 2>/dev/null)
  if echo "$result" | grep -qE "$expect"; then
    echo "  ✅ $name"
  else
    echo "  ❌ $name — got: $(echo $result | head -c 80)"
  fi
}

echo ""
echo "1. Health checks"
check "Backend health" "curl -sf $BASE/health" '"ok"'

echo ""
echo "2. API endpoints"
check "Thresholds endpoint" "curl -sf $BASE/api/metrics/thresholds" 'faithfulness'
check "Evaluations list" "curl -sf $BASE/api/evaluations" 'run_name|"id"|\[\]'

echo ""
echo "3. Start evaluation run"
RES=$(curl -sf -X POST "$BASE/api/evaluate" \
  -H "Content-Type: application/json" \
  -d '{"run_name":"test-run","num_test_questions":3,"use_synthetic_dataset":true}' 2>/dev/null)
if echo "$RES" | grep -q "report_id"; then
  echo "  ✅ Evaluation started"
  REPORT_ID=$(echo $RES | python3 -c "import sys,json; print(json.load(sys.stdin)['report_id'])" 2>/dev/null)
  sleep 3
  check "Report polling" "curl -sf $BASE/api/evaluations/$REPORT_ID" '"state"'
else
  echo "  ❌ Evaluation start failed: $RES"
fi

echo ""
echo "4. Python imports"
cd backend && source ../venv/bin/activate 2>/dev/null
python3 -c "from app.models import EvaluationReport, MetricScores; print('  ✅ Models import OK')" 2>/dev/null || echo "  ❌ Models import failed"
python3 -c "from app.metrics import GeminiJudge; print('  ✅ Metrics import OK')" 2>/dev/null || echo "  ❌ Metrics import failed"
python3 -c "from app.evaluator import EvaluationOrchestrator; print('  ✅ Evaluator import OK')" 2>/dev/null || echo "  ❌ Evaluator import failed"
cd ..

echo ""
echo "✅ Test suite complete"
