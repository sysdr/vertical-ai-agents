#!/usr/bin/env bash
set -e
# Avoid duplicate services
pkill -f "uvicorn backend.api.app" 2>/dev/null || true
[ -f /tmp/l43_backend.pid ]  && kill $(cat /tmp/l43_backend.pid) 2>/dev/null || true
[ -f /tmp/l43_frontend.pid ] && kill $(cat /tmp/l43_frontend.pid) 2>/dev/null || true
rm -f /tmp/l43_backend.pid /tmp/l43_frontend.pid
sleep 1
ROOT="/home/systemdr5/git/vertical-ai-agents/lesson43/l43-agentic-rag"
[ -f "${ROOT}/.env" ] && set -a && source "${ROOT}/.env" && set +a
source "${ROOT}/.venv/bin/activate"
if [ -z "${GEMINI_API_KEY}" ]; then
  echo "[L43] ERROR: GEMINI_API_KEY is not set. Export a valid key first, e.g.: export GEMINI_API_KEY=your_key"
  echo "       Get a key at: https://aistudio.google.com/apikey"
  exit 1
fi
export PYTHONPATH="${ROOT}"
export XDG_CACHE_HOME="${ROOT}/.cache"
mkdir -p "${ROOT}/.cache"

echo "[L43] Seeding knowledge base..."
python3 "${ROOT}/backend/seed_knowledge_base.py"

echo "[L43] Starting backend on port 8043..."
cd "${ROOT}"
uvicorn backend.api.app:app --host 0.0.0.0 --port 8043 --reload &
echo $! > /tmp/l43_backend.pid

echo "[L43] Starting frontend on port 3043..."
cd "${ROOT}/frontend" && PORT=3043 npm start &
echo $! > /tmp/l43_frontend.pid

echo ""
echo "✓ Backend:  http://localhost:8043"
echo "✓ Frontend: http://localhost:3043"
echo "✓ API Docs: http://localhost:8043/docs"
echo "✓ Health:   http://localhost:8043/health"
