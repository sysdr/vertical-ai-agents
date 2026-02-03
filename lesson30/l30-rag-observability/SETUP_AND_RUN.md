# L30 Setup and Run Guide

## Summary of Fixes Applied

### setup.sh Fixes
- **Line 83**: Fixed `duration_ms: Optional<br/>` → `duration_ms: Optional[float]` (invalid Python)
- **frontend/public**: Added to mkdir - was missing, caused setup failure
- **python-dotenv**: Added to requirements.txt (used by main.py)
- **pytest**: Fixed version conflict (pytest>=7.4.0,<8 for pytest-asyncio compatibility)
- **__init__.py**: Added for backend/models, backend/services, backend/middleware
- **load_dotenv**: Now loads from backend/.env via Path
- **stats.model_dump()**: Pydantic v2 API (was .dict())
- **docker-compose**: Removed obsolete version attribute
- **Project root .env**: Created for docker-compose GEMINI_API_KEY
- **MetricsDashboard**: No-data state shows "—" and banner instead of zeros
- **Scripts**: Project root detection, demo.sh for populating metrics

### Startup Script Fixes
- **start.sh**: cd to project root via SCRIPT_DIR, prefer Docker when available
- **test.sh**: Wait for backend with retry, run from project root
- **demo.sh**: New script to run queries and populate dashboard
- **stop.sh**: Project root, suppress pkill errors

## Run Instructions

### 1. Build (Docker - first time takes ~10 min for torch/sentence-transformers)
```bash
cd /home/systemdrllp5/git/vertical-ai-agents/lesson30/l30-rag-observability
./scripts/build.sh
# Or: docker-compose build
```

### 2. Start Services
```bash
./scripts/start.sh
```
- Uses Docker if available (recommended)
- Backend: http://localhost:8000
- Frontend: http://localhost:3000

### 3. Populate Dashboard Metrics
The dashboard shows "—" until queries are run. Either:
- **Option A**: Use Query Interface in the UI - enter a question and click Submit
- **Option B**: Run demo script:
```bash
./scripts/demo.sh
```

### 4. Run Tests
```bash
./scripts/test.sh
```
(Requires backend running on port 8000)

### 5. Stop Services
```bash
./scripts/stop.sh
```

## Duplicate Services Check
Before starting, check for existing processes:
```bash
lsof -i :8000 -i :3000
# Or: docker ps | grep -E "8000|3000"
```

## Dashboard Validation
- **Before queries**: Shows "No metrics yet" banner, "—" for all values
- **After queries**: Total Requests, Avg Latency, Tokens, Cost, etc. update in real-time
- WebSocket streams metrics every 2 seconds
