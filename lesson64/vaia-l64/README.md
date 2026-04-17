# VAIA L64 Standalone App

This folder is standalone and does not depend on `setup.sh` at runtime.

## Run (Docker)

1. Create `.env` from `.env.example` and set `GEMINI_API_KEY`.
2. Start services:
   - `./start.sh`
3. Open dashboard:
   - `http://localhost:3000` (run `npm run dev` inside `frontend`)

## Build Image

- `./build.sh`

## Stop Services

- `./stop.sh`

## Cleanup

- `./cleanup.sh`

## Minimal files required for dashboard flow

- `docker-compose.yml`
- `src/`
- `frontend/`
- `requirements.txt`
- `start.sh`, `stop.sh`, `build.sh`
- `.env` (local only; do not commit)
