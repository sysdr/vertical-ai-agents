# VAIA · Lesson 69 — Agent observability and logging

Runnable app: **FastAPI** + **OpenTelemetry** + **SQLite** + **React** (traces, metrics, demo queries).  
**`setup.sh`** lives in the **parent** directory (`../setup.sh` from here); it creates or refreshes this tree.

## First run (after clone)

From the **lesson 69** folder (one level up — same directory as `setup.sh`):

```bash
../setup.sh
```

If your shell is already in `lesson69/`:

```bash
./setup.sh
```

This creates **`.venv/`**, installs Node dependencies, prepares **`data/`** (see `data/.gitkeep`), and runs pytest. SQLite under **`data/`** is not committed; it is created on first use.

## Optional — Gemini API

Not stored in git. Either:

```bash
export GEMINI_API_KEY="your-key"
```

or copy **`.env.example`** to **`.env`** in this directory and set the variable.  
If unset, the agent still returns a **demo** response so traces and the dashboard work.

## Run, stop, test (from this directory)

```bash
./start.sh
./stop.sh
./test.sh
```

- API docs: <http://localhost:8069/docs>  
- Dashboard: <http://localhost:3069>

## Docker (from this directory)

```bash
docker compose up --build
```

Tear down and prune (see `cleanup.sh` header comment):

```bash
./cleanup.sh
```

## What to commit

- This directory’s source and config: `package.json`, `package-lock.json`, backend, frontend, `tests/`, Docker files, **`README.md`**, `requirements.txt`, `cleanup.sh`, **`.gitignore`**, `start.sh`, `stop.sh`, `test.sh`, etc.
- Plus **`../setup.sh`** in the parent lesson folder.  
- Do not commit **`.venv/`**, **`node_modules/`**, **`.pytest_cache/`**, **`__pycache__/`**, **`data/*.db*`** — see **`.gitignore`**.

## Python dependencies

See **`requirements.txt`** here; `setup.sh` rewrites it to match the lesson template when you (re)run setup.
