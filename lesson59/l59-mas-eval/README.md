# Lesson 59 - MAS Evaluation & Debugging

This directory contains the Lesson 59 generated project and helper scripts.

## Quick start

```bash
cd lesson59/l59-mas-eval
bash scripts/start.sh
```

Dashboard:

- http://localhost:3059

API docs:

- http://localhost:8059/docs

## Run tests

```bash
cd lesson59/l59-mas-eval
bash scripts/test.sh
```

## Environment variables

Set Gemini API key if you want live model calls:

```bash
export GEMINI_API_KEY="your_api_key_here"
```

If no key is set, the app uses deterministic fallback responses for demo flow.

## Cleanup

Stop services and clean Docker resources:

```bash
cd lesson59/l59-mas-eval
./cleanup.sh
```
