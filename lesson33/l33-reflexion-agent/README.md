# L33: Implementing Self-Correction (Reflexion)

Self-correcting VAIA agent with iterative reflection and autonomous error recovery.

## Quick Start

### Automated Setup
```bash
source venv/bin/activate
bash scripts/build.sh
bash scripts/start.sh
```

### Manual Setup

**Backend:**
```bash
pip install -r requirements.txt
python -m backend.api
```

**Frontend:**
```bash
cd frontend
npm install
npm start
```

## Features

- ✅ Reflexion loop with LLM-powered critique
- ✅ Reflection memory across attempts
- ✅ Automatic plan refinement
- ✅ Production-ready error recovery
- ✅ Real-time dashboard

## Architecture

```
ReflexionAgent
├── ReActAgent (from L32)
├── ReflectionEngine (LLM critique)
└── ReflectionMemory (attempt history)
```

## Usage

```python
from backend.reflexion_agent import ReflexionAgent
from backend.tools import DEFAULT_TOOLS

agent = ReflexionAgent(tools=DEFAULT_TOOLS, max_reflections=3)
result = agent.run("Find the CEO of Anthropic")

print(f"Success: {result['success']}")
print(f"Attempts: {result['attempts']}")
print(f"Result: {result['result']}")
```

## Testing

```bash
bash scripts/test.sh
```

## Next Lesson

L34 adds iteration limits, token budgeting, and cost controls to the reflexion system.
