# VAIA L1: Python Setup for AI

Enterprise-grade Python development environment for building Vertical AI Agents.

## Quick Start
```bash
# Activate environment
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Verify setup
python src/validate_environment.py

# Run application
python src/main.py

# Run tests
./scripts/test.sh
```

## Project Structure
```
vaia-l1-python-setup/
├── venv/                 # Virtual environment
├── src/                  # Source code
│   ├── agents/          # AI agents (future lessons)
│   ├── services/        # API services (future lessons)
│   ├── utils/           # Utilities
│   ├── main.py          # Main entry point
│   └── validate_environment.py
├── tests/               # Test suite
├── scripts/             # Automation scripts
├── config/              # Configuration files
├── requirements.txt     # Core dependencies
└── requirements-dev.txt # Development dependencies
```

## Dependencies

- Python 3.12+
- FastAPI 0.109.0
- Pydantic 2.5.3
- NumPy 1.26.3
- Pandas 2.1.4
- Gemini AI SDK 0.3.2

## Next Steps

Proceed to **L2: FastAPI Fundamentals** to build your first AI service.

## Automation Scripts

- `scripts/build.sh` - Build project (future lessons)
- `scripts/start.sh` - Start application
- `scripts/stop.sh` - Stop and cleanup
- `scripts/test.sh` - Run test suite
