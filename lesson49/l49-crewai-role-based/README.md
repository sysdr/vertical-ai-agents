# Lesson49 - CrewAI Role-Based Execution

This app is a 3-agent CrewAI pipeline:

- Researcher -> Analyst -> Writer
- FastAPI backend
- React dashboard frontend

## Files in this directory

- `build.sh`, `start.sh`, `stop.sh`, `test.sh` — run the stack
- `cleanup.sh` — stops services, removes local caches/artifacts, prunes unused Docker resources
- `.gitignore` — Python/Node/cache/log/env ignores
- `requirements.txt` — prerequisite tooling notes

The project **generator** lives one level up: `../setup.sh` (stays in the `lesson49` folder).

## Quick Start

From the parent `lesson49` directory (first-time or regenerate):

```bash
cd lesson49
bash setup.sh
```

From **this** directory:

```bash
cd l49-crewai-role-based
./build.sh
./start.sh
./test.sh
```

## Cleanup

From this directory:

```bash
./cleanup.sh
```

This stops compose for this project, stops any running Docker containers, removes local cache artifacts under this tree, and prunes unused Docker resources.
