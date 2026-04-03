"""
L49 FastAPI Backend
Endpoints: /run-crew (POST), /health (GET), /stream-crew (GET SSE)
"""
import asyncio
import json
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

load_dotenv()

from crew import run_crew, build_crew

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("L49 CrewAI server starting — Gemini 2.0 Flash backend")
    yield
    print("L49 server shutdown")

app = FastAPI(
    title="L49 CrewAI Role-Based Execution",
    description="VAIA Module 5 — Multi-Agent Architectures",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class CrewRequest(BaseModel):
    topic: str


class CrewResponse(BaseModel):
    topic: str
    final_output: str
    task_outputs: list
    agent_count: int
    task_count: int
    status: str = "complete"


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "lesson": "L49",
        "framework": "CrewAI",
        "llm": "gemini-2.0-flash",
        "process": "sequential",
    }


@app.post("/run-crew", response_model=CrewResponse)
async def execute_crew(request: CrewRequest):
    """Run a 3-agent sequential crew on the given topic."""
    if not request.topic.strip():
        raise HTTPException(status_code=400, detail="Topic cannot be empty")
    try:
        result = await asyncio.to_thread(run_crew, request.topic.strip())
        return CrewResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def stream_crew_events(topic: str) -> AsyncGenerator[str, None]:
    """SSE generator — emits progress events as crew executes."""
    agents_meta = [
        {"step": 1, "role": "Senior Technical Researcher", "icon": "🔬"},
        {"step": 2, "role": "Principal Solution Architect", "icon": "🏗️"},
        {"step": 3, "role": "Technical Content Strategist", "icon": "✍️"},
    ]

    yield f"data: {json.dumps({'type': 'start', 'topic': topic})}\n\n"
    await asyncio.sleep(0.2)

    result_holder: dict = {}

    async def run_in_thread():
        result_holder["result"] = await asyncio.to_thread(run_crew, topic)

    task = asyncio.create_task(run_in_thread())

    # Emit agent start events with approximate timing
    intervals = [2, 14, 26]
    for meta, delay in zip(agents_meta, intervals):
        await asyncio.sleep(delay)
        yield f"data: {json.dumps({'type': 'agent_start', **meta})}\n\n"

    await task

    result = result_holder["result"]

    # Emit per-task results
    for task_out in result["task_outputs"]:
        yield f"data: {json.dumps({'type': 'task_complete', **task_out})}\n\n"
        await asyncio.sleep(0.1)

    yield f"data: {json.dumps({'type': 'crew_complete', 'final_output': result['final_output']})}\n\n"


@app.get("/stream-crew")
async def stream_crew(topic: str):
    if not topic.strip():
        raise HTTPException(status_code=400, detail="Topic required")
    return StreamingResponse(
        stream_crew_events(topic.strip()),
        media_type="text/event-stream",
    )


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("BACKEND_PORT", 8049))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
