# L25: Advanced Prompting for RAG

System prompts that guide LLMs to use retrieved context effectively, avoiding hallucinations and staying grounded.

## Quick Start

```bash
./build.sh   # Install dependencies
./start.sh   # Start services
./test.sh    # Run tests
./stop.sh    # Stop services
```

## Architecture

- **Backend**: FastAPI with LangChain RAG pipelines
- **Frontend**: React dashboard for testing grounding
- **Vector Store**: ChromaDB with sample knowledge base
- **LLM**: Gemini 1.5 Flash

## Features

- Three grounding domains (strict/moderate/creative)
- Adversarial testing with misleading contexts
- Citation extraction and validation
- Grounding confidence scoring
- Real-time grounding state visualization

## API Endpoints

- `POST /query` - Execute RAG query with grounding
- `POST /test-prompt` - Test custom prompt templates
- `GET /stats` - System statistics

## Testing Grounding

The adversarial test mode intentionally provides irrelevant context to verify the LLM refuses to answer when grounding is impossible.

Built on L24's modular RAG pipeline.
Prepares for L26's multi-turn conversations.
