FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN mkdir -p data/traces

ENV TRACE_DB_PATH=data/traces/audit.db
ENV TRACE_JSONL_PATH=data/traces/spans.jsonl
ENV PORT=8042

EXPOSE 8042
CMD ["uvicorn", "backend.api.app:app", "--host", "0.0.0.0", "--port", "8042"]
