FROM python:3.11-slim

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/
RUN mkdir -p data

WORKDIR /app/backend
EXPOSE 8069
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8069"]
