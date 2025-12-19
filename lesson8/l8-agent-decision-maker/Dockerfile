FROM python:3.11-slim

WORKDIR /app

# Backend
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/
COPY tests/ ./tests/

EXPOSE 8000

CMD ["python", "backend/main.py"]
