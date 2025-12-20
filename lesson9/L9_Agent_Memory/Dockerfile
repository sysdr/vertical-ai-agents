FROM python:3.11-slim

WORKDIR /app

# Install backend dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY backend/ ./backend/

# Create data directory
RUN mkdir -p data/users

EXPOSE 8000

CMD ["python", "backend/api.py"]
