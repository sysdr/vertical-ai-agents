FROM python:3.11-slim

WORKDIR /app

# Install Node.js
RUN apt-get update && apt-get install -y nodejs npm curl

# Copy backend files
COPY requirements.txt .
COPY backend/ backend/
COPY config/ config/
COPY .env .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy frontend files
COPY frontend/package.json frontend/
WORKDIR /app/frontend
RUN npm install

COPY frontend/ .
RUN npm run build

# Return to app directory
WORKDIR /app

# Expose ports
EXPOSE 8000 3000

# Start script
COPY docker-start.sh .
RUN chmod +x docker-start.sh

CMD ["./docker-start.sh"]
