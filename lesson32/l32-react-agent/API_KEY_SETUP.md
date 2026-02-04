# API Key Setup

The ReAct agent uses **Google Gemini API**. The previous hardcoded key has expired.

## Get a New API Key

1. Go to [Google AI Studio](https://aistudio.google.com/apikey)
2. Sign in with your Google account
3. Click **Create API Key**
4. Copy the generated key

## Configure the Application

### Option 1: Using .env file (recommended)

```bash
cd /home/systemdrllp5/git/vertical-ai-agents/lesson32/l32-react-agent

# Create .env from template
cp .env.example .env

# Edit .env and replace your_api_key_here with your actual key
nano .env   # or: code .env, vim .env, etc.
```

Set:
```
GEMINI_API_KEY=AIzaSy...your_actual_key...
```

### Option 2: Using setup script

```bash
./scripts/setup_api_key.sh
# Then edit .env with your key
```

### Option 3: Export in shell

```bash
export GEMINI_API_KEY=your_actual_key
./scripts/start.sh
```

## Restart the Application

After adding your key:

```bash
./scripts/stop.sh
./scripts/start.sh
```

For Docker:
```bash
docker-compose down
docker-compose up -d
```

## Verify

- Backend: http://localhost:8000/health
- Dashboard: http://localhost:3000
- Run a demo query (e.g., "What is the current price of Google stock?")
