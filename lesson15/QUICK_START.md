# Quick Start Guide

## 🚀 Getting Started

### 1. Build Dependencies
```bash
./build.sh
```

### 2. Start Services
```bash
./start.sh
```

### 3. Access Dashboard
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Health**: http://localhost:8000/health
- **API Docs**: http://localhost:8000/docs

### 4. Stop Services
```bash
./stop.sh
```

## 📝 Usage

### Creating a Conversation
1. Open http://localhost:3000
2. Enter a User ID
3. Click "Create Conversation"

### Setting Goals
Type in the message box:
```
/goal Learn Python programming
```

### Using Step-by-Step Reasoning
Ask questions like:
- "Explain step-by-step how goals work"
- "Tell me about conversational agents with detailed reasoning"
- "Break down how state management works"

## 🔧 Configuration

### Update API Key (Optional)
To enable full AI capabilities:
```bash
./update_api_key.sh YOUR_GEMINI_API_KEY
```

Get API key from: https://aistudio.google.com/app/apikey

**Note**: The system works in intelligent fallback mode even without a valid API key!

## ✅ Features

- ✅ Intelligent fallback mode (works without API key)
- ✅ Step-by-step reasoning responses
- ✅ Goal tracking and management
- ✅ Real-time dashboard metrics
- ✅ Conversation state management
- ✅ Persistent memory across sessions

## 🐛 Troubleshooting

### Services won't start
```bash
# Check if ports are in use
netstat -tuln | grep -E ":(8000|3000)"

# Kill existing processes
./stop.sh
```

### Goals not showing
- Refresh the browser
- Check browser console for errors
- Verify backend is running: `curl http://localhost:8000/health`

### Active Goals showing 0
- Set a goal using `/goal <description>`
- The count should update immediately
- Check the goals panel on the left side

## 📊 Dashboard Metrics

The dashboard shows:
- **State**: Conversation state (initializing, active, goal_seeking, completed)
- **Messages**: Total message count
- **Active Goals**: Number of incomplete goals
- **Tokens**: Estimated token usage

All metrics update in real-time!

