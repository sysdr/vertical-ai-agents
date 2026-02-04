# API Documentation

## Endpoints

### Execute Agent Query
```
POST /agent/query
```

Request:
```json
{
  "query": "What is the current price of Google stock?",
  "max_iterations": 10,
  "session_id": "optional-session-id"
}
```

Response:
```json
{
  "success": true,
  "result": "Current stock price for GOOGL: $175.32",
  "reasoning_trace": [
    {
      "step_number": 1,
      "thought": "I need to look up Google's stock price",
      "action": "StockPrice",
      "action_input": {"symbol": "GOOGL"},
      "observation": "Current stock price for GOOGL: $175.32",
      "timestamp": "2025-02-03T10:30:00"
    }
  ],
  "iterations_used": 2,
  "session_id": "session_20250203_103000_001"
}
```

### List Tools
```
GET /agent/tools
```

Response:
```json
[
  {
    "name": "Wikipedia",
    "description": "Search Wikipedia and retrieve article summaries",
    "parameters": {
      "query": {
        "type": "string",
        "description": "Search term or article title"
      }
    }
  }
]
```

### Get Session History
```
GET /agent/history/{session_id}
```

Response:
```json
{
  "query": "Original query",
  "result": {...},
  "timestamp": "2025-02-03T10:30:00"
}
```
