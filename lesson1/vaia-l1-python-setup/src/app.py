"""
VAIA L1: FastAPI Application with Dashboard
"""
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Dict
import time
import random
import asyncio
from datetime import datetime

app = FastAPI(title="VAIA L1 Dashboard", version="1.0.0")

# Metrics storage
metrics = {
    "requests_total": 0,
    "requests_per_second": 0.0,
    "demo_executions": 0,
    "successful_operations": 0,
    "average_response_time": 0.0,
    "uptime_seconds": 0,
    "last_update": datetime.now().isoformat()
}

start_time = time.time()

class DemoRequest(BaseModel):
    action: str = "run"

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Dashboard HTML page"""
    return """
<!DOCTYPE html>
<html>
<head>
    <title>VAIA L1 Dashboard</title>
    <meta http-equiv="refresh" content="2">
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        h1 {
            color: white;
            text-align: center;
            margin-bottom: 30px;
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .metric-card {
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            transition: transform 0.2s;
        }
        .metric-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        }
        .metric-label {
            font-size: 14px;
            color: #666;
            margin-bottom: 10px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .metric-value {
            font-size: 32px;
            font-weight: bold;
            color: #667eea;
        }
        .metric-unit {
            font-size: 14px;
            color: #999;
            margin-left: 5px;
        }
        .status {
            text-align: center;
            color: white;
            padding: 10px;
            border-radius: 5px;
            margin-top: 20px;
        }
        .status.active {
            background: #10b981;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 VAIA L1 Dashboard</h1>
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">Total Requests</div>
                <div class="metric-value" id="requests_total">0</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Requests/Second</div>
                <div class="metric-value" id="requests_per_second">0.0</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Demo Executions</div>
                <div class="metric-value" id="demo_executions">0</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Successful Operations</div>
                <div class="metric-value" id="successful_operations">0</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Avg Response Time</div>
                <div class="metric-value" id="average_response_time">0.0<span class="metric-unit">ms</span></div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Uptime</div>
                <div class="metric-value" id="uptime_seconds">0<span class="metric-unit">s</span></div>
            </div>
        </div>
        <div class="status active">
            ✅ System Operational - Last Update: <span id="last_update"></span>
        </div>
    </div>
    <script>
        async function updateMetrics() {
            try {
                const response = await fetch('/api/metrics');
                const data = await response.json();
                document.getElementById('requests_total').textContent = data.requests_total;
                document.getElementById('requests_per_second').textContent = data.requests_per_second.toFixed(2);
                document.getElementById('demo_executions').textContent = data.demo_executions;
                document.getElementById('successful_operations').textContent = data.successful_operations;
                document.getElementById('average_response_time').textContent = data.average_response_time.toFixed(2);
                document.getElementById('uptime_seconds').textContent = Math.floor(data.uptime_seconds);
                document.getElementById('last_update').textContent = new Date(data.last_update).toLocaleTimeString();
            } catch (error) {
                console.error('Error updating metrics:', error);
            }
        }
        updateMetrics();
        setInterval(updateMetrics, 1000);
    </script>
</body>
</html>
"""

@app.get("/api/metrics")
async def get_metrics():
    """Get current metrics"""
    current_time = time.time()
    metrics["uptime_seconds"] = current_time - start_time
    if metrics["uptime_seconds"] > 0:
        metrics["requests_per_second"] = metrics["requests_total"] / metrics["uptime_seconds"]
    metrics["last_update"] = datetime.now().isoformat()
    return metrics

@app.post("/api/demo")
async def run_demo(demo_request: DemoRequest):
    """Run demo operation"""
    start = time.time()
    
    # Simulate demo operation
    await asyncio.sleep(0.1)
    
    # Update metrics
    metrics["demo_executions"] += 1
    metrics["requests_total"] += 1
    metrics["successful_operations"] += 1
    
    # Calculate response time
    response_time = (time.time() - start) * 1000
    if metrics["requests_total"] > 1:
        metrics["average_response_time"] = (
            (metrics["average_response_time"] * (metrics["requests_total"] - 1) + response_time) 
            / metrics["requests_total"]
        )
    else:
        metrics["average_response_time"] = response_time
    
    return {
        "status": "success",
        "message": f"Demo {demo_request.action} executed successfully",
        "metrics": metrics
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "metrics": metrics}

if __name__ == "__main__":
    import uvicorn
    import asyncio
    uvicorn.run(app, host="0.0.0.0", port=8000)
