#!/bin/bash

################################################################################
# VAIA Curriculum - L1: Python Setup for AI
# Automated Environment Setup Script
# 
# This script creates a production-grade Python development environment for
# enterprise AI agent development. It establishes patterns used throughout
# the 90-lesson VAIA curriculum.
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Print banner
echo "=================================================="
echo "  VAIA L1: Python Setup for AI"
echo "  Enterprise Development Environment"
echo "=================================================="
echo ""

# Check if Python 3.12+ is available
log_info "Checking Python version..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)
    if [ "$PYTHON_MAJOR" -ge 3 ] && [ "$PYTHON_MINOR" -ge 12 ]; then
        PYTHON_CMD=python3
    else
        log_error "Python 3.12+ required. Found: $PYTHON_VERSION"
        exit 1
    fi
elif command -v python3.12 &> /dev/null; then
    PYTHON_CMD=python3.12
else
    log_error "Python 3 not found. Please install Python 3.12+"
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD --version)
log_success "Found: $PYTHON_VERSION"

# Create project directory structure
log_info "Creating project structure..."
PROJECT_DIR="vaia-l1-python-setup"
mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"

mkdir -p src/{agents,services,utils}
mkdir -p tests
mkdir -p scripts
mkdir -p config
mkdir -p logs

log_success "Project structure created"

# Create requirements.txt with pinned versions (May 2025 compatible)
log_info "Creating requirements.txt..."
cat > requirements.txt << 'EOF'
# Core Web Framework
fastapi==0.109.0
uvicorn[standard]==0.27.0
pydantic==2.5.3
pydantic-settings==2.1.0

# AI/ML Libraries
google-generativeai==0.3.2
numpy==1.26.3
pandas==2.1.4

# HTTP Client
httpx==0.26.0
aiohttp==3.9.1

# Utilities
python-dotenv==1.0.0
python-multipart==0.0.6
python-jose[cryptography]==3.3.0

# Logging
loguru==0.7.2
EOF

log_success "requirements.txt created"

# Create requirements-dev.txt for development tools
log_info "Creating requirements-dev.txt..."
cat > requirements-dev.txt << 'EOF'
# Testing
pytest==7.4.3
pytest-asyncio==0.21.1
pytest-cov==4.1.0

# Linting and Formatting
black==23.12.1
ruff==0.1.9
mypy==1.8.0

# Development
ipython==8.19.0
ipdb==0.13.13
EOF

log_success "requirements-dev.txt created"

# Determine activation command based on OS
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    ACTIVATE_CMD="venv\\Scripts\\activate"
    PYTHON_VENV="venv\\Scripts\\python"
    PIP_VENV="venv\\Scripts\\pip"
else
    ACTIVATE_CMD="venv/bin/activate"
    PYTHON_VENV="venv/bin/python"
    PIP_VENV="venv/bin/pip"
fi

# Create virtual environment
log_info "Creating virtual environment..."
if ! $PYTHON_CMD -m venv venv 2>/dev/null; then
    log_warning "venv creation failed, trying with --without-pip..."
    $PYTHON_CMD -m venv --without-pip venv || {
        log_error "Failed to create virtual environment"
        exit 1
    }
    log_info "Installing pip manually..."
    if command -v curl &> /dev/null; then
        curl -sS https://bootstrap.pypa.io/get-pip.py | $PYTHON_VENV - || {
            log_error "Failed to install pip"
            exit 1
        }
    else
        log_error "curl not found. Please install python3-venv package: sudo apt install python3-venv"
        exit 1
    fi
fi
log_success "Virtual environment created"

# Upgrade pip in virtual environment
log_info "Upgrading pip..."
$PYTHON_VENV -m pip install --upgrade pip setuptools wheel > /dev/null 2>&1
log_success "Pip upgraded"

# Install dependencies
log_info "Installing dependencies (this may take a minute)..."
$PIP_VENV install -r requirements.txt > /dev/null 2>&1
log_success "Core dependencies installed"

log_info "Installing development dependencies..."
$PIP_VENV install -r requirements-dev.txt > /dev/null 2>&1
log_success "Development dependencies installed"

# Create .env file template
log_info "Creating environment configuration..."
cat > .env.example << 'EOF'
# Gemini AI Configuration
GEMINI_API_KEY=AIzaSyDGswqDT4wQw_bd4WZtIgYAawRDZ0Gisn8
GEMINI_MODEL=gemini-pro

# Application Configuration
APP_ENV=development
LOG_LEVEL=INFO
DEBUG=True

# Server Configuration
HOST=0.0.0.0
PORT=8000
EOF

cp .env.example .env
log_success "Environment configuration created"

# Create sample source files
log_info "Creating sample source files..."

# Create main application stub
cat > src/main.py << 'EOF'
"""
VAIA L1: Main Application Entry Point
This is a placeholder that will be expanded in L2 (FastAPI Fundamentals)
"""

def main():
    print("🚀 VAIA Python Environment Ready!")
    print("📚 Lesson 1: Python Setup Complete")
    print("➡️  Next: Lesson 2 - FastAPI Fundamentals")
    
    # Verify key imports work
    try:
        import fastapi
        import pydantic
        import numpy as np
        import pandas as pd
        import google.generativeai as genai
        
        print("\n✅ All core dependencies verified:")
        print(f"   - FastAPI: {fastapi.__version__}")
        print(f"   - Pydantic: {pydantic.__version__}")
        print(f"   - NumPy: {np.__version__}")
        print(f"   - Pandas: {pd.__version__}")
        print(f"   - Gemini AI: SDK loaded")
        
        return True
    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        return False

if __name__ == "__main__":
    main()
EOF

# Create validation script
cat > src/validate_environment.py << 'EOF'
"""
Environment Validation Script
Verifies that all dependencies are correctly installed
"""
import sys
import importlib
from typing import List, Tuple

REQUIRED_PACKAGES = [
    ('fastapi', 'FastAPI'),
    ('pydantic', 'Pydantic'),
    ('numpy', 'NumPy'),
    ('pandas', 'Pandas'),
    ('google.generativeai', 'Gemini AI'),
    ('uvicorn', 'Uvicorn'),
    ('httpx', 'HTTPX'),
    ('dotenv', 'Python-dotenv'),
]

def validate_python_version() -> bool:
    """Check Python version is 3.12+"""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 12:
        print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"❌ Python version: {version.major}.{version.minor}.{version.micro} (requires 3.12+)")
        return False

def validate_packages() -> Tuple[List[str], List[str]]:
    """Validate all required packages can be imported"""
    success = []
    failed = []
    
    for package, display_name in REQUIRED_PACKAGES:
        try:
            mod = importlib.import_module(package)
            version = getattr(mod, '__version__', 'unknown')
            print(f"✅ {display_name}: {version}")
            success.append(display_name)
        except ImportError:
            print(f"❌ {display_name}: Not installed")
            failed.append(display_name)
    
    return success, failed

def main():
    print("=" * 50)
    print("  VAIA Environment Validation")
    print("=" * 50)
    print()
    
    # Check Python version
    python_ok = validate_python_version()
    print()
    
    # Check packages
    print("Checking required packages...")
    success, failed = validate_packages()
    print()
    
    # Summary
    if python_ok and not failed:
        print("🎉 Environment validation successful!")
        print(f"   {len(success)} packages verified")
        return 0
    else:
        print("⚠️  Environment validation failed!")
        if failed:
            print(f"   Missing packages: {', '.join(failed)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
EOF

# Create FastAPI application with dashboard
log_info "Creating FastAPI application with dashboard..."
cat > src/app.py << 'EOF'
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
EOF

# Create demo script
cat > src/demo.py << 'EOF'
"""
VAIA L1: Demo Script
Runs demo operations to update dashboard metrics
"""
import asyncio
import httpx
import time
import sys

async def run_demo_operations(base_url: str = "http://localhost:8000", count: int = 10):
    """Run multiple demo operations"""
    print(f"🚀 Running {count} demo operations...")
    
    async with httpx.AsyncClient() as client:
        for i in range(count):
            try:
                response = await client.post(f"{base_url}/api/demo", json={"action": "run"})
                if response.status_code == 200:
                    data = response.json()
                    print(f"✅ Demo {i+1}/{count}: {data['message']}")
                else:
                    print(f"❌ Demo {i+1}/{count}: Failed with status {response.status_code}")
            except Exception as e:
                print(f"❌ Demo {i+1}/{count}: Error - {e}")
            
            # Small delay between operations
            await asyncio.sleep(0.5)
    
    # Get final metrics
    try:
        response = await client.get(f"{base_url}/api/metrics")
        if response.status_code == 200:
            metrics = response.json()
            print("\n📊 Final Metrics:")
            print(f"   Total Requests: {metrics['requests_total']}")
            print(f"   Demo Executions: {metrics['demo_executions']}")
            print(f"   Successful Operations: {metrics['successful_operations']}")
            print(f"   Requests/Second: {metrics['requests_per_second']:.2f}")
            print(f"   Avg Response Time: {metrics['average_response_time']:.2f}ms")
    except Exception as e:
        print(f"❌ Error fetching metrics: {e}")

if __name__ == "__main__":
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    asyncio.run(run_demo_operations(count=count))
EOF

log_success "FastAPI application and demo created"

log_success "Sample source files created"

# Create automation scripts
log_info "Creating automation scripts..."

# build.sh - No build step needed for L1, but create for consistency
cat > scripts/build.sh << 'EOF'
#!/bin/bash
# L1: No build step required
# Future lessons will use this for compiling assets, etc.
echo "✅ Build complete (no build step for L1)"
EOF
chmod +x scripts/build.sh

# start.sh - Start FastAPI server
cat > scripts/start.sh << 'EOF'
#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "🚀 Starting VAIA L1 Environment..."
echo "Project directory: $PROJECT_DIR"
echo ""

cd "$PROJECT_DIR"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run setup.sh first."
    exit 1
fi

# Activate virtual environment
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
    PYTHON_CMD="venv/Scripts/python"
else
    source venv/bin/activate
    PYTHON_CMD="venv/bin/python"
fi

# Check if Python executable exists
if [ ! -f "$PYTHON_CMD" ]; then
    echo "❌ Python executable not found at $PYTHON_CMD"
    exit 1
fi

# Run validation
echo "Running environment validation..."
$PYTHON_CMD src/validate_environment.py

# Check if port is already in use
if command -v lsof &> /dev/null; then
    if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null ; then
        echo "⚠️  Port 8000 is already in use. Attempting to use existing service..."
    fi
elif command -v netstat &> /dev/null; then
    if netstat -tuln | grep -q ':8000 '; then
        echo "⚠️  Port 8000 is already in use. Attempting to use existing service..."
    fi
fi

# Start FastAPI server
echo ""
echo "🌐 Starting FastAPI server on http://0.0.0.0:8000"
echo "📊 Dashboard: http://localhost:8000"
echo "🔍 Health: http://localhost:8000/health"
echo "📈 Metrics API: http://localhost:8000/api/metrics"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

$PYTHON_CMD -m uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload
EOF
chmod +x scripts/start.sh

# stop.sh - Stop FastAPI server
cat > scripts/stop.sh << 'EOF'
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "🛑 Stopping VAIA L1 Environment..."

cd "$PROJECT_DIR"

# Find and kill processes on port 8000
if command -v lsof &> /dev/null; then
    PID=$(lsof -ti:8000)
    if [ ! -z "$PID" ]; then
        echo "Stopping process on port 8000 (PID: $PID)..."
        kill $PID 2>/dev/null || true
        sleep 1
        # Force kill if still running
        kill -9 $PID 2>/dev/null || true
        echo "✅ Server stopped"
    else
        echo "ℹ️  No process found on port 8000"
    fi
elif command -v fuser &> /dev/null; then
    fuser -k 8000/tcp 2>/dev/null || echo "ℹ️  No process found on port 8000"
else
    echo "⚠️  Could not find lsof or fuser. Please stop the server manually."
fi

deactivate 2>/dev/null || true
echo "✅ Environment stopped"
EOF
chmod +x scripts/stop.sh

# test.sh - Run tests
cat > scripts/test.sh << 'EOF'
#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "🧪 Running tests..."
echo ""

cd "$PROJECT_DIR"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run setup.sh first."
    exit 1
fi

# Activate virtual environment
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
    PYTHON_CMD="venv/Scripts/python"
else
    source venv/bin/activate
    PYTHON_CMD="venv/bin/python"
fi

# Check if Python executable exists
if [ ! -f "$PYTHON_CMD" ]; then
    echo "❌ Python executable not found at $PYTHON_CMD"
    exit 1
fi

# Run pytest
echo "Running pytest..."
$PYTHON_CMD -m pytest tests/ -v --tb=short
EOF
chmod +x scripts/test.sh

# Create demo script
cat > scripts/demo.sh << 'EOF'
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_DIR"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run setup.sh first."
    exit 1
fi

# Activate virtual environment
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
    PYTHON_CMD="venv/Scripts/python"
else
    source venv/bin/activate
    PYTHON_CMD="venv/bin/python"
fi

# Check if Python executable exists
if [ ! -f "$PYTHON_CMD" ]; then
    echo "❌ Python executable not found at $PYTHON_CMD"
    exit 1
fi

# Run demo
COUNT=${1:-10}
echo "🎬 Running demo with $COUNT operations..."
$PYTHON_CMD src/demo.py $COUNT
EOF
chmod +x scripts/demo.sh

log_success "Automation scripts created"

# Create sample test
mkdir -p tests
cat > tests/test_environment.py << 'EOF'
"""
Basic environment tests
"""
import pytest
import sys

def test_python_version():
    """Verify Python 3.12+"""
    assert sys.version_info >= (3, 12), "Python 3.12+ required"

def test_imports():
    """Verify core packages import successfully"""
    import fastapi
    import pydantic
    import numpy
    import pandas
    import google.generativeai
    
    # If we get here, all imports succeeded
    assert True

def test_environment_setup():
    """Verify basic environment is functional"""
    import os
    assert os.path.exists('venv'), "Virtual environment should exist"
    assert os.path.exists('requirements.txt'), "Requirements file should exist"
    assert os.path.exists('.env'), "Environment file should exist"
EOF

# Create application tests
cat > tests/test_app.py << 'EOF'
"""
Application and dashboard tests
"""
import pytest
from fastapi.testclient import TestClient
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from app import app

client = TestClient(app)

def test_root_endpoint():
    """Test dashboard HTML is served"""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "VAIA L1 Dashboard" in response.text

def test_metrics_endpoint():
    """Test metrics API endpoint"""
    response = client.get("/api/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "requests_total" in data
    assert "demo_executions" in data
    assert "successful_operations" in data
    assert "uptime_seconds" in data
    assert isinstance(data["requests_total"], int)
    assert isinstance(data["demo_executions"], int)

def test_demo_endpoint():
    """Test demo endpoint updates metrics"""
    # Get initial metrics
    initial_response = client.get("/api/metrics")
    initial_data = initial_response.json()
    initial_demo_count = initial_data["demo_executions"]
    
    # Run demo
    demo_response = client.post("/api/demo", json={"action": "run"})
    assert demo_response.status_code == 200
    demo_data = demo_response.json()
    assert demo_data["status"] == "success"
    
    # Verify metrics updated
    updated_response = client.get("/api/metrics")
    updated_data = updated_response.json()
    assert updated_data["demo_executions"] == initial_demo_count + 1
    assert updated_data["requests_total"] > initial_data["requests_total"]

def test_health_endpoint():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "metrics" in data

def test_metrics_not_zero_after_demo():
    """Test that metrics are not zero after running demo"""
    # Run a few demos
    for _ in range(5):
        client.post("/api/demo", json={"action": "run"})
    
    response = client.get("/api/metrics")
    data = response.json()
    assert data["demo_executions"] > 0, "Demo executions should not be zero"
    assert data["requests_total"] > 0, "Total requests should not be zero"
    assert data["successful_operations"] > 0, "Successful operations should not be zero"
EOF

# Create pytest configuration
cat > pytest.ini << 'EOF'
[pytest]
testpaths = tests
python_files = test_*.py
python_functions = test_*
addopts = -v --tb=short
EOF

log_success "Test infrastructure created"

# Create README
cat > README.md << 'EOF'
# VAIA L1: Python Setup for AI

Enterprise-grade Python development environment for building Vertical AI Agents.

## Quick Start
```bash
# Activate environment
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Verify setup
python src/validate_environment.py

# Run application
python src/main.py

# Run tests
./scripts/test.sh
```

## Project Structure
```
vaia-l1-python-setup/
├── venv/                 # Virtual environment
├── src/                  # Source code
│   ├── agents/          # AI agents (future lessons)
│   ├── services/        # API services (future lessons)
│   ├── utils/           # Utilities
│   ├── main.py          # Main entry point
│   └── validate_environment.py
├── tests/               # Test suite
├── scripts/             # Automation scripts
├── config/              # Configuration files
├── requirements.txt     # Core dependencies
└── requirements-dev.txt # Development dependencies
```

## Dependencies

- Python 3.12+
- FastAPI 0.109.0
- Pydantic 2.5.3
- NumPy 1.26.3
- Pandas 2.1.4
- Gemini AI SDK 0.3.2

## Next Steps

Proceed to **L2: FastAPI Fundamentals** to build your first AI service.

## Automation Scripts

- `scripts/build.sh` - Build project (future lessons)
- `scripts/start.sh` - Start application
- `scripts/stop.sh` - Stop and cleanup
- `scripts/test.sh` - Run test suite
EOF

log_success "README created"

# Create .gitignore
cat > .gitignore << 'EOF'
# Virtual Environment
venv/
env/
ENV/

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Environment
.env
.env.local

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# Logs
logs/
*.log

# OS
.DS_Store
Thumbs.db

# Testing
.pytest_cache/
.coverage
htmlcov/

# MyPy
.mypy_cache/
EOF

log_success ".gitignore created"

# Run initial validation
log_info "Running environment validation..."
$PYTHON_VENV src/validate_environment.py

# Verify all required files were created
log_info "Verifying all files were created..."
REQUIRED_FILES=(
    "requirements.txt"
    "requirements-dev.txt"
    ".env"
    ".env.example"
    "src/main.py"
    "src/app.py"
    "src/demo.py"
    "src/validate_environment.py"
    "scripts/build.sh"
    "scripts/start.sh"
    "scripts/stop.sh"
    "scripts/test.sh"
    "scripts/demo.sh"
    "tests/test_environment.py"
    "tests/test_app.py"
    "pytest.ini"
    "README.md"
    ".gitignore"
)

MISSING_FILES=()
for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        MISSING_FILES+=("$file")
    fi
done

if [ ${#MISSING_FILES[@]} -gt 0 ]; then
    log_error "Missing files: ${MISSING_FILES[*]}"
    exit 1
else
    log_success "All required files created"
fi

# Print success message and next steps
echo ""
echo "=================================================="
log_success "L1 Setup Complete! 🎉"
echo "=================================================="
echo ""
echo "📁 Project created: $PROJECT_DIR"
echo ""
echo "Next steps:"
echo "  1. cd $PROJECT_DIR"
echo "  2. source venv/bin/activate  (Linux/Mac)"
echo "     OR venv\\Scripts\\activate  (Windows)"
echo "  3. python src/main.py"
echo "  4. ./scripts/test.sh"
echo ""
echo "📚 Ready for L2: FastAPI Fundamentals"
echo "=================================================="