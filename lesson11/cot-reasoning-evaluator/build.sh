#!/bin/bash
set -e

echo "🔨 Building L11 CoT Reasoning Evaluator..."

if command -v docker &> /dev/null && docker ps &> /dev/null; then
    echo "Building with Docker..."
    if command -v docker-compose &> /dev/null; then
        docker-compose build
    elif docker compose version &> /dev/null; then
        docker compose build
    else
        echo "Docker Compose not available, building without Docker..."
        BUILD_WITHOUT_DOCKER=true
    fi
else
    BUILD_WITHOUT_DOCKER=true
fi

if [ "$BUILD_WITHOUT_DOCKER" = "true" ]; then
    echo "Building without Docker..."
    
    # Get absolute path
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    
    # Backend
    cd "$SCRIPT_DIR/backend"
    if python3 -m venv venv 2>/dev/null; then
        source venv/bin/activate
        pip install -r requirements.txt
    else
        echo "Warning: venv not available, installing packages with --break-system-packages flag"
        # Check if pip is available
        if python3 -m pip --version &>/dev/null; then
            echo "Installing backend dependencies..."
            python3 -m pip install --break-system-packages -r requirements.txt
        elif [ -f ~/.local/bin/pip ]; then
            echo "Installing backend dependencies..."
            ~/.local/bin/pip install --break-system-packages -r requirements.txt
        else
            echo "Installing pip with --break-system-packages (required for this environment)..."
            curl -sS https://bootstrap.pypa.io/get-pip.py | python3 - --break-system-packages 2>/dev/null
            if python3 -m pip --version &>/dev/null; then
                echo "Installing backend dependencies..."
                python3 -m pip install --break-system-packages -r requirements.txt
            else
                echo "ERROR: Cannot install Python packages. Please install python3-venv: sudo apt install python3.12-venv"
                exit 1
            fi
        fi
    fi
    cd ..
    
    # Frontend
    cd "$SCRIPT_DIR/frontend"
    npm install
    cd ..
fi

echo "✅ Build complete!"
