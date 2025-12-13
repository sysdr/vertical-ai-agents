#!/bin/bash
set -e

echo "📦 Building L4 Parameter Analysis Platform..."

# Backend setup
cd backend

# Check if venv already exists
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    # Try to create venv with pip
    if ! python3 -m venv venv 2>/dev/null; then
        echo "⚠️  venv module not available, trying venv without pip..."
        # Try creating venv without pip first
        if python3 -m venv venv --without-pip 2>/dev/null; then
            echo "Installing pip in virtual environment..."
            # Bootstrap pip using ensurepip
            venv/bin/python3 -m ensurepip --upgrade 2>/dev/null || {
                # If ensurepip fails, download get-pip.py
                echo "Downloading pip installer..."
                curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
                venv/bin/python3 /tmp/get-pip.py || {
                    echo "❌ Failed to install pip in virtual environment."
                    echo "Please install python3-venv and python3-pip:"
                    echo "  sudo apt install python3-venv python3-pip"
                    exit 1
                }
                rm -f /tmp/get-pip.py
            }
        else
            echo "❌ Failed to create virtual environment."
            echo "Please install python3-venv: sudo apt install python3-venv"
            exit 1
        fi
    fi
fi

source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
deactivate
cd ..

# Frontend setup
cd frontend
npm install
cd ..

echo "✅ Build complete!"
echo "Run ./start.sh to launch the application"
