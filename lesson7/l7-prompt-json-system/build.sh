#!/bin/bash
set -e

echo "🏗️  Building L7 Prompt Engineering System..."

# Backend setup
echo "Setting up Python backend..."
cd backend
if [ ! -d "venv" ]; then
    python3 -m venv venv --without-pip || python3 -m venv venv
    source venv/bin/activate
    # Install pip if not available
    if ! command -v pip &> /dev/null; then
        curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
        python3 get-pip.py
        rm get-pip.py
    fi
    pip install --upgrade pip
else
    source venv/bin/activate
fi
pip install -r requirements.txt
deactivate
cd ..

# Frontend setup
echo "Setting up React frontend..."
cd frontend
npm install
cd ..

echo "✅ Build complete!"
echo ""
echo "Next steps:"
echo "  ./start.sh  - Start the system"
echo "  ./test.sh   - Run tests"
