#!/bin/bash
echo "🔨 Building Few-Shot Learning System..."

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

# Backend
cd backend || exit 1
if [ ! -d "venv" ] || [ ! -f "venv/bin/activate" ]; then
    echo "Creating Python virtual environment..."
    if command -v virtualenv >/dev/null 2>&1; then
        virtualenv -p python3 venv
        source venv/bin/activate
        pip install -q -r requirements.txt
        deactivate
    elif python3 -m venv venv 2>&1; then
        source venv/bin/activate
        pip install -q -r requirements.txt
        deactivate
    else
        echo "⚠️  venv not available. Installing packages with --break-system-packages flag..."
        pip3 install --break-system-packages -q -r requirements.txt
        # Create a dummy venv structure for start.sh compatibility
        mkdir -p venv/bin
        echo '#!/bin/bash' > venv/bin/activate
        echo 'export PATH="/usr/local/bin:$PATH"' >> venv/bin/activate
        echo 'alias python=python3' >> venv/bin/activate
        echo 'alias pip=pip3' >> venv/bin/activate
        chmod +x venv/bin/activate
    fi
else
    source venv/bin/activate
    pip install -q -r requirements.txt
    deactivate
fi
cd "$SCRIPT_DIR" || exit 1

# Frontend
cd frontend || exit 1
npm install
cd "$SCRIPT_DIR" || exit 1

echo "✅ Build complete!"
