#!/bin/bash
set -e

echo "🔨 Building L8 Agent Decision Maker..."

# Backend
echo "Installing Python dependencies..."
cd backend
# Try to create venv, if it fails, install packages with --user flag
if python3 -m venv venv 2>&1 | grep -q "ensurepip"; then
    echo "Creating venv without pip..."
    python3 -m venv venv --without-pip
    source venv/bin/activate
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3
    pip install --quiet --upgrade pip
    pip install --quiet -r requirements.txt
    deactivate
elif [ -d venv ]; then
    source venv/bin/activate
    pip install --quiet --upgrade pip
    pip install --quiet -r requirements.txt
    deactivate
else
    echo "Warning: Could not create venv, installing with --user flag..."
    pip3 install --user --quiet --upgrade pip setuptools wheel
    pip3 install --user --quiet -r requirements.txt
    # Create a dummy venv directory for start.sh
    mkdir -p venv/bin
    echo '#!/bin/bash' > venv/bin/activate
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> venv/bin/activate
    echo 'alias python=python3' >> venv/bin/activate
    echo 'alias pip=pip3' >> venv/bin/activate
    chmod +x venv/bin/activate
fi
cd ..

# Frontend
echo "Installing Node dependencies..."
cd frontend
npm install --silent
cd ..

echo "✅ Build complete!"
