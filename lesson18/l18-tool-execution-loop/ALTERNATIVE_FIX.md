# 🔄 Alternative Approaches to Fix API Key

## Method 1: Non-Interactive Script (Easiest)

### Option A: Pass API key as argument
```bash
./scripts/set_api_key.sh YOUR_API_KEY_HERE
```

### Option B: Use environment variable
```bash
GEMINI_API_KEY=your_key_here ./scripts/set_api_key.sh
```

### Option C: Pipe from file
```bash
echo "your_api_key" | ./scripts/quick_setup.sh
# Or
cat api_key.txt | ./scripts/quick_setup.sh
```

## Method 2: One-Liner (Fastest)

```bash
cd backend && echo "GEMINI_API_KEY=your_key_here" > .env && pkill -f "python main.py" && source venv/bin/activate && python main.py &
```

## Method 3: Environment Variable (No File)

```bash
export GEMINI_API_KEY=your_key_here
pkill -f "python main.py"
cd backend
source venv/bin/activate
python main.py
```

## Method 4: Using Python Directly

```python
# Create a Python script to set it
import os
with open('backend/.env', 'w') as f:
    f.write(f"GEMINI_API_KEY={os.getenv('GEMINI_API_KEY', 'your_key_here')}\n")
```

## Method 5: Copy from Clipboard (Linux)

If you have your API key in clipboard:
```bash
xclip -o | ./scripts/quick_setup.sh
# Or with xsel:
xsel -b | ./scripts/quick_setup.sh
```

## Method 6: Read from Secure Input

```bash
# Using read with -s flag (hidden input)
read -s -p "Enter API key: " key && ./scripts/set_api_key.sh "$key"
```

## 🎯 Recommended: Method 1 Option A

Just run:
```bash
./scripts/set_api_key.sh YOUR_API_KEY
```

This will:
- ✅ Save the key
- ✅ Validate it
- ✅ Restart backend automatically
- ✅ Test that it works

## 📝 Get Your API Key

1. Visit: https://makersuite.google.com/app/apikey
2. Sign in with Google
3. Click "Create API Key"
4. Copy the key

Then use any method above!


