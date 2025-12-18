#!/usr/bin/env python3
"""Find which Gemini model actually works"""
import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("ERROR: GEMINI_API_KEY not found")
    exit(1)

genai.configure(api_key=api_key)

print("Testing Gemini models...\n")

# List of models to try
models_to_test = [
    'gemini-1.5-flash',
    'gemini-1.5-pro', 
    'gemini-pro',
    'models/gemini-1.5-flash',
    'models/gemini-1.5-pro',
    'models/gemini-pro',
]

working_model = None

for model_name in models_to_test:
    try:
        print(f"Testing: {model_name}...", end=" ")
        model = genai.GenerativeModel(model_name)
        # Try a simple generation to verify it works
        response = model.generate_content("Say hello")
        print(f"✅ WORKS!")
        working_model = model_name
        break
    except Exception as e:
        error_msg = str(e)
        if "404" in error_msg or "not found" in error_msg.lower():
            print(f"❌ Not found")
        else:
            print(f"❌ Error: {error_msg[:80]}")

if working_model:
    print(f"\n✅ Working model found: {working_model}")
    print(f"\nUpdate your code to use: genai.GenerativeModel('{working_model}')")
else:
    print("\n❌ No working model found. Checking available models...")
    try:
        print("\nAvailable models:")
        for model in genai.list_models():
            if 'generateContent' in model.supported_generation_methods:
                print(f"  - {model.name}")
    except Exception as e:
        print(f"Error listing models: {e}")

