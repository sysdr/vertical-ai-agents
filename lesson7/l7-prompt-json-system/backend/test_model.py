#!/usr/bin/env python3
"""Test script to find available Gemini models"""
import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("ERROR: GEMINI_API_KEY not found")
    exit(1)

genai.configure(api_key=api_key)

print("Testing available models...\n")

models_to_test = [
    'gemini-pro',
    'gemini-1.5-pro',
    'gemini-1.5-flash',
    'models/gemini-pro',
    'models/gemini-1.5-pro',
    'models/gemini-1.5-flash',
]

for model_name in models_to_test:
    try:
        model = genai.GenerativeModel(model_name)
        print(f"✅ {model_name} - SUCCESS")
    except Exception as e:
        print(f"❌ {model_name} - FAILED: {str(e)[:100]}")

print("\n" + "="*60)
print("Listing all available models from API:")
print("="*60)

try:
    for model in genai.list_models():
        if 'generateContent' in model.supported_generation_methods:
            print(f"  - {model.name}")
except Exception as e:
    print(f"Error listing models: {e}")

