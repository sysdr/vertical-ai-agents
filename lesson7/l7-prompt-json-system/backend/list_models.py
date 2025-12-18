#!/usr/bin/env python3
"""List available Gemini models"""
import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("ERROR: GEMINI_API_KEY not found")
    exit(1)

genai.configure(api_key=api_key)

print("Available Gemini models that support generateContent:\n")
try:
    models = genai.list_models()
    available_models = []
    for model in models:
        if 'generateContent' in model.supported_generation_methods:
            model_name = model.name
            # Extract just the model name part
            if '/' in model_name:
                short_name = model_name.split('/')[-1]
            else:
                short_name = model_name
            available_models.append((model_name, short_name))
            print(f"  Full name: {model_name}")
            print(f"  Short name: {short_name}")
            print()
    
    if available_models:
        print(f"\n✅ Found {len(available_models)} available model(s)")
        print(f"\nRecommended model to use: {available_models[0][1]}")
    else:
        print("\n❌ No models found with generateContent support")
        
except Exception as e:
    print(f"Error listing models: {e}")
    print("\nTrying direct model initialization...")
    test_models = ['gemini-pro', 'models/gemini-pro']
    for tm in test_models:
        try:
            m = genai.GenerativeModel(tm)
            print(f"✅ {tm} works!")
            break
        except Exception as e2:
            print(f"❌ {tm}: {str(e2)[:100]}")

