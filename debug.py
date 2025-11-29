import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load .env if present
load_dotenv()

api_key = os.environ.get("GOOGLE_API_KEY")

if not api_key:
    print("❌ GOOGLE_API_KEY not found in environment.")
    print("Please set it via: export GOOGLE_API_KEY='your_key'")
else:
    print(f"🔑 Key found: {api_key[:5]}...{api_key[-4:]}")
    try:
        genai.configure(api_key=api_key)
        print("\nAttempting to list models...")
        models = list(genai.list_models())
        
        print(f"\n✅ Found {len(models)} models available to this key:")
        found_flash = False
        found_pro = False
        
        for m in models:
            print(f" - {m.name}")
            if "flash" in m.name: found_flash = True
            if "pro" in m.name: found_pro = True
            
        print("\nDiagnostics:")
        if found_flash:
            print("✅ Gemini Flash is available.")
        else:
            print("❌ Gemini Flash NOT found. You might need to enable it in Google AI Studio.")
            
        if found_pro:
            print("✅ Gemini Pro is available.")
        else:
            print("❌ Gemini Pro NOT found.")
            
    except Exception as e:
        print(f"\n❌ Error listing models: {e}")