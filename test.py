import os
import google.generativeai as genai

API_KEY = os.environ.get("GOOGLE_API_KEY")

print("=========================================")
print("🔍 GEMINI API KEY TEST")
print("=========================================")

if not API_KEY:
    print("❌ ERROR: GOOGLE_API_KEY is NOT set in environment variables.")
    print("Set it first:")
    print("  Windows CMD:  set GOOGLE_API_KEY=your_key_here")
    print("  PowerShell:   $env:GOOGLE_API_KEY='your_key_here'")
    print("  Linux/macOS:  export GOOGLE_API_KEY=your_key_here")
    exit()

print("🔑 API Key detected.")
genai.configure(api_key=API_KEY)

# -----------------------------------------------------
# 1) Test Chat Completion (Gemini-Pro)
# -----------------------------------------------------
print("\n🧪 TEST 1 — Chat Completion (gemini-pro)")
try:
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content("Say 'Hello from Gemini'")
    print("✅ Chat request successful!")
    print("Model replied:", response.text)
except Exception as e:
    print("❌ Chat request FAILED")
    print("Error:", e)

# -----------------------------------------------------
# 2) Test Embeddings (embedding-001)
# -----------------------------------------------------
print("\n🧪 TEST 2 — Embeddings (models/embedding-001)")
try:
    emb = genai.embed_content(
        model="models/embedding-001",
        content="test embedding"
    )
    print("✅ Embeddings request successful!")
    print("Embedding length:", len(emb['embedding']))
except Exception as e:
    print("❌ Embeddings request FAILED")
    print("Error:", e)

print("\n=========================================")
print("🔍 TEST FINISHED")
print("=========================================")
