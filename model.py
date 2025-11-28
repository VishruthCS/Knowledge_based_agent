# test_models_and_quota.py
import os
import google.generativeai as genai
import json
from pprint import pprint

API_KEY = os.environ.get("GOOGLE_API_KEY")
if not API_KEY:
    raise SystemExit("Set GOOGLE_API_KEY in your env first.")

genai.configure(api_key=API_KEY)

print("Listing models available to this API key...\n")
try:
    # library exposes list_models()
    models = genai.list_models()
except Exception as e:
    print("ERROR listing models:", e)
    models = None

if models:
    # models may be a dict or list depending on lib version
    if isinstance(models, dict) and "models" in models:
        models_list = models["models"]
    elif isinstance(models, list):
        models_list = models
    else:
        models_list = [models]

    print(f"Found {len(models_list)} models. Showing id and supported methods (if present):\n")
    for m in models_list:
        # model object/dict printing - handle both shapes
        try:
            mid = m.get("name") or m.get("model") or m.get("id") or str(m)
        except Exception:
            mid = str(m)
        # print minimal info
        print("MODEL ID:", mid)
        # try to show supported methods / modalities
        try:
            pprint(m)
        except Exception:
            print(m)
        print("-" * 60)
else:
    print("No models returned. Check API key / permissions / API enabled in GCP.")

# Try to find one model that supports text generation (generateContent) and embeddings (embedContent)
print("\nSearching for a model that supports generateContent or embedding...")
candidate = None
if models:
    for m in models_list:
        # Different API responses have different fields; attempt to detect supported methods
        mstr = json.dumps(m) if not isinstance(m, str) else m
        if "generate" in mstr.lower() or "generateContent" in mstr or "text-bison" in mstr.lower() or "gemini" in mstr.lower():
            candidate = m
            break

if not candidate:
    print("No obvious candidate model found in the model list. Pick one from the printed list and test with it.")
    raise SystemExit()

# determine model id
if isinstance(candidate, dict):
    model_id = candidate.get("name") or candidate.get("model") or candidate.get("id")
else:
    model_id = str(candidate)

print("\nTesting model id:", model_id)

# Test a small generate (if supported)
try:
    print("\n-- Testing generate / chat --")
    model = genai.GenerativeModel(model_id)
    resp = model.generate_content("Hello — send back a short test message.")
    # resp structure varies with lib version
    if hasattr(resp, "text"):
        print("Generate OK. Text:", resp.text)
    else:
        print("Generate response object:", resp)
except Exception as e:
    print("Generate failed:", e)

# Test embeddings (if available)
try:
    print("\n-- Testing embeddings --")
    emb = genai.embed_content(model="models/embedding-001", content="hello")
    print("Embeddings OK; length:", len(emb.get("embedding") if isinstance(emb, dict) else emb))
except Exception as e:
    print("Embeddings failed:", e)

print("\nDone. If embedding failed with 429, your project does not have embedding quota (see steps below).")
