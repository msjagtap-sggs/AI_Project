import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
key = os.getenv('GOOGLE_API_KEY')
print('KEY SET:', bool(key))
if not key:
    raise SystemExit('No GOOGLE_API_KEY in environment')

try:
    genai.configure(api_key=key)
    models = genai.ModelService.list_models()
    print('MODEL COUNT:', len(models))
    for m in models:
        try:
            name = m.name
        except Exception:
            name = str(m)
        print(name)
except Exception as e:
    print('ERROR:', type(e).__name__, e)
