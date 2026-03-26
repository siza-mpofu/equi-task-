from __future__ import annotations
 
import json
import os
from typing import Any, Dict
 
from google import genai
from google.genai import types
 
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
 
 
def call_llm_structured(
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.2,
) -> Dict[str, Any]:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set")
 
    client = genai.Client(api_key=api_key)
 
    prompt = f"{system_prompt}\n\n{user_prompt}"
 
    response = client.models.generate_content(
        model=model or DEFAULT_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=temperature,
            response_mime_type="application/json",
        ),
    )
 
    raw_text = (response.text or "").strip()
    print("GEMINI RAW RESPONSE:", raw_text)
 
    if not raw_text:
        raise RuntimeError("Gemini returned an empty response")
 
    data = json.loads(raw_text)
 
    if not isinstance(data, dict):
        raise RuntimeError("Gemini response was not a JSON object")
 
    return data
 