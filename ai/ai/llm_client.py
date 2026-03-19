from __future__ import annotations

import json
import os
from typing import Any, Dict

from google import genai
from google.genai import types


def task_simplifier_schema() -> Dict[str, Any]:
    """
    JSON schema for Gemini structured output.
    Gemini supports a subset of JSON Schema for structured outputs.
    """
    return {
        "type": "object",
        "properties": {
            "task_id": {"type": "string"},
            "confidence_score": {"type": "number"},
            "simplified_steps": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "step_number": {"type": "integer"},
                        "instruction": {"type": "string"},
                    },
                    "required": ["step_number", "instruction"],
                    "additionalProperties": False,
                },
            },
            "clarification_needed": {"type": "boolean"},
            "clarification_question": {"type": "string"},
        },
        "required": [
            "task_id",
            "confidence_score",
            "simplified_steps",
            "clarification_needed",
            "clarification_question",
        ],
        "additionalProperties": False,
    }


def call_llm_structured(
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.2,
) -> Dict[str, Any]:
    """
    Real Gemini API call using official Google GenAI SDK.
    Structured output is requested using:
    - response_mime_type="application/json"
    - response_json_schema=<schema>
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set.")

    client = genai.Client(api_key=api_key)

    schema = task_simplifier_schema()

    response = client.models.generate_content(
        model=model,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=temperature,
            response_mime_type="application/json",
            response_json_schema=schema,
        ),
    )

    text = response.text or ""
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Gemini returned invalid JSON: {e}\nRaw output: {text}") from e
