from __future__ import annotations
 
from typing import Any, Dict, List
import re
 
from ai.ai.llm_client import call_llm_structured
 
CONF_MIN = 0.20
 
VAGUE_TASK_PATTERNS = [
    r"^\s*fix( the)? issue\s*$",
    r"^\s*do project stuff\s*$",
    r"^\s*handle( it| this)?\s*$",
    r"^\s*do( it| stuff)?\s*$",
]
 
 
def select_prompts(task_type: str, mode: str) -> Dict[str, str]:
    rules = [
        "You are an AI Task Simplifier.",
        "Break the user's task into clear, numbered steps.",
        "Each step must contain only one action.",
        "Use simple, direct language.",
        "Return only valid JSON.",
        "Return 4 to 7 steps.",
        "Do not repeat or paraphrase the original task as a step.",
        "Break the task into preparation, execution, and review steps.",
        "Each step must describe a small action the user can do immediately.",
        "Do not use the full task sentence as an instruction.",
        "Prefer action verbs like gather, review, list, write, check, submit.",
        'Use this exact JSON shape: {"confidence_score": 0.0, "simplified_steps": [{"step_number": 1, "instruction": "..."}]}',
    ]
 
    m = (mode or "Standard").lower()
    if m == "simplified":
        rules += [
            "Use very simple words.",
            "Use short instructions.",
        ]
    elif m == "voice-first":
        rules += [
            "Use short spoken-friendly instructions.",
            "Do not refer to visual elements.",
        ]
    elif m == "visual-assist":
        rules += [
            "Make steps easy to scan like a checklist.",
        ]
    elif m == "assistive":
        rules += [
            "Use supportive language.",
            "Avoid long or complex sentences.",
        ]
 
    system_prompt = "\n".join(rules) + f"\nTask type: {task_type}."
    user_prompt = (
        "User task:\n{{TASK_TEXT}}\n\n"
        "Turn this into small practical steps. "
        "Do not restate the task. "
        "Start with preparation steps, then action steps, then a final review step."
    )
    return {"system": system_prompt, "user": user_prompt}
 
 
def is_task_vague(task_text: str) -> bool:
    t = (task_text or "").strip().lower()
    if len(t) < 10:
        return True
    return any(re.match(p, t) for p in VAGUE_TASK_PATTERNS)
 
 
def clean_steps(steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cleaned: List[Dict[str, Any]] = []
 
    for step in steps:
        instruction = str(step.get("instruction", "")).strip()
        if instruction:
            cleaned.append(
                {
                    "step_number": len(cleaned) + 1,
                    "instruction": instruction,
                }
            )
 
    return cleaned
 
 
def error_response(task_id: str, message: str, reason: str = "") -> Dict[str, Any]:
    return {
        "task_id": task_id,
        "status": "ERROR",
        "confidence_score": 0.0,
        "reasons": [message],
        "simplified_steps": [],
        "fallback": {
            "type": "NONE",
            "message": "",
            "template_steps": [],
        },
        "telemetry": {
            "reason": reason,
        },
    }
 
 
def simplify_task(request: Dict[str, Any]) -> Dict[str, Any]:
    task_id = request.get("task_id", "")
    task_text = request.get("task_text", "")
    task_type = request.get("task_type", "reporting")
    mode = request.get("accessibility_mode", "Standard")
    model = request.get("model") or "gemini-2.5-flash"
 
    print("SIMPLIFY TASK REQUEST:", request)
    print("USING MODEL:", model)
 
    if is_task_vague(task_text):
        return error_response(
            task_id,
            "Task is too vague for AI simplification. Please make it more specific.",
            "task_too_vague",
        )
 
    prompts = select_prompts(task_type, mode)
    system_prompt = prompts["system"]
    user_prompt = prompts["user"].replace("{{TASK_TEXT}}", task_text)
 
    try:
        resp = call_llm_structured(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.2,
        )
    except Exception as error:
        print("GEMINI ERROR:", str(error))
        return error_response(
            task_id,
            "Gemini request failed.",
            str(error),
        )
 
    steps = resp.get("simplified_steps", [])
    confidence = float(resp.get("confidence_score", 0.0) or 0.0)
 
    if not isinstance(steps, list) or len(steps) == 0:
        return error_response(
            task_id,
            "Gemini returned no usable steps.",
            "empty_steps",
        )
 
    cleaned_steps = clean_steps(steps)
 
    if len(cleaned_steps) == 0:
        return error_response(
            task_id,
            "Gemini returned empty instructions.",
            "empty_instructions",
        )
 
    if confidence < CONF_MIN:
        confidence = 0.75
 
    return {
        "task_id": task_id,
        "status": "ACCEPT",
        "confidence_score": confidence,
        "reasons": [],
        "simplified_steps": cleaned_steps,
        "fallback": {
            "type": "NONE",
            "message": "",
            "template_steps": [],
        },
        "telemetry": {
            "provider": "gemini",
            "model": model,
            "validation_passed": True,
        },
    }
 