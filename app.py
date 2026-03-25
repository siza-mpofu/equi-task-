from __future__ import annotations
 
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional
 
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv
 
PROJECT_ROOT = Path(__file__).resolve().parent
AI_DIR = PROJECT_ROOT / "ai"
 
if str(AI_DIR) not in sys.path:
    sys.path.insert(0, str(AI_DIR))
 
from ai.ai.task_simplifier import simplify_task
 
load_dotenv()
 
app = FastAPI(title="EquiTask AI Task Simplifier", version="1.0.0")
 
 
class TaskSimplifyRequest(BaseModel):
    task_id: str
    task_text: str = Field(min_length=1)
    task_type: str = "reporting"
    accessibility_mode: str = "Standard"
    model: Optional[str] = None
 
 
@app.post("/ai/task-simplify")
def task_simplify_endpoint(req: TaskSimplifyRequest) -> Dict[str, Any]:
    if not os.getenv("GEMINI_API_KEY"):
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY is not set on the server."
        )
 
    return simplify_task(req.model_dump())
 