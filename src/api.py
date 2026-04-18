from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .runtime import PortRuntime, StageStatus


app = FastAPI(
    title="PromToon Inheritance Engine API",
    description="Automated Webtoon Production Pipeline Backend",
    version="1.0.0"
)


class InheritanceRequest(BaseModel):
    synopsis_text: str = Field(..., description="The initial story synopsis to process")
    target_model: str = Field("flux-klein-v2", description="The image generation model to target")


class InheritanceResponse(BaseModel):
    status: str = Field(..., description="Final status: 'completed' or 'review_failed'")
    story: Dict[str, Any] = Field(..., description="The generated story data")
    review: Dict[str, Any] = Field(..., description="The quality review report")
    image_prompts: Optional[List[str]] = Field(None, description="Generated image prompts (if successful)")


@app.post("/api/v1/inheritance-engine/start", response_model=InheritanceResponse)
async def start_pipeline(request: InheritanceRequest):
    """
    Starts the 5-stage Inheritence Engine pipeline for a given synopsis.
    Runs the full loop until completion or review failure.
    """
    runtime = PortRuntime()
    
    # We use a turn loop to simulate the pipeline stages.
    # Stage 1: Story Creator
    # Stage 2: Story Reviewer
    # Stage 3-5: Cut/Compilers
    results = runtime.run_turn_loop(request.synopsis_text, max_turns=5)
    
    if not results:
        raise HTTPException(status_code=500, detail="Engine failed to produce any results.")
    
    last_result = results[-1]
    output_text = last_result.output
    
    # Parse the state summary from the output
    try:
        state_section = output_text.split("--- Inheritance Engine State ---")[-1]
        state_json = json.loads(state_section.split("\n\n")[0] if "\n\n" in state_section else state_section)
    except (IndexError, json.JSONDecodeError):
        # Fallback state if parsing fails
        state_json = {"status": "runtime_error", "current_stage": 0, "story_data": {}}

    # Extract info for response
    final_status = state_json.get("status", "unknown")
    
    # Simple extraction for story and review from textual output (Mocked for port)
    story_info = {"raw_output": output_text[:200] + "..."}
    review_info = {"passed": final_status == "completed" or (final_status == "running" and state_json.get("current_stage") >= 3)}
    
    if "Stage 2 Quality Review FAILED" in output_text:
        review_info["passed"] = False
        review_info["reason"] = "Story Reviewer rejected the content."
    
    image_prompts = []
    if final_status == "completed":
        # Extract cut lines
        for line in output_text.split("\n"):
            if line.startswith("Cut "):
                image_prompts.append(line)

    return InheritanceResponse(
        status="completed" if final_status == "completed" else "review_failed",
        story=story_info,
        review=review_info,
        image_prompts=image_prompts if image_prompts else None
    )


@app.get("/health")
async def health_check():
    return {"status": "healthy", "engine": "Inheritance Engine v1"}
