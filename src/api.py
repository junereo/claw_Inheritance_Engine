from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .runtime import PortRuntime

app = FastAPI(
    title="PromToon Inheritance Engine API",
    description="Automated Webtoon Production Pipeline Backend (SSE Streaming)",
    version="1.2.0",
)

class InheritanceRequest(BaseModel):
    synopsis_text: str = Field(..., description="The initial story synopsis to process")
    target_model: str = Field("flux-klein-v2", description="The image generation model to target")
    use_llm_polish: bool = Field(True, description="Reserved for downstream compiler stages")

@app.post("/api/v1/inheritance-engine/start")
async def start_pipeline(request: InheritanceRequest):
    synopsis_text = request.synopsis_text.strip()
    if not synopsis_text:
        raise HTTPException(status_code=400, detail="synopsis_text must not be empty.")

    runtime = PortRuntime()
    
    # Return an SSE stream delegating directly to the Agent's stream loop
    return StreamingResponse(
        runtime.stream_turn_loop(synopsis_text, max_turns=5),
        media_type="text/event-stream"
    )

@app.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "healthy", "engine": "Inheritance Engine v1.2 (SSE)"}
