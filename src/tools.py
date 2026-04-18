from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from .models import PortingBacklog, PortingModule
from .permissions import ToolPermissionContext


# --- Schemas ---

class StoryPhase(BaseModel):
    name: str = Field(..., description="The name of the story phase (e.g., Intro, Climax)")
    description: str = Field(..., description="A brief summary of what happens in this phase")
    events: List[str] = Field(..., description="A list of specific major events in this phase")

class StoryJSON(BaseModel):
    meta: Dict[str, Any] = Field(..., description="Metadata including genre, target audience, and theme")
    phases: List[StoryPhase] = Field(..., description="The 3-Phase story structure (e.g., Beginning, Middle, End)")
    ending: str = Field(..., description="The final resolution of the story")

class ReviewResult(BaseModel):
    passed: bool = Field(..., description="Whether the story passed the quality gate")
    report: str = Field(..., description="Detailed feedback or scoring report")

class CompilerInputs(BaseModel):
    shared_assets: Dict[str, Any] = Field(..., description="Global assets like character descriptions and world rules")
    relational_cuts: List[Dict[str, Any]] = Field(..., description="List of cut definitions with local overrides and relations")


# --- Tool Implementation Logic ---

def validate_story_json(story: StoryJSON) -> Dict[str, Any]:
    """
    Validates that the story JSON follows the 3-Phase structure.
    """
    if len(story.phases) != 3:
        return {"valid": False, "error": f"Story must have exactly 3 phases, found {len(story.phases)}."}
    return {"valid": True, "message": "Story JSON validation passed."}

def run_story_reviewer(text: str) -> ReviewResult:
    """
    Simulates a quality review of the provided story text.
    """
    # Simple mock logic
    if len(text) < 20:
        return ReviewResult(passed=False, report="Story text is too short to be meaningful.")
    
    if "cliche" in text.lower():
        return ReviewResult(passed=False, report="Story contains excessive clichés.")
        
    return ReviewResult(passed=True, report="Story passed initial quality check with a score of 8/10.")

def run_deterministic_compiler(shared_assets: Dict[str, Any], relational_cuts: List[Dict[str, Any]]) -> str:
    """
    Compiles final prompt strings by merging shared assets with cut-specific instructions.
    """
    results = []
    for i, cut in enumerate(relational_cuts):
        char_name = cut.get("character", "Unknown")
        char_desc = shared_assets.get("characters", {}).get(char_name, "Standard appearance")
        action = cut.get("action", "Standing still")
        prompt = f"Cut {i+1}: {char_name} ({char_desc}) is {action}. [Background: {shared_assets.get('world_setting', 'Default')}]"
        results.append(prompt)
    
    return "\n".join(results)


# --- Porting Infrastructure Shims ---

@dataclass(frozen=True)
class ToolExecution:
    name: str
    source_hint: str
    payload: str
    handled: bool
    message: str


PORTED_TOOLS = (
    PortingModule(name='validate_story_json', responsibility='Validates 3-Phase Story structure', source_hint='Inheritance Engine', status='active'),
    PortingModule(name='run_story_reviewer', responsibility='Performs quality review of story text', source_hint='Inheritance Engine', status='active'),
    PortingModule(name='run_deterministic_compiler', responsibility='Compiles inherited prompts from assets and cuts', source_hint='Inheritance Engine', status='active'),
)


def build_tool_backlog() -> PortingBacklog:
    return PortingBacklog(title='Webtoon Tool surface', modules=list(PORTED_TOOLS))


def tool_names() -> list[str]:
    return [module.name for module in PORTED_TOOLS]


def get_tool(name: str) -> PortingModule | None:
    needle = name.lower()
    for module in PORTED_TOOLS:
        if module.name.lower() == needle:
            return module
    return None


def get_tools(
    simple_mode: bool = False,
    include_mcp: bool = True,
    permission_context: ToolPermissionContext | None = None,
) -> tuple[PortingModule, ...]:
    return PORTED_TOOLS


def execute_tool(name: str, payload: str = '') -> ToolExecution:
    module = get_tool(name)
    if module is None:
        return ToolExecution(name=name, source_hint='', payload=payload, handled=False, message=f'Unknown tool: {name}')
    
    # In a real system, this would call the actual functions implemented above.
    # For now, we return a mock execution message.
    return ToolExecution(name=module.name, source_hint=module.source_hint, payload=payload, handled=True, message=f"Executed {module.name} with payload.")


def render_tool_index(limit: int = 20, query: str | None = None) -> str:
    lines = [f'Webtoon Tool entries: {len(PORTED_TOOLS)}', '']
    lines.extend(f'- {module.name} — {module.responsibility}' for module in PORTED_TOOLS)
    return '\n'.join(lines)
