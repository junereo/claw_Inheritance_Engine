from __future__ import annotations
import json

from dataclasses import dataclass

from .commands import PORTED_COMMANDS
from .llm_client import ask_agentic_llm_json
from .tools import validate_story_json, run_story_reviewer, run_deterministic_compiler, assemble_image_prompt_artifact, StoryJSON, CompilerInputs
from .context import PortContext, build_port_context, render_context, inject_inheritance_context, WebtoonContextManager
from .history import HistoryLog
from .models import PermissionDenial, PortingModule, AgentState, StageStatus
from .query_engine import QueryEngineConfig, QueryEnginePort, TurnResult
from .setup import SetupReport, WorkspaceSetup, run_setup
from .system_init import build_system_init_message
from .tools import PORTED_TOOLS
from .execution_registry import build_execution_registry


@dataclass(frozen=True)
class RoutedMatch:
    kind: str
    name: str
    source_hint: str
    score: int


@dataclass
class RuntimeSession:
    prompt: str
    context: PortContext
    setup: WorkspaceSetup
    setup_report: SetupReport
    system_init_message: str
    history: HistoryLog
    routed_matches: list[RoutedMatch]
    turn_result: TurnResult
    command_execution_messages: tuple[str, ...]
    tool_execution_messages: tuple[str, ...]
    stream_events: tuple[dict[str, object], ...]
    persisted_session_path: str

    def as_markdown(self) -> str:
        lines = [
            '# Runtime Session',
            '',
            f'Prompt: {self.prompt}',
            '',
            '## Context',
            render_context(self.context),
            '',
            '## Setup',
            f'- Python: {self.setup.python_version} ({self.setup.implementation})',
            f'- Platform: {self.setup.platform_name}',
            f'- Test command: {self.setup.test_command}',
            '',
            '## Startup Steps',
            *(f'- {step}' for step in self.setup.startup_steps()),
            '',
            '## System Init',
            self.system_init_message,
            '',
            '## Routed Matches',
        ]
        if self.routed_matches:
            lines.extend(
                f'- [{match.kind}] {match.name} ({match.score}) — {match.source_hint}'
                for match in self.routed_matches
            )
        else:
            lines.append('- none')
        lines.extend([
            '',
            '## Command Execution',
            *(self.command_execution_messages or ('none',)),
            '',
            '## Tool Execution',
            *(self.tool_execution_messages or ('none',)),
            '',
            '## Stream Events',
            *(f"- {event['type']}: {event}" for event in self.stream_events),
            '',
            '## Turn Result',
            self.turn_result.output,
            '',
            f'Persisted session path: {self.persisted_session_path}',
            '',
            self.history.as_markdown(),
        ])
        return '\n'.join(lines)


from enum import Enum
from pydantic import BaseModel, Field


class StageStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    REVIEW_FAILED = "review_failed"


class AgentState(BaseModel):
    current_stage: int = Field(1, ge=1, le=5)
    status: StageStatus = StageStatus.RUNNING
    story_data: dict = Field(default_factory=dict)
    history: list[str] = Field(default_factory=list)


class PortRuntime:
    def route_prompt(self, prompt: str, limit: int = 5) -> list[RoutedMatch]:
        tokens = {token.lower() for token in prompt.replace('/', ' ').replace('-', ' ').split() if token}
        by_kind = {
            'command': self._collect_matches(tokens, PORTED_COMMANDS, 'command'),
            'tool': self._collect_matches(tokens, PORTED_TOOLS, 'tool'),
        }

        selected: list[RoutedMatch] = []
        for kind in ('command', 'tool'):
            if by_kind[kind]:
                selected.append(by_kind[kind].pop(0))

        leftovers = sorted(
            [match for matches in by_kind.values() for match in matches],
            key=lambda item: (-item.score, item.kind, item.name),
        )
        selected.extend(leftovers[: max(0, limit - len(selected))])
        return selected[:limit]

    def bootstrap_session(self, prompt: str, limit: int = 5) -> RuntimeSession:
        context = build_port_context()
        setup_report = run_setup(trusted=True)
        setup = setup_report.setup
        history = HistoryLog()
        engine = QueryEnginePort.from_workspace()
        history.add('context', f'python_files={context.python_file_count}, archive_available={context.archive_available}')
        history.add('registry', f'commands={len(PORTED_COMMANDS)}, tools={len(PORTED_TOOLS)}')
        matches = self.route_prompt(prompt, limit=limit)
        registry = build_execution_registry()
        command_execs = tuple(registry.command(match.name).execute(prompt) for match in matches if match.kind == 'command' and registry.command(match.name))
        tool_execs = tuple(registry.tool(match.name).execute(prompt) for match in matches if match.kind == 'tool' and registry.tool(match.name))
        denials = tuple(self._infer_permission_denials(matches))
        stream_events = tuple(engine.stream_submit_message(
            prompt,
            matched_commands=tuple(match.name for match in matches if match.kind == 'command'),
            matched_tools=tuple(match.name for match in matches if match.kind == 'tool'),
            denied_tools=denials,
        ))
        turn_result = engine.submit_message(
            prompt,
            matched_commands=tuple(match.name for match in matches if match.kind == 'command'),
            matched_tools=tuple(match.name for match in matches if match.kind == 'tool'),
            denied_tools=denials,
        )
        persisted_session_path = engine.persist_session()
        history.add('routing', f'matches={len(matches)} for prompt={prompt!r}')
        history.add('execution', f'command_execs={len(command_execs)} tool_execs={len(tool_execs)}')
        history.add('turn', f'commands={len(turn_result.matched_commands)} tools={len(turn_result.matched_tools)} denials={len(turn_result.permission_denials)} stop={turn_result.stop_reason}')
        history.add('session_store', persisted_session_path)
        return RuntimeSession(
            prompt=prompt,
            context=context,
            setup=setup,
            setup_report=setup_report,
            system_init_message=build_system_init_message(trusted=True),
            history=history,
            routed_matches=matches,
            turn_result=turn_result,
            command_execution_messages=command_execs,
            tool_execution_messages=tool_execs,
            stream_events=stream_events,
            persisted_session_path=persisted_session_path,
        )

    async def stream_turn_loop(self, prompt: str, max_turns: int = 10):
        import asyncio
        from .llm_client import ask_llm_decision, ask_llm_generate, _detect_repetition

        state = AgentState()
        previous_cut_assets = {}
        context_mgr = WebtoonContextManager()

        # ── System prompts ─────────────────────────────────────────────
        decision_system = (
            "당신은 PrompToon 총괄 웹툰 연출가입니다.\n"
            "현재 파이프라인 상태를 읽고, 다음에 호출할 도구만 결정하세요.\n"
            "절대 tool_payload를 포함하지 마세요.\n\n"
            "사용 가능한 도구:\n"
            "- validate_story_json (Stage 1→2)\n"
            "- run_story_reviewer (Stage 2→3)\n"
            "- run_deterministic_compiler (Stage 3→5)\n"
            "- none (작업 완료)\n\n"
            "출력 형식 (JSON only):\n"
            '{"thought": "추론", "tool_to_use": "도구명"}'
        )

        story_gen_system = (
            "You are a Story Creator for the PromToon Inheritance Engine.\n"
            "Generate a structured story text using the exactly following TAG format.\n"
            "DO NOT OUTPUT JSON. Use only the tags below.\n\n"
            "[FORMAT]\n"
            "# META\n"
            "TITLE: <text>\n"
            "SUBTITLE: <text>\n"
            "GENRE: <text>\n"
            "SYNOPSIS: <text>\n"
            "SUMMARY: <text>\n\n"
            "# PHASE 1\n"
            "[NARRATION] <text>\n"
            "[IMAGE] <description of visual>\n"
            "[DIALOGUE | SpeakerName] <lines>\n"
            "... (repeat for 5-6 scenes)\n\n"
            "# PHASE 1 CHOICE\n"
            "QUESTION: <choice question>\n"
            "IMAGE_DESCRIPTION: <visual for choice>\n"
            "---\n"
            "OPTION 1: Label | Subtext | Reaction | [MAP: boundary, action, control, connection]\n"
            "OPTION 2: Label | Subtext | Reaction | [MAP: boundary, action, control, connection]\n\n"
            "(Repeat # PHASE 2, # PHASE 3 in the same way)\n\n"
            "# ENDING POSTER\n"
            "TITLE_KO: <text>\n"
            "TITLE_EN: <text>\n"
            "SYNOPSIS: <text>\n"
            "CREDIT: <text>\n"
            "IMAGE_DESCRIPTION: <text>\n"
            "FOOTER: <text>\n\n"
            "# ENDING LIST\n"
            "- ID: ending-a | CONDITION: <hint> | BADGE: <badge> | TAGLINE: <text> | LINES: l1, l2, l3, l4\n"
            "(Provide 3 or 4 endings)\n\n"
            "# INTERFACE\n"
            "BUTTONS: <btn1>, <btn2>\n"
            "BRAND_TEXT: <text>\n\n"
            "[RULES]\n"
            "- psychologyMapping (MAP) values are integers (typically -1, 0, 1).\n"
            "- Each phase MUST have 5-6 scenes and exactly 2 options.\n"
            "- imageDescription must be descriptive and visual.\n"
            "- Keep narration and dialogue concise.\n"
        )

        # ── Structured retry state ─────────────────────────────────────
        retry_state = {
            "last_tool_name": None,
            "last_payload": None,
            "last_error_type": None,
            "json_retry_count": 0,
            "stage_retry_count": 0,
            "last_valid_story": None,
        }

        def _build_history_summary() -> str:
            parts = []
            if retry_state["last_tool_name"]:
                parts.append(f"Last tool: {retry_state['last_tool_name']}")
            if retry_state["last_error_type"]:
                parts.append(f"Last error: {retry_state['last_error_type']}")
            if retry_state["json_retry_count"] > 0:
                parts.append(f"JSON retries: {retry_state['json_retry_count']}")
            if retry_state["last_valid_story"]:
                parts.append("Has valid story from previous stage.")
            return " | ".join(parts) if parts else "No prior actions."

        yield f"data: {json.dumps({'type': 'init', 'message': 'Engine initialized.'})}\n\n"

        for turn in range(max_turns):
            if state.status != StageStatus.RUNNING:
                break

            # Abort if too many stage retries
            if retry_state["stage_retry_count"] >= 3:
                state.status = StageStatus.REVIEW_FAILED
                yield f"data: {json.dumps({'type': 'error', 'message': 'Too many stage retries. Pipeline aborted.'})}\n\n"
                break

            pinned_context = ""
            if state.current_stage == 3:
                mock_current_cut = {"inheritsFromCutId": "PREVIOUS_CUT"}
                pinned_context = inject_inheritance_context(mock_current_cut, previous_cut_assets)

            history = _build_history_summary()
            state_info = f"\n[State: Stage {state.current_stage}, Status: {state.status.value}]\n[History: {history}]"
            turn_prompt = context_mgr.assemble_messages(prompt + state_info, pinned_context)

            # ── Phase A: Decision (short, stable) ──────────────────────
            yield f"data: {json.dumps({'type': 'thinking', 'turn': turn+1, 'message': f'Stage {state.current_stage}: Deciding next tool...'})}\n\n"

            decision = await asyncio.to_thread(ask_llm_decision, decision_system, turn_prompt)
            thought = decision.get("thought", "Processing...")
            tool_name = decision.get("tool_to_use", "none").lower()

            yield f"data: {json.dumps({'type': 'decision', 'turn': turn+1, 'thought': thought, 'tool': tool_name})}\n\n"

            # ── Exit conditions ────────────────────────────────────────
            if tool_name in ("none", "error"):
                yield f"data: {json.dumps({'type': 'info', 'message': 'Agent elected to stop.'})}\n\n"
                break

            # ── Phase B: Generate payload (only for content-heavy tools)
            tool_result_data = None

            if tool_name == "validate_story_json":
                yield f"data: {json.dumps({'type': 'generating', 'turn': turn+1, 'message': 'Generating story JSON...'})}\n\n"

                # ── Text Output → Python Template Parser Logic ──
                raw_text = await asyncio.to_thread(
                    ask_llm_generate,
                    story_gen_system,
                    f"시놉시스:\n{prompt}",
                    max_tokens=StoryMaxTokens if 'StoryMaxTokens' in locals() else 5000,
                    num_ctx=32768,
                    json_mode=False
                )

                if not raw_text or len(raw_text) < 100:
                    retry_state["json_retry_count"] += 1
                    retry_state["last_error_type"] = "empty_generation"
                    retry_state["stage_retry_count"] += 1
                    msg = f"Story generation returned empty. Retry {retry_state['json_retry_count']}/2"
                    yield f"data: {json.dumps({'type': 'retry', 'turn': turn+1, 'message': msg})}\n\n"
                    continue

                # Server-side repetition check on raw text
                if _detect_repetition(raw_text, check_field_lengths=False):
                    retry_state["json_retry_count"] += 1
                    retry_state["last_error_type"] = "repetition_detected"
                    retry_state["stage_retry_count"] += 1
                    msg = f"Repetition detected in story. Discarding and retrying. ({retry_state['json_retry_count']}/2)"
                    yield f"data: {json.dumps({'type': 'retry', 'turn': turn+1, 'message': msg})}\n\n"
                    continue

                try:
                    from .tools import parse_text_to_story_json
                    payload = parse_text_to_story_json(raw_text)
                    
                    yield (
                        f"data: {json.dumps({'type': 'story_generated', 'turn': turn+1, 'stage': state.current_stage, 'payload': payload}, ensure_ascii=False)}\n\n"
                    )
                    story_obj = StoryJSON(**payload)
                    tool_result_data = validate_story_json(story_obj)
                    if tool_result_data.get("valid"):
                        state.current_stage = 2
                        state.story_data = payload
                        retry_state["last_valid_story"] = payload
                        retry_state["json_retry_count"] = 0
                        retry_state["stage_retry_count"] = 0
                except Exception as e:
                    retry_state["json_retry_count"] += 1
                    retry_state["last_error_type"] = f"parsing_validation_error: {e}"
                    retry_state["stage_retry_count"] += 1
                    tool_result_data = {"valid": False, "error": str(e), "raw": raw_text[:500]}
                    msg = f"Story parsing/validation failed: {e}. Retry {retry_state['json_retry_count']}/2"
                    yield f"data: {json.dumps({'type': 'retry', 'turn': turn+1, 'message': msg})}\n\n"
                    continue

            elif tool_name == "run_story_reviewer":
                review_input = prompt
                if retry_state.get("last_valid_story"):
                    review_input = retry_state["last_valid_story"]
                review_res = run_story_reviewer(review_input)
                tool_result_data = review_res.model_dump(mode="json")

                if not review_res.passed:
                    state.status = StageStatus.REVIEW_FAILED
                    review_summary = " | ".join(review_res.warnings) if review_res.warnings else "Review failed."
                    yield f"data: {json.dumps({'type': 'error', 'message': f'[HARD GATE] Stage 2 Review FAILED: {review_summary}'})}\n\n"
                else:
                    state.current_stage = 3
                    previous_cut_assets = {
                        "location_anchor": "Detected location from review",
                        "character_dna": {"trait": "Consistent from description"},
                    }

            elif tool_name == "run_deterministic_compiler":
                # Decision-only: ask LLM for the compiler payload
                yield f"data: {json.dumps({'type': 'generating', 'turn': turn+1, 'message': 'Generating compiler inputs...'})}\n\n"
                compiler_system = (
                    "You are a Cut Architect for a webtoon production pipeline (v1.1-local schema).\n"
                    "Generate deterministic compiler input from the given story data.\n\n"
                    "[CRITICAL RULES]\n"
                    "1. Output ONLY valid JSON with root keys shared_assets and relational_cuts.\n"
                    "2. shared_assets must follow schemaVersion=1.0 and use array-based locations/characters/palettes/lightingPresets.\n"
                    "3. relational_cuts must follow schemaVersion=1.1-local and use root shape {schemaVersion, storyId, sharedAssetsRef, cuts[]}.\n"
                    "4. Every cut object must include cutId, sceneId, cutType, locationId, summary, continuityLock, frameRelation.\n"
                    "5. Use real story-derived ids and descriptions. Never use placeholders.\n"
                    "6. location anchors, character silhouettes, palette, lighting, and inheritance should be explicit when inferable.\n\n"
                    "JSON Shape:\n"
                    "{"
                    "\"shared_assets\":{"
                    "\"schemaVersion\":\"1.0\","
                    "\"storyId\":\"story-slug\","
                    "\"storyTitle\":\"str\","
                    "\"locations\":[{\"id\":\"str\",\"label\":\"str\",\"baseStructure\":\"str\",\"anchors\":[{\"id\":\"str\",\"description\":\"str\",\"firstAppearanceCutId\":\"str?\"}],\"defaultPaletteId\":\"str?\",\"defaultLightingId\":\"str?\"}],"
                    "\"characters\":[{\"id\":\"str\",\"role\":\"protagonist|named|group|extra\",\"silhouetteDescription\":\"str\",\"signatureProps\":[\"str\"],\"firstAppearanceCutId\":\"str?\"}],"
                    "\"palettes\":[{\"id\":\"str\",\"warmLight\":\"#RRGGBB?\",\"base\":\"#RRGGBB?\",\"shadow\":\"#RRGGBB?\",\"accent\":\"#RRGGBB?\",\"description\":\"str?\"}],"
                    "\"lightingPresets\":[{\"id\":\"str\",\"description\":\"str\",\"mood\":\"str?\"}],"
                    "\"globalStyle\":{\"styleBlock\":\"str\",\"globalRules\":[\"str\"]}"
                    "},"
                    "\"relational_cuts\":{"
                    "\"schemaVersion\":\"1.1-local\","
                    "\"storyId\":\"story-slug\","
                    "\"sharedAssetsRef\":\"./story.shared-assets.json\","
                    "\"cuts\":[{"
                    "\"schemaVersion\":\"1.1-local\","
                    "\"storyId\":\"story-slug\","
                    "\"sceneId\":\"scene-id\","
                    "\"cutId\":\"cut-id\","
                    "\"cutType\":\"establishing|environment_focus|dialogue|interaction|reaction|insert|transition|cliffhanger\","
                    "\"locationId\":\"location-id\","
                    "\"summary\":\"str\","
                    "\"continuityLock\":{\"keepLocation\":true,\"keepAnchors\":[\"str\"],\"keepCharacters\":[\"str\"],\"keepProps\":[\"str\"],\"keepPalette\":true,\"keepLighting\":true,\"keepMood\":true},"
                    "\"frameRelation\":{\"inheritsFromCutId\":\"str?\",\"temporalRelation\":\"continuous|short_pause|time_jump|flashback|parallel\",\"spatialRelation\":\"same_location_same_axis|same_location_focus_shift|same_location_camera_shift|same_location_new_frame|new_location\",\"shotType\":\"establishing_wide|wide|medium|medium_close_up|close_up|insert|first_person_pov|over_shoulder\"},"
                    "\"paletteSignature\":{\"paletteId\":\"str?\"},"
                    "\"characterDeltas\":[],"
                    "\"propDeltas\":[],"
                    "\"actionDeltas\":[],"
                    "\"editingNotes\":[],"
                    "\"sourceEvidence\":{\"sourceTextIds\":[\"str\"],\"sourceExcerpt\":\"str?\"},"
                    "\"reviewHints\":{\"ambiguityFlags\":[\"str\"],\"manualReviewRequired\":false,\"suggestedChecks\":[\"str\"]}"
                    "}]"
                    "}"
                    "}\n\n"
                    "Output ONLY valid JSON. No markdown fences."
                )
                payload = await asyncio.to_thread(ask_llm_generate, compiler_system, f"Story:\n{json.dumps(state.story_data or {}, ensure_ascii=False)}")
                try:
                    yield (
                        f"data: {json.dumps({'type': 'compiler_generated', 'turn': turn+1, 'stage': state.current_stage, 'payload': payload}, ensure_ascii=False)}\n\n"
                    )
                    compiler_obj = CompilerInputs.model_validate(payload)
                    compiled_preview = run_deterministic_compiler(
                        compiler_obj.shared_assets,
                        compiler_obj.relational_cuts,
                    )
                    story_meta = (state.story_data or {}).get("meta", {})
                    image_prompts = assemble_image_prompt_artifact(
                        compiler_obj.shared_assets,
                        compiler_obj.relational_cuts,
                        compiled_preview,
                        story_title=story_meta.get("title"),
                        genre=story_meta.get("genre"),
                    )
                    tool_result_data = {
                        "compiled_preview": compiled_preview.model_dump(mode="json"),
                        "image_prompts": image_prompts.model_dump(mode="json", by_alias=True),
                    }
                    state.current_stage = 5
                    state.status = StageStatus.COMPLETED
                except Exception as e:
                    tool_result_data = {"error": str(e)}

            else:
                yield f"data: {json.dumps({'type': 'info', 'message': f'Unknown tool: {tool_name}'})}\n\n"
                continue

            retry_state["last_tool_name"] = tool_name
            yield f"data: {json.dumps({'type': 'tool_result', 'tool': tool_name, 'status': state.status.value, 'stage': state.current_stage, 'result': tool_result_data})}\n\n"

        final_state = state.model_dump()
        final_state["status"] = state.status.value
        yield f"data: {json.dumps({'type': 'completed', 'state': final_state})}\n\n"

    def run_turn_loop(self, prompt: str, limit: int = 5, max_turns: int = 5, structured_output: bool = True) -> list[TurnResult]:
        engine = QueryEnginePort.from_workspace()
        results: list[TurnResult] = []
        
        # Initialize Webtoon Pipeline State
        state = AgentState()
        previous_cut_assets = {}
        context_mgr = WebtoonContextManager()
        
        system_instructions = (
            "당신은 PrompToon의 총괄 웹툰 연출가이자 'Cut Architect'입니다.\n"
            "5단계 파이프라인(Story Creator -> Story Reviewer -> Cut Architect -> Deterministic Compile -> Polish)을 관리합니다.\n"
            "현재 상태(AgentState)를 확인하고, 다음에 호출할 도구(tool_to_use)와 파라미터(tool_payload)를 JSON으로 반환하세요.\n"
            "\n"
            "[STAGE 1: validate_story_json — Story Creator]\n"
            "- 시놉시스를 바탕으로 전체 스토리를 생성합니다.\n"
            "- 반드시 아래의 [TEXT TAG FORMAT]을 지켜야 합니다. JSON 괄호는 절대 쓰지 마세요.\n"
            "\n"
            "[TEXT TAG FORMAT]\n"
            "# META\n"
            "TITLE: <제목>\n"
            "SYNOPSIS: <시놉시스>\n"
            "... (GENRE, SUMMARY 등)\n\n"
            "# PHASE 1\n"
            "[NARRATION] <내용>\n"
            "[IMAGE] <이미지 묘사>\n"
            "[DIALOGUE | 인물명] <대사>\n\n"
            "# PHASE 1 CHOICE\n"
            "QUESTION: <질문>\n"
            "IMAGE_DESCRIPTION: <이미지 묘사>\n"
            "---\n"
            "OPTION 1: 레이블 | 서브텍스트 | 반응 | [MAP: 0, 0, 0, 0]\n"
            "OPTION 2: 레이블 | 서브텍스트 | 반응 | [MAP: 0, 0, 0, 0]\n\n"
            "(PHASE 2, 3 동일 반복)\n\n"
            "# ENDING POSTER\n"
            "TITLE_KO: <한국어 제목>\n"
            "IMAGE_DESCRIPTION: <포스터 묘사>\n\n"
            "# ENDING LIST\n"
            "- ID: a | CONDITION: <조건> | BADGE: <배지> | TAGLINE: <문구> | LINES: l1, l2, l3, l4\n\n"
            "# INTERFACE\n"
            "BUTTONS: 버튼1, 버튼2\n"
            "BRAND_TEXT: <브랜드 문구>\n"
            "\n"
            "출력 형식 (JSON strictly for Director Response):\n"
            "{\n"
            "  \"thought\": \"현재 상황 추론\",\n"
            "  \"tool_to_use\": \"도구 이름\",\n"
            "  \"tool_payload\": \"위의 [TEXT TAG FORMAT]에 따른 텍스트 본문 (JSON string 필드에 담으세요)\"\n"
            "}"
        )

        # 에이전트가 이전 턴에서 무엇을 했는지 기억하기 위한 히스토리 변수
        agent_history_context = ""

        for turn in range(max_turns):
            if state.status != StageStatus.RUNNING:
                break
            
            pinned_context = ""
            if state.current_stage == 3:
                mock_current_cut = {"inheritsFromCutId": "PREVIOUS_CUT"}
                pinned_context = inject_inheritance_context(mock_current_cut, previous_cut_assets)
            
            # 이전 턴의 실행 결과를 프롬프트에 누적하여 에이전트가 같은 도구를 반복하지 않도록 함
            state_info = f"\n[Current State: Stage {state.current_stage}, Status: {state.status.value}]\n{agent_history_context}"
            turn_prompt = context_mgr.assemble_messages(prompt + state_info, pinned_context)
            
            # LLM 호출
            agent_response = ask_agentic_llm_json(system_instructions, turn_prompt)
            thought = agent_response.get("thought", "Processing...")
            tool_name = agent_response.get("tool_to_use", "none").lower()

            # 서버 터미널에서 에이전트의 생각을 실시간으로 보기 위한 출력
            print(f"\n🧠 [Turn {turn+1}] ----------------------")
            print(f"🤔 Thought: {thought}")
            print(f"🛠️ Tool: {tool_name}")
            print("--------------------------------------\n")

            payload = agent_response.get("tool_payload", {})
            
            output_msg = f"\n=== Turn {turn+1} ===\n[Director's Thought] {thought}\n[Selected Tool] {tool_name}\n"
            tool_result_data = None
            
            # 도구 실행 분기
            if tool_name == "validate_story_json":
                try:
                    from .tools import parse_text_to_story_json
                    raw_text = payload if isinstance(payload, str) else json.dumps(payload)
                    parsed_payload = parse_text_to_story_json(raw_text)
                    
                    story_obj = StoryJSON(**parsed_payload)
                    tool_result_data = validate_story_json(story_obj)
                    tool_result_data["story"] = parsed_payload
                    if tool_result_data.get("valid"):
                        state.current_stage = 2
                        state.story_data = parsed_payload
                except Exception as e:
                    tool_result_data = {"valid": False, "error": str(e)}
            
            elif tool_name == "run_story_reviewer":
                review_input = payload if payload else state.story_data or prompt
                if isinstance(payload, dict) and "text" in payload and len(payload) == 1:
                    review_input = payload.get("text", prompt)
                review_res = run_story_reviewer(review_input)
                tool_result_data = review_res.model_dump(mode="json")
                
                if not review_res.passed:
                    state.status = StageStatus.REVIEW_FAILED
                    review_summary = " | ".join(review_res.warnings) if review_res.warnings else "Review failed."
                    output_msg += f"\n[CRITICAL] Stage 2 Quality Review FAILED: {review_summary}\n"
                else:
                    state.current_stage = 3
                    previous_cut_assets = {
                        "location_anchor": "Detected location from review",
                        "character_dna": {"trait": "Consistent from description"}
                    }
            
            elif tool_name == "run_deterministic_compiler":
                try:
                    compiler_obj = CompilerInputs.model_validate(payload)
                    compiled_preview = run_deterministic_compiler(
                        compiler_obj.shared_assets,
                        compiler_obj.relational_cuts,
                    )
                    story_meta = (state.story_data or {}).get("meta", {})
                    image_prompts = assemble_image_prompt_artifact(
                        compiler_obj.shared_assets,
                        compiler_obj.relational_cuts,
                        compiled_preview,
                        story_title=story_meta.get("title"),
                        genre=story_meta.get("genre"),
                    )
                    tool_result_data = {
                        "compiled_preview": compiled_preview.model_dump(mode="json"),
                        "image_prompts": image_prompts.model_dump(mode="json", by_alias=True),
                    }
                    output_msg += f"\n--- Image Prompt Artifact ---\n{image_prompts.model_dump_json(indent=2, by_alias=True)}\n"
                    state.current_stage = 5
                    state.status = StageStatus.COMPLETED
                except Exception as e:
                    tool_result_data = {"error": str(e)}
            
            elif tool_name == "none" or tool_name == "error":
                 # 모델이 할 일을 마쳤거나 에러가 났을 때 루프 탈출
                 output_msg += "\n[System] Agent elected to stop or encountered an error."
                 break 
            else:
                output_msg += f"\n[System] Unknown tool requested: {tool_name}"

            # 방금 실행한 도구의 결과를 agent_history_context에 누적 (다음 턴에서 모델이 이걸 읽음)
            execution_record = f"Past Action - Executed {tool_name}. Result: {json.dumps(tool_result_data, ensure_ascii=False)}\n"
            agent_history_context += execution_record

            turn_result = TurnResult(
                prompt=turn_prompt,
                output=output_msg + f"\n[Tool Result] {json.dumps(tool_result_data, ensure_ascii=False)}",
                matched_commands=(),
                matched_tools=(tool_name,) if tool_name != "none" else (),
                permission_denials=(),
                usage={},
                stop_reason="tool_use"
            )
            results.append(turn_result)

        # 루프 종료 후 최종 상태 요약
        if results:
            last_result = results[-1]
            state_summary = f"\n\n--- Inheritance Engine Final State ---\n{state.model_dump_json(indent=2)}"
            
            output_msg = last_result.output
            if state.status == StageStatus.REVIEW_FAILED:
                output_msg += "\n[SYSTEM] Stage 2 Quality Review FAILED. Stopping pipeline."
                
            results[-1] = TurnResult(
                prompt=last_result.prompt,
                output=output_msg + state_summary,
                matched_commands=last_result.matched_commands,
                matched_tools=last_result.matched_tools,
                permission_denials=last_result.permission_denials,
                usage=last_result.usage,
                stop_reason="completed"
            )
            
        return results

    def _infer_permission_denials(self, matches: list[RoutedMatch]) -> list[PermissionDenial]:
        denials: list[PermissionDenial] = []
        for match in matches:
            if match.kind == 'tool' and 'bash' in match.name.lower():
                denials.append(PermissionDenial(tool_name=match.name, reason='destructive shell execution remains gated in the Python port'))
        return denials

    def _collect_matches(self, tokens: set[str], modules: tuple[PortingModule, ...], kind: str) -> list[RoutedMatch]:
        matches: list[RoutedMatch] = []
        for module in modules:
            score = self._score(tokens, module)
            if score > 0:
                matches.append(RoutedMatch(kind=kind, name=module.name, source_hint=module.source_hint, score=score))
        matches.sort(key=lambda item: (-item.score, item.name))
        return matches

    @staticmethod
    def _score(tokens: set[str], module: PortingModule) -> int:
        haystacks = [module.name.lower(), module.source_hint.lower(), module.responsibility.lower()]
        score = 0
        for token in tokens:
            if any(token in haystack for haystack in haystacks):
                score += 1
        return score
