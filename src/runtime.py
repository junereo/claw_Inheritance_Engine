from __future__ import annotations
import json

from dataclasses import dataclass

from .commands import PORTED_COMMANDS
from .llm_client import ask_agentic_llm_json
from .tools import validate_story_json, run_story_reviewer, run_deterministic_compiler, StoryJSON, CompilerInputs, ReviewResult
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

    def run_turn_loop(self, prompt: str, limit: int = 5, max_turns: int = 5, structured_output: bool = True) -> list[TurnResult]:
        engine = QueryEnginePort.from_workspace()
        results: list[TurnResult] = []
        
        # Initialize Webtoon Pipeline State
        state = AgentState()
        previous_cut_assets = {}
        context_mgr = WebtoonContextManager()
        
        system_instructions = (
            "당신은 PrompToon의 총괄 웹툰 연출가이자 'Cut Architect'입니다.\n"
            "당신은 5단계 웹툰 제작 파이프라인(Story Creator -> Story Reviewer -> Cut Architect -> Deterministic Compile -> Polish)을 관리합니다.\n"
            "현재 상태(AgentState)를 확인하고, 다음에 호출할 도구(tool_to_use)와 그 파라미터(tool_payload)를 JSON 형식으로 반환하세요.\n"
            "더 이상 호출할 도구가 없거나 파이프라인이 완료되었다면 tool_to_use를 'none'으로 반환하세요.\n"
            "\n"
            "사용 가능한 도구 및 Payload 규칙:\n"
            "1. validate_story_json: [Stage 1 -> 2 진행용]\n"
            "   - 당신이 직접 사용자의 시놉시스를 바탕으로 3-Phase 스토리를 창작하여 payload로 전달합니다.\n"
            "   - payload 형식: {\"meta\": {\"genre\": \"\"}, \"phases\": [{\"name\": \"\", \"description\": \"\", \"events\": []}, ...총 3개...], \"ending\": \"\"}\n"
            "2. run_story_reviewer: [Stage 2 -> 3 진행용]\n"
            "   - 창작된 스토리의 품질을 검수합니다.\n"
            "   - payload 형식: {\"text\": \"전체 스토리 요약 텍스트\"}\n"
            "3. run_deterministic_compiler: [Stage 3 -> 5 진행용]\n"
            "   - 캐릭터/배경 자산과 컷 관계를 조립해 프롬프트를 생성합니다.\n"
            "   - payload 형식: {\"shared_assets\": {\"characters\": {}, \"locations\": {}}, \"relational_cuts\": [{\"character\": \"\", \"action\": \"\"}]}\n"
            "\n"
            "출력 형식 (JSON strictly):\n"
            "{\n"
            "  \"thought\": \"현재 단계와 실패/성공 여부에 따른 다음 행동 추론\",\n"
            "  \"tool_to_use\": \"도구 이름 또는 'none'\",\n"
            "  \"tool_payload\": { 도구에 전달할 데이터 }\n"
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
                    story_obj = StoryJSON(**payload)
                    tool_result_data = validate_story_json(story_obj)
                    if tool_result_data.get("valid"):
                        state.current_stage = 2
                except Exception as e:
                    tool_result_data = {"valid": False, "error": str(e)}
            
            elif tool_name == "run_story_reviewer":
                text_to_review = payload.get("text", prompt)
                review_res = run_story_reviewer(text_to_review)
                tool_result_data = {"passed": review_res.passed, "report": review_res.report}
                
                if not review_res.passed:
                    state.status = StageStatus.REVIEW_FAILED
                    output_msg += f"\n[CRITICAL] Stage 2 Quality Review FAILED: {review_res.report}\n"
                else:
                    state.current_stage = 3
                    previous_cut_assets = {
                        "location_anchor": "Detected location from review",
                        "character_dna": {"trait": "Consistent from description"}
                    }
            
            elif tool_name == "run_deterministic_compiler":
                try:
                    compiled_prompts = run_deterministic_compiler(
                        payload.get("shared_assets", {}), 
                        payload.get("relational_cuts", [])
                    )
                    tool_result_data = {"prompts": compiled_prompts}
                    output_msg += f"\n--- Compiled Prompts ---\n{compiled_prompts}\n"
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
