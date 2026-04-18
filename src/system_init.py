from __future__ import annotations

import json
from typing import Any

from .commands import built_in_command_names, get_commands
from .setup import run_setup
from .tools import get_tools


def get_system_prompt() -> str:
    """
    Returns the system prompt for the Inheritance Engine (PromToon Persona).
    """
    persona = "너는 PromToon의 총괄 웹툰 연출가이자 Cut Architect다."
    
    pipeline_stages = [
        "Stage 1: Story Creator (시놉시스 -> 3-Phase Story JSON)",
        "Stage 2: Story Reviewer (품질 검수. Fail 시 즉시 중단)",
        "Stage 3: Cut Architect (Shared Assets, Relational Cuts 도출)",
        "Stage 4: Deterministic Compile (수식 기반 프롬프트 조립 - Tool 호출)",
        "Stage 5: Prompt Polish (자연어 다듬기)"
    ]
    
    hard_gate_rule = (
        "Stage 2(Story Reviewer)에서 passed=False가 나오면 더 이상 도구를 호출하지 말고 "
        "즉시 상태를 'review_failed'로 설정 후 최종 응답을 반환하라."
    )
    
    output_format = "최종 산출물은 반드시 JSON 형태로 반환하라."
    
    system_message = [
        f"## Persona\n{persona}",
        "\n## 5-Stage Pipeline",
        *[f"- {stage}" for stage in pipeline_stages],
        f"\n## Hard Gate Rule\n{hard_gate_rule}",
        f"\n## Output Requirement\n{output_format}",
        "\n## Operational Instructions",
        "1. 모든 단계는 순차적으로 진행되어야 한다.",
        "2. 품질 검수(Stage 2)는 가장 엄격한 기준을 적용한다.",
        "3. JSON 응답 외에 추가적인 자연어 설명은 최소화하고, 모든 데이터 구조를 정확히 지켜야 한다."
    ]
    
    return "\n".join(system_message)


def build_system_init_message(trusted: bool = True) -> str:
    setup = run_setup(trusted=trusted)
    commands = get_commands()
    tools = get_tools()
    
    system_prompt = get_system_prompt()
    
    lines = [
        '# System Init (Inheritance Engine)',
        '',
        system_prompt,
        '',
        '---',
        f'Trusted: {setup.trusted}',
        f'Built-in command names: {len(built_in_command_names())}',
        f'Loaded command entries: {len(commands)}',
        f'Loaded tool entries: {len(tools)}',
        '',
        'Startup steps:',
        *(f'- {step}' for step in setup.setup.startup_steps()),
    ]
    return '\n'.join(lines)
