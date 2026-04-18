from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class PortContext:
    source_root: Path
    tests_root: Path
    assets_root: Path
    archive_root: Path
    python_file_count: int
    test_file_count: int
    asset_file_count: int
    archive_available: bool


def build_port_context(base: Path | None = None) -> PortContext:
    root = base or Path(__file__).resolve().parent.parent
    source_root = root / 'src'
    tests_root = root / 'tests'
    assets_root = root / 'assets'
    archive_root = root / 'archive' / 'claude_code_ts_snapshot' / 'src'
    return PortContext(
        source_root=source_root,
        tests_root=tests_root,
        assets_root=assets_root,
        archive_root=archive_root,
        python_file_count=sum(1 for path in source_root.rglob('*.py') if path.is_file()),
        test_file_count=sum(1 for path in tests_root.rglob('*.py') if path.is_file()),
        asset_file_count=sum(1 for path in assets_root.rglob('*') if path.is_file()),
        archive_available=archive_root.exists(),
    )


def render_context(context: PortContext) -> str:
    return '\n'.join([
        f'Source root: {context.source_root}',
        f'Test root: {context.tests_root}',
        f'Assets root: {context.assets_root}',
        f'Archive root: {context.archive_root}',
        f'Python files: {context.python_file_count}',
        f'Test files: {context.test_file_count}',
        f'Assets: {context.asset_file_count}',
        f'Archive available: {context.archive_available}',
    ])


# --- Webtoon Inheritance Chain Context ---

def inject_inheritance_context(current_cut: Dict[str, Any], previous_cut_assets: Dict[str, Any]) -> str:
    """
    Constructs a pinned context block for visual continuity between cuts.
    """
    if "inheritsFromCutId" not in current_cut:
        return ""
    
    inherited_id = current_cut["inheritsFromCutId"]
    location = previous_cut_assets.get("location_anchor", "N/A")
    character_dna = previous_cut_assets.get("character_dna", {})
    
    dna_str = ", ".join([f"{k}: {v}" for k, v in character_dna.items()]) if character_dna else "Standard DNA"
    
    pinned_block = [
        f"### 🔗 INHERITANCE CONTEXT (From Cut: {inherited_id})",
        f"**Location Anchor**: {location}",
        f"**Character DNA**: {dna_str}",
        "---",
        "**Instruction**: 상기 'Location Anchor'와 'Character DNA'를 엄격히 준수하여 비주얼 일관성을 유지하라."
    ]
    
    return "\n".join(pinned_block)


class WebtoonContextManager:
    """
    Manages the assembly of prompt messages with pinned context blocks.
    """
    @staticmethod
    def assemble_messages(base_prompt: str, pinned_context: str = "") -> str:
        if not pinned_context:
            return base_prompt
            
        return f"{pinned_context}\n\n{base_prompt}"
