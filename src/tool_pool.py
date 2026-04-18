from __future__ import annotations

from dataclasses import dataclass

from .models import PortingModule
from .permissions import ToolPermissionContext
from .tools import get_tools


@dataclass(frozen=True)
class ToolPool:
    tools: tuple[PortingModule, ...]

    def as_markdown(self) -> str:
        lines = [
            '# Webtoon Tool Pool',
            '',
            f'Tool count: {len(self.tools)}',
        ]
        lines.extend(f'- {tool.name} — {tool.responsibility}' for tool in self.tools)
        return '\n'.join(lines)


def assemble_tool_pool(
    simple_mode: bool = False,
    include_mcp: bool = True,
    permission_context: ToolPermissionContext | None = None,
) -> ToolPool:
    # We ignore simple_mode and include_mcp as our pipeline tools are the only ones active.
    return ToolPool(
        tools=get_tools(simple_mode=simple_mode, include_mcp=include_mcp, permission_context=permission_context),
    )
