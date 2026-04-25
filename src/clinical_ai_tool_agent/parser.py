"""ReAct output parser.

ChatOllama (TinyLLaMA in particular) does not support native function-
calling. The agent uses a text-based ReAct protocol where the model
emits either:

    Action: <tool_name>
    Action Input: {"key": "value"}

or:

    Final Answer: <free text>

`parse_action_or_final` extracts whichever one is present.
"""

from __future__ import annotations

import json
import re
from typing import Optional


ACTION_RE = re.compile(
    r"(?s)Action:\s*([A-Za-z0-9_\-]+)\s*\n\s*Action Input:\s*(\{.*?\})"
)
FINAL_RE = re.compile(r"(?s)Final Answer:\s*(.+?)(?:\n\s*Action:|\Z)")


def parse_action_or_final(
    text: str,
) -> tuple[Optional[str], Optional[dict], Optional[str]]:
    """Return `(tool_name, tool_args, final_answer)`. Exactly one is non-None
    when parsing succeeds; all three are None when the text contains neither.
    Final Answer takes precedence if both appear.
    """
    if not text:
        return None, None, None

    final_match = FINAL_RE.search(text)
    if final_match:
        return None, None, final_match.group(1).strip()

    action_match = ACTION_RE.search(text)
    if not action_match:
        return None, None, None

    name = action_match.group(1).strip()
    raw_args = action_match.group(2).strip()
    try:
        args = json.loads(raw_args)
    except json.JSONDecodeError:
        args = None
    return name, args, None
