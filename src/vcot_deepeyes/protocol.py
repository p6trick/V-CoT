"""DeepEyes V2 tool prompt and trace-friendly protocol helpers.

The system prompt and turn policy mirror the public DeepEyes evaluation demo
(`eval/eval_vstar.py`).  Keeping this protocol separate prevents accidentally
falling back to one-shot Qwen2.5-VL inference.
"""

from __future__ import annotations

import json
import re
from typing import Any

SYSTEM_PROMPT_V2 = """You are a helpful assistant.

# Tools
You may call one or more functions to assist with the user query.
You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type":"function","function":{"name":"image_zoom_in_tool","description":"Zoom in on a specific region of an image by cropping it based on a bounding box (bbox) and an optional object label.","parameters":{"type":"object","properties":{"bbox_2d":{"type":"array","items":{"type":"number"},"minItems":4,"maxItems":4,"description":"The bounding box of the region to zoom in, as [x1, y1, x2, y2], where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner."},"label":{"type":"string","description":"The name or label of the object in the specified bounding box (optional)."}},"required":["bbox"]}}}
</tools>

# How to call a tool
Return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

**Example**:
<tool_call>
{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [10, 20, 100, 200], "label": "the apple on the desk"}}
</tool_call>"""

USER_PROMPT_V2 = (
    "\nThink first, call **image_zoom_in_tool** if needed, then answer. "
    "Format strictly as: <think>...</think> <tool_call>...</tool_call> "
    "(if tools needed) <answer>...</answer> "
)

TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def format_question(question: str, choices: list[str]) -> str:
    options = "\n".join(f"{chr(65 + index)}. {choice}" for index, choice in enumerate(choices))
    return f"Question: {question}\nOptions:\n{options}\n{USER_PROMPT_V2}"


def parse_last_tool_call(text: str) -> dict[str, Any] | None:
    """Return the last syntactically valid DeepEyes tool call, if present."""
    matches = TOOL_CALL_RE.findall(text)
    if not matches:
        return None
    try:
        value = json.loads(matches[-1].strip())
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def extract_answer(text: str) -> str | None:
    matches = ANSWER_RE.findall(text)
    return matches[-1].strip() if matches else None
