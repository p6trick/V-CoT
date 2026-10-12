"""Client-side reproduction of the official DeepEyes multi-turn zoom loop."""

from __future__ import annotations

import base64
import hashlib
import io
import time
from pathlib import Path
from typing import Any, Protocol

from PIL import Image

from .protocol import SYSTEM_PROMPT_V2, USER_PROMPT_V2, extract_answer, format_question, parse_last_tool_call


class ChatClient(Protocol):
    def complete(self, messages: list[dict[str, Any]], **kwargs: Any) -> tuple[str, int | None]: ...


class OpenAIChatClient:
    def __init__(self, api_base: str, api_key: str) -> None:
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key, base_url=api_base)

    def complete(self, messages: list[dict[str, Any]], **kwargs: Any) -> tuple[str, int | None]:
        response = self.client.chat.completions.create(messages=messages, **kwargs)
        usage = getattr(response, "usage", None)
        generated_tokens = getattr(usage, "completion_tokens", None) if usage else None
        return response.choices[0].message.content or "", generated_tokens


def image_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _crop(image: Image.Image, bbox: list[float | int]) -> Image.Image:
    if len(bbox) != 4:
        raise ValueError("bbox_2d must contain exactly four coordinates")
    return image.crop(tuple(bbox))


def run_deepeyes_episode(
    *,
    client: ChatClient,
    image: Image.Image,
    question: str,
    choices: list[str],
    model: str,
    temperature: float,
    max_tokens_per_turn: int,
    max_turns: int,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Execute official-style model → crop → model turns and return complete trace."""
    prompt = format_question(question, choices)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT_V2},
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": image_data_url(image)}}, {"type": "text", "text": prompt}]},
    ]
    trace: list[dict[str, Any]] = []
    total_tokens: int | None = 0
    started_at = time.perf_counter()
    final_response = ""
    status = "max_turns"

    for turn_index in range(max_turns):
        response, generated_tokens = client.complete(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens_per_turn,
            stop=["<|im_end|>"],
        )
        final_response = response
        if generated_tokens is None:
            total_tokens = None
        elif total_tokens is not None:
            total_tokens += generated_tokens
        turn: dict[str, Any] = {"turn": turn_index, "assistant_response": response, "generated_tokens": generated_tokens}
        tool_call = parse_last_tool_call(response)
        if tool_call is None:
            trace.append(turn)
            status = "answered" if extract_answer(response) is not None else "stopped_without_tool_or_answer"
            break

        try:
            if tool_call.get("name") != "image_zoom_in_tool":
                raise ValueError("unsupported tool name")
            arguments = tool_call["arguments"]
            bbox = arguments["bbox_2d"]
            cropped = _crop(image, bbox)
            crop_bytes = io.BytesIO()
            cropped.save(crop_bytes, format="PNG")
            tool_record: dict[str, Any] = {
                "name": "image_zoom_in_tool",
                "arguments": arguments,
                "status": "success",
                "crop_size": list(cropped.size),
                "crop_sha256": hashlib.sha256(crop_bytes.getvalue()).hexdigest(),
            }
            if output_dir is not None:
                output_dir.mkdir(parents=True, exist_ok=True)
                crop_path = output_dir / f"turn_{turn_index:02d}_zoom.png"
                crop_path.write_bytes(crop_bytes.getvalue())
                tool_record["crop_path"] = str(crop_path)
            turn["tool"] = tool_record
            messages.extend([
                {"role": "assistant", "content": response},
                {"role": "user", "content": [
                    {"type": "text", "text": "<tool_response>"},
                    {"type": "image_url", "image_url": {"url": image_data_url(cropped)}},
                    {"type": "text", "text": USER_PROMPT_V2},
                    {"type": "text", "text": "</tool_response>"},
                ]},
            ])
        except (KeyError, TypeError, ValueError) as error:
            turn["tool"] = {"status": "error", "error": str(error), "raw_call": tool_call}
            trace.append(turn)
            status = "invalid_tool_call"
            break
        trace.append(turn)

    latency_seconds = time.perf_counter() - started_at
    return {
        "question": question,
        "choices": choices,
        "initial_user_prompt": prompt,
        "protocol": "DeepEyes_eval_vstar_V2_image_zoom_in_tool",
        "model_response": extract_answer(final_response) or final_response,
        "final_raw_response": final_response,
        "generated_token_count": total_tokens,
        "reasoning_turn_count": len(trace),
        "zoom_tool_call_count": sum(1 for turn in trace if turn.get("tool", {}).get("status") == "success"),
        "latency_seconds": latency_seconds,
        "status": status,
        "trace": trace,
    }
