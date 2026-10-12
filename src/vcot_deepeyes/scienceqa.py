"""Streaming ScienceQA test loader with an explicit image-only filter."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any


def has_image(row: dict[str, Any]) -> bool:
    """Support decoded PIL images and datasets Image feature dictionaries."""
    image = row.get("image")
    if image is None:
        return False
    if isinstance(image, dict):
        return bool(image.get("bytes") or image.get("path"))
    return hasattr(image, "size")


def iter_image_test_samples(
    dataset_id: str,
    split: str,
    max_samples: int,
    streaming: bool = True,
) -> Iterator[dict[str, Any]]:
    """Yield at most ``max_samples`` image-bearing samples without materializing test."""
    from datasets import load_dataset

    dataset = load_dataset(dataset_id, split=split, streaming=streaming)
    emitted = 0
    for row in dataset:
        if not has_image(row):
            continue
        yield row
        emitted += 1
        if emitted >= max_samples:
            return


def normalize_choices(row: dict[str, Any]) -> list[str]:
    choices = row.get("choices", row.get("options", []))
    if not isinstance(choices, list) or not all(isinstance(item, str) for item in choices):
        raise ValueError("ScienceQA sample has no string choices/options field")
    return choices


def normalize_answer(row: dict[str, Any], choices: list[str]) -> dict[str, Any]:
    raw = row.get("answer")
    index = raw if isinstance(raw, int) else None
    label = chr(65 + index) if index is not None and 0 <= index < len(choices) else None
    return {"raw": raw, "index": index, "label": label, "text": choices[index] if label else None}
