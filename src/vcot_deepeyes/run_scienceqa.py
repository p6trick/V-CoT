"""CLI for a bounded, streaming ScienceQA clean DeepEyes baseline."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import yaml

from .runner import OpenAIChatClient, run_deepeyes_episode
from .scienceqa import iter_image_test_samples, normalize_answer, normalize_choices


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    config: dict[str, Any] = yaml.safe_load(args.config.read_text())
    model_config, dataset_config, output_config = config["model"], config["dataset"], config["output"]
    api_key = os.environ.get(model_config["api_key_env"], "EMPTY")
    client = OpenAIChatClient(model_config["api_base"], api_key)
    output_dir = Path(output_config["directory"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "results.jsonl"

    with output_file.open("w", encoding="utf-8") as handle:
        for sample_index, row in enumerate(iter_image_test_samples(
            dataset_id=dataset_config["id"],
            split=dataset_config["split"],
            max_samples=dataset_config["max_samples"],
            streaming=dataset_config["streaming"],
        )):
            choices = normalize_choices(row)
            episode_dir = output_dir / f"sample_{sample_index:04d}" if output_config.get("save_crops") else None
            result = run_deepeyes_episode(
                client=client,
                image=row["image"], question=row["question"], choices=choices,
                model=model_config["id"], temperature=model_config["temperature"],
                max_tokens_per_turn=model_config["max_tokens_per_turn"], max_turns=model_config["max_turns"],
                output_dir=episode_dir,
            )
            result.update({"sample_index": sample_index, "sample_id": row.get("id"), "gold_answer": normalize_answer(row, choices)})
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()


if __name__ == "__main__":
    main()
