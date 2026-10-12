from pathlib import Path

from PIL import Image

from vcot_deepeyes.runner import run_deepeyes_episode


class FakeClient:
    def __init__(self) -> None:
        self.calls = 0

    def complete(self, messages, **kwargs):
        self.calls += 1
        if self.calls == 1:
            return '<think>zoom</think><tool_call>{"name":"image_zoom_in_tool","arguments":{"bbox_2d":[0,0,8,8],"label":"corner"}}</tool_call>', 7
        return '<think>done</think><answer>A</answer>', 3


def test_preserves_multiturn_zoom_and_trace(tmp_path: Path) -> None:
    result = run_deepeyes_episode(
        client=FakeClient(), image=Image.new("RGB", (16, 16)), question="Q?", choices=["yes", "no"],
        model="fake", temperature=0.0, max_tokens_per_turn=10, max_turns=4, output_dir=tmp_path,
    )
    assert result["model_response"] == "A"
    assert result["generated_token_count"] == 10
    assert result["reasoning_turn_count"] == 2
    assert result["zoom_tool_call_count"] == 1
    assert (tmp_path / "turn_00_zoom.png").is_file()
