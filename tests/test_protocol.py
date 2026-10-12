from vcot_deepeyes.protocol import extract_answer, parse_last_tool_call


def test_parses_last_valid_deepeyes_tool_call() -> None:
    output = '<tool_call>{"name":"old", "arguments":{}}</tool_call><tool_call>{"name":"image_zoom_in_tool", "arguments":{"bbox_2d":[1,2,3,4]}}</tool_call>'
    assert parse_last_tool_call(output)["name"] == "image_zoom_in_tool"


def test_extracts_tagged_answer() -> None:
    assert extract_answer("<think>x</think><answer>B</answer>") == "B"
