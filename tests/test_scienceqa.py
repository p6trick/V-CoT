from vcot_deepeyes.scienceqa import has_image, normalize_answer, normalize_choices


def test_image_filter_rejects_missing_image() -> None:
    assert not has_image({"image": None})
    assert has_image({"image": {"path": "sample.png", "bytes": None}})


def test_normalizes_scienceqa_choice_answer() -> None:
    row = {"choices": ["red", "blue"], "answer": 1}
    choices = normalize_choices(row)
    assert normalize_answer(row, choices) == {"raw": 1, "index": 1, "label": "B", "text": "blue"}
