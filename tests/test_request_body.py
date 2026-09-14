from copy import deepcopy

import pytest

from core.request_body import merge_request_body, request_field


def test_nested_merge_preserves_siblings_and_isolates_requests():
    payload = {"generationConfig": {"imageConfig": {"imageSize": "4K"}}}
    extra = {"generationConfig": {"temperature": 0.4}, "quality": "high"}
    original = deepcopy((payload, extra))
    result = merge_request_body(payload, extra)
    assert result["generationConfig"]["imageConfig"]["imageSize"] == "4K"
    assert result["generationConfig"]["temperature"] == 0.4
    assert result["quality"] == "high"
    result["generationConfig"]["imageConfig"]["imageSize"] = "1K"
    assert (payload, extra) == original


@pytest.mark.parametrize("invalid", ["{}", [], 3, True])
def test_invalid_request_options_fail_explicitly(invalid):
    with pytest.raises(ValueError, match="JSON object"):
        merge_request_body({}, invalid)


@pytest.mark.parametrize(
    "value,expected",
    [(True, "true"), (3, "3"), ({"x": 1}, '{"x": 1}'), ("high", "high")],
)
def test_form_fields_keep_json_types(value, expected):
    assert request_field(value) == expected
