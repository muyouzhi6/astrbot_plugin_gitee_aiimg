import io

import pytest
from PIL import Image as PILImage

from core.output_spec import (
    OutputIntent,
    detect_aspect_ratio_from_image,
    format_output_intent,
    merge_output_intents,
    parse_output,
    parse_output_intent,
    select_allowed_size,
    split_prompt_output_suffix,
)


def test_parse_output_intent_supports_adaptive_and_exact_forms():
    adaptive = parse_output_intent("16:9 4k")
    exact = parse_output_intent("2048×1152")

    assert adaptive == OutputIntent(aspect_ratio="16:9", resolution="4K")
    assert format_output_intent(adaptive) == "16:9 4K"
    assert exact == OutputIntent(exact_size="2048x1152")
    assert parse_output("4K") == (None, "4K")
    assert parse_output("2048x1152") == ("2048x1152", None)


def test_parse_output_intent_rejects_conflicts():
    with pytest.raises(ValueError, match="exact size cannot be combined"):
        parse_output_intent("2048x1152 16:9")
    with pytest.raises(ValueError, match="conflicting resolution"):
        parse_output_intent("2K 4K")


def test_split_prompt_output_suffix_only_consumes_trailing_controls():
    prompt, intent = split_prompt_output_suffix("画一张 16:9 的电影海报 4K")
    untouched, empty = split_prompt_output_suffix("画一张 4K 电影海报")
    limited_prompt, limited_intent = split_prompt_output_suffix(
        "画一张海报 1:1 16:9 4K"
    )

    assert prompt == "画一张 16:9 的电影海报"
    assert intent == OutputIntent(resolution="4K")
    assert untouched == "画一张 4K 电影海报"
    assert empty.is_empty
    assert limited_prompt == "画一张海报 1:1"
    assert limited_intent == OutputIntent(aspect_ratio="16:9", resolution="4K")


def test_merge_output_intents_fills_only_missing_adaptive_fields():
    merged = merge_output_intents(
        OutputIntent(aspect_ratio="16:9"),
        OutputIntent(resolution="4K"),
        OutputIntent(aspect_ratio="1:1", resolution="1K"),
    )
    exact = merge_output_intents(
        OutputIntent(exact_size="2048x1152"),
        OutputIntent(aspect_ratio="1:1", resolution="1K"),
    )

    assert merged == OutputIntent(aspect_ratio="16:9", resolution="4K")
    assert exact == OutputIntent(exact_size="2048x1152")


def test_select_allowed_size_uses_ratio_and_resolution_target():
    allowed = ["1024x1024", "2048x2048", "1024x576", "2048x1152"]

    assert (
        select_allowed_size(
            OutputIntent(aspect_ratio="16:9", resolution="4K"),
            allowed,
            default_size="1024x1024",
        )
        == "2048x1152"
    )
    assert (
        select_allowed_size(
            OutputIntent(resolution="1K"),
            allowed,
            default_size="1024x1024",
        )
        == "1024x1024"
    )


def test_detect_aspect_ratio_from_image_matches_common_ratio():
    output = io.BytesIO()
    PILImage.new("RGB", (1600, 900), "white").save(output, format="PNG")

    assert detect_aspect_ratio_from_image(output.getvalue()) == "16:9"
    assert detect_aspect_ratio_from_image(b"not-an-image") is None
