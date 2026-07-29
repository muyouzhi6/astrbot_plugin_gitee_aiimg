import io

import pytest
from PIL import Image as PILImage

from core.output_spec import (
    OutputIntent,
    detect_aspect_ratio_from_image,
    extract_output_intent_from_prompt,
    format_output_intent,
    merge_output_intents,
    parse_output,
    parse_output_intent,
    resolve_gpt_image_2_size,
    resolve_llm_output_intent,
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


def test_extract_output_intent_from_natural_prompt():
    assert extract_output_intent_from_prompt(
        "电影感海边日落, 画面比例 16:9, 输出 4K 高清"
    ) == OutputIntent(aspect_ratio="16:9", resolution="4K")
    assert extract_output_intent_from_prompt("横屏海报 2048x1152") == OutputIntent(
        exact_size="2048x1152"
    )
    assert extract_output_intent_from_prompt("普通方形图") == OutputIntent()


def test_resolve_llm_output_intent_prefers_explicit_prompt_controls():
    assert resolve_llm_output_intent(
        "电影海报, 画面比例 16:9",
        output="1024x1024",
    ) == OutputIntent(aspect_ratio="16:9", resolution="1K")


def test_resolve_llm_output_intent_uses_structured_fields_and_ignores_auto():
    assert resolve_llm_output_intent(
        "竖屏人像",
        output="auto",
        aspect_ratio="9:16",
        resolution="2K",
    ) == OutputIntent(aspect_ratio="9:16", resolution="2K")
    assert resolve_llm_output_intent(
        "普通图片",
        output="default",
        aspect_ratio="auto",
        resolution="auto",
    ) == OutputIntent()


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

    default_size = merge_output_intents(
        OutputIntent(aspect_ratio="16:9"),
        OutputIntent(exact_size="1024x1024"),
    )
    assert default_size == OutputIntent(aspect_ratio="16:9", resolution="1K")


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


def test_resolve_gpt_image_2_size_maps_adaptive_controls():
    assert (
        resolve_gpt_image_2_size(
            OutputIntent(aspect_ratio="16:9", resolution="4K")
        )
        == "3840x2160"
    )
    assert (
        resolve_gpt_image_2_size(
            OutputIntent(aspect_ratio="3:4", resolution="2K")
        )
        == "1536x2048"
    )


def test_resolve_gpt_image_2_size_preserves_exact_and_rejects_unknown_ratio():
    assert (
        resolve_gpt_image_2_size(OutputIntent(exact_size="1280x720"))
        == "1280x720"
    )
    assert (
        resolve_gpt_image_2_size(
            OutputIntent(aspect_ratio="7:5", resolution="2K")
        )
        is None
    )
