import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "core" / "llm_batch_planner.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("llm_batch_planner_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_planned_prompt_items_from_code_fence():
    mod = _load_module()

    items = mod.parse_planned_prompt_items(
        """```json
[
  {"title":"正面微笑","prompt":"prompt-a","variation_focus":["pose","expression"],"aspect_ratio":"3:4"},
  {"title":"侧身回头","prompt":"prompt-b","variation_focus":["angle"],"aspect_ratio":"16:9"}
]
```"""
    )

    assert len(items) == 2
    assert items[0].title == "正面微笑"
    assert items[0].aspect_ratio == "3:4"
    assert items[1].prompt == "prompt-b"


def test_validate_planned_prompt_items_rejects_duplicates():
    mod = _load_module()

    items = [
        mod.PlannedPromptItem(
            title="正面微笑",
            prompt="same prompt",
            variation_focus=[],
            aspect_ratio="3:4",
        ),
        mod.PlannedPromptItem(
            title="正面微笑",
            prompt="same prompt",
            variation_focus=[],
            aspect_ratio="16:9",
        ),
    ]

    error = mod.validate_planned_prompt_items(items, expected_count=2)

    assert error is not None


def test_validate_planned_prompt_items_requires_mixed_ratios_without_fixed_ratio():
    mod = _load_module()
    items = [
        mod.PlannedPromptItem("近景", "prompt-a", [], "3:4"),
        mod.PlannedPromptItem("全身", "prompt-b", [], "3:4"),
    ]

    assert (
        mod.validate_planned_prompt_items(items, expected_count=2)
        == "batch planner must use at least two aspect ratios"
    )


def test_validate_planned_prompt_items_honors_fixed_ratio():
    mod = _load_module()
    items = [
        mod.PlannedPromptItem("近景", "prompt-a", [], "16:9"),
        mod.PlannedPromptItem("全身", "prompt-b", [], "16:9"),
    ]

    assert (
        mod.validate_planned_prompt_items(
            items,
            expected_count=2,
            fixed_aspect_ratio="16:9",
        )
        is None
    )


def test_build_batch_prompt_requests_per_item_aspect_ratios():
    mod = _load_module()

    adaptive = mod.build_batch_planning_prompt(
        mode="selfie_ref",
        user_prompt="拍几张看看你",
        count=3,
    )
    fixed = mod.build_batch_planning_prompt(
        mode="selfie_ref",
        user_prompt="拍三张横屏照",
        count=3,
        fixed_aspect_ratio="16:9",
    )

    assert '"aspect_ratio": "3:4"' in adaptive
    assert "整组至少使用两种比例" in adaptive
    assert "aspect_ratio 必须固定为 16:9" in fixed
