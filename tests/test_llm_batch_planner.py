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
  {"title":"正面微笑","prompt":"prompt-a","variation_focus":["pose","expression"]},
  {"title":"侧身回头","prompt":"prompt-b","variation_focus":["angle"]}
]
```"""
    )

    assert len(items) == 2
    assert items[0].title == "正面微笑"
    assert items[1].prompt == "prompt-b"


def test_validate_planned_prompt_items_rejects_duplicates():
    mod = _load_module()

    items = [
        mod.PlannedPromptItem(title="正面微笑", prompt="same prompt", variation_focus=[]),
        mod.PlannedPromptItem(title="正面微笑", prompt="same prompt", variation_focus=[]),
    ]

    error = mod.validate_planned_prompt_items(items, expected_count=2)

    assert error is not None
