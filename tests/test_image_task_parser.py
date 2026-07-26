import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "core" / "image_task_parser.py"
PACKAGE_NAME = "image_task_parser_testpkg"
CORE_PACKAGE_NAME = f"{PACKAGE_NAME}.core"
MODULE_NAME = f"{CORE_PACKAGE_NAME}.image_task_parser"


def _load_module():
    import importlib.util

    for name in list(sys.modules):
        if name.startswith(PACKAGE_NAME):
            sys.modules.pop(name, None)

    package = types.ModuleType(PACKAGE_NAME)
    package.__path__ = [str(ROOT)]
    sys.modules[PACKAGE_NAME] = package
    core_package = types.ModuleType(CORE_PACKAGE_NAME)
    core_package.__path__ = [str(ROOT / "core")]
    sys.modules[CORE_PACKAGE_NAME] = core_package

    spec = importlib.util.spec_from_file_location(MODULE_NAME, MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


def test_parse_draw_preset_command_with_extra_prompt():
    mod = _load_module()

    parsed = mod.parse_image_request(
        "/文生图 手办 让她站在便利店门口\n低头看手机",
        draw_presets={"手办": "Transform into figurine style"},
        edit_presets={},
        known_provider_ids=set(),
    )

    assert parsed is not None
    assert parsed.batch_count == 1
    assert parsed.spec.mode == "draw"
    assert parsed.spec.preset_name == "手办"
    assert parsed.spec.user_prompt == "让她站在便利店门口\n低头看手机"
    assert parsed.spec.effective_prompt == (
        "Transform into figurine style\n\n补充要求：\n让她站在便利店门口\n低头看手机"
    )


def test_parse_batch_compact_prefix_and_edit_preset():
    mod = _load_module()

    parsed = mod.parse_image_request(
        "/批量8 手办化 加金色边框",
        draw_presets={},
        edit_presets={"手办化": "Transform into figurine style"},
        known_provider_ids=set(),
    )

    assert parsed is not None
    assert parsed.batch_count == 8
    assert parsed.spec.mode == "edit"
    assert parsed.spec.preset_name == "手办化"
    assert parsed.spec.effective_prompt == "Transform into figurine style, 加金色边框"


def test_parse_draw_command_with_provider_override():
    mod = _load_module()

    parsed = mod.parse_image_request(
        "/文生图 @grok_chat 手办 一张柜台前的展示图",
        draw_presets={"手办": "Transform into figurine style"},
        edit_presets={},
        known_provider_ids={"grok_chat"},
    )

    assert parsed is not None
    assert parsed.spec.provider_id == "grok_chat"
    assert parsed.spec.preset_name == "手办"
    assert parsed.spec.effective_prompt.endswith("一张柜台前的展示图")


def test_parse_batch_draw_output_suffix_without_touching_prompt_body():
    mod = _load_module()

    parsed = mod.parse_image_request(
        "/批量3 aiimg 画一张 16:9 的电影海报 16:9 4K",
        known_provider_ids=set(),
    )

    assert parsed is not None
    assert parsed.batch_count == 3
    assert parsed.spec.user_prompt == "画一张 16:9 的电影海报"
    assert parsed.spec.output == "16:9 4K"
