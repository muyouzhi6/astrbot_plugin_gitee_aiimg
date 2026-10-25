import sys
import types
from pathlib import Path

import pytest

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


def test_parse_edit_command_with_provider_and_resolution_suffix():
    mod = _load_module()

    parsed = mod.parse_image_request(
        "/aiedit @provider_id 把背景换成夜景街道 4K",
        known_provider_ids={"provider_id"},
    )

    assert parsed is not None
    assert parsed.spec.provider_id == "provider_id"
    assert parsed.spec.user_prompt == "把背景换成夜景街道"
    assert parsed.spec.output == "4K"


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


@pytest.mark.parametrize(
    "command,mode", [("aiimg", "draw"), ("aiedit", "edit"), ("自拍", "selfie_ref")]
)
@pytest.mark.parametrize("prefix,count", [("/", 1), (".", 1), ("", 1), ("/批量3 ", 3)])
def test_compact_provider_commands_preserve_prompt_output_and_batch(
    command, mode, prefix, count
):
    mod = _load_module()
    parsed = mod.parse_image_request(
        f"{prefix}{command}美年达sd\t窗边的人像\n保持蓝色外套 3:4 4K",
        known_provider_ids={"美年达", "美年达sd"},
    )
    assert parsed.batch_count == count
    assert parsed.spec.mode == mode
    assert parsed.spec.provider_id == "美年达sd"
    assert parsed.spec.user_prompt == "窗边的人像\n保持蓝色外套"
    assert parsed.spec.output == "3:4 4K"


@pytest.mark.parametrize(
    "token", ["aiimg美年达不存在", "aiedit美年达不存在", "自拍美年达不存在", "自拍参考"]
)
def test_unknown_compact_provider_is_not_routed_to_default(token):
    mod = _load_module()
    assert (
        mod.parse_image_request(f"/{token} 文字", known_provider_ids={"美年达"}) is None
    )


def test_compact_provider_does_not_consume_prompt_provider_mentions():
    mod = _load_module()
    parsed = mod.parse_image_request(
        "/aiedit美年达sd @other 保留文字", known_provider_ids={"美年达sd", "other"}
    )
    assert parsed.spec.provider_id == "美年达sd"
    assert parsed.spec.user_prompt == "@other 保留文字"


def test_existing_preset_name_takes_precedence_over_compact_syntax():
    mod = _load_module()
    parsed = mod.parse_image_request(
        "/自拍照 额外要求", edit_presets={"自拍照": "preset"}, known_provider_ids={"照"}
    )
    assert parsed.spec.preset_name == "自拍照"
    assert parsed.spec.provider_id is None


def test_selfie_reference_management_is_reserved_even_with_matching_provider():
    mod = _load_module()
    assert (
        mod.parse_image_request("/自拍参考 查看", known_provider_ids={"参考"}) is None
    )
