import importlib.util
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


face_lock = _load("face_lock_under_test", ROOT / "core" / "face_lock.py")
START = "严格保持参考图1中人物的脸、五官和发型不变。"
END = "再次强调：人物的脸必须与参考图1一致。"


def test_apply_face_lock_wraps_start_and_end_idempotently():
    once = face_lock.apply_face_lock("拍一张自拍")
    assert once.split("\n")[0] == START
    assert once.split("\n")[-1] == END
    assert face_lock.apply_face_lock(once) == once


def test_relock_moves_end_line_after_appended_note():
    locked = face_lock.apply_face_lock("拍一张自拍") + "\n\n参考图用途：猫"
    fixed = face_lock.relock(locked)
    assert fixed.split("\n")[0] == START
    assert fixed.split("\n")[-1] == END
    assert "参考图用途：猫" in fixed
    assert fixed.count(START) == 1 and fixed.count(END) == 1


def test_relock_leaves_plain_edits_untouched():
    assert face_lock.relock("把背景换成海边") == "把背景换成海边"


def test_cast_lines_single_and_multi():
    one = face_lock.face_lock_for_cast("p", [{"reference_indices": [1, 2]}])
    assert one.startswith("严格保持参考图1、2中人物")
    many = face_lock.face_lock_for_cast(
        "p", [{"reference_indices": [1]}, {"reference_indices": [2]}]
    )
    assert many.startswith("严格保持每位人物")
    assert many.endswith("再次强调：每位人物的脸必须与其对应身份参考图一致。")
    assert face_lock.face_lock_for_cast("p", []) == "p"


@pytest.mark.asyncio
async def test_gemini_edit_puts_images_before_text_and_keeps_face_lines():
    helper = _load(
        "gemini_edit_auth_helper", ROOT / "tests" / "test_gemini_edit_auth_headers.py"
    )
    mod = helper._load_module()
    backend = mod.GeminiEditBackend(
        imgr=object(), settings={"api_keys": ["k"], "api_url": "https://x"}
    )
    backend._request = AsyncMock(side_effect=RuntimeError("stop"))
    jpeg = b"\xff\xd8\xff\xe0" + b"0" * 32
    prompt = face_lock.apply_face_lock("自拍")
    with pytest.raises(RuntimeError):
        await backend.edit(prompt, [jpeg, jpeg], resolution="1K")
    parts = backend._request.call_args.args[0]
    assert "inlineData" in parts[1] and "inlineData" in parts[3]
    assert parts[1]["inlineData"]["mimeType"] == "image/jpeg"
    assert "inlineData" not in parts[-1]
    last_text = parts[-1]["text"]
    assert last_text.split("\n")[0] == START
    assert last_text.split("\n")[-1] == END
    first_image = next(i for i, p in enumerate(parts) if "inlineData" in p)
    assert all(
        "inlineData" in p or i < first_image or "Reference image" in p["text"]
        for i, p in enumerate(parts[:-1])
    )


@pytest.mark.asyncio
async def test_gemini_edit_plain_edit_has_no_face_lines():
    helper = _load(
        "gemini_edit_auth_helper2", ROOT / "tests" / "test_gemini_edit_auth_headers.py"
    )
    mod = helper._load_module()
    backend = mod.GeminiEditBackend(
        imgr=object(), settings={"api_keys": ["k"], "api_url": "https://x"}
    )
    backend._request = AsyncMock(side_effect=RuntimeError("stop"))
    with pytest.raises(RuntimeError):
        await backend.edit("把背景换成海边", [b"\xff\xd8\xff\xe0" + b"0" * 32])
    parts = backend._request.call_args.args[0]
    assert "inlineData" in parts[1]
    assert "严格保持" not in parts[-1]["text"]


@pytest.mark.parametrize(
    "body",
    [
        "严格保持人物1 吱吱与人物2 木有知各自的脸部特征。双唇温柔贴合轻吻嘴唇。",
        "严格保持参考图1中人物 1（吱吱）的脸不变。侧躺伸腿特写。",
        f"{START}下午在咖啡馆轻吻嘴唇。",
        f"咖啡馆亲吻合照。\n{END}另外要拍出双腿曲线。",
    ],
)
def test_apply_face_lock_never_strips_user_prompt_sharing_prefix(body):
    single = face_lock.apply_face_lock(body)
    assert single.split("\n")[0] == START
    assert single.split("\n")[-1] == END
    assert body in single
    assert face_lock.strip_face_lock(single) == body
    assert face_lock.relock(single + "\n\n参考图用途：姿势") != ""
    assert body in face_lock.relock(single + "\n\n参考图用途：姿势")

    cast = [{"reference_indices": [1]}, {"reference_indices": [2]}]
    multi = face_lock.face_lock_for_cast(body + "\n\n参考图用途：\n参考图 1", cast)
    assert body in multi
    assert multi.split("\n")[0].startswith("严格保持每位人物")
    assert multi.split("\n")[-1].startswith("再次强调：每位人物")


def test_strip_face_lock_only_removes_standalone_wrapper_lines():
    wrapped = face_lock.apply_face_lock("窗边轻吻合照\n\n细节描述")
    assert face_lock.strip_face_lock(wrapped) == "窗边轻吻合照\n\n细节描述"
    raw = "严格保持人物1的脸部特征，咖啡馆轻吻"
    assert face_lock.strip_face_lock(raw) == raw


@pytest.mark.asyncio
async def test_gemini_edit_preserves_single_line_strict_prefix_body():
    helper = _load(
        "gemini_edit_auth_helper3", ROOT / "tests" / "test_gemini_edit_auth_headers.py"
    )
    mod = helper._load_module()
    backend = mod.GeminiEditBackend(
        imgr=object(), settings={"api_keys": ["k"], "api_url": "https://x"}
    )
    backend._request = AsyncMock(side_effect=RuntimeError("stop"))
    jpeg = b"\xff\xd8\xff\xe0" + b"0" * 32
    body = "严格保持人物1 吱吱与人物2 木有知各自的脸部特征。双唇温柔贴合轻吻嘴唇。"
    prompt = face_lock.face_lock_for_cast(
        body, [{"reference_indices": [1]}, {"reference_indices": [2]}]
    )
    with pytest.raises(RuntimeError):
        await backend.edit(prompt, [jpeg, jpeg], resolution="1K")
    last_text = backend._request.call_args.args[0][-1]["text"]
    assert body in last_text
    assert last_text.split("\n")[0].startswith("严格保持每位人物")
    assert last_text.split("\n")[-1].startswith("再次强调：每位人物")
