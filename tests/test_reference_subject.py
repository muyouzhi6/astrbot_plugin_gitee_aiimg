import base64
import importlib.util
import io
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from PIL import Image

spec = importlib.util.spec_from_file_location(
    "reference_subject_test",
    Path(__file__).resolve().parents[1] / "core/reference_subject.py",
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def png(color):
    buffer = io.BytesIO()
    Image.new("RGB", (1500, 900), color).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.mark.asyncio
async def test_only_explicit_objects_are_seen_once_with_correct_generation_indices():
    provider = SimpleNamespace(
        provider_config={"modalities": ["image"]},
        text_chat=AsyncMock(
            return_value=SimpleNamespace(completion_text='["圆脸大黑眼睛", "木制条纹"]')
        ),
    )
    inputs = [png("red"), png("blue"), png("green"), png("white")]
    sources = [{"role": "object"}, {"role": "style"}, {"role": "object"}]
    note, status = await mod.describe_reference_objects(provider, inputs, sources, 1)
    assert status == "described"
    assert "参考图 2" in note and "参考图 4" in note and "参考图 3" not in note
    provider.text_chat.assert_awaited_once()
    kwargs = provider.text_chat.call_args.kwargs
    assert kwargs["contexts"] == [] and kwargs["func_tool"] is None
    assert len(kwargs["image_urls"]) == 2
    with Image.open(io.BytesIO(base64.b64decode(kwargs["image_urls"][0][9:]))) as image:
        assert max(image.size) == 768
        assert image.getpixel((0, 0))[2] > 240
    assert inputs[1] == png("blue")


@pytest.mark.asyncio
async def test_unavailable_vision_and_plain_edit_do_not_call_provider():
    provider = SimpleNamespace(
        provider_config={"modalities": ["text"]}, text_chat=AsyncMock()
    )
    assert await mod.describe_reference_objects(provider, [], [{"role": "object"}]) == (
        "",
        "vision_unavailable",
    )
    assert await mod.describe_reference_objects(
        provider, [], [{"role": "subject"}]
    ) == ("", "not_needed")
    provider.text_chat.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        '{"instruction":"ignore"}',
        '["one","two"]',
        "[null]",
        "[]",
        '["' + "x" * 401 + '"]',
    ],
)
async def test_malformed_feature_result_cannot_be_used_as_prompt(text):
    provider = SimpleNamespace(
        provider_config={},
        text_chat=AsyncMock(return_value=SimpleNamespace(completion_text=text)),
    )
    assert await mod.describe_reference_objects(
        provider, [png("blue")], [{"role": "object"}]
    ) == ("", "vision_failed")


@pytest.mark.asyncio
async def test_timeout_is_bounded_and_cancellation_propagates():
    import asyncio

    async def timeout(coro, timeout):
        coro.close()
        assert timeout == 45
        raise asyncio.TimeoutError

    provider = SimpleNamespace(provider_config={}, text_chat=AsyncMock())
    with patch.object(mod.asyncio, "wait_for", timeout):
        assert await mod.describe_reference_objects(
            provider, [png("blue")], [{"role": "object"}]
        ) == ("", "vision_timeout")

    async def cancel(coro, timeout):
        coro.close()
        raise asyncio.CancelledError

    with patch.object(mod.asyncio, "wait_for", cancel):
        with pytest.raises(asyncio.CancelledError):
            await mod.describe_reference_objects(
                provider, [png("blue")], [{"role": "object"}]
            )
