import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import core.debouncer as debouncer_module
from test_background_pipeline_event import _Event, _plugin
from test_main_initialize_request_mode import _load_module


@pytest.fixture
def runtime(monkeypatch):
    main, _ = _load_module()
    plugin = _plugin(main, None)
    clock = [0.0]
    monkeypatch.setattr(
        debouncer_module, "time", SimpleNamespace(time=lambda: clock[0])
    )
    plugin.debouncer = debouncer_module.Debouncer(plugin.config)
    plugin._concurrency_lock = asyncio.Lock()
    plugin._image_inflight = {}
    plugin._video_inflight = {}
    plugin._character_binding_error = lambda *args: ""
    plugin._prepare_context_reference_request = AsyncMock(return_value=None)
    plugin._background_manager_for_event = lambda event: None
    plugin._has_message_images = AsyncMock(return_value=False)
    plugin._signal_llm_tool_failure = AsyncMock()
    plugin.draw = SimpleNamespace(generate=AsyncMock())
    for name in ("mark_processing", "mark_success", "mark_failed"):
        monkeypatch.setattr(main, name, AsyncMock())
    return main, plugin, clock


@pytest.mark.asyncio
@pytest.mark.parametrize("duration", [121, 601])
@pytest.mark.parametrize("message_id", ["source-message", ""])
async def test_slow_failed_provider_runs_once_for_same_event(
    runtime, duration, message_id
):
    main, plugin, clock = runtime
    event = _Event()
    event.message_obj.message_id = message_id

    async def fail_provider(*args, **kwargs):
        clock[0] += duration
        raise RuntimeError("provider timeout")

    plugin.draw.generate.side_effect = fail_provider
    first = await plugin.aiimg_generate(event, prompt="portrait", mode="text")
    second = await plugin.aiimg_generate(event, prompt="portrait", mode="text")

    assert "failed and has ended" in first.content[0].text
    assert "already handled" in second.content[0].text
    assert plugin.draw.generate.await_count == 1
    plugin._signal_llm_tool_failure.assert_awaited_once()
    main.mark_success.assert_not_awaited()
    assert plugin._image_inflight == {}

    new_event = _Event()
    new_event.message_obj.message_id = "new-user-message"
    await plugin.aiimg_generate(new_event, prompt="portrait", mode="text")
    assert plugin.draw.generate.await_count == 2


@pytest.mark.asyncio
async def test_slow_single_failure_does_not_accept_batch_for_same_event(runtime):
    _, plugin, clock = runtime
    event = _Event()

    async def fail_provider(*args, **kwargs):
        clock[0] += 601
        raise RuntimeError("provider timeout")

    plugin.draw.generate.side_effect = fail_provider
    await plugin.aiimg_generate(event, prompt="portrait", mode="text")
    plugin._background_manager_for_event = lambda current_event: object()
    plugin._resolve_llm_batch_mode = AsyncMock(return_value="draw")
    plugin._accept_background_batch = AsyncMock(return_value="accepted")

    result = await plugin.aiimg_batch_generate(
        event, prompt="four portraits", count=4, mode="text"
    )

    assert "already handled" in result.content[0].text
    plugin._accept_background_batch.assert_not_awaited()


@pytest.mark.asyncio
async def test_slow_batch_failure_does_not_call_single_provider(runtime):
    _, plugin, clock = runtime
    event = _Event()
    plugin._background_manager_for_event = lambda current_event: object()
    plugin._resolve_llm_batch_mode = AsyncMock(return_value="draw")

    async def fail_acceptance(*args, **kwargs):
        clock[0] += 601
        raise RuntimeError("queue timeout")

    plugin._accept_background_batch = AsyncMock(side_effect=fail_acceptance)
    first = await plugin.aiimg_batch_generate(
        event, prompt="four portraits", count=4, mode="text"
    )
    plugin._background_manager_for_event = lambda current_event: None
    result = await plugin.aiimg_generate(event, prompt="portrait", mode="text")

    assert "could not be prepared" in first.content[0].text
    assert "already handled" in result.content[0].text
    plugin.draw.generate.assert_not_awaited()


@pytest.mark.asyncio
async def test_character_argument_correction_still_uses_same_event(runtime):
    _, plugin, _ = runtime
    event = _Event()
    plugin._character_binding_error = lambda event, prompt, ids, mode, roles: (
        "missing character IDs" if prompt == "invalid" else ""
    )
    plugin._background_manager_for_event = lambda current_event: object()
    plugin._accept_background_single = AsyncMock(return_value="accepted")

    first = await plugin.aiimg_generate(event, prompt="invalid", mode="text")
    second = await plugin.aiimg_generate(event, prompt="corrected", mode="text")

    assert "missing character IDs" in first.content[0].text
    assert second == "accepted"
    plugin._accept_background_single.assert_awaited_once()
