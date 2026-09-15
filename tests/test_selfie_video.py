import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from test_studio import picture
from test_studio import runtime as studio_runtime

runtime = studio_runtime


@pytest.mark.asyncio
async def test_selfie_frame_precedes_video_and_only_generated_frame_is_passed(
    runtime, monkeypatch, tmp_path
):
    main, _, _, plugin, _ = runtime
    order = []
    frame = tmp_path / "frame.png"
    frame.write_bytes(picture("green"))
    snapshot = (
        [picture("blue")],
        "frozen face and schedule outfit",
        {
            "default_output": "3:4 4K",
            "chain_override": [{"provider_id": "photo"}],
            "task_types": ["id"],
        },
        {},
    )

    async def edit(**kwargs):
        order.append("selfie")
        assert kwargs["images"] == snapshot[0]
        assert kwargs["prompt"] == snapshot[1]
        assert kwargs["default_output"] == "3:4 4K"
        assert kwargs["infer_source_aspect"] is False
        return frame

    async def video(**kwargs):
        order.append("video")
        assert kwargs["image_bytes"] == picture("green")
        assert kwargs["image_bytes"] != snapshot[0][0]
        return "https://cdn.example/dance.mp4"

    async def send(*args, **kwargs):
        order.append("send")

    plugin.edit = SimpleNamespace(edit=edit)
    plugin.registry = SimpleNamespace(
        get_video_backend=lambda _: SimpleNamespace(generate_video_url=video)
    )
    plugin.background_tasks = None
    plugin._remember_last_image = lambda event, path: None
    plugin._send_video_result = send
    plugin._video_end = AsyncMock()
    monkeypatch.setattr(main, "mark_success", AsyncMock())
    monkeypatch.setattr(main, "mark_failed", AsyncMock())
    await plugin._async_generate_video(
        SimpleNamespace(),
        "dance naturally",
        "user",
        provider_id="agnes",
        selfie_snapshot=snapshot,
    )
    assert order == ["selfie", "video", "send"]
    plugin._video_end.assert_awaited_once_with("user")


@pytest.mark.asyncio
async def test_selfie_failure_never_creates_text_video(runtime, monkeypatch):
    main, _, _, plugin, _ = runtime
    plugin.edit = SimpleNamespace(
        edit=AsyncMock(side_effect=RuntimeError("photo failed"))
    )
    plugin.registry = SimpleNamespace(get_video_backend=AsyncMock())
    plugin.background_tasks = None
    plugin._video_end = AsyncMock()
    monkeypatch.setattr(main, "mark_failed", AsyncMock())
    await plugin._async_generate_video(
        SimpleNamespace(),
        "dance",
        "user",
        provider_id="agnes",
        selfie_snapshot=([picture("blue")], "start", {}, {}),
    )
    plugin.registry.get_video_backend.assert_not_called()
    main.mark_failed.assert_awaited_once()
    plugin._video_end.assert_awaited_once()


@pytest.mark.asyncio
async def test_tool_accepts_one_selfie_video_task_and_freezes_preparation(
    runtime, monkeypatch
):
    main, _, _, plugin, _ = runtime
    plugin.config["features"]["video"] = {"enabled": True, "llm_tool_enabled": True}
    plugin.config["features"]["selfie"] = {"enabled": True, "llm_tool_enabled": True}
    plugin.debouncer = SimpleNamespace(hit=lambda _: False)
    plugin._debounce_key = lambda *args: "request"
    plugin._video_begin = AsyncMock(return_value=True)
    plugin._video_end = AsyncMock()
    plugin._video_tasks = set()
    prepared = ([picture("red")], "frozen schedule", {}, {})
    plugin._prepare_background_selfie = AsyncMock(return_value=prepared)
    plugin._capture_video_image_snapshot = AsyncMock()
    plugin._async_generate_video = AsyncMock()
    monkeypatch.setattr(main, "mark_processing", AsyncMock())
    event = SimpleNamespace(get_sender_id=lambda: "user")
    await plugin.grok_generate_video(
        event,
        "slow dance",
        mode="selfie",
        selfie_prompt="full body standing on a clear floor",
    )
    await asyncio.gather(*plugin._video_tasks)
    plugin._prepare_background_selfie.assert_awaited_once()
    assert "full body" in plugin._prepare_background_selfie.call_args.args[1]
    plugin._capture_video_image_snapshot.assert_not_awaited()
    plugin._async_generate_video.assert_awaited_once()
    assert plugin._async_generate_video.call_args.kwargs["selfie_snapshot"] == prepared


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure", [ValueError("missing identity"), asyncio.CancelledError()]
)
async def test_missing_selfie_identity_rejects_before_video_submission(
    runtime, monkeypatch, failure
):
    main, _, _, plugin, _ = runtime
    plugin.config["features"]["video"] = {"enabled": True, "llm_tool_enabled": True}
    plugin.config["features"]["selfie"] = {"enabled": True, "llm_tool_enabled": True}
    plugin.debouncer = SimpleNamespace(hit=lambda _: False)
    plugin._debounce_key = lambda *args: "request"
    plugin._video_begin = AsyncMock(return_value=True)
    plugin._video_end = AsyncMock()
    plugin._prepare_background_selfie = AsyncMock(side_effect=failure)
    plugin._async_generate_video = AsyncMock()
    plugin._signal_llm_tool_failure = AsyncMock()
    monkeypatch.setattr(main, "mark_processing", AsyncMock())
    if isinstance(failure, asyncio.CancelledError):
        with pytest.raises(asyncio.CancelledError):
            await plugin.grok_generate_video(
                SimpleNamespace(get_sender_id=lambda: "user"), "dance", mode="selfie"
            )
    else:
        await plugin.grok_generate_video(
            SimpleNamespace(get_sender_id=lambda: "user"), "dance", mode="selfie"
        )
    plugin._async_generate_video.assert_not_awaited()
    plugin._video_end.assert_awaited_once()
