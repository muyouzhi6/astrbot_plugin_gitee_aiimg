from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from test_background_pipeline_event import _Event
from test_main_initialize_request_mode import _load_module


class CommandEvent(_Event):
    def __init__(self, main, text, *, group=True, image_first=False, extra_text=""):
        super().__init__(group=group)
        self.message_str = text + extra_text
        self.is_at_or_wake_command = True
        self.should_call_llm = Mock()
        self.chain = ([main.Image()] if image_first else []) + [main.Plain(text)]
        if extra_text:
            self.chain += [main.Image(), main.Plain(extra_text)]

    def get_messages(self):
        return self.chain

    def is_private_chat(self):
        return not self._group


@pytest.fixture
def runtime(monkeypatch):
    main, _ = _load_module()
    plugin = object.__new__(main.GiteeAIImagePlugin)
    plugin._wake_prefixes = ("/", ".", "!!")
    plugin.config = {"features": {"selfie": {"enabled": True}}}
    plugin.registry = SimpleNamespace(
        provider_ids=lambda: ["美年达sd", "美年达画图", "other"]
    )
    plugin.draw = SimpleNamespace(
        generate=AsyncMock(return_value=Path("/tmp/draw.png"))
    )
    plugin.edit = SimpleNamespace(
        get_preset_names=lambda: [], edit=AsyncMock(return_value=Path("/tmp/edit.png"))
    )
    plugin.debouncer = SimpleNamespace(hit=Mock(return_value=False))
    plugin._begin_user_job = AsyncMock(return_value=True)
    plugin._end_user_job = AsyncMock()
    plugin._remember_last_image = Mock()
    plugin._send_image_with_fallback = AsyncMock(return_value=True)
    plugin._save_last_image_task_meta = AsyncMock()
    plugin._generate_selfie_image_with_meta = AsyncMock(
        return_value=(Path("/tmp/selfie.png"), {"mode": "selfie_ref"})
    )
    monkeypatch.setattr(main, "mark_failed", AsyncMock())
    monkeypatch.setattr(main, "mark_processing", AsyncMock())
    monkeypatch.setattr(main, "mark_success", AsyncMock())
    monkeypatch.setattr(
        main,
        "get_images_from_event",
        AsyncMock(
            return_value=[
                SimpleNamespace(convert_to_base64=AsyncMock(return_value="c291cmNl"))
            ]
        ),
    )
    monkeypatch.setattr(main, "decode_base64_image_payload", lambda value: b"source")
    return main, plugin


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["aiimg", "aiedit", "自拍"])
@pytest.mark.parametrize(
    "prefix,group,image_first",
    [("/", True, False), (".", True, True), ("!!", True, False), ("", False, False)],
)
async def test_compact_commands_reach_the_selected_service_once(
    runtime, command, prefix, group, image_first
):
    main, plugin = runtime
    event = CommandEvent(
        main,
        f"{prefix}{command}美年达sd\t窗边的人像\n蓝色外套 3:4 4K",
        group=group,
        image_first=image_first,
    )
    assert main.ImageCommandWakePrefixFilter().filter(
        event, {"wake_prefix": list(plugin._wake_prefixes)}
    )
    await plugin.provider_image_command(event)
    await plugin.provider_image_command(event)
    if command == "aiimg":
        call = plugin.draw.generate.await_args
        assert call.args == ("窗边的人像\n蓝色外套",)
        assert call.kwargs["provider_id"] == "美年达sd"
        plugin.draw.generate.assert_awaited_once()
        plugin.edit.edit.assert_not_awaited()
    elif command == "aiedit":
        call = plugin.edit.edit.await_args
        assert call.kwargs["prompt"] == "窗边的人像\n蓝色外套"
        assert call.kwargs["backend"] == "美年达sd"
        assert call.kwargs["images"] == [b"source"]
        plugin.edit.edit.assert_awaited_once()
        plugin.draw.generate.assert_not_awaited()
    else:
        call = plugin._generate_selfie_image_with_meta.await_args
        assert call.args == (event, "窗边的人像\n蓝色外套", "美年达sd")
        plugin._generate_selfie_image_with_meta.assert_awaited_once()
        plugin.draw.generate.assert_not_awaited()
        plugin.edit.edit.assert_not_awaited()
    assert call.kwargs["output_intent"].aspect_ratio == "3:4"
    assert call.kwargs["output_intent"].resolution == "4K"
    plugin._send_image_with_fallback.assert_awaited_once()
    plugin._begin_user_job.assert_awaited_once()
    plugin._end_user_job.assert_awaited_once()
    main.mark_failed.assert_not_awaited()
    assert event.is_stopped()


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["aiimg", "aiedit", "自拍"])
async def test_compact_channel_is_not_replaced_by_a_prompt_mention(runtime, command):
    main, plugin = runtime
    event = CommandEvent(main, f"/{command}美年达sd @other 保留文字")
    await plugin.provider_image_command(event)
    if command == "aiimg":
        call = plugin.draw.generate.await_args
        assert call.args[0] == "@other 保留文字"
        assert call.kwargs["provider_id"] == "美年达sd"
    elif command == "aiedit":
        call = plugin.edit.edit.await_args
        assert call.kwargs["prompt"] == "@other 保留文字"
        assert call.kwargs["backend"] == "美年达sd"
    else:
        call = plugin._generate_selfie_image_with_meta.await_args
        assert call.args[1:] == ("@other 保留文字", "美年达sd")


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["aiimg", "aiedit", "自拍"])
async def test_unknown_channel_reports_error_without_requesting_default(
    runtime, command
):
    main, plugin = runtime
    event = CommandEvent(main, f"/{command}不存在 提示词")
    await plugin.provider_image_command(event)
    assert event.sent_result.chain[0].text == "未配置图片渠道: 不存在"
    plugin._begin_user_job.assert_not_awaited()
    plugin.draw.generate.assert_not_awaited()
    plugin.edit.edit.assert_not_awaited()
    plugin._generate_selfie_image_with_meta.assert_not_awaited()
    event.should_call_llm.assert_called_once_with(False)
    assert event.is_stopped()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        "/自拍参考 查看",
        "聊天提到 /aiimg美年达sd 猫",
        "aiimg美年达sd 猫",
        "/aiimg 猫",
        "/aiedit @美年达sd 猫",
        "/批量3 aiimg美年达sd 猫",
        "/aiimg美年达sdExtra 猫",
    ],
)
async def test_other_commands_and_chat_do_not_make_image_requests(runtime, text):
    main, plugin = runtime
    event = CommandEvent(main, text)
    await plugin.provider_image_command(event)
    plugin.draw.generate.assert_not_awaited()
    plugin.edit.edit.assert_not_awaited()
    plugin._generate_selfie_image_with_meta.assert_not_awaited()


@pytest.mark.asyncio
async def test_provider_command_respects_prefix_gate(runtime):
    main, plugin = runtime
    gate = main.ImageCommandWakePrefixFilter()
    plugin._wake_prefixes = (".",)
    for text in ("/aiimg美年达sd 猫", "aiimg美年达sd 猫", "聊天提到 .aiimg美年达sd 猫"):
        event = CommandEvent(main, text)
        assert not gate.filter(event, {"wake_prefix": ["."]})
        await plugin.provider_image_command(event)
    plugin.draw.generate.assert_not_awaited()


@pytest.mark.asyncio
async def test_split_text_after_image_keeps_all_prompt_parts(runtime):
    main, plugin = runtime
    event = CommandEvent(
        main, "/aiedit美年达sd 换背景", image_first=True, extra_text="保留人物 16:9"
    )
    await plugin.provider_image_command(event)
    assert plugin.edit.edit.await_args.kwargs["prompt"] == "换背景\n保留人物"
    assert plugin.edit.edit.await_args.kwargs["output_intent"].aspect_ratio == "16:9"


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["aiimg", "aiedit", "自拍"])
async def test_legacy_at_provider_with_whitespace_keeps_working(runtime, command):
    main, plugin = runtime
    prompt = "@美年达画图\t橘猫\n蓝色背景 16:9"
    event = CommandEvent(main, f"/{command}\n{prompt}")
    if command == "aiimg":
        await plugin.generate_image_command(event, "@美年达画图")
        assert plugin.draw.generate.await_args.kwargs["provider_id"] == "美年达画图"
        assert plugin.draw.generate.await_args.args[0] == "橘猫\n蓝色背景"
    elif command == "aiedit":
        await plugin.edit_image_default(event, "@美年达画图")
        assert plugin.edit.edit.await_args.kwargs["backend"] == "美年达画图"
        assert plugin.edit.edit.await_args.kwargs["prompt"] == "橘猫\n蓝色背景"
    else:
        await plugin.selfie_command(event)
        assert plugin._generate_selfie_image_with_meta.await_args.args[1:] == (
            "橘猫\n蓝色背景",
            "美年达画图",
        )


@pytest.mark.asyncio
async def test_explicit_provider_failure_is_terminal_and_releases_job(runtime):
    main, plugin = runtime
    plugin.draw.generate.side_effect = RuntimeError("upstream unavailable")
    event = CommandEvent(main, "/aiimg美年达sd 猫")
    await plugin.provider_image_command(event)
    plugin.draw.generate.assert_awaited_once()
    plugin._end_user_job.assert_awaited_once()
    plugin._send_image_with_fallback.assert_not_awaited()
    main.mark_failed.assert_awaited_once()
    assert event.is_stopped()
