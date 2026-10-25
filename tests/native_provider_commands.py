"""Exercise channel commands through AstrBot's real wake and handler filters."""

# ruff: noqa: E402
import asyncio
import importlib
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

ROOT = Path(__file__).resolve().parents[1]
os.umask(0o077)
Path(os.environ["ASTRBOT_ROOT"]).mkdir(parents=True, exist_ok=True)
os.chdir(os.environ["ASTRBOT_ROOT"])
sys.path.insert(0, str(ROOT.parent))

from astrbot.api.event import AstrMessageEvent
from astrbot.api.star import Context
from astrbot.core.message.components import Image, Plain
from astrbot.core.pipeline.waking_check.stage import WakingCheckStage
from astrbot.core.platform.astrbot_message import AstrBotMessage, MessageMember
from astrbot.core.platform.message_type import MessageType
from astrbot.core.platform.platform_metadata import PlatformMetadata
from astrbot.core.star.star import StarMetadata, star_map

main = importlib.import_module(ROOT.name + ".main")


class Event(AstrMessageEvent):
    def __init__(self, text, *, group=True, image_first=False):
        message = AstrBotMessage()
        message.type = (
            MessageType.GROUP_MESSAGE if group else MessageType.FRIEND_MESSAGE
        )
        message.self_id = "bot"
        message.session_id = "test"
        message.message_id = "native-command"
        message.group_id = "group" if group else ""
        message.sender = MessageMember(user_id="user", nickname="Test")
        message.message = ([Image.fromBase64("c291cmNl")] if image_first else []) + [
            Plain(text)
        ]
        message.message_str = text
        message.raw_message = {}
        super().__init__(
            text, message, PlatformMetadata("aiocqhttp", "test", "test"), "test"
        )
        self.sent = []

    async def send(self, message):
        self.sent.append(message)


async def check():
    context = object.__new__(Context)
    context.registered_web_apis = []
    context.provider_manager = SimpleNamespace(provider_insts=[])
    context.get_config = lambda: {"wake_prefix": ["."]}
    initialized = main.GiteeAIImagePlugin(context, {})
    try:
        await initialized.initialize()
        assert initialized.studio is not None
        assert any(
            api[0].endswith("/studio/state") for api in context.registered_web_apis
        )
    finally:
        await initialized.terminate()

    plugin = object.__new__(main.GiteeAIImagePlugin)
    plugin._wake_prefixes = (".",)
    plugin.registry = SimpleNamespace(provider_ids=lambda: ["美年达sd", "美年达画图"])
    plugin.edit = SimpleNamespace(get_preset_names=lambda: [])
    for name in ("_do_draw", "_do_edit", "_do_selfie"):
        setattr(plugin, name, AsyncMock())
    main.mark_failed = AsyncMock()
    star_map[main.__name__] = StarMetadata(
        name=ROOT.name, activated=True, star_cls=plugin
    )
    stage = WakingCheckStage()
    await stage.initialize(
        SimpleNamespace(
            db_helper=None,
            astrbot_config_id="native-test",
            astrbot_config={
                "wake_prefix": ["."],
                "admins_id": [],
                "platform_settings": {},
                "plugin_set": ["*"],
            },
        )
    )
    count = 0
    for command, method in (
        ("aiimg", "_do_draw"),
        ("aiedit", "_do_edit"),
        ("自拍", "_do_selfie"),
    ):
        for group, image_first, prefix in (
            (True, False, "."),
            (True, True, "."),
            (False, False, ""),
        ):
            for name in ("_do_draw", "_do_edit", "_do_selfie"):
                getattr(plugin, name).reset_mock()
            event = Event(
                f"{prefix}{command}美年达sd 场景要求\n保留蓝色衣服 3:4",
                group=group,
                image_first=image_first,
            )
            await stage.process(event)
            handlers = event.get_extra("activated_handlers", [])
            selected = [
                h for h in handlers if h.handler_name == "provider_image_command"
            ]
            assert len(selected) == 1, (
                command,
                group,
                image_first,
                [h.handler_name for h in handlers],
            )
            assert not any(
                h.handler_name
                in {"generate_image_command", "edit_image_default", "selfie_command"}
                for h in handlers
            )
            await selected[0].handler(plugin, event)
            await selected[0].handler(plugin, event)
            target = getattr(plugin, method)
            target.assert_awaited_once()
            assert target.await_args.args[1] == "场景要求\n保留蓝色衣服 3:4"
            assert target.await_args.kwargs == {
                "provider_id" if command == "aiimg" else "backend": "美年达sd"
            }
            assert event.is_stopped()
            count += 1
    for text, expected in (
        ("aiimg美年达sd 猫", False),
        ("/aiimg美年达sd 猫", False),
        ("聊天提到 .aiimg美年达sd 猫", False),
        (".自拍参考 查看", True),
        (".aiedit @美年达sd 保留蓝色衣服", True),
    ):
        event = Event(text)
        await stage.process(event)
        selected = [
            h
            for h in event.get_extra("activated_handlers", [])
            if h.handler_name == "provider_image_command"
        ]
        for handler in selected:
            await handler.handler(plugin, event)
        assert not event.get_extra("_gitee_provider_command_handled", False)
        if not expected:
            assert not selected
        count += 1
    print(
        f"Native channel commands: {count} wake, dispatch, attachment and dedup cases passed"
    )


asyncio.run(check())
