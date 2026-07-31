import asyncio

import pytest
from core.emoji_feedback import (
    _CONFIG,
    EmojiID,
    _get_message_id,
    configure_emoji_feedback,
    mark_failed,
    mark_processing,
    mark_success,
    set_emoji,
)


class FakeBot:
    def __init__(self):
        self.calls = []

    async def set_msg_emoji_like(self, **kwargs):
        self.calls.append(kwargs)


class FakeMessageObj:
    def __init__(self, message_id):
        self.raw_message = {"message_id": message_id}


class FakeEvent:
    def __init__(self, message_id=100, bot=None):
        self.message_obj = FakeMessageObj(message_id)
        self.bot = bot or FakeBot()


def test_configure_defaults_reset_everything():
    configure_emoji_feedback(None)
    assert _CONFIG["enabled"] is True
    assert _CONFIG["emoji_type"] == "1"
    assert _CONFIG["processing"] == EmojiID.PROCESSING
    assert _CONFIG["success"] == EmojiID.SUCCESS
    assert _CONFIG["failed"] == EmojiID.FAILED


def test_configure_applies_custom_emoji_ids():
    configure_emoji_feedback(
        {
            "enabled": False,
            "emoji_type": "326",
            "processing": "10",
            "success": 20,
            "failed": "30",
        }
    )
    assert _CONFIG["enabled"] is False
    assert _CONFIG["emoji_type"] == "326"
    assert _CONFIG["processing"] == 10
    assert _CONFIG["success"] == 20
    assert _CONFIG["failed"] == 30


def test_configure_ignores_invalid_emoji_ids():
    configure_emoji_feedback({"processing": "abc", "success": "-1", "failed": 999999})
    assert _CONFIG["processing"] == EmojiID.PROCESSING
    assert _CONFIG["success"] == EmojiID.SUCCESS
    assert _CONFIG["failed"] == EmojiID.FAILED


def test_get_message_id_extracts_raw_message():
    configure_emoji_feedback(None)
    assert asyncio.run(_get_message_id(FakeEvent(message_id=42))) == 42


@pytest.mark.asyncio
async def test_disabled_skips_all_stickers():
    configure_emoji_feedback({"enabled": False})
    bot = FakeBot()
    event = FakeEvent(bot=bot)
    assert await mark_processing(event) is False
    assert await mark_success(event) is False
    assert await mark_failed(event) is False
    assert bot.calls == []


@pytest.mark.asyncio
async def test_mark_uses_configured_ids_and_type():
    configure_emoji_feedback(
        {
            "enabled": True,
            "emoji_type": "326",
            "processing": 10,
            "success": 20,
            "failed": 30,
        }
    )
    bot = FakeBot()
    event = FakeEvent(bot=bot)
    assert await mark_processing(event) is True
    assert await mark_success(event) is True
    assert await mark_failed(event) is True
    ids = [call["emoji_id"] for call in bot.calls]
    types = [call["emoji_type"] for call in bot.calls]
    assert ids == [10, 20, 30]
    assert types == ["326", "326", "326"]


@pytest.mark.asyncio
async def test_bot_without_api_is_skipped():
    configure_emoji_feedback(None)

    class NoApiBot:
        pass

    event = FakeEvent(bot=NoApiBot())
    assert await set_emoji(event, EmojiID.SUCCESS) is False
