"""
表情反馈工具模块

提供基于 CQHTTP set_msg_emoji_like API 的表情反馈功能
用于在消息上贴表情来表示任务状态
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from astrbot.api import logger

if TYPE_CHECKING:
    from astrbot.core.platform.astr_message_event import AstrMessageEvent


# 表情 ID 常量 (避免与 parser 插件仲裁协议冲突: 289, 124)
class EmojiID:
    """QQ 表情 ID"""

    PROCESSING = 125  # 🔄 处理中 (转圈)
    SUCCESS = 79  # ✌️ 成功 (胜利)
    FAILED = 106  # 😞 失败 (委屈)


# 运行期配置（由 configure_emoji_feedback 注入）
_CONFIG: dict[str, Any] = {
    "enabled": True,
    "emoji_type": "1",
    "processing": EmojiID.PROCESSING,
    "success": EmojiID.SUCCESS,
    "failed": EmojiID.FAILED,
}


def configure_emoji_feedback(config: dict | None) -> None:
    """从插件配置中读取贴表情反馈设置。

    Args:
        config: 插件配置的 emoji_feedback 段；传空则使用内置默认值。
    """
    _CONFIG.clear()
    _CONFIG.update(
        {
            "enabled": True,
            "emoji_type": "1",
            "processing": EmojiID.PROCESSING,
            "success": EmojiID.SUCCESS,
            "failed": EmojiID.FAILED,
        }
    )
    if not isinstance(config, dict):
        return

    enabled = config.get("enabled")
    if enabled is not None:
        _CONFIG["enabled"] = bool(enabled)

    emoji_type = str(config.get("emoji_type") or "").strip()
    if emoji_type:
        _CONFIG["emoji_type"] = emoji_type

    for key in ("processing", "success", "failed"):
        raw = config.get(key)
        if raw in (None, ""):
            continue
        try:
            value = int(raw)
        except (TypeError, ValueError):
            logger.warning("[emoji_feedback] 忽略无效的 %s 表情 ID: %r", key, raw)
            continue
        if 0 <= value <= 65535:
            _CONFIG[key] = value
        else:
            logger.warning("[emoji_feedback] 忽略越界的 %s 表情 ID: %r", key, raw)


async def _get_message_id(event: AstrMessageEvent) -> int | None:
    """从事件中提取消息 ID"""
    try:
        # AiocqhttpMessageEvent 有 message_obj.raw_message
        if hasattr(event, "message_obj"):
            raw = event.message_obj.raw_message
            logger.debug(
                f"[emoji_feedback] raw_message type={type(raw).__name__}, value={raw}"
            )
            if isinstance(raw, dict) and "message_id" in raw:
                return int(raw["message_id"])
            else:
                logger.debug("[emoji_feedback] raw_message 不是 dict 或无 message_id")
        else:
            logger.debug("[emoji_feedback] event 无 message_obj 属性")
    except Exception as e:
        logger.debug(f"[emoji_feedback] 获取消息ID失败: {e}")
    return None


async def _get_bot(event: AstrMessageEvent) -> Any | None:
    """从事件中获取 bot 实例"""
    try:
        if hasattr(event, "bot"):
            return event.bot
    except Exception:
        pass
    return None


async def set_emoji(
    event: AstrMessageEvent,
    emoji_id: int,
    emoji_type: str = "1",
) -> bool:
    """
    给消息贴表情

    Args:
        event: 消息事件
        emoji_id: 表情 ID
        emoji_type: 表情类型，默认 "1"

    Returns:
        是否成功
    """
    if not _CONFIG.get("enabled", True):
        return False
    emoji_type = str(_CONFIG.get("emoji_type") or emoji_type or "1")

    message_id = await _get_message_id(event)
    if message_id is None:
        logger.debug("[emoji_feedback] 无法获取消息ID，跳过贴表情")
        return False

    bot = await _get_bot(event)
    if bot is None:
        logger.debug("[emoji_feedback] 无法获取bot实例，跳过贴表情")
        return False

    # 检查 bot 是否支持 set_msg_emoji_like
    if not hasattr(bot, "set_msg_emoji_like"):
        logger.debug("[emoji_feedback] bot不支持set_msg_emoji_like，跳过贴表情")
        return False

    try:
        await bot.set_msg_emoji_like(
            message_id=message_id,
            emoji_id=emoji_id,
            emoji_type=emoji_type,
            set=True,
        )
        logger.debug(
            f"[emoji_feedback] 贴表情成功: message_id={message_id}, emoji_id={emoji_id}"
        )
        return True
    except Exception as e:
        logger.debug(f"[emoji_feedback] 贴表情失败: {e}")
        return False


async def mark_processing(event: AstrMessageEvent) -> bool:
    """标记消息为处理中状态"""
    return await set_emoji(event, int(_CONFIG.get("processing", EmojiID.PROCESSING)))


async def mark_success(event: AstrMessageEvent) -> bool:
    """标记消息为成功状态"""
    return await set_emoji(event, int(_CONFIG.get("success", EmojiID.SUCCESS)))


async def mark_failed(event: AstrMessageEvent) -> bool:
    """标记消息为失败状态"""
    return await set_emoji(event, int(_CONFIG.get("failed", EmojiID.FAILED)))
