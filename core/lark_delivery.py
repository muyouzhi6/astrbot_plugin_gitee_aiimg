import inspect
import json

from astrbot.api import logger


async def send_lark_image(adapter, target, chain) -> None:
    """Proactively send one image and require a successful upload and send."""
    from astrbot.core import sp
    from astrbot.core.platform.sources.lark.lark_event import LarkMessageEvent

    client = adapter.lark_api
    # The adapter's generic send_by_session silently drops images that failed upload.
    content = await LarkMessageEvent._convert_to_lark(chain, client)
    if not any(part.get("image_key") for row in content for part in row):
        raise RuntimeError("Feishu image upload failed")

    send = LarkMessageEvent._send_im_message
    route = {
        "receive_id": target.group_id or target.session_id,
        "receive_id_type": "chat_id" if target.group_id else "open_id",
    }
    if not target.group_id and "fallback_chat_id" in inspect.signature(send).parameters:
        try:
            route["fallback_chat_id"] = await sp.get_async(
                "lark",
                f"{adapter.meta().id}:{adapter.appid}",
                f"private_chat:{target.session_id}",
                None,
            )
        except Exception as exc:
            logger.warning(
                "[GiteeAIImage] Failed to look up the Feishu private route: %s", exc
            )
    sent = await send(
        client,
        content=json.dumps({"zh_cn": {"title": "", "content": content}}),
        msg_type="post",
        **route,
    )
    if not sent:
        raise RuntimeError("Feishu did not accept the image message")
