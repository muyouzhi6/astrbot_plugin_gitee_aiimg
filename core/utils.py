from astrbot.core.message.components import Image, Reply
from astrbot.core.platform.astr_message_event import AstrMessageEvent


async def get_images_from_event(event: AstrMessageEvent) -> list[Image]:
    """从消息事件中提取图片二进制数据列表

    支持：
    1. 回复/引用消息中的图片（优先）
    2. 当前消息中的图片
    3. 多图输入
    4. base64 格式图片
    """
    image_segs: list[Image] = []
    chain = event.get_messages()

    for seg in chain:
        if isinstance(seg, Reply) and seg.chain:
            for chain_item in seg.chain:
                if isinstance(chain_item, Image):
                    image_segs.append(chain_item)

        if isinstance(seg, Image):
            image_segs.append(seg)

    return image_segs


