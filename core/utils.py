"""
工具函数模块

提供图片获取、头像下载等通用功能
"""

import asyncio
import base64
import io

import aiohttp

from astrbot.api import logger
from astrbot.core.message.components import At, Image, Reply
from astrbot.core.platform.astr_message_event import AstrMessageEvent

from .net_safety import URLFetchPolicy, ensure_url_allowed

# 尝试导入 PIL 用于 GIF 处理
try:
    from PIL import Image as PILImage
except ImportError:
    PILImage = None


# HTTP 会话单例
_http_session: aiohttp.ClientSession | None = None
_session_lock = asyncio.Lock()


async def _get_session() -> aiohttp.ClientSession:
    """获取或创建 HTTP 会话（单例模式）"""
    global _http_session
    if _http_session is None or _http_session.closed:
        async with _session_lock:
            if _http_session is None or _http_session.closed:
                timeout = aiohttp.ClientTimeout(total=30, connect=10)
                connector = aiohttp.TCPConnector(limit=10, limit_per_host=5)
                _http_session = aiohttp.ClientSession(
                    timeout=timeout,
                    connector=connector,
                )
    return _http_session


async def close_session() -> None:
    """关闭 HTTP 会话"""
    global _http_session
    if _http_session and not _http_session.closed:
        await _http_session.close()
        _http_session = None


async def download_image(url: str, retries: int = 3) -> bytes | None:
    """下载图片，带重试机制

    Args:
        url: 图片 URL
        retries: 重试次数

    Returns:
        图片字节数据，失败返回 None
    """
    session = await _get_session()

    policy = URLFetchPolicy(
        allow_private=False,
        trusted_origins=frozenset(),
        allowed_hosts=frozenset(),
        dns_timeout_seconds=2.0,
    )
    max_redirects = 5
    max_bytes = 50 * 1024 * 1024

    for i in range(retries):
        try:
            current = str(url or "").strip()
            redirects = 0
            while True:
                await ensure_url_allowed(current, policy=policy)
                async with session.get(current, allow_redirects=False) as resp:
                    if resp.status in {301, 302, 303, 307, 308}:
                        if redirects >= max_redirects:
                            raise RuntimeError("Too many redirects")
                        loc = (resp.headers.get("location") or "").strip()
                        if not loc:
                            raise RuntimeError("Redirect without location")
                        current = (
                            aiohttp.client.URL(current)
                            .join(aiohttp.client.URL(loc))
                            .human_repr()
                        )
                        redirects += 1
                        continue

                    if resp.status != 200:
                        logger.warning(
                            f"[下载图片] HTTP {resp.status}: {current[:60]}..."
                        )
                        break

                    total = 0
                    chunks: list[bytes] = []
                    async for chunk in resp.content.iter_chunked(1024 * 256):
                        if not chunk:
                            continue
                        total += len(chunk)
                        if total > max_bytes:
                            raise RuntimeError("Image too large")
                        chunks.append(chunk)
                    return b"".join(chunks)
        except asyncio.TimeoutError:
            logger.warning(f"[下载图片] 超时 (第{i + 1}次): {url[:60]}...")
        except Exception as e:
            if i < retries - 1:
                await asyncio.sleep(1)
            else:
                logger.error(f"[下载图片] 失败: {url[:60]}..., 错误: {e}")
    return None


async def get_avatar(user_id: str) -> bytes | None:
    """获取 QQ 用户头像

    使用 q4.qlogo.cn，更稳定

    Args:
        user_id: QQ 号

    Returns:
        头像图片字节数据，失败返回 None
    """
    if not str(user_id).isdigit():
        return None

    avatar_url = f"https://q4.qlogo.cn/headimg_dl?dst_uin={user_id}&spec=640"
    raw = await download_image(avatar_url)

    if raw:
        # 处理 GIF 头像，提取第一帧
        return await _extract_first_frame(raw)
    return None


def _extract_first_frame_sync(raw: bytes) -> bytes:
    """提取 GIF 第一帧（同步方法，供线程池调用）"""
    if PILImage is None:
        return raw
    try:
        img = PILImage.open(io.BytesIO(raw))
        if getattr(img, "is_animated", False):
            img.seek(0)
        img = img.convert("RGB")
        out = io.BytesIO()
        img.save(out, format="JPEG", quality=85)
        return out.getvalue()
    except Exception:
        return raw


async def _extract_first_frame(raw: bytes) -> bytes:
    """提取 GIF 第一帧（异步包装）"""
    return await asyncio.to_thread(_extract_first_frame_sync, raw)


async def get_images_from_event(
    event: AstrMessageEvent,
    include_avatar: bool = True,
    include_sender_avatar_fallback: bool = True,
) -> list[Image]:
    """从消息事件中提取图片组件列表

    图片来源（全部收集，不互斥）：
    1. 回复/引用消息中的图片
    2. 当前消息中的图片
    3. @用户头像（有@时获取被@者头像）
    4. 发送者头像（无图片且无@时，作为兜底）

    Args:
        event: 消息事件
        include_avatar: 是否包含头像，默认 True
        include_sender_avatar_fallback: include_avatar=True 且无图片无@时，是否用发送者头像兜底

    Returns:
        Image 组件列表
    """
    image_segs: list[Image] = []
    chain = event.get_messages()

    logger.debug(
        f"[get_images] 消息链长度: {len(chain)}, 内容: {[type(seg).__name__ for seg in chain]}"
    )

    # 获取机器人自己的 ID（用于过滤@机器人）
    self_id = ""
    if hasattr(event, "get_self_id"):
        try:
            self_id = str(event.get_self_id()).strip()
        except Exception:
            pass

    # 收集所有有效的 @用户（排除@机器人自己和@all）
    at_user_ids: list[str] = []
    for seg in chain:
        if isinstance(seg, At) and hasattr(seg, "qq") and seg.qq != "all":
            uid = str(seg.qq)
            # 排除@机器人自己
            if uid != self_id and uid not in at_user_ids:
                at_user_ids.append(uid)

    # 1. 回复链中的图片
    for seg in chain:
        if isinstance(seg, Reply) and seg.chain:
            for chain_item in seg.chain:
                if isinstance(chain_item, Image):
                    image_segs.append(chain_item)
                    logger.debug("[get_images] 从回复中获取图片")

    # 2. 当前消息中的图片
    for seg in chain:
        if isinstance(seg, Image):
            image_segs.append(seg)
            logger.debug(
                f"[get_images] 从当前消息获取图片: url={getattr(seg, 'url', 'N/A')[:50] if getattr(seg, 'url', None) else 'N/A'}"
            )

    logger.debug(f"[get_images] 图片段数量: {len(image_segs)}, @用户: {at_user_ids}")

    # 3. 头像处理
    if include_avatar:
        if at_user_ids:
            # 有@用户：获取所有被@者的头像（与图片共存）
            for uid in at_user_ids:
                avatar_bytes = await get_avatar(uid)
                if avatar_bytes:
                    b64 = base64.b64encode(avatar_bytes).decode()
                    image_segs.append(Image.fromBase64(b64))
                    logger.debug(f"[get_images] 获取@用户头像成功: {uid}")
        elif include_sender_avatar_fallback and not image_segs:
            # 无@用户且无图片：获取发送者自己的头像（兜底）
            sender_id = event.get_sender_id()
            if sender_id:
                avatar_bytes = await get_avatar(str(sender_id))
                if avatar_bytes:
                    b64 = base64.b64encode(avatar_bytes).decode()
                    image_segs.append(Image.fromBase64(b64))
                    logger.debug(f"[get_images] 获取发送者头像成功: {sender_id}")

    logger.debug(f"[get_images] 最终返回 {len(image_segs)} 个图片段")
    return image_segs
