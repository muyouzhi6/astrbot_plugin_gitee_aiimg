"""
Utility helpers for image extraction and downloads.
"""

import asyncio
import base64
import io
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

import aiohttp

from astrbot.api import logger
from astrbot.core.message.components import At, Image, Reply
from astrbot.core.platform.astr_message_event import AstrMessageEvent

from .net_safety import URLFetchPolicy, ensure_url_allowed

try:
    from PIL import Image as PILImage
except ImportError:
    PILImage = None


_http_session: aiohttp.ClientSession | None = None
_session_lock = asyncio.Lock()


async def _get_session() -> aiohttp.ClientSession:
    """Return the shared HTTP session."""
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
    """Close the shared HTTP session."""
    global _http_session
    if _http_session and not _http_session.closed:
        await _http_session.close()
        _http_session = None


async def download_image(url: str, retries: int = 3) -> bytes | None:
    """Download an image with URL safety checks and bounded size."""
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
                            f"[download_image] HTTP {resp.status}: {current[:60]}..."
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
            logger.warning(f"[download_image] timeout (attempt {i + 1}): {url[:60]}...")
        except Exception as e:
            if i < retries - 1:
                await asyncio.sleep(1)
            else:
                logger.error(f"[download_image] failed: {url[:60]}..., error: {e}")
    return None


async def get_avatar(user_id: str) -> bytes | None:
    """Fetch a QQ avatar image."""
    if not str(user_id).isdigit():
        return None

    avatar_url = f"https://q4.qlogo.cn/headimg_dl?dst_uin={user_id}&spec=640"
    raw = await download_image(avatar_url)
    if raw:
        return await _extract_first_frame(raw)
    return None


def _extract_first_frame_sync(raw: bytes) -> bytes:
    """Convert animated avatars to a static JPEG first frame."""
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
    """Async wrapper for first-frame extraction."""
    return await asyncio.to_thread(_extract_first_frame_sync, raw)


def _image_from_ref(ref: str) -> Image | None:
    value = str(ref or "").strip()
    if not value:
        return None
    if value.startswith("base64://"):
        payload = value[len("base64://") :].strip()
        return Image.fromBase64(payload) if payload else None
    if value.startswith("data:image/") and "," in value:
        header, payload = value.split(",", 1)
        if ";base64" in header.lower() and payload.strip():
            return Image.fromBase64(payload.strip())
        return None
    if value.startswith(("http://", "https://")):
        return Image.fromURL(value)
    if value.startswith("file://"):
        parsed = urlsplit(value)
        local_path = unquote(parsed.path or "")
        if local_path.startswith("/") and len(local_path) > 3 and local_path[2] == ":":
            local_path = local_path[1:]
        if local_path and Path(local_path).exists():
            return Image.fromFileSystem(local_path)
        return None
    if Path(value).exists():
        return Image.fromFileSystem(value)
    return None


def _extract_message_components(message: Any) -> list[Any]:
    """Normalize different message container shapes into a component list."""
    if message is None:
        return []
    if hasattr(message, "message_obj") and hasattr(message.message_obj, "message"):
        components = message.message_obj.message
    elif isinstance(message, list):
        components = message
    elif hasattr(message, "message"):
        components = message.message
    else:
        components = []
    return list(components or [])


async def _fetch_reply_message_images(
    event: AstrMessageEvent,
    reply_seg: Reply,
) -> list[Image]:
    """Fetch images from the replied message when Reply.chain is empty."""
    try:
        from astrbot.core.utils.quoted_message_parser import (
            extract_quoted_message_images,
        )
    except Exception:
        extract_quoted_message_images = None

    if callable(extract_quoted_message_images):
        try:
            refs = await extract_quoted_message_images(event, reply_seg)
        except Exception as e:
            logger.debug(f"[get_images] quoted parser fallback failed: {e}")
        else:
            images = [
                image
                for ref in refs
                for image in [_image_from_ref(ref)]
                if image is not None
            ]
            if images:
                logger.debug(
                    f"[get_images] fetched {len(images)} image(s) from quoted parser"
                )
                return images

    reply_id = getattr(reply_seg, "id", None)
    if not reply_id:
        return []

    bot = getattr(event, "bot", None)
    if bot is None or not hasattr(bot, "get_message"):
        return []

    try:
        target_msg = await bot.get_message(reply_id)
    except Exception as e:
        logger.warning(
            f"[get_images] failed to fetch replied message id={reply_id}: {e}"
        )
        return []

    images: list[Image] = []
    for comp in _extract_message_components(target_msg):
        if isinstance(comp, Image):
            images.append(comp)

    if images:
        logger.debug(
            f"[get_images] fetched {len(images)} image(s) from reply id={reply_id}"
        )
    return images


async def get_images_from_event(
    event: AstrMessageEvent,
    include_avatar: bool = True,
    include_sender_avatar_fallback: bool = True,
) -> list[Image]:
    """Collect image segments from reply/current message/avatar sources."""
    image_segs: list[Image] = []

    try:
        chain = list(event.get_messages() or [])
    except Exception:
        chain = []

    logger.debug(
        f"[get_images] chain_len={len(chain)}, types={[type(seg).__name__ for seg in chain]}"
    )

    self_id = ""
    if hasattr(event, "get_self_id"):
        try:
            self_id = str(event.get_self_id()).strip()
        except Exception:
            pass

    at_user_ids: list[str] = []
    for seg in chain:
        if isinstance(seg, At) and hasattr(seg, "qq") and seg.qq != "all":
            uid = str(seg.qq)
            if uid != self_id and uid not in at_user_ids:
                at_user_ids.append(uid)

    for seg in chain:
        if not isinstance(seg, Reply):
            continue

        found_in_reply = False
        for chain_item in getattr(seg, "chain", None) or []:
            if isinstance(chain_item, Image):
                image_segs.append(chain_item)
                found_in_reply = True
                logger.debug("[get_images] image from reply.chain")

        if found_in_reply:
            continue

        fetched_images = await _fetch_reply_message_images(event, seg)
        if fetched_images:
            image_segs.extend(fetched_images)

    for seg in chain:
        if isinstance(seg, Image):
            image_segs.append(seg)
            logger.debug(
                f"[get_images] image from current message url={getattr(seg, 'url', 'N/A')[:50] if getattr(seg, 'url', None) else 'N/A'}"
            )

    logger.debug(f"[get_images] image_count={len(image_segs)}, at_users={at_user_ids}")

    if include_avatar:
        if at_user_ids:
            for uid in at_user_ids:
                avatar_bytes = await get_avatar(uid)
                if avatar_bytes:
                    b64 = base64.b64encode(avatar_bytes).decode()
                    image_segs.append(Image.fromBase64(b64))
                    logger.debug(f"[get_images] avatar loaded for @{uid}")
        elif include_sender_avatar_fallback and not image_segs:
            sender_id = event.get_sender_id()
            if sender_id:
                avatar_bytes = await get_avatar(str(sender_id))
                if avatar_bytes:
                    b64 = base64.b64encode(avatar_bytes).decode()
                    image_segs.append(Image.fromBase64(b64))
                    logger.debug(
                        f"[get_images] sender avatar fallback loaded: {sender_id}"
                    )

    logger.debug(f"[get_images] final_count={len(image_segs)}")
    return image_segs
