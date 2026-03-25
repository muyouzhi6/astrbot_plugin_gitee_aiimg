"""
Utility helpers for image extraction and downloads.
"""

import asyncio
import base64
import io
import os
from typing import Any, Awaitable, Callable

import aiohttp

from astrbot.api import logger
from astrbot.core.message.components import At, Image, Reply
from astrbot.core.platform.astr_message_event import AstrMessageEvent

from .net_safety import URLFetchPolicy, ensure_url_allowed

try:
    from astrbot.core.utils.quoted_message_parser import (
        extract_quoted_message_images as _astrbot_extract_quoted_message_images,
    )
except Exception:
    _astrbot_extract_quoted_message_images = None

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


def _safe_getattr(obj: Any, attr: str, default: Any = None) -> Any:
    try:
        return getattr(obj, attr)
    except Exception as e:
        logger.debug(
            "[get_images] 读取属性失败: %s.%s err=%s",
            type(obj).__name__,
            attr,
            e,
        )
        return default


def _get_event_chain(event: AstrMessageEvent) -> list[Any]:
    try:
        chain = event.get_messages()
        if isinstance(chain, list):
            return chain
    except Exception:
        pass

    message_obj = getattr(event, "message_obj", None)
    chain = getattr(message_obj, "message", None)
    return chain if isinstance(chain, list) else []


def _append_unique_string(target: list[str], value: Any) -> None:
    s = str(value or "").strip()
    if s and s not in target:
        target.append(s)


def _normalize_image_ref(value: Any) -> str:
    s = str(value or "").strip()
    if not s:
        return ""
    if s.startswith("data:image/"):
        _, _, payload = s.partition(",")
        return f"base64://{payload}" if payload else ""
    if s.startswith(("http://", "https://", "file:///", "base64://")):
        return s
    if os.path.exists(s):
        return os.path.abspath(s)
    return ""


def _collect_image_refs_from_mapping(mapping: dict[str, Any], refs: list[str]) -> None:
    for key in ("url", "file", "path", "image", "file_id", "id"):
        _append_unique_string(refs, mapping.get(key))


def _extract_images_from_structure(
    value: Any,
    image_segs: list[Image],
    image_refs: list[str],
    *,
    seen: set[int] | None = None,
) -> None:
    if value is None:
        return

    if seen is None:
        seen = set()

    try:
        marker = id(value)
    except Exception:
        marker = None
    if marker is not None:
        if marker in seen:
            return
        seen.add(marker)

    if isinstance(value, Image):
        image_segs.append(value)
        return

    if isinstance(value, dict):
        seg_type = str(value.get("type") or "").strip().lower()
        data = value.get("data")
        if seg_type == "image":
            if isinstance(data, dict):
                _collect_image_refs_from_mapping(data, image_refs)
            else:
                _collect_image_refs_from_mapping(value, image_refs)
            return

        for key in ("message", "messages", "chain", "content", "elements", "segments"):
            if key in value:
                _extract_images_from_structure(
                    value.get(key), image_segs, image_refs, seen=seen
                )
        return

    if isinstance(value, (list, tuple, set)):
        for item in value:
            _extract_images_from_structure(item, image_segs, image_refs, seen=seen)
        return

    for attr in ("chain", "message", "messages", "content", "elements", "segments"):
        nested = _safe_getattr(value, attr, None)
        if nested is not None and nested is not value:
            _extract_images_from_structure(nested, image_segs, image_refs, seen=seen)


def _resolve_call_action_candidates(
    event: AstrMessageEvent,
) -> list[Callable[..., Awaitable[Any]]]:
    bot = getattr(event, "bot", None)
    api = getattr(bot, "api", None)
    funcs: list[Callable[..., Awaitable[Any]]] = []
    seen: set[tuple[int, int]] = set()

    for owner in (api, bot):
        func = getattr(owner, "call_action", None)
        if not callable(func):
            continue
        marker = (
            id(getattr(func, "__self__", None)),
            id(getattr(func, "__func__", func)),
        )
        if marker in seen:
            continue
        seen.add(marker)
        funcs.append(func)

    return funcs


def _looks_like_call_action_signature_error(exc: Exception) -> bool:
    if not isinstance(exc, TypeError):
        return False
    msg = str(exc)
    markers = (
        "positional argument",
        "unexpected keyword argument",
        "missing 1 required positional argument",
        "got multiple values for argument",
    )
    return any(marker in msg for marker in markers)


async def _invoke_call_action(
    call_action: Callable[..., Awaitable[Any]],
    action: str,
    params: dict[str, Any],
) -> Any:
    try:
        return await call_action(action, **params)
    except Exception as e:
        if not _looks_like_call_action_signature_error(e):
            raise
    return await call_action(action=action, **params)


async def _call_action_compat(
    event: AstrMessageEvent,
    action: str,
    params_list: list[dict[str, Any]],
) -> Any:
    funcs = _resolve_call_action_candidates(event)
    if not funcs:
        return None

    for params in params_list:
        for func in funcs:
            try:
                result = await _invoke_call_action(func, action, params)
                if isinstance(result, dict):
                    return result
            except Exception as e:
                logger.debug(
                    "[get_images] call_action failed: action=%s params=%s err=%s",
                    action,
                    {k: str(v)[:80] for k, v in params.items()},
                    e,
                )
    return None


def _unwrap_action_data(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    data = payload.get("data")
    return data if isinstance(data, dict) else payload


def _build_reply_lookup_params(message_id: Any) -> list[dict[str, Any]]:
    message_id_str = str(message_id or "").strip()
    if not message_id_str:
        return []
    params_list: list[dict[str, Any]] = [
        {"message_id": message_id_str},
        {"id": message_id_str},
    ]
    if message_id_str.isdigit():
        numeric_id = int(message_id_str)
        params_list.extend(
            [{"message_id": numeric_id}, {"id": numeric_id}]
        )
    return params_list


def _build_image_resolve_actions(
    event: AstrMessageEvent,
    image_ref: str,
) -> list[tuple[str, dict[str, Any]]]:
    candidates: list[str] = []
    cleaned = str(image_ref or "").strip()
    if cleaned:
        candidates.append(cleaned)
        stem, ext = os.path.splitext(cleaned)
        if ext and stem and stem not in candidates:
            candidates.append(stem)

    actions: list[tuple[str, dict[str, Any]]] = []
    for candidate in candidates:
        actions.extend(
            [
                ("get_image", {"file": candidate}),
                ("get_image", {"file_id": candidate}),
                ("get_image", {"id": candidate}),
                ("get_image", {"image": candidate}),
                ("get_file", {"file_id": candidate}),
                ("get_file", {"file": candidate}),
            ]
        )

    try:
        group_id = event.get_group_id()
    except Exception:
        group_id = None
    if isinstance(group_id, str) and group_id.isdigit():
        group_id = int(group_id)
    if group_id:
        for candidate in candidates:
            actions.append(
                ("get_group_file_url", {"group_id": group_id, "file_id": candidate})
            )
    for candidate in candidates:
        actions.append(("get_private_file_url", {"file_id": candidate}))
    return actions


async def _resolve_image_ref(event: AstrMessageEvent, image_ref: str) -> str:
    normalized = _normalize_image_ref(image_ref)
    if normalized:
        return normalized

    for action, params in _build_image_resolve_actions(event, image_ref):
        payload = await _call_action_compat(event, action, [params])
        data = _unwrap_action_data(payload)
        for key in ("url", "file", "path"):
            normalized = _normalize_image_ref(data.get(key))
            if normalized:
                return normalized
    return ""


def _image_identity(seg: Image) -> str:
    direct = _normalize_image_ref(
        getattr(seg, "url", None) or getattr(seg, "file", None) or getattr(seg, "path", None)
    )
    if direct:
        return direct
    return f"obj:{id(seg)}"


def _append_unique_images(target: list[Image], extras: list[Image]) -> None:
    seen = {_image_identity(seg) for seg in target}
    for seg in extras:
        ident = _image_identity(seg)
        if ident in seen:
            continue
        seen.add(ident)
        target.append(seg)


async def _build_images_from_refs(
    event: AstrMessageEvent,
    refs: list[str],
) -> list[Image]:
    images: list[Image] = []
    for ref in refs:
        resolved = await _resolve_image_ref(event, ref)
        if not resolved:
            continue
        if resolved.startswith(("http://", "https://")):
            images.append(Image.fromURL(resolved))
        elif resolved.startswith("base64://"):
            images.append(Image.fromBase64(resolved.removeprefix("base64://")))
        elif resolved.startswith("file:///"):
            images.append(Image(file=resolved, path=resolved[8:]))
        else:
            images.append(Image.fromFileSystem(resolved))
    return images


async def _extract_reply_images(
    event: AstrMessageEvent,
    reply_seg: Reply,
) -> list[Image]:
    image_segs: list[Image] = []
    image_refs: list[str] = []

    embedded_chain = _safe_getattr(reply_seg, "chain", None)
    _extract_images_from_structure(embedded_chain, image_segs, image_refs)
    if image_refs:
        _append_unique_images(
            image_segs,
            await _build_images_from_refs(event, image_refs),
        )
    if image_segs:
        return image_segs

    if _astrbot_extract_quoted_message_images is not None:
        try:
            refs = await _astrbot_extract_quoted_message_images(
                event, reply_component=reply_seg
            )
            _append_unique_images(image_segs, await _build_images_from_refs(event, refs))
        except Exception as e:
            logger.debug(
                "[get_images] astrbot quoted_message_parser failed: reply_id=%s err=%s",
                _safe_getattr(reply_seg, "id", ""),
                e,
            )
    if image_segs:
        return image_segs

    reply_id = _safe_getattr(reply_seg, "id", "")
    params_list = _build_reply_lookup_params(reply_id)
    if not params_list:
        return image_segs

    try:
        payload = await _call_action_compat(event, "get_msg", params_list)
    except Exception as e:
        logger.warning(
            "[get_images] failed to fetch replied message id=%s: %s",
            reply_id,
            e,
        )
        return image_segs

    data = _unwrap_action_data(payload)
    _extract_images_from_structure(data, image_segs, image_refs)
    if image_refs:
        _append_unique_images(
            image_segs,
            await _build_images_from_refs(event, image_refs),
        )
    return image_segs
async def get_images_from_event(
    event: AstrMessageEvent,
    include_avatar: bool = True,
    include_sender_avatar_fallback: bool = True,
) -> list[Image]:
    """Collect image segments from reply/current message/avatar sources."""
    image_segs: list[Image] = []
    chain = _get_event_chain(event)

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
        reply_images = await _extract_reply_images(event, seg)
        if reply_images:
            _append_unique_images(image_segs, reply_images)
            logger.debug("[get_images] image(s) from reply count=%s", len(reply_images))

    for seg in chain:
        if isinstance(seg, Image):
            _append_unique_images(image_segs, [seg])
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
                    logger.debug(f"[get_images] sender avatar fallback loaded: {sender_id}")

    logger.debug(f"[get_images] final_count={len(image_segs)}")
    return image_segs
