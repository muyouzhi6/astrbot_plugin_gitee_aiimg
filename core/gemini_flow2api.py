"""
Gemini Flow2API 后端（OpenAI Chat Completions + SSE 流式出图 URL）

用于无法直连 Google 官方 Gemini API 时的替代方案：
- 请求形态：POST /v1/chat/completions (OpenAI 兼容)
  - payload: {"model": "...", "messages": [...], "stream": true}
- 返回：SSE 分片，delta.content 里逐步输出图片 URL（常见），或输出 markdown / data:image

备注：
- Flow2API 在部分实现里“带图输入”可能输出 video/mp4（图生视频），本后端会识别并报错。
"""

from __future__ import annotations

import asyncio
import base64
import json
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlsplit, urlunsplit

import aiohttp

from astrbot.api import logger

from .image_format import guess_image_mime_and_ext

_MD_IMAGE_RE = re.compile(r"!\[.*?\]\((.*?)\)")
_DATA_IMAGE_RE = re.compile(r"(data:image/[^\s)]+)")
_HTML_IMG_RE = re.compile(r'<img[^>]*src=["\']([^"\'>]+)["\']', re.IGNORECASE)
_IMG_URL_RE = re.compile(
    r"(https?://[^\s<>\"')\]]+?\.(?:png|jpg|jpeg|webp|gif)(?:\?[^\s<>\"')\]]*)?)",
    re.IGNORECASE,
)
_JSON_URL_FIELD_RE = re.compile(
    r'"(?:image_url|imageUrl|url|image|src|uri|link|href|fifeUrl|fife_url|final_image_url|origin_image_url)"\s*:\s*"([^"]+)"'
)
_HTML_VIDEO_RE = re.compile(r'<video[^>]*src=["\']([^"\'>]+)["\']', re.IGNORECASE)
_VIDEO_URL_RE = re.compile(
    r"(https?://[^\s<>\"')\]]+?\.(?:mp4|webm|mov)(?:\?[^\s<>\"')\]]*)?)",
    re.IGNORECASE,
)
_BASE64_PREFIX_RE = re.compile(r"^(?:b64|base64)\s*:\s*", re.IGNORECASE)
_LOCAL_MEDIA_HOSTS = {"0.0.0.0", "127.0.0.1", "localhost"}


def _strip_markdown_target(target: str) -> str | None:
    s = (target or "").strip()
    if not s:
        return None
    if s.startswith("<") and ">" in s:
        right = s.find(">")
        if right > 1:
            s = s[1:right].strip()
    m = re.match(r'^(?P<url>\S+)(?:\s+(?:"[^"]*"|\'[^\']*\'))?\s*$', s)
    if m:
        s = m.group("url")
    s = s.strip().strip('"').strip("'")
    return s or None


def _decode_base64_bytes(text: str) -> bytes:
    s = re.sub(r"\s+", "", str(text or "").strip())
    if not s:
        return b""
    candidates = [s, s.replace("-", "+").replace("_", "/")]
    for cand in candidates:
        pad = "=" * ((4 - len(cand) % 4) % 4)
        try:
            raw = base64.b64decode(cand + pad, validate=False)
            if raw:
                return raw
        except Exception:
            continue
    try:
        raw = base64.urlsafe_b64decode(s + ("=" * ((4 - len(s) % 4) % 4)))
        if raw:
            return raw
    except Exception:
        pass
    return b""


def _guess_mime_from_magic(image_bytes: bytes) -> str | None:
    if len(image_bytes) >= 3 and image_bytes[0:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if len(image_bytes) >= 8 and image_bytes[0:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if len(image_bytes) >= 6 and (
        image_bytes[0:6] == b"GIF87a" or image_bytes[0:6] == b"GIF89a"
    ):
        return "image/gif"
    if (
        len(image_bytes) >= 12
        and image_bytes[0:4] == b"RIFF"
        and image_bytes[8:12] == b"WEBP"
    ):
        return "image/webp"
    return None


def _base64_to_data_image_ref(text: str, *, min_length: int = 128) -> str | None:
    s = (text or "").strip().strip('"').strip("'")
    s = _BASE64_PREFIX_RE.sub("", s).strip()
    s = re.sub(r"\s+", "", s)
    if len(s) < min_length:
        return None
    raw = _decode_base64_bytes(s)
    if not raw:
        return None
    mime = _guess_mime_from_magic(raw)
    if not mime:
        return None
    return f"data:{mime};base64,{base64.b64encode(raw).decode()}"


def _is_valid_data_image_ref(ref: str) -> bool:
    s = str(ref or "").strip()
    if not s.startswith("data:image/"):
        return False
    if "," not in s:
        return False
    _header, b64 = s.split(",", 1)
    b64 = re.sub(r"\s+", "", (b64 or "").strip())
    if not b64 or b64 == "...":
        return False
    if len(b64) < 16:
        return False
    if not re.fullmatch(r"[A-Za-z0-9+/=_-]+", b64[:2048]):
        return False
    if len(b64) < 128 and not _decode_base64_bytes(b64):
        return False
    return True


def _looks_like_video_url(url: str) -> bool:
    u = (url or "").strip().lower()
    if not u.startswith(("http://", "https://")):
        return False
    if any(ext in u for ext in (".mp4", ".webm", ".mov")):
        return True
    if "generated_video" in u:
        return True
    return False


def _looks_like_relative_image_ref(value: str) -> bool:
    s = (value or "").strip()
    if not s or s.startswith(("http://", "https://", "data:image/")):
        return False
    if not s.startswith(("/", "./", "../", "tmp/")):
        return False
    base = s.split("?", 1)[0].lower()
    return base.endswith((".png", ".jpg", ".jpeg", ".webp", ".gif"))


def _looks_like_relative_video_ref(value: str) -> bool:
    s = (value or "").strip()
    if not s or s.startswith(("http://", "https://")):
        return False
    if not s.startswith(("/", "./", "../", "tmp/")):
        return False
    base = s.split("?", 1)[0].lower()
    return base.endswith((".mp4", ".webm", ".mov"))


def _origin_from_url(raw: str) -> str:
    s = str(raw or "").strip()
    if not s:
        return ""
    try:
        parts = urlsplit(s)
    except Exception:
        return ""
    if not parts.scheme or not parts.netloc:
        return ""
    return urlunsplit((parts.scheme, parts.netloc, "", "", "")).rstrip("/")


def _is_local_media_host(host: str) -> bool:
    h = str(host or "").strip().lower()
    if not h:
        return False
    return h in _LOCAL_MEDIA_HOSTS or h.endswith(".localhost") or h.endswith(".local")


def _rewrite_flow2api_media_ref(ref: str, *, endpoint_url: str) -> str:
    s = str(ref or "").strip()
    if not s:
        return ""

    origin = _origin_from_url(endpoint_url)
    if not origin:
        return s

    try:
        parts = urlsplit(s)
    except Exception:
        return s

    if parts.scheme and parts.netloc:
        if not _is_local_media_host(parts.hostname or ""):
            return s
        origin_parts = urlsplit(origin)
        return urlunsplit(
            (
                origin_parts.scheme,
                origin_parts.netloc,
                parts.path or "/",
                parts.query,
                parts.fragment,
            )
        )

    if _looks_like_relative_image_ref(s) or _looks_like_relative_video_ref(s):
        return urljoin(origin + "/", s)

    return s


def _nested_value(obj: Any, *path: str) -> Any:
    current = obj
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _extract_first_image_ref(text: str) -> str | None:
    s = (text or "").strip()
    if not s:
        return None
    if s.startswith("data:image/"):
        compact = re.sub(r"\s+", "", s)
        if _is_valid_data_image_ref(compact):
            return compact
    m = _MD_IMAGE_RE.search(s)
    if m:
        ref = _strip_markdown_target(m.group(1)) or m.group(1).strip()
        if ref.startswith("data:image/"):
            ref = re.sub(r"\s+", "", ref)
            if _is_valid_data_image_ref(ref):
                return ref
        elif not _looks_like_video_url(ref):
            return ref or None
    for m in _DATA_IMAGE_RE.finditer(s):
        ref = re.sub(r"\s+", "", m.group(1).strip())
        if _is_valid_data_image_ref(ref):
            return ref
    m = _HTML_IMG_RE.search(s)
    if m:
        ref = m.group(1).strip()
        return None if _looks_like_video_url(ref) else (ref or None)
    m = _IMG_URL_RE.search(s)
    if m:
        ref = m.group(1).strip()
        return None if _looks_like_video_url(ref) else (ref or None)
    if _looks_like_relative_image_ref(s):
        return s
    if s.startswith(("http://", "https://")) and not _looks_like_video_url(s):
        return s

    for m in _JSON_URL_FIELD_RE.finditer(s):
        cand = m.group(1).strip().replace("\\/", "/")
        cand = _strip_markdown_target(cand) or cand
        if cand.startswith("data:image/"):
            cand = re.sub(r"\s+", "", cand)
            if _is_valid_data_image_ref(cand):
                return cand
        if _looks_like_relative_image_ref(cand):
            return cand
        if cand.startswith(("http://", "https://")) and not _looks_like_video_url(cand):
            return cand

    if (s.startswith("{") and s.endswith("}")) or (
        s.startswith("[") and s.endswith("]")
    ):
        try:
            parsed = json.loads(s)
        except Exception:
            parsed = None
        if parsed is not None:
            candidates: list[str] = []
            seen: set[int] = set()

            def walk(x: Any) -> None:
                if x is None:
                    return
                oid = id(x)
                if oid in seen:
                    return
                seen.add(oid)
                if isinstance(x, str):
                    candidates.append(x)
                    return
                if isinstance(x, dict):
                    for v in x.values():
                        walk(v)
                    return
                if isinstance(x, list):
                    for v in x:
                        walk(v)
                    return

            walk(parsed)
            for cand in candidates:
                ref = _extract_first_image_ref(cand)
                if ref:
                    return ref
    ref = _base64_to_data_image_ref(s)
    if ref:
        return ref
    return None


def _extract_first_video_ref(text: str) -> str | None:
    s = (text or "").strip()
    if not s:
        return None
    m = _HTML_VIDEO_RE.search(s)
    if m:
        ref = m.group(1).strip()
        return ref if _looks_like_video_url(ref) else None
    m = _VIDEO_URL_RE.search(s)
    if m:
        ref = m.group(1).strip()
        return ref if _looks_like_video_url(ref) else None
    if _looks_like_relative_video_ref(s):
        return s
    if _looks_like_video_url(s):
        return s
    return None


def _iter_strings(obj: Any) -> list[str]:
    out: list[str] = []
    seen: set[int] = set()

    def walk(x: Any) -> None:
        if x is None:
            return
        oid = id(x)
        if oid in seen:
            return
        seen.add(oid)
        if isinstance(x, str):
            out.append(x)
            return
        if isinstance(x, dict):
            for v in x.values():
                walk(v)
            return
        if isinstance(x, list):
            for v in x:
                walk(v)
            return

    walk(obj)
    return out


def _extract_first_image_ref_from_obj(obj: Any) -> str | None:
    if obj is None:
        return None
    if isinstance(obj, str):
        return _extract_first_image_ref(obj)
    if isinstance(obj, list):
        for item in obj:
            ref = _extract_first_image_ref_from_obj(item)
            if ref:
                return ref
        return None
    if isinstance(obj, dict):
        for path in (
            ("generated_assets", "upscaled_image", "local_url"),
            ("generated_assets", "upscaled_image", "url"),
            ("generated_assets", "final_image_url"),
            ("generated_assets", "local_url"),
            ("upscaled_image", "local_url"),
            ("upscaled_image", "url"),
            ("final_image_url",),
            ("local_url",),
        ):
            value = _nested_value(obj, *path)
            ref = _extract_first_image_ref_from_obj(value)
            if ref:
                return ref

        for path in (
            ("generated_assets", "upscaled_image", "base64"),
            ("upscaled_image", "base64"),
        ):
            value = _nested_value(obj, *path)
            if isinstance(value, str) and value.strip():
                ref = _base64_to_data_image_ref(value, min_length=1)
                if ref:
                    return ref

        for key in ("b64_json", "b64", "base64", "image_b64", "image_base64"):
            b64 = obj.get(key)
            if isinstance(b64, str) and b64.strip():
                ref = _base64_to_data_image_ref(b64, min_length=1)
                if ref:
                    return ref

        for key in (
            "url",
            "image_url",
            "image",
            "src",
            "uri",
            "link",
            "href",
            "fifeUrl",
            "fife_url",
            "final_image_url",
            "origin_image_url",
            "thumbnail",
        ):
            value = obj.get(key)
            if isinstance(value, str):
                ref = _extract_first_image_ref(value)
                if ref:
                    return ref
            ref = _extract_first_image_ref_from_obj(value)
            if ref:
                return ref

        for key in (
            "images",
            "image_urls",
            "attachments",
            "generated_assets",
            "media",
            "result",
            "response",
            "content",
            "delta",
            "message",
            "tool_calls",
            "choices",
            "parts",
            "candidates",
        ):
            ref = _extract_first_image_ref_from_obj(obj.get(key))
            if ref:
                return ref

        for s in _iter_strings(obj):
            ref = _extract_first_image_ref(s)
            if ref:
                return ref
    return None


def _extract_first_video_ref_from_obj(obj: Any) -> str | None:
    if obj is None:
        return None
    if isinstance(obj, str):
        return _extract_first_video_ref(obj)
    if isinstance(obj, list):
        for item in obj:
            ref = _extract_first_video_ref_from_obj(item)
            if ref:
                return ref
        return None
    if isinstance(obj, dict):
        for path in (
            ("generated_assets", "final_video_url"),
            ("generated_assets", "local_url"),
            ("generated_assets", "url"),
            ("final_video_url",),
            ("local_url",),
        ):
            value = _nested_value(obj, *path)
            ref = _extract_first_video_ref_from_obj(value)
            if ref:
                return ref

        for key in ("video_url", "file_url", "url", "href", "download_url"):
            value = obj.get(key)
            if isinstance(value, str):
                ref = _extract_first_video_ref(value)
                if ref:
                    return ref
            ref = _extract_first_video_ref_from_obj(value)
            if ref:
                return ref

        for key in (
            "media",
            "result",
            "response",
            "content",
            "delta",
            "message",
            "tool_calls",
            "choices",
            "parts",
            "candidates",
        ):
            ref = _extract_first_video_ref_from_obj(obj.get(key))
            if ref:
                return ref

        for s in _iter_strings(obj):
            ref = _extract_first_video_ref(s)
            if ref:
                return ref
    return None


def _clamp_int(value: Any, *, default: int, min_value: int, max_value: int) -> int:
    try:
        value_int = int(value)
    except (TypeError, ValueError):
        return default
    return max(min_value, min(max_value, value_int))


def _parse_api_keys(conf: dict) -> list[str]:
    """Parse Flow2API api key settings.

    Accepts:
    - api_keys: ["k1", "k2"]  (preferred)
    - api_keys: "k1,k2"      (legacy / convenience)
    - api_key: "k1"          (legacy)
    """
    if not isinstance(conf, dict):
        return []
    raw = conf.get("api_keys", None)
    if raw is None or raw == []:
        raw = conf.get("api_key", None)

    if isinstance(raw, str):
        return [k.strip() for k in raw.split(",") if k.strip()]
    if isinstance(raw, list):
        return [str(k).strip() for k in raw if str(k).strip()]
    return []


def normalize_flow2api_chat_url(raw: str) -> str:
    """Normalize Flow2API chat.completions endpoint URL.

    Flow2API README uses:
      POST http://host:8000/v1/chat/completions

    Users may paste either:
    - http://host:8000
    - http://host:8000/v1
    - http://host:8000/v1/chat/completions
    """
    s = str(raw or "").strip()
    if not s:
        return ""
    s = s.rstrip("/")

    try:
        parts = urlsplit(s)
    except Exception:
        return s

    if not parts.scheme or not parts.netloc:
        return s

    path = (parts.path or "").rstrip("/")
    lower = path.lower()

    if lower.endswith("/v1/chat/completions"):
        final_path = path
    elif lower.endswith("/v1"):
        final_path = f"{path}/chat/completions"
    else:
        final_path = f"{path}/v1/chat/completions"

    return urlunsplit((parts.scheme, parts.netloc, final_path, "", "")).rstrip("/")


class GeminiFlow2ApiBackend:
    """Flow2API 风格的 Gemini 出图后端（支持文生图 + 图生图）。"""

    def __init__(self, *, imgr, settings: dict):
        self.imgr = imgr
        conf = settings if isinstance(settings, dict) else {}

        self.api_url: str = normalize_flow2api_chat_url(conf.get("api_url"))
        self.model: str = str(conf.get("model") or "").strip()
        self.timeout: int = _clamp_int(
            conf.get("timeout", 120), default=120, min_value=1, max_value=3600
        )

        self.use_proxy: bool = bool(conf.get("use_proxy", False))
        self.proxy_url: str = str(conf.get("proxy_url") or "").strip()

        self.api_keys = _parse_api_keys(conf)
        self._key_index = 0
        self._key_lock = asyncio.Lock()

        self._session: aiohttp.ClientSession | None = None
        self._session_lock = asyncio.Lock()

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            async with self._session_lock:
                if self._session is None or self._session.closed:
                    timeout = aiohttp.ClientTimeout(total=float(self.timeout))
                    connector = aiohttp.TCPConnector(
                        limit=10, limit_per_host=5, ttl_dns_cache=300
                    )
                    self._session = aiohttp.ClientSession(
                        timeout=timeout, connector=connector
                    )
        return self._session

    async def _next_key(self) -> str:
        async with self._key_lock:
            if not self.api_keys:
                raise RuntimeError("Flow2API API Key 未配置")
            key = self.api_keys[self._key_index]
            self._key_index = (self._key_index + 1) % len(self.api_keys)
            return key

    def _proxy(self) -> str | None:
        return self.proxy_url if self.use_proxy and self.proxy_url else None

    @staticmethod
    def _resolution_hint(resolution: str | None) -> str:
        r = (resolution or "").strip().upper()
        if not r:
            return ""
        if r in {"1K", "2K", "4K"}:
            return f" Target resolution: {r}."
        if "X" in r:
            return f" Target size: {r}."
        return ""

    def _build_user_text(self, prompt: str, *, resolution: str | None) -> str:
        # 尽量与官方示例保持一致：content 直接使用用户提示词（避免网关对提示词模板敏感导致失败）。
        # 分辨率提示不强制拼接，以免改变模型行为；需要的话请在 prompt 内自行表达。
        p = (prompt or "").strip()
        return p or "a high quality image"

    async def _request_stream_text(self, payload: dict, headers: dict) -> str:
        session = await self._get_session()
        proxy = self._proxy()
        t0 = time.perf_counter()

        async with session.post(
            self.api_url, json=payload, headers=headers, proxy=proxy
        ) as resp:
            if resp.status != 200:
                text = await resp.text()
                if resp.status == 405:
                    raise RuntimeError(
                        "Flow2API 请求失败 HTTP 405(Method Not Allowed)："
                        f"{text[:300]}；请确认 api_url 指向 /v1/chat/completions 且为 POST"
                        f"（当前: {self.api_url}）"
                    )
                raise RuntimeError(
                    f"Flow2API 请求失败 HTTP {resp.status}: {text[:300]} (url={self.api_url})"
                )

            ctype = (resp.headers.get("content-type") or "").lower()
            if "application/json" in ctype:
                data = await resp.json()
                image_ref = _extract_first_image_ref_from_obj(data)
                if image_ref:
                    logger.info(
                        "[GeminiFlow2API] 非流式 JSON 命中图片引用, 耗时: %.2fs",
                        time.perf_counter() - t0,
                    )
                    return image_ref
                video_ref = _extract_first_video_ref_from_obj(data)
                if video_ref:
                    logger.info(
                        "[GeminiFlow2API] 非流式 JSON 命中视频引用, 耗时: %.2fs",
                        time.perf_counter() - t0,
                    )
                    return video_ref
                content = ((data.get("choices") or [{}])[0].get("message") or {}).get(
                    "content"
                ) or ""
                logger.info(
                    "[GeminiFlow2API] 非流式 JSON 响应耗时: %.2fs",
                    time.perf_counter() - t0,
                )
                return str(content)

            buffer = ""
            full = ""
            max_chars = 80_000_000  # 4K data:image 可能超过 27MB，保留保护上限

            async for chunk in resp.content.iter_chunked(1024):
                if not chunk:
                    continue
                buffer += chunk.decode("utf-8", errors="ignore")
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if not line.startswith("data:"):
                        continue
                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        logger.info(
                            "[GeminiFlow2API] SSE 结束, 耗时: %.2fs",
                            time.perf_counter() - t0,
                        )
                        return full
                    try:
                        obj = json.loads(data_str)
                    except Exception:
                        continue

                    chunk_image_ref = _extract_first_image_ref_from_obj(obj)
                    if chunk_image_ref:
                        chunk_image_ref = _rewrite_flow2api_media_ref(
                            chunk_image_ref, endpoint_url=self.api_url
                        )
                    if chunk_image_ref and chunk_image_ref not in full:
                        full += f"\n{chunk_image_ref}"
                    chunk_video_ref = _extract_first_video_ref_from_obj(obj)
                    if chunk_video_ref:
                        chunk_video_ref = _rewrite_flow2api_media_ref(
                            chunk_video_ref, endpoint_url=self.api_url
                        )
                    if chunk_video_ref and chunk_video_ref not in full:
                        full += f"\n{chunk_video_ref}"

                    choice0 = (obj.get("choices") or [{}])[0]
                    delta = choice0.get("delta") or {}
                    message = choice0.get("message") or {}
                    delta_content = (
                        delta.get("content")
                        if "content" in delta
                        else message.get("content")
                    )
                    if delta_content is None and "reasoning_content" in delta:
                        delta_content = delta.get("reasoning_content")
                    if delta_content is None and "reasoning_content" in message:
                        delta_content = message.get("reasoning_content")

                    def _content_to_text(value: Any) -> str:
                        if value is None:
                            return ""
                        if isinstance(value, str):
                            return value
                        if isinstance(value, list):
                            return "".join(_content_to_text(x) for x in value)
                        if isinstance(value, dict):
                            # multimodal chunks: {"type":"text","text":...}
                            text = value.get("text")
                            if isinstance(text, str) and text:
                                return text
                            # {"type":"image_url","image_url":{"url":"..."}}
                            image_url = value.get("image_url")
                            if isinstance(image_url, dict):
                                url = image_url.get("url")
                                if isinstance(url, str) and url:
                                    return url
                            url = value.get("url")
                            if isinstance(url, str) and url:
                                return url
                            return str(value)
                        return str(value)

                    full += _content_to_text(delta_content)

                    image_ref = _extract_first_image_ref(full)
                    video_ref = _extract_first_video_ref(full)

                    # 仅当出现 URL 类媒体引用时提前结束；
                    # data:image 在流式中可能尚未完整，提前返回会导致 base64 被截断。
                    if (
                        chunk_image_ref
                        and chunk_image_ref.startswith(("http://", "https://"))
                    ) or chunk_video_ref:
                        logger.info(
                            "[GeminiFlow2API] 提前命中结构化媒体引用, 耗时: %.2fs",
                            time.perf_counter() - t0,
                        )
                        return full
                    if (
                        image_ref and image_ref.startswith(("http://", "https://"))
                    ) or video_ref:
                        logger.info(
                            "[GeminiFlow2API] 提前命中媒体引用, 耗时: %.2fs",
                            time.perf_counter() - t0,
                        )
                        return full

                    if len(full) > max_chars:
                        raise RuntimeError(
                            "Flow2API 返回内容过大，已终止解析（可能是服务异常输出）"
                        )

            logger.info(
                "[GeminiFlow2API] SSE 读完但无 [DONE], 耗时: %.2fs",
                time.perf_counter() - t0,
            )
            tail = buffer.strip()
            if tail.startswith("data:"):
                data_str = tail[5:].strip()
                if data_str and data_str != "[DONE]":
                    try:
                        obj = json.loads(data_str)
                    except Exception:
                        pass
                    else:
                        chunk_image_ref = _extract_first_image_ref_from_obj(obj)
                        if chunk_image_ref:
                            chunk_image_ref = _rewrite_flow2api_media_ref(
                                chunk_image_ref, endpoint_url=self.api_url
                            )
                        if chunk_image_ref and chunk_image_ref not in full:
                            full += f"\n{chunk_image_ref}"
                        chunk_video_ref = _extract_first_video_ref_from_obj(obj)
                        if chunk_video_ref:
                            chunk_video_ref = _rewrite_flow2api_media_ref(
                                chunk_video_ref, endpoint_url=self.api_url
                            )
                        if chunk_video_ref and chunk_video_ref not in full:
                            full += f"\n{chunk_video_ref}"

                        choice0 = (obj.get("choices") or [{}])[0]
                        delta = choice0.get("delta") or {}
                        message = choice0.get("message") or {}
                        delta_content = (
                            delta.get("content")
                            if "content" in delta
                            else message.get("content")
                        )
                        if delta_content is None and "reasoning_content" in delta:
                            delta_content = delta.get("reasoning_content")
                        if delta_content is None and "reasoning_content" in message:
                            delta_content = message.get("reasoning_content")

                        def _content_to_text_tail(value: Any) -> str:
                            if value is None:
                                return ""
                            if isinstance(value, str):
                                return value
                            if isinstance(value, list):
                                return "".join(_content_to_text_tail(x) for x in value)
                            if isinstance(value, dict):
                                text = value.get("text")
                                if isinstance(text, str) and text:
                                    return text
                                image_url = value.get("image_url")
                                if isinstance(image_url, dict):
                                    url = image_url.get("url")
                                    if isinstance(url, str) and url:
                                        return url
                                url = value.get("url")
                                if isinstance(url, str) and url:
                                    return url
                                return str(value)
                            return str(value)

                        full += _content_to_text_tail(delta_content)
            return full

    async def _save_from_content(self, content: str) -> Path:
        ref = _extract_first_image_ref(content)
        if not ref:
            video = _extract_first_video_ref(content)
            if video:
                raise RuntimeError(
                    f"Flow2API 返回了视频而不是图片：{video}（如果想要视频请用 /视频；如果想要图片请换模型/网关或改用 Gemini 原生）"
                )
            snippet = (content or "").strip().replace("\n", " ")[:200]
            raise RuntimeError(f"Flow2API 未返回图片：{snippet}")

        ref = _rewrite_flow2api_media_ref(ref, endpoint_url=self.api_url)

        if ref.startswith("data:image/"):
            ref = re.sub(r"\s+", "", ref)
            try:
                _header, b64_data = ref.split(",", 1)
            except ValueError:
                raise RuntimeError(
                    "Flow2API 返回 data:image 但缺少 base64 数据"
                ) from None
            image_bytes = _decode_base64_bytes((b64_data or "").strip())
            if not image_bytes:
                raise RuntimeError("Flow2API 返回 data:image 但 base64 解码失败")
            return await self.imgr.save_image(image_bytes)

        if ref.startswith(("http://", "https://")):
            return await self.imgr.download_image(ref)

        raise RuntimeError("Flow2API 返回的图片引用格式不支持")

    async def generate(
        self, prompt: str, *, resolution: str | None = None, **_
    ) -> Path:
        if not self.api_url:
            raise RuntimeError("未配置 Flow2API 地址（flow2api.api_url）")
        if not self.model:
            raise RuntimeError("未配置 Flow2API 模型（flow2api.model）")

        key = await self._next_key()
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        }

        user_text = self._build_user_text(prompt, resolution=resolution)
        payload = {
            "model": self.model,
            # Flow2API README 示例：纯文生图时 content 为 string
            "messages": [{"role": "user", "content": user_text}],
            "stream": True,
        }

        content = await self._request_stream_text(payload, headers)
        return await self._save_from_content(content)

    async def edit(
        self, prompt: str, images: list[bytes], *, resolution: str | None = None, **_
    ) -> Path:
        if not images:
            raise ValueError("至少需要一张图片")
        if not self.api_url:
            raise RuntimeError("未配置 Flow2API 地址（flow2api.api_url）")
        if not self.model:
            raise RuntimeError("未配置 Flow2API 模型（flow2api.model）")

        key = await self._next_key()
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        }

        user_text = self._build_user_text(prompt, resolution=resolution)
        parts: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
        for b in images:
            mime, _ = guess_image_mime_and_ext(b)
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime};base64,{base64.b64encode(b).decode()}"
                    },
                }
            )

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": parts}],
            "stream": True,
        }

        content = await self._request_stream_text(payload, headers)
        return await self._save_from_content(content)


class Flow2ApiVideoBackend:
    """Flow2API 视频后端（OpenAI Chat Completions + SSE 流式输出视频 URL）。"""

    def __init__(self, *, settings: dict):
        conf = settings if isinstance(settings, dict) else {}

        self.api_url: str = normalize_flow2api_chat_url(conf.get("api_url"))
        self.model: str = str(conf.get("model") or "").strip()
        self.timeout: int = _clamp_int(
            conf.get("timeout", 300), default=300, min_value=1, max_value=3600
        )

        self.use_proxy: bool = bool(conf.get("use_proxy", False))
        self.proxy_url: str = str(conf.get("proxy_url") or "").strip()

        self.api_keys = _parse_api_keys(conf)
        self._key_index = 0
        self._key_lock = asyncio.Lock()

        self._session: aiohttp.ClientSession | None = None
        self._session_lock = asyncio.Lock()

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            async with self._session_lock:
                if self._session is None or self._session.closed:
                    timeout = aiohttp.ClientTimeout(total=float(self.timeout))
                    connector = aiohttp.TCPConnector(
                        limit=10, limit_per_host=5, ttl_dns_cache=300
                    )
                    self._session = aiohttp.ClientSession(
                        timeout=timeout, connector=connector
                    )
        return self._session

    async def _next_key(self) -> str:
        async with self._key_lock:
            if not self.api_keys:
                raise RuntimeError("Flow2API API Key 未配置")
            key = self.api_keys[self._key_index]
            self._key_index = (self._key_index + 1) % len(self.api_keys)
            return key

    def _proxy(self) -> str | None:
        return self.proxy_url if self.use_proxy and self.proxy_url else None

    async def _request_stream_text(self, payload: dict, headers: dict) -> str:
        session = await self._get_session()
        proxy = self._proxy()
        t0 = time.perf_counter()

        async with session.post(
            self.api_url, json=payload, headers=headers, proxy=proxy
        ) as resp:
            if resp.status != 200:
                text = await resp.text()
                if resp.status == 405:
                    raise RuntimeError(
                        "Flow2API 请求失败 HTTP 405(Method Not Allowed)："
                        f"{text[:300]}；请确认 api_url 指向 /v1/chat/completions 且为 POST"
                        f"（当前: {self.api_url}）"
                    )
                raise RuntimeError(
                    f"Flow2API 请求失败 HTTP {resp.status}: {text[:300]} (url={self.api_url})"
                )

            ctype = (resp.headers.get("content-type") or "").lower()
            if "application/json" in ctype:
                data = await resp.json()
                video_ref = _extract_first_video_ref_from_obj(data)
                if video_ref:
                    logger.info(
                        "[Flow2API-Video] 非流式 JSON 命中视频引用, 耗时: %.2fs",
                        time.perf_counter() - t0,
                    )
                    return video_ref
                image_ref = _extract_first_image_ref_from_obj(data)
                if image_ref:
                    logger.info(
                        "[Flow2API-Video] 非流式 JSON 命中图片引用, 耗时: %.2fs",
                        time.perf_counter() - t0,
                    )
                    return image_ref
                content = ((data.get("choices") or [{}])[0].get("message") or {}).get(
                    "content"
                ) or ""
                logger.info(
                    "[Flow2API-Video] 非流式 JSON 响应耗时: %.2fs",
                    time.perf_counter() - t0,
                )
                return str(content)

            buffer = ""
            full = ""
            max_chars = 8_000_000

            async for chunk in resp.content.iter_chunked(1024):
                if not chunk:
                    continue
                buffer += chunk.decode("utf-8", errors="ignore")
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if not line.startswith("data:"):
                        continue
                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        logger.info(
                            "[Flow2API-Video] SSE 结束, 耗时: %.2fs",
                            time.perf_counter() - t0,
                        )
                        return full
                    try:
                        obj = json.loads(data_str)
                    except Exception:
                        continue

                    chunk_video_ref = _extract_first_video_ref_from_obj(obj)
                    if chunk_video_ref:
                        chunk_video_ref = _rewrite_flow2api_media_ref(
                            chunk_video_ref, endpoint_url=self.api_url
                        )
                    if chunk_video_ref and chunk_video_ref not in full:
                        full += f"\n{chunk_video_ref}"
                    chunk_image_ref = _extract_first_image_ref_from_obj(obj)
                    if chunk_image_ref:
                        chunk_image_ref = _rewrite_flow2api_media_ref(
                            chunk_image_ref, endpoint_url=self.api_url
                        )
                    if chunk_image_ref and chunk_image_ref not in full:
                        full += f"\n{chunk_image_ref}"

                    choice0 = (obj.get("choices") or [{}])[0]
                    delta = choice0.get("delta") or {}
                    message = choice0.get("message") or {}
                    delta_content = (
                        delta.get("content")
                        if "content" in delta
                        else message.get("content")
                    )
                    if delta_content is None and "reasoning_content" in delta:
                        delta_content = delta.get("reasoning_content")
                    if delta_content is None and "reasoning_content" in message:
                        delta_content = message.get("reasoning_content")

                    def _content_to_text(value: Any) -> str:
                        if value is None:
                            return ""
                        if isinstance(value, str):
                            return value
                        if isinstance(value, list):
                            return "".join(_content_to_text(x) for x in value)
                        if isinstance(value, dict):
                            text = value.get("text")
                            if isinstance(text, str) and text:
                                return text
                            image_url = value.get("image_url")
                            if isinstance(image_url, dict):
                                url = image_url.get("url")
                                if isinstance(url, str) and url:
                                    return url
                            url = value.get("url")
                            if isinstance(url, str) and url:
                                return url
                            return str(value)
                        return str(value)

                    full += _content_to_text(delta_content)

                    if len(full) > max_chars:
                        raise RuntimeError(
                            "Flow2API 返回内容过大，已终止解析（可能是服务异常输出）"
                        )

                    # 视频优先：命中 URL 类媒体引用即可提前结束
                    video_ref = _extract_first_video_ref(full)
                    image_ref = _extract_first_image_ref(full)
                    if chunk_video_ref or (
                        chunk_image_ref
                        and chunk_image_ref.startswith(("http://", "https://"))
                    ):
                        logger.info(
                            "[Flow2API-Video] 提前命中结构化媒体引用, 耗时: %.2fs",
                            time.perf_counter() - t0,
                        )
                        return full
                    if video_ref or (
                        image_ref and image_ref.startswith(("http://", "https://"))
                    ):
                        logger.info(
                            "[Flow2API-Video] 提前命中媒体引用, 耗时: %.2fs",
                            time.perf_counter() - t0,
                        )
                        return full

            logger.info(
                "[Flow2API-Video] SSE 读完但无 [DONE], 耗时: %.2fs",
                time.perf_counter() - t0,
            )
            tail = buffer.strip()
            if tail.startswith("data:"):
                data_str = tail[5:].strip()
                if data_str and data_str != "[DONE]":
                    try:
                        obj = json.loads(data_str)
                    except Exception:
                        pass
                    else:
                        chunk_video_ref = _extract_first_video_ref_from_obj(obj)
                        if chunk_video_ref:
                            chunk_video_ref = _rewrite_flow2api_media_ref(
                                chunk_video_ref, endpoint_url=self.api_url
                            )
                        if chunk_video_ref and chunk_video_ref not in full:
                            full += f"\n{chunk_video_ref}"
                        chunk_image_ref = _extract_first_image_ref_from_obj(obj)
                        if chunk_image_ref:
                            chunk_image_ref = _rewrite_flow2api_media_ref(
                                chunk_image_ref, endpoint_url=self.api_url
                            )
                        if chunk_image_ref and chunk_image_ref not in full:
                            full += f"\n{chunk_image_ref}"

                        choice0 = (obj.get("choices") or [{}])[0]
                        delta = choice0.get("delta") or {}
                        message = choice0.get("message") or {}
                        delta_content = (
                            delta.get("content")
                            if "content" in delta
                            else message.get("content")
                        )
                        if delta_content is None and "reasoning_content" in delta:
                            delta_content = delta.get("reasoning_content")
                        if delta_content is None and "reasoning_content" in message:
                            delta_content = message.get("reasoning_content")

                        def _content_to_text_tail(value: Any) -> str:
                            if value is None:
                                return ""
                            if isinstance(value, str):
                                return value
                            if isinstance(value, list):
                                return "".join(_content_to_text_tail(x) for x in value)
                            if isinstance(value, dict):
                                text = value.get("text")
                                if isinstance(text, str) and text:
                                    return text
                                image_url = value.get("image_url")
                                if isinstance(image_url, dict):
                                    url = image_url.get("url")
                                    if isinstance(url, str) and url:
                                        return url
                                url = value.get("url")
                                if isinstance(url, str) and url:
                                    return url
                                return str(value)
                            return str(value)

                        full += _content_to_text_tail(delta_content)
            return full

    async def generate_video_url(
        self, *, prompt: str, image_bytes: bytes | None = None
    ) -> str:
        if not self.api_url:
            raise RuntimeError("未配置 Flow2API 地址（flow2api_video.api_url）")
        if not self.model:
            raise RuntimeError("未配置 Flow2API 模型（flow2api_video.model）")

        key = await self._next_key()
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        }

        p = (prompt or "").strip()
        if not p:
            raise ValueError("缺少提示词")

        if image_bytes:
            mime, _ = guess_image_mime_and_ext(image_bytes)
            image_url = f"data:{mime};base64,{base64.b64encode(image_bytes).decode()}"
            content: Any = [
                {"type": "text", "text": p},
                {"type": "image_url", "image_url": {"url": image_url}},
            ]
        else:
            # 与官方示例一致：纯文本 prompt
            content = p

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "stream": True,
        }

        content_text = await self._request_stream_text(payload, headers)
        ref = _extract_first_video_ref(content_text)
        if ref:
            return _rewrite_flow2api_media_ref(ref, endpoint_url=self.api_url)
        img = _extract_first_image_ref(content_text)
        if img:
            raise RuntimeError("Flow2API 返回了图片而不是视频")
        snippet = (content_text or "").strip().replace("\n", " ")[:200]
        raise RuntimeError(f"Flow2API 未返回视频：{snippet}")
