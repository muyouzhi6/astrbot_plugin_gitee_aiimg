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
from urllib.parse import urlsplit, urlunsplit

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
_HTML_VIDEO_RE = re.compile(r'<video[^>]*src=["\']([^"\'>]+)["\']', re.IGNORECASE)
_VIDEO_URL_RE = re.compile(
    r"(https?://[^\s<>\"')\]]+?\.(?:mp4|webm|mov)(?:\?[^\s<>\"')\]]*)?)",
    re.IGNORECASE,
)


def _looks_like_video_url(url: str) -> bool:
    u = (url or "").strip().lower()
    if not u.startswith(("http://", "https://")):
        return False
    if any(ext in u for ext in (".mp4", ".webm", ".mov")):
        return True
    if "generated_video" in u:
        return True
    return False


def _extract_first_image_ref(text: str) -> str | None:
    s = (text or "").strip()
    if not s:
        return None
    m = _MD_IMAGE_RE.search(s)
    if m:
        ref = m.group(1).strip()
        return None if _looks_like_video_url(ref) else (ref or None)
    m = _DATA_IMAGE_RE.search(s)
    if m:
        return m.group(1).strip()
    m = _HTML_IMG_RE.search(s)
    if m:
        ref = m.group(1).strip()
        return None if _looks_like_video_url(ref) else (ref or None)
    m = _IMG_URL_RE.search(s)
    if m:
        ref = m.group(1).strip()
        return None if _looks_like_video_url(ref) else (ref or None)
    if s.startswith(("http://", "https://")) and not _looks_like_video_url(s):
        return s
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
    if _looks_like_video_url(s):
        return s
    return None


def _clamp_int(value: Any, *, default: int, min_value: int, max_value: int) -> int:
    try:
        value_int = int(value)
    except (TypeError, ValueError):
        return default
    return max(min_value, min(max_value, value_int))


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

        raw_keys = conf.get("api_keys", [])
        self.api_keys = [str(k).strip() for k in (raw_keys or []) if str(k).strip()]
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
                raise RuntimeError("Gemini(Flow2API) API Key 未配置")
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
        p = (prompt or "").strip() or "a high quality image"
        hint = self._resolution_hint(resolution)
        return (
            f"{p}\n\n"
            "Return ONLY one direct https image URL (png/jpg/webp/gif) OR one data:image/...;base64,... .\n"
            "Do NOT return video/mp4, HTML, or any explanation.\n"
            f"{hint}"
        )

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
            max_chars = 8_000_000  # 防止异常输出导致内存膨胀

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

                    choice0 = (obj.get("choices") or [{}])[0]
                    delta = choice0.get("delta") or {}
                    delta_content = delta.get("content")
                    if isinstance(delta_content, str):
                        full += delta_content
                    elif delta_content is not None:
                        full += str(delta_content)

                    if len(full) > max_chars:
                        raise RuntimeError(
                            "Flow2API 返回内容过大，已终止解析（可能是服务异常输出）"
                        )

                    # 提前发现 URL / data:image 就可提前结束（避免一直等）
                    if _extract_first_image_ref(full) or _extract_first_video_ref(full):
                        logger.info(
                            "[GeminiFlow2API] 提前命中媒体引用, 耗时: %.2fs",
                            time.perf_counter() - t0,
                        )
                        return full

            logger.info(
                "[GeminiFlow2API] SSE 读完但无 [DONE], 耗时: %.2fs",
                time.perf_counter() - t0,
            )
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

        if ref.startswith("data:image/"):
            try:
                _header, b64_data = ref.split(",", 1)
            except ValueError:
                raise RuntimeError(
                    "Flow2API 返回 data:image 但缺少 base64 数据"
                ) from None
            image_bytes = base64.b64decode((b64_data or "").strip())
            return await self.imgr.save_image(image_bytes)

        if ref.startswith(("http://", "https://")):
            return await self.imgr.download_image(ref)

        raise RuntimeError("Flow2API 返回的图片引用格式不支持")

    async def generate(
        self, prompt: str, *, resolution: str | None = None, **_
    ) -> Path:
        if not self.api_url:
            raise RuntimeError("未配置 Flow2API 地址（gemini_native.api_url）")
        if not self.model:
            raise RuntimeError("未配置 Flow2API 模型（gemini_native.model）")

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
            raise RuntimeError("未配置 Flow2API 地址（gemini_native.api_url）")
        if not self.model:
            raise RuntimeError("未配置 Flow2API 模型（gemini_native.model）")

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
