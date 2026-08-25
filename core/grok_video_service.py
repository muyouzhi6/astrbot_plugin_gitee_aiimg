"""
Grok 视频生成服务（grok-imagine-0.9）

职责：
- 预设提示词拼接
- Grok /v1/chat/completions 调用
- 超时与重试
- 从响应中提取视频 URL
"""

from __future__ import annotations

import asyncio
import base64
import random
import re
import time
from collections import deque
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qs, quote, urljoin, urlsplit

import httpx

from astrbot.api import logger


def _clamp_int(value: Any, *, default: int, min_value: int, max_value: int) -> int:
    try:
        value_int = int(value)
    except (TypeError, ValueError):
        return default
    return max(min_value, min(max_value, value_int))


def _guess_image_mime(data: bytes) -> str:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    return "image/jpeg"


def _build_data_url(image_bytes: bytes) -> str:
    mime = _guess_image_mime(image_bytes)
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def _normalize_video_resolution(value: Any) -> str:
    """Normalize common numeric resolution values to the xAI enum spelling."""
    raw = str(value or "").strip()
    if not raw:
        return ""
    normalized = raw.lower()
    if normalized in {"480", "480p"}:
        return "480p"
    if normalized in {"720", "720p"}:
        return "720p"
    if normalized in {"1080", "1080p"}:
        return "1080p"
    return raw


def _chat_completions_endpoint(server_url: str) -> str:
    base = (server_url or "https://api.x.ai").strip().rstrip("/")
    path = urlsplit(base).path.rstrip("/")
    if path.endswith("/chat/completions"):
        return base
    if path.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def _video_generations_endpoint(server_url: str) -> str:
    base = (server_url or "https://api.x.ai").strip().rstrip("/")
    path = urlsplit(base).path.rstrip("/")
    if path.endswith("/videos/generations"):
        return base
    if path.endswith("/v1"):
        return f"{base}/videos/generations"
    return f"{base}/v1/videos/generations"


def _origin_from_url(url: str) -> str:
    parts = urlsplit(url)
    if not parts.scheme or not parts.netloc:
        return ""
    return f"{parts.scheme}://{parts.netloc}"


def _format_exception(exc: Exception) -> str:
    """Keep transport failures diagnosable when their string is empty."""
    detail = str(exc).strip()
    if detail:
        return detail
    return type(exc).__name__


@dataclass(frozen=True)
class VideoResult:
    """Video URL plus optional headers required to download it."""

    url: str
    download_headers: dict[str, str] | None = None


def _looks_like_proxy_video_url(url: str) -> bool:
    lowered = (url or "").strip().lower()
    if "generated_video" in lowered:
        return True

    # Some gateways return extension-less links like:
    # https://.../images/p_<base64(/users/.../generated_video.mp4)>
    try:
        path = urlsplit(url).path or ""
    except Exception:
        path = ""
    match = re.search(r"/images/p_([A-Za-z0-9+/_=-]+)", path)
    if not match:
        return False

    token = match.group(1)
    padded = token + ("=" * (-len(token) % 4))
    for decoder in (base64.urlsafe_b64decode, base64.b64decode):
        try:
            decoded = decoder(padded.encode("ascii")).decode("utf-8", errors="ignore")
        except Exception:
            continue
        decoded_l = decoded.lower()
        if "generated_video" in decoded_l:
            return True
        if any(ext in decoded_l for ext in (".mp4", ".webm", ".mov")):
            return True
    return False


def _is_valid_video_url(url: str) -> bool:
    if not isinstance(url, str):
        return False
    url = url.strip()
    if len(url) < 10:
        return False
    if not url.startswith(("http://", "https://")):
        return False
    lowered = url.lower()
    if any(c in url for c in ["<", ">", '"', "'", "\n", "\r", "\t"]):
        return False
    if any(ext in lowered for ext in (".mp4", ".webm", ".mov")):
        return True

    try:
        parsed = urlsplit(url)
        if (parsed.path or "").rstrip("/") == "/v1/files/video" and parse_qs(
            parsed.query
        ).get("id"):
            return True
    except Exception:
        pass

    if _looks_like_proxy_video_url(url):
        return True
    return False


_VIDEO_URL_RE = re.compile(
    r"(https?://[^\s<>\"')\]\}]+?\.(?:mp4|webm|mov)(?:\?[^\s<>\"')\]\}]*)?)",
    re.IGNORECASE,
)
_GENERIC_URL_RE = re.compile(
    r"(https?://[^\s<>\"')\]\}]+)",
    re.IGNORECASE,
)


def _extract_video_url_from_content(content: str) -> str | None:
    if not content:
        return None

    # HTML <video src="...">
    if "<video" in content and "src=" in content:
        html_patterns = [
            r'<video[^>]*src=["\']([^"\'>]+)["\'][^>]*>',
            r'src=["\']([^"\'>]+\.mp4[^"\'>]*)["\']',
        ]
        for pattern in html_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                url = match.group(1).strip()
                if _is_valid_video_url(url):
                    return url

    # Direct URL
    match = _VIDEO_URL_RE.search(content)
    if match:
        url = match.group(1).strip()
        if _is_valid_video_url(url):
            return url

    # Markdown [text](url)
    md_patterns = [
        r"!?\[[^\]]*\]\(([^\)]+\.(?:mp4|webm|mov)[^\)]*)\)",
        r"!?\[[^\]]*\]:\s*([^\s]+\.(?:mp4|webm|mov)[^\s]*)",
    ]
    for pattern in md_patterns:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            url = match.group(1).strip()
            if _is_valid_video_url(url):
                return url

    # Generic URL fallback (for extension-less proxy video URLs)
    for match in _GENERIC_URL_RE.finditer(content):
        url = match.group(1).strip().rstrip(".,;")
        if _is_valid_video_url(url):
            return url

    return None


def _deep_find_video_url(
    data: Any, *, max_depth: int = 6, max_nodes: int = 2000
) -> str | None:
    """在不确定响应结构时，做一次有限深度的全局扫描，尽量找到视频 URL。"""
    queue: deque[tuple[Any, int]] = deque([(data, 0)])
    seen = 0

    while queue:
        obj, depth = queue.popleft()
        seen += 1
        if seen > max_nodes:
            return None
        if depth > max_depth:
            continue

        if isinstance(obj, str):
            url = _extract_video_url_from_content(obj) or (
                obj.strip() if _is_valid_video_url(obj) else None
            )
            if url:
                return url
            continue

        if isinstance(obj, dict):
            for key in ("video_url", "file_url", "url", "href", "download_url"):
                val = obj.get(key)
                if isinstance(val, str) and _is_valid_video_url(val):
                    return val.strip()
                if isinstance(val, dict):
                    nested_url = val.get("url") or val.get("file_url")
                    if isinstance(nested_url, str) and _is_valid_video_url(nested_url):
                        return nested_url.strip()

            for val in obj.values():
                queue.append((val, depth + 1))
            continue

        if isinstance(obj, list):
            for item in obj:
                queue.append((item, depth + 1))
            continue

    return None


def _extract_video_url_from_response(
    response_data: Any,
) -> tuple[str | None, str | None]:
    """
    Returns: (video_url, error_message)
    """
    try:
        if not isinstance(response_data, dict):
            return None, f"无效的响应格式: {type(response_data).__name__}"

        direct = response_data.get("video_url")
        if isinstance(direct, str) and _is_valid_video_url(direct):
            return direct, None

        choices = response_data.get("choices")
        if not isinstance(choices, list) or not choices:
            return None, "API 响应缺少 choices"

        choice0 = choices[0]
        if not isinstance(choice0, dict):
            return None, "choices[0] 格式错误"

        message = choice0.get("message")
        if not isinstance(message, dict):
            return None, "choices[0] 缺少 message"

        content = message.get("content")
        if isinstance(content, str):
            url = _extract_video_url_from_content(content)
            if url:
                return url, None
        elif isinstance(content, list):
            # OpenAI 风格：content = [{"type":"text","text":"..."}, ...]
            for part in content:
                if isinstance(part, str):
                    url = _extract_video_url_from_content(part)
                    if url:
                        return url, None
                if isinstance(part, dict):
                    part_url = (
                        part.get("url")
                        or part.get("video_url")
                        or (
                            part.get("video_url", {})
                            if isinstance(part.get("video_url"), dict)
                            else None
                        )
                    )
                    if isinstance(part_url, str) and _is_valid_video_url(part_url):
                        return part_url, None
                    if isinstance(part_url, dict):
                        nested = part_url.get("url")
                        if isinstance(nested, str) and _is_valid_video_url(nested):
                            return nested, None
                    text = part.get("text")
                    if isinstance(text, str):
                        url = _extract_video_url_from_content(text)
                        if url:
                            return url, None

        # 结构化字段（不同代理/实现可能放在这里）
        for field in ("attachments", "media", "files"):
            items = message.get(field)
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict):
                        url = (
                            item.get("url")
                            or item.get("file_url")
                            or item.get("video_url")
                        )
                        if isinstance(url, str) and _is_valid_video_url(url):
                            return url, None

        # 兜底：全局扫描
        deep = _deep_find_video_url(response_data)
        if deep:
            return deep, None

        content_preview = ""
        if isinstance(content, str):
            content_preview = content[:200]
        logger.warning(
            f"[GrokVideo] 未能提取视频 URL，content 片段: {content_preview}..."
        )
        return None, "未能从 API 响应中提取到有效的视频 URL"
    except Exception as e:
        logger.warning(f"[GrokVideo] URL 提取异常: {e}")
        return None, f"URL 提取失败: {e}"


class GrokVideoService:
    """xAI-compatible asynchronous video generation service."""

    _DONE = {"done", "completed", "succeeded", "success"}
    _FAILED = {"failed", "expired", "error", "cancelled", "canceled", "rejected"}

    def __init__(self, *, settings: dict):
        self.settings = settings if isinstance(settings, dict) else {}
        self.server_url = str(
            self.settings.get("server_url", "https://api.x.ai")
        ).strip()
        self.api_key = str(self.settings.get("api_key", "")).strip()
        self.model = (
            str(self.settings.get("model", "grok-imagine-video-1.5")).strip()
            or "grok-imagine-video-1.5"
        )
        self.duration = _clamp_int(
            self.settings.get("duration", 5),
            default=5,
            min_value=1,
            max_value=15,
        )
        self.aspect_ratio = str(self.settings.get("aspect_ratio") or "").strip()
        self.resolution = _normalize_video_resolution(self.settings.get("resolution"))
        self.timeout_seconds = _clamp_int(
            self.settings.get("timeout_seconds", 600),
            default=600,
            min_value=30,
            max_value=3600,
        )
        self.poll_interval_seconds = max(
            1.0, min(float(self.settings.get("poll_interval_seconds", 5)), 120.0)
        )
        self.max_retries = _clamp_int(
            self.settings.get("max_retries", 2),
            default=2,
            min_value=0,
            max_value=10,
        )
        self.create_max_retries = _clamp_int(
            self.settings.get("create_max_retries", 0),
            default=0,
            min_value=0,
            max_value=3,
        )
        self.retry_delay = _clamp_int(
            self.settings.get("retry_delay", 2),
            default=2,
            min_value=0,
            max_value=60,
        )
        self.presets = self._load_presets()
        self.api_url = _video_generations_endpoint(self.server_url)
        self.base_origin = _origin_from_url(self.api_url)

        logger.info(
            "[GrokVideo] Initialized: model=%s, endpoint=%s, duration=%ss, timeout=%ss",
            self.model,
            self.api_url,
            self.duration,
            self.timeout_seconds,
        )

    def _load_presets(self) -> dict[str, str]:
        presets: dict[str, str] = {}
        for item in self.settings.get("presets", []):
            if isinstance(item, str) and ":" in item:
                key, value = item.split(":", 1)
                if key.strip() and value.strip():
                    presets[key.strip()] = value.strip()
        return presets

    def get_preset_names(self) -> list[str]:
        return list(self.presets.keys())

    def build_prompt(self, prompt: str, preset: str | None = None) -> str:
        value = (prompt or "").strip()
        if preset and preset in self.presets:
            prefix = self.presets[preset]
            return f"{prefix}, {value}" if value else prefix
        return value

    def _absolute_video_url(self, value: Any) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        if text.startswith(("http://", "https://")):
            return text
        if text.startswith("/") and self.base_origin:
            return urljoin(self.base_origin + "/", text.lstrip("/"))
        return ""

    def _video_result(self, url: str, headers: dict[str, str]) -> str | VideoResult:
        parts = urlsplit(url)
        if (
            _origin_from_url(url) == self.base_origin
            and "/v1/videos/" in parts.path
            and parts.path.rstrip("/").endswith("/content")
        ):
            return VideoResult(url, {"Authorization": headers["Authorization"]})
        return url

    def _extract_async_video_url(self, data: Any) -> str:
        if not isinstance(data, dict):
            return ""
        candidates: list[Any] = [data.get("video_url"), data.get("url")]
        video = data.get("video")
        if isinstance(video, dict):
            candidates.extend([video.get("url"), video.get("video_url")])
        metadata = data.get("metadata")
        if isinstance(metadata, dict):
            candidates.extend([metadata.get("url"), metadata.get("video_url")])
        for candidate in candidates:
            url = self._absolute_video_url(candidate)
            if url:
                return url
        return ""

    @staticmethod
    def _task_id(data: Any) -> str:
        if not isinstance(data, dict):
            return ""
        for key in ("request_id", "task_id", "id"):
            value = str(data.get(key) or "").strip()
            if value:
                return value
        return ""

    @staticmethod
    def _error_text(data: Any) -> str:
        if not isinstance(data, dict):
            return str(data)[:300]
        error = data.get("error")
        if isinstance(error, dict):
            return str(error.get("message") or error.get("code") or error)[:300]
        return str(data.get("message") or error or data)[:300]

    async def _request_json(
        self,
        client: httpx.AsyncClient,
        method: str,
        url: str,
        *,
        headers: dict[str, str],
        payload: dict[str, Any] | None = None,
    ) -> Any:
        response = await client.request(
            method,
            url,
            headers=headers,
            json=payload,
        )
        if response.status_code >= 400:
            detail = response.text[:500]
            raise RuntimeError(
                f"Grok API 请求失败 HTTP {response.status_code}: {detail}"
            )
        try:
            return response.json()
        except Exception as exc:
            raise RuntimeError(
                f"Grok API 响应 JSON 解析失败: {exc}, body={response.text[:200]}"
            ) from exc

    async def generate_video_url(
        self,
        prompt: str,
        image_bytes: bytes | None = None,
        *,
        preset: str | None = None,
    ) -> str | VideoResult:
        if not self.api_key:
            raise RuntimeError("Missing API key for video provider (api_key)")
        final_prompt = self.build_prompt(prompt, preset=preset)
        if not final_prompt:
            raise ValueError("缺少提示词")

        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": final_prompt,
            "duration": self.duration,
        }
        if image_bytes:
            payload["image"] = {"url": _build_data_url(image_bytes)}
        if self.aspect_ratio:
            payload["aspect_ratio"] = self.aspect_ratio
        if self.resolution:
            payload["resolution"] = self.resolution

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        timeout = httpx.Timeout(
            connect=10.0,
            read=float(self.timeout_seconds),
            write=10.0,
            pool=float(self.timeout_seconds) + 10.0,
        )
        deadline = time.monotonic() + self.timeout_seconds
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            data: Any | None = None
            last_error: Exception | None = None
            for attempt in range(self.create_max_retries + 1):
                try:
                    logger.info(
                        "[GrokVideo] 创建任务: endpoint=%s, model=%s, duration=%ss",
                        self.api_url,
                        self.model,
                        self.duration,
                    )
                    data = await self._request_json(
                        client,
                        "POST",
                        self.api_url,
                        headers=headers,
                        payload=payload,
                    )
                    break
                except Exception as exc:
                    last_error = exc
                    if attempt >= self.create_max_retries:
                        raise
                    delay = self.retry_delay * (2**attempt) + random.uniform(0, 0.5)
                    logger.warning(
                        "[GrokVideo] 创建失败: %s，%.1fs 后重试",
                        _format_exception(exc),
                        delay,
                    )
                    await asyncio.sleep(delay)

            if data is None:
                raise last_error or RuntimeError("Grok 视频任务创建失败")

            status = (
                str(data.get("status") or "").strip().lower()
                if isinstance(data, dict)
                else ""
            )
            video_url = self._extract_async_video_url(data)
            if video_url and (not status or status in self._DONE):
                return self._video_result(video_url, headers)
            if status in self._FAILED:
                raise RuntimeError(f"Grok 视频任务失败: {self._error_text(data)}")

            task_id = self._task_id(data)
            if not task_id:
                raise RuntimeError(f"Grok API 未返回 request_id: {str(data)[:300]}")
            status_base = self.api_url.removesuffix("/generations")
            status_url = f"{status_base}/{quote(task_id, safe='')}"

            while True:
                if time.monotonic() >= deadline:
                    raise RuntimeError(f"Grok 视频任务超时: request_id={task_id}")
                await asyncio.sleep(self.poll_interval_seconds)
                for poll_attempt in range(self.max_retries + 1):
                    try:
                        data = await self._request_json(
                            client,
                            "GET",
                            status_url,
                            headers=headers,
                        )
                        break
                    except Exception as exc:
                        if poll_attempt >= self.max_retries:
                            raise
                        delay = self.retry_delay * (2**poll_attempt) + random.uniform(
                            0, 0.5
                        )
                        logger.warning(
                            "[GrokVideo] 轮询失败: %s，%.1fs 后重试",
                            _format_exception(exc),
                            delay,
                        )
                        await asyncio.sleep(delay)
                status = (
                    str(data.get("status") or "").strip().lower()
                    if isinstance(data, dict)
                    else ""
                )
                video_url = self._extract_async_video_url(data)
                if video_url and (not status or status in self._DONE):
                    logger.info("[GrokVideo] 成功: request_id=%s", task_id)
                    return self._video_result(video_url, headers)
                if status in self._FAILED:
                    raise RuntimeError(
                        f"Grok 视频任务失败: request_id={task_id}, {self._error_text(data)}"
                    )
                logger.info(
                    "[GrokVideo] 等待任务: request_id=%s, status=%s",
                    task_id,
                    status or "unknown",
                )
