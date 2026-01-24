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
from typing import Any
from urllib.parse import urljoin

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


def _is_valid_video_url(url: str) -> bool:
    if not isinstance(url, str):
        return False
    url = url.strip()
    if len(url) < 10:
        return False
    if not url.startswith(("http://", "https://")):
        return False
    lowered = url.lower()
    if not any(ext in lowered for ext in (".mp4", ".webm", ".mov")):
        return False
    if any(c in url for c in ['<', '>', '"', "'", "\n", "\r", "\t"]):
        return False
    return True


_VIDEO_URL_RE = re.compile(
    r"(https?://[^\s<>\"')\]\}]+?\.(?:mp4|webm|mov)(?:\?[^\s<>\"')\]\}]*)?)",
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
        r'!?\[[^\]]*\]\(([^\)]+\.(?:mp4|webm|mov)[^\)]*)\)',
        r'!?\[[^\]]*\]:\s*([^\s]+\.(?:mp4|webm|mov)[^\s]*)',
    ]
    for pattern in md_patterns:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            url = match.group(1).strip()
            if _is_valid_video_url(url):
                return url

    return None


def _deep_find_video_url(data: Any, *, max_depth: int = 6, max_nodes: int = 2000) -> str | None:
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
            url = _extract_video_url_from_content(obj) or (obj.strip() if _is_valid_video_url(obj) else None)
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


def _extract_video_url_from_response(response_data: Any) -> tuple[str | None, str | None]:
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
                    part_url = part.get("url") or part.get("video_url") or (part.get("video_url", {}) if isinstance(part.get("video_url"), dict) else None)
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
                        url = item.get("url") or item.get("file_url") or item.get("video_url")
                        if isinstance(url, str) and _is_valid_video_url(url):
                            return url, None

        # 兜底：全局扫描
        deep = _deep_find_video_url(response_data)
        if deep:
            return deep, None

        content_preview = ""
        if isinstance(content, str):
            content_preview = content[:200]
        logger.warning(f"[GrokVideo] 未能提取视频 URL，content 片段: {content_preview}...")
        return None, "未能从 API 响应中提取到有效的视频 URL"
    except Exception as e:
        logger.warning(f"[GrokVideo] URL 提取异常: {e}")
        return None, f"URL 提取失败: {e}"


class GrokVideoService:
    def __init__(self, config: dict):
        self.config = config
        self.vconf = config.get("video", {})

        self.enabled: bool = bool(self.vconf.get("enabled", True))
        self.server_url: str = str(self.vconf.get("server_url", "https://api.x.ai")).rstrip("/")
        self.api_key: str = str(self.vconf.get("api_key", "")).strip()
        self.model: str = str(self.vconf.get("model", "grok-imagine-0.9")).strip() or "grok-imagine-0.9"

        self.timeout_seconds: int = _clamp_int(
            self.vconf.get("timeout_seconds", 180),
            default=180,
            min_value=1,
            max_value=3600,
        )
        self.max_retries: int = _clamp_int(
            self.vconf.get("max_retries", 2),
            default=2,
            min_value=0,
            max_value=10,
        )
        self.empty_response_retry: int = _clamp_int(
            self.vconf.get("empty_response_retry", 2),
            default=2,
            min_value=0,
            max_value=10,
        )
        self.retry_delay: int = _clamp_int(
            self.vconf.get("retry_delay", 2),
            default=2,
            min_value=0,
            max_value=60,
        )

        self.presets: dict[str, str] = self._load_presets()

        self.api_url = urljoin(self.server_url + "/", "v1/chat/completions")

        logger.info(
            f"[GrokVideo] 初始化完成: enabled={self.enabled}, model={self.model}, "
            f"timeout={self.timeout_seconds}s, retries={self.max_retries}, "
            f"empty_retry={self.empty_response_retry}, presets={len(self.presets)}"
        )

    def _load_presets(self) -> dict[str, str]:
        presets: dict[str, str] = {}
        items = self.vconf.get("presets", [])
        for item in items:
            if isinstance(item, str) and ":" in item:
                key, val = item.split(":", 1)
                key = key.strip()
                val = val.strip()
                if key and val:
                    presets[key] = val
        return presets

    def get_preset_names(self) -> list[str]:
        return list(self.presets.keys())

    def build_prompt(self, prompt: str, preset: str | None = None) -> str:
        prompt = (prompt or "").strip()
        if preset and preset in self.presets:
            preset_prompt = self.presets[preset]
            if prompt:
                return f"{preset_prompt}, {prompt}"
            return preset_prompt
        return prompt

    async def generate_video_url(
        self,
        prompt: str,
        image_bytes: bytes,
        *,
        preset: str | None = None,
    ) -> str:
        if not self.enabled:
            raise RuntimeError("视频功能已禁用")
        if not self.api_key:
            raise RuntimeError("未配置 Grok API Key（video.api_key）")
        if not image_bytes:
            raise ValueError("缺少参考图")

        final_prompt = self.build_prompt(prompt, preset=preset)
        if not final_prompt:
            raise ValueError("缺少提示词")

        image_url = _build_data_url(image_bytes)

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": final_prompt},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ],
        }

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

        async def _request_once() -> Any:
            async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
                resp = await client.post(self.api_url, json=payload, headers=headers)

            if resp.status_code != 200:
                detail = resp.text[:500]
                if resp.status_code == 401:
                    raise RuntimeError("Grok API Key 无效或已过期 (401)")
                if resp.status_code == 403:
                    raise RuntimeError("Grok API 访问被拒绝 (403)")
                raise RuntimeError(f"Grok API 请求失败 HTTP {resp.status_code}: {detail}")

            try:
                return resp.json()
            except Exception as e:
                raise RuntimeError(f"API 响应 JSON 解析失败: {e}, body={resp.text[:200]}") from e

        async def _request_with_retries() -> Any:
            last_exc: Exception | None = None
            for attempt in range(self.max_retries + 1):
                try:
                    logger.info(
                        f"[GrokVideo] 调用 API attempt={attempt + 1}/{self.max_retries + 1}, "
                        f"prompt={final_prompt[:60]}..."
                    )
                    return await _request_once()
                except Exception as e:
                    last_exc = e
                    if attempt >= self.max_retries:
                        break
                    delay = max(0, self.retry_delay) * (2**attempt) + random.uniform(0, 0.5)
                    logger.warning(f"[GrokVideo] 请求失败: {e}，{delay:.1f}s 后重试...")
                    await asyncio.sleep(delay)
            raise last_exc or RuntimeError("请求失败")

        t_start = time.perf_counter()
        last_parse_error: str | None = None

        # 对「200但没有视频 URL」做额外重试（与网络重试分离，提升成功率）
        for attempt in range(self.empty_response_retry + 1):
            data = await _request_with_retries()
            video_url, parse_error = _extract_video_url_from_response(data)
            if video_url:
                t_end = time.perf_counter()
                logger.info(f"[GrokVideo] 成功: 耗时={t_end - t_start:.2f}s, url={video_url[:80]}...")
                return video_url

            last_parse_error = parse_error or "API 响应未包含视频 URL"
            if attempt >= self.empty_response_retry:
                break

            delay = max(0, self.retry_delay) * (2**attempt) + random.uniform(0, 0.5)
            logger.warning(f"[GrokVideo] 响应无视频URL: {last_parse_error}，{delay:.1f}s 后重试...")
            await asyncio.sleep(delay)

        raise RuntimeError(f"Grok 视频生成失败: {last_parse_error}")
