"""3365 SD2.0 asynchronous text/image-to-video service."""

from __future__ import annotations

import asyncio
import base64
import email.utils
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, ClassVar
from urllib.parse import quote, urljoin, urlsplit

import httpx
from astrbot.api import logger

from .image_format import guess_image_mime_and_ext


@dataclass(frozen=True)
class VideoResult:
    """Video URL plus headers needed by an authenticated content endpoint."""

    url: str
    download_headers: dict[str, str] | None = None


def _clamp_int(value: Any, *, default: int, min_value: int, max_value: int) -> int:
    try:
        value_int = int(value)
    except (TypeError, ValueError):
        return default
    return max(min_value, min(max_value, value_int))


def _clamp_float(
    value: Any, *, default: float, min_value: float, max_value: float
) -> float:
    try:
        value_float = float(value)
    except (TypeError, ValueError):
        return default
    return max(min_value, min(max_value, value_float))


def _api_keys(settings: dict[str, Any]) -> list[str]:
    values = settings.get("api_keys", [])
    if not values and settings.get("api_key"):
        values = [settings.get("api_key")]
    if not isinstance(values, list):
        values = [values]
    return [str(value).strip() for value in values if str(value or "").strip()]


def _env_key(settings: dict[str, Any]) -> str:
    name = str(settings.get("api_key_env") or "").strip()
    return str(os.environ.get(name) or "").strip() if name else ""


def _endpoint(base_url: str) -> str:
    base = (base_url or "https://api.3365api.cn").strip().rstrip("/")
    if base.endswith("/v1/video/generations"):
        return base
    if base.endswith("/v1"):
        return f"{base}/video/generations"
    return f"{base}/v1/video/generations"


def _origin(url: str) -> str:
    parts = urlsplit(url)
    return f"{parts.scheme}://{parts.netloc}" if parts.scheme and parts.netloc else ""


def _absolute_url(value: Any, *, base_origin: str) -> str:
    text = str(value or "").strip()
    if text.startswith(("http://", "https://")):
        return text
    if text.startswith("/") and base_origin:
        return urljoin(base_origin + "/", text.lstrip("/"))
    return ""


def _walk_values(value: Any):
    if isinstance(value, dict):
        yield value
        for nested in value.values():
            yield from _walk_values(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _walk_values(nested)


def _extract_task_id(data: Any) -> str:
    for item in _walk_values(data):
        for key in ("task_id", "request_id", "id"):
            value = str(item.get(key) or "").strip()
            if value:
                return value
    return ""


def _extract_video_url(data: Any, *, base_origin: str) -> str:
    preferred = ("result_url", "video_url", "download_url", "file_url", "url")
    for item in _walk_values(data):
        for key in preferred:
            value = _absolute_url(item.get(key), base_origin=base_origin)
            if value:
                return value
    return ""


def _extract_status(data: Any) -> str:
    for item in _walk_values(data):
        value = str(item.get("status") or "").strip().lower()
        if value:
            return value
    return ""


def _error_text(data: Any) -> str:
    if not isinstance(data, dict):
        return str(data)[:300]
    error = data.get("error")
    if isinstance(error, dict):
        return str(error.get("message") or error.get("code") or error)[:300]
    return str(data.get("message") or data.get("msg") or error or data)[:300]


def _retry_after(value: Any) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        seconds = float(text)
        return seconds if seconds >= 0 else None
    except ValueError:
        pass
    try:
        date = email.utils.parsedate_to_datetime(text)
        if date.tzinfo is None:
            date = date.replace(tzinfo=timezone.utc)
        return max(0.0, (date - datetime.now(timezone.utc)).total_seconds())
    except (TypeError, ValueError):
        return None


class SD20APIError(RuntimeError):
    def __init__(
        self, message: str, status_code: int, retry_after_seconds: float | None = None
    ):
        super().__init__(message)
        self.status_code = status_code
        self.retry_after_seconds = retry_after_seconds


class SD20VideoService:
    """3365's documented ``/v1/video/generations`` async API."""

    _DONE: ClassVar[frozenset[str]] = frozenset(
        {"success", "succeeded", "completed", "done"}
    )
    _FAILED: ClassVar[frozenset[str]] = frozenset(
        {"failure", "failed", "error", "cancelled", "canceled", "rejected"}
    )
    _KEY_RETRY: ClassVar[frozenset[int]] = frozenset({401, 403, 429})

    def __init__(self, *, settings: dict[str, Any]):
        self.settings = settings if isinstance(settings, dict) else {}
        self.api_url = _endpoint(str(self.settings.get("base_url") or ""))
        self.base_origin = _origin(self.api_url)
        self.api_keys = _api_keys(self.settings)
        self.model = str(self.settings.get("model") or "video-v1-5s").strip()
        self.ratio = str(self.settings.get("ratio") or "16:9").strip() or "16:9"
        self.timeout_seconds = _clamp_int(
            self.settings.get("timeout_seconds") or self.settings.get("timeout") or 600,
            default=600,
            min_value=30,
            max_value=3600,
        )
        self.request_timeout_seconds = _clamp_int(
            self.settings.get("request_timeout_seconds", 120),
            default=120,
            min_value=10,
            max_value=600,
        )
        self.poll_interval_seconds = _clamp_float(
            self.settings.get("poll_interval_seconds", 15),
            default=15.0,
            min_value=1.0,
            max_value=120.0,
        )
        self.create_max_retries = _clamp_int(
            self.settings.get("create_max_retries", 0),
            default=0,
            min_value=0,
            max_value=3,
        )
        self.max_retries = _clamp_int(
            self.settings.get("max_retries", 2), default=2, min_value=0, max_value=10
        )
        self.retry_delay = _clamp_float(
            self.settings.get("retry_delay", 2),
            default=2.0,
            min_value=0.0,
            max_value=60.0,
        )
        self._key_cursor = 0
        self._disabled_until: dict[int, float] = {}

        logger.info(
            "[SD20Video] Initialized: model=%s, endpoint=%s, ratio=%s, timeout=%ss",
            self.model,
            self.api_url,
            self.ratio,
            self.timeout_seconds,
        )

    def _key_candidates(self) -> list[tuple[int, str]]:
        env_key = _env_key(self.settings)
        if env_key:
            return [(-1, env_key)]
        if not self.api_keys:
            raise RuntimeError("未配置 3365 SD2.0 API Key（api_keys 或 api_key）")
        now = time.monotonic()
        cursor = self._key_cursor % len(self.api_keys)
        candidates = [
            ((cursor + offset) % len(self.api_keys), key)
            for offset, key in enumerate(self.api_keys)
        ]
        active = [
            item for item in candidates if self._disabled_until.get(item[0], 0) <= now
        ]
        if active:
            return active
        wait = max(1, int(min(self._disabled_until.values()) - now))
        raise RuntimeError(f"3365 API Key 暂不可用，请约 {wait}s 后重试")

    def _mark_used(self, index: int) -> None:
        if index >= 0 and self.api_keys:
            self._key_cursor = (index + 1) % len(self.api_keys)

    def _disable_after_error(self, index: int, exc: Exception) -> None:
        if index < 0:
            return
        status = getattr(exc, "status_code", None)
        if status in {401, 403}:
            self._disabled_until[index] = time.monotonic() + 3600
        elif status == 429:
            retry = getattr(exc, "retry_after_seconds", None)
            self._disabled_until[index] = time.monotonic() + (
                float(retry) if isinstance(retry, (int, float)) and retry >= 0 else 60
            )

    async def _request_json(
        self,
        client: httpx.AsyncClient,
        method: str,
        url: str,
        *,
        headers: dict[str, str],
        payload: dict[str, Any] | None = None,
    ) -> Any:
        response = await client.request(method, url, headers=headers, json=payload)
        if response.status_code >= 400:
            detail = response.text[:500]
            retry_after = _retry_after(response.headers.get("retry-after"))
            raise SD20APIError(
                f"3365 SD2.0 API 请求失败 HTTP {response.status_code}: {detail}",
                response.status_code,
                retry_after,
            )
        try:
            return response.json()
        except Exception as exc:
            raise RuntimeError(
                f"3365 SD2.0 API 响应 JSON 解析失败: {exc}, body={response.text[:200]}"
            ) from exc

    async def _request_with_retries(
        self,
        client: httpx.AsyncClient,
        method: str,
        url: str,
        *,
        headers: dict[str, str],
        payload: dict[str, Any] | None,
        label: str,
        retries: int,
    ) -> Any:
        last_error: Exception | None = None
        for attempt in range(max(0, retries) + 1):
            try:
                return await self._request_json(
                    client, method, url, headers=headers, payload=payload
                )
            except (httpx.HTTPError, RuntimeError, ValueError) as exc:
                last_error = exc
                if attempt >= retries:
                    break
                delay = self.retry_delay * (2**attempt) + random.uniform(0, 0.5)
                logger.warning(
                    "[SD20Video] %s 失败: %s，%.1fs 后重试", label, exc, delay
                )
                await asyncio.sleep(delay)
        raise last_error or RuntimeError(f"{label} 失败")

    async def generate_video_url(
        self,
        prompt: str,
        image_bytes: bytes | None = None,
        *,
        preset: str | None = None,
    ) -> str | VideoResult:
        del preset
        final_prompt = (prompt or "").strip()
        if not final_prompt:
            raise ValueError("缺少视频提示词")

        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": final_prompt,
            "ratio": self.ratio,
        }
        if image_bytes:
            mime, _ = guess_image_mime_and_ext(image_bytes)
            encoded = base64.b64encode(image_bytes).decode("ascii")
            payload["image"] = f"data:{mime};base64,{encoded}"

        timeout = httpx.Timeout(
            connect=10.0,
            read=float(self.request_timeout_seconds),
            write=float(self.request_timeout_seconds),
            pool=float(self.request_timeout_seconds) + 10.0,
        )
        deadline = time.monotonic() + self.timeout_seconds
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            response: Any | None = None
            headers: dict[str, str] | None = None
            key_candidates = self._key_candidates()
            errors: list[str] = []
            for offset, (key_index, api_key) in enumerate(key_candidates):
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                }
                try:
                    response = await self._request_with_retries(
                        client,
                        "POST",
                        self.api_url,
                        headers=headers,
                        payload=payload,
                        label="创建 SD2.0 视频任务",
                        retries=self.create_max_retries,
                    )
                    self._mark_used(key_index)
                    break
                except Exception as exc:
                    if getattr(exc, "status_code", None) not in self._KEY_RETRY:
                        raise
                    self._disable_after_error(key_index, exc)
                    errors.append(str(exc))
                    if offset >= len(key_candidates) - 1:
                        raise RuntimeError(
                            "3365 API Key 池不可用: " + "; ".join(errors[-3:])
                        ) from exc

            if response is None or headers is None:
                raise RuntimeError("3365 SD2.0 视频任务创建失败")

            task_id = _extract_task_id(response)
            if not task_id:
                direct_url = _extract_video_url(response, base_origin=self.base_origin)
                if direct_url:
                    return direct_url
                raise RuntimeError(f"3365 SD2.0 未返回 task_id: {str(response)[:300]}")

            status_url = f"{self.api_url}/{quote(task_id, safe='')}"
            while True:
                if time.monotonic() >= deadline:
                    raise RuntimeError(f"3365 SD2.0 视频任务超时: task_id={task_id}")
                await asyncio.sleep(self.poll_interval_seconds)
                data = await self._request_with_retries(
                    client,
                    "GET",
                    status_url,
                    headers=headers,
                    payload=None,
                    label=f"查询 SD2.0 视频任务 {task_id}",
                    retries=self.max_retries,
                )
                status = _extract_status(data)
                video_url = _extract_video_url(data, base_origin=self.base_origin)
                if video_url and (not status or status in self._DONE):
                    parts = urlsplit(video_url)
                    if (
                        _origin(video_url) == self.base_origin
                        and "/v1/videos/" in parts.path
                    ):
                        return VideoResult(
                            video_url, {"Authorization": headers["Authorization"]}
                        )
                    return video_url
                if status in self._DONE:
                    content_url = f"{self.base_origin}/v1/videos/{quote(task_id, safe='')}/content"
                    return VideoResult(
                        content_url, {"Authorization": headers["Authorization"]}
                    )
                if status in self._FAILED:
                    raise RuntimeError(
                        f"3365 SD2.0 视频任务失败: task_id={task_id}, {_error_text(data)}"
                    )
                logger.info(
                    "[SD20Video] 等待任务: task_id=%s, status=%s",
                    task_id,
                    status or "unknown",
                )
