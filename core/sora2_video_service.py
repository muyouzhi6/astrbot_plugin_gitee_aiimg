from __future__ import annotations

import asyncio
import email.utils
import json
import os
import random
import time
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote, urljoin, urlsplit

import httpx

from astrbot.api import logger

from .image_format import guess_image_mime_and_ext


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


def _as_key_list(settings: dict) -> list[str]:
    api_keys = settings.get("api_keys", [])
    if not api_keys and settings.get("api_key"):
        api_keys = [settings.get("api_key")]
    if not isinstance(api_keys, list):
        api_keys = [api_keys]
    return [str(key).strip() for key in api_keys if str(key or "").strip()]


def _env_key(settings: dict) -> str:
    env_name = str(settings.get("api_key_env") or "").strip()
    if not env_name:
        return ""
    return str(os.environ.get(env_name) or "").strip()


def _videos_endpoint(base_url: str) -> str:
    base = (base_url or "https://x666.me").strip().rstrip("/")
    if base.endswith("/v1/videos"):
        return base
    if base.endswith("/v1"):
        return f"{base}/videos"
    return f"{base}/v1/videos"


def _origin_from_url(url: str) -> str:
    parts = urlsplit(url)
    if not parts.scheme or not parts.netloc:
        return ""
    return f"{parts.scheme}://{parts.netloc}"


def _form_value(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _to_form_fields(payload: dict[str, Any]) -> dict[str, str]:
    return {
        str(key): _form_value(value)
        for key, value in payload.items()
        if value is not None
    }


def _parse_retry_after_seconds(value: Any) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        seconds = float(text)
        if seconds >= 0:
            return seconds
    except ValueError:
        pass

    try:
        dt = email.utils.parsedate_to_datetime(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return max(0.0, (dt - datetime.now(timezone.utc)).total_seconds())
    except (TypeError, ValueError):
        return None


def _absolute_url(value: Any, *, base_origin: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if text.startswith(("http://", "https://")):
        return text
    if text.startswith("/") and base_origin:
        return urljoin(base_origin + "/", text.lstrip("/"))
    return ""


def _extract_video_url(data: Any, *, base_origin: str) -> str:
    if isinstance(data, dict):
        for key in ("video_url", "url", "download_url", "file_url"):
            url = _absolute_url(data.get(key), base_origin=base_origin)
            if url:
                return url

        for key in ("data", "output", "videos", "result"):
            value = data.get(key)
            if isinstance(value, list):
                for item in value:
                    url = _extract_video_url(item, base_origin=base_origin)
                    if url:
                        return url
            elif isinstance(value, dict):
                url = _extract_video_url(value, base_origin=base_origin)
                if url:
                    return url

    if isinstance(data, list):
        for item in data:
            url = _extract_video_url(item, base_origin=base_origin)
            if url:
                return url

    return ""


def _extract_task_id(data: Any) -> str:
    if not isinstance(data, dict):
        return ""
    for key in ("task_id", "id"):
        value = str(data.get(key) or "").strip()
        if value:
            return value
    return ""


def _format_upstream_error(data: Any) -> str:
    if not isinstance(data, dict):
        return str(data)[:300]

    error = data.get("error")
    if isinstance(error, dict):
        code = str(error.get("code") or "").strip()
        message = str(error.get("message") or error.get("error") or "").strip()
        if code and message:
            return f"{code}: {message}"
        return message or code or str(error)[:300]
    if isinstance(error, str) and error.strip():
        return error.strip()

    for key in ("message", "msg", "detail", "reason"):
        value = str(data.get(key) or "").strip()
        if value:
            return value[:300]

    return str(data)[:300]


class Sora2APIError(RuntimeError):
    def __init__(
        self,
        message: str,
        status_code: int,
        *,
        retry_after_seconds: float | None = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.retry_after_seconds = retry_after_seconds


class Sora2VideoService:
    """OpenAI-compatible /v1/videos async task provider for Sora2-style gateways."""

    _DONE = {"completed", "succeeded", "success", "done"}
    _FAILED = {"failed", "error", "cancelled", "canceled", "rejected"}
    _KEY_RETRY_STATUS = {401, 403, 429}
    _AUTH_KEY_COOLDOWN_SECONDS = 3600.0
    _RATE_LIMIT_KEY_COOLDOWN_SECONDS = 60.0

    def __init__(self, *, settings: dict):
        self.settings = settings if isinstance(settings, dict) else {}

        self.base_url = str(self.settings.get("base_url") or "https://x666.me").strip()
        self.api_url = _videos_endpoint(self.base_url)
        self.base_origin = _origin_from_url(self.api_url)
        self.api_keys = _as_key_list(self.settings)
        self.model = str(self.settings.get("model") or "sora-2").strip() or "sora-2"
        self.seconds = str(self.settings.get("seconds") or "5").strip() or "5"
        self.size = str(self.settings.get("size") or "720x1280").strip() or "720x1280"
        self.n = _clamp_int(
            self.settings.get("n", 1), default=1, min_value=1, max_value=4
        )
        self.timeout_seconds = _clamp_int(
            self.settings.get("timeout_seconds")
            or self.settings.get("timeout")
            or 300,
            default=300,
            min_value=30,
            max_value=3600,
        )
        self.request_timeout_seconds = _clamp_int(
            self.settings.get("request_timeout_seconds", 60),
            default=60,
            min_value=10,
            max_value=600,
        )
        self.poll_interval_seconds = _clamp_float(
            self.settings.get("poll_interval_seconds", 10),
            default=10.0,
            min_value=1.0,
            max_value=120.0,
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
        self.retry_delay = _clamp_float(
            self.settings.get("retry_delay", 2),
            default=2.0,
            min_value=0.0,
            max_value=60.0,
        )
        extra_body = self.settings.get("extra_body")
        self.extra_body = extra_body if isinstance(extra_body, dict) else {}
        self._key_cursor = 0
        self._key_disabled_until: dict[int, float] = {}

        logger.info(
            "[Sora2Video] Initialized: model=%s, endpoint=%s, size=%s, seconds=%s, timeout=%ss",
            self.model,
            self.api_url,
            self.size,
            self.seconds,
            self.timeout_seconds,
        )

    def _key_candidates(self) -> list[tuple[int, str]]:
        env_key = _env_key(self.settings)
        if env_key:
            return [(-1, env_key)]
        if not self.api_keys:
            raise RuntimeError("未配置 Sora2 API Key（api_keys 或 api_key）")
        now = time.monotonic()
        cursor = self._key_cursor % len(self.api_keys)
        ordered = self.api_keys[cursor:] + self.api_keys[:cursor]
        candidates = [
            ((cursor + offset) % len(self.api_keys), key)
            for offset, key in enumerate(ordered)
        ]
        active = [
            (key_index, key)
            for key_index, key in candidates
            if self._key_disabled_until.get(key_index, 0.0) <= now
        ]
        if active:
            return active

        next_ready = min(self._key_disabled_until.values(), default=now)
        wait_seconds = max(1, int(next_ready - now))
        raise RuntimeError(f"Sora2 API Key 池暂时不可用，请约 {wait_seconds}s 后重试")

    def _mark_key_used(self, key_index: int) -> None:
        if key_index >= 0 and self.api_keys:
            self._key_cursor = (key_index + 1) % len(self.api_keys)

    def _disable_key_after_error(self, key_index: int, exc: Exception) -> None:
        if key_index < 0:
            return
        status_code = getattr(exc, "status_code", None)
        if status_code in {401, 403}:
            cooldown = self._AUTH_KEY_COOLDOWN_SECONDS
        elif status_code == 429:
            retry_after = getattr(exc, "retry_after_seconds", None)
            cooldown = (
                float(retry_after)
                if isinstance(retry_after, (int, float)) and retry_after >= 0
                else self._RATE_LIMIT_KEY_COOLDOWN_SECONDS
            )
        else:
            return

        self._key_disabled_until[key_index] = time.monotonic() + cooldown

    @classmethod
    def _should_try_next_key(cls, exc: Exception) -> bool:
        status_code = getattr(exc, "status_code", None)
        return isinstance(status_code, int) and status_code in cls._KEY_RETRY_STATUS

    async def _request_json(
        self,
        client: httpx.AsyncClient,
        method: str,
        url: str,
        *,
        headers: dict[str, str],
        json_body: dict[str, Any] | None = None,
        data_fields: dict[str, str] | None = None,
        files: dict[str, tuple[str, bytes, str]] | None = None,
    ) -> Any:
        resp = await client.request(
            method,
            url,
            headers=headers,
            json=json_body,
            data=data_fields,
            files=files,
        )
        if resp.status_code >= 400:
            detail = resp.text[:500]
            if resp.status_code == 401:
                raise Sora2APIError("Sora2 API Key 无效或已过期 (401)", 401)
            if resp.status_code == 403:
                raise Sora2APIError("Sora2 API 访问被拒绝 (403)", 403)
            if resp.status_code == 429:
                retry_after = _parse_retry_after_seconds(resp.headers.get("retry-after"))
                raise Sora2APIError(
                    f"Sora2 API 达到限流或额度限制 (429): {detail}",
                    429,
                    retry_after_seconds=retry_after,
                )
            raise Sora2APIError(
                f"Sora2 API 请求失败 HTTP {resp.status_code}: {detail}",
                resp.status_code,
            )
        try:
            return resp.json()
        except Exception as e:
            raise RuntimeError(
                f"Sora2 API 响应 JSON 解析失败: {e}, body={resp.text[:200]}"
            ) from e

    async def _request_json_with_retries(
        self,
        client: httpx.AsyncClient,
        method: str,
        url: str,
        *,
        headers: dict[str, str],
        json_body: dict[str, Any] | None = None,
        data_fields: dict[str, str] | None = None,
        files: dict[str, tuple[str, bytes, str]] | None = None,
        label: str,
        max_retries: int | None = None,
        retry_key_errors: bool = True,
    ) -> Any:
        last_exc: Exception | None = None
        retries = self.max_retries if max_retries is None else max(0, max_retries)
        for attempt in range(retries + 1):
            try:
                return await self._request_json(
                    client,
                    method,
                    url,
                    headers=headers,
                    json_body=json_body,
                    data_fields=data_fields,
                    files=files,
                )
            except Exception as e:
                last_exc = e
                if not retry_key_errors and self._should_try_next_key(e):
                    break
                if attempt >= retries:
                    break
                delay = self.retry_delay * (2**attempt) + random.uniform(0, 0.5)
                logger.warning(
                    "[Sora2Video] %s 失败: %s，%.1fs 后重试...",
                    label,
                    e,
                    delay,
                )
                await asyncio.sleep(delay)
        raise last_exc or RuntimeError(f"{label} 失败")

    async def generate_video_url(
        self,
        prompt: str,
        image_bytes: bytes | None = None,
        *,
        preset: str | None = None,
    ) -> str:
        final_prompt = (prompt or "").strip()
        if not final_prompt:
            raise ValueError("缺少视频提示词")

        payload: dict[str, Any] = dict(self.extra_body)
        payload.update(
            {
                "model": self.model,
                "prompt": final_prompt,
                "seconds": str(self.seconds),
                "size": self.size,
                "n": self.n,
            }
        )
        request_json_body: dict[str, Any] | None = payload
        request_data_fields: dict[str, str] | None = None
        request_files: dict[str, tuple[str, bytes, str]] | None = None
        if image_bytes:
            mime, ext = guess_image_mime_and_ext(image_bytes)
            request_json_body = None
            request_data_fields = _to_form_fields(payload)
            request_files = {
                "input_reference": (f"input_reference.{ext}", image_bytes, mime)
            }

        timeout = httpx.Timeout(
            connect=10.0,
            read=float(self.request_timeout_seconds),
            write=10.0,
            pool=float(self.request_timeout_seconds) + 10.0,
        )

        t_start = time.perf_counter()
        deadline = time.monotonic() + self.timeout_seconds
        async with httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=True,
        ) as client:
            logger.info(
                "[Sora2Video] 创建任务: model=%s, size=%s, seconds=%s, prompt=%s...",
                self.model,
                self.size,
                self.seconds,
                final_prompt[:60],
            )
            data: Any | None = None
            headers: dict[str, str] | None = None
            key_errors: list[str] = []
            key_candidates = self._key_candidates()
            for candidate_offset, (key_index, api_key) in enumerate(key_candidates):
                headers = {"Authorization": f"Bearer {api_key}"}
                if request_json_body is not None:
                    headers["Content-Type"] = "application/json"
                try:
                    data = await self._request_json_with_retries(
                        client,
                        "POST",
                        self.api_url,
                        headers=headers,
                        json_body=request_json_body,
                        data_fields=request_data_fields,
                        files=request_files,
                        label="创建视频任务",
                        max_retries=self.create_max_retries,
                        retry_key_errors=False,
                    )
                    self._mark_key_used(key_index)
                    break
                except Exception as e:
                    if not self._should_try_next_key(e):
                        raise
                    self._disable_key_after_error(key_index, e)
                    key_errors.append(str(e))
                    if candidate_offset >= len(key_candidates) - 1:
                        raise RuntimeError(
                            "Sora2 API Key 池全部不可用: "
                            + "; ".join(key_errors[-3:])
                        ) from e
                    logger.warning(
                        "[Sora2Video] 创建任务鉴权/额度失败，切换下一个 API Key: %s",
                        e,
                    )

            if data is None or headers is None:
                raise RuntimeError("Sora2 视频任务创建失败")

            video_url = _extract_video_url(data, base_origin=self.base_origin)
            if video_url:
                logger.info("[Sora2Video] 创建响应直接返回视频 URL")
                return video_url

            task_id = _extract_task_id(data)
            if not task_id:
                raise RuntimeError(f"Sora2 API 未返回 task_id: {str(data)[:300]}")

            status_url = f"{self.api_url}/{quote(task_id, safe='')}"
            while True:
                if time.monotonic() >= deadline:
                    raise RuntimeError(f"Sora2 视频任务超时: task_id={task_id}")

                await asyncio.sleep(self.poll_interval_seconds)
                data = await self._request_json_with_retries(
                    client,
                    "GET",
                    status_url,
                    headers=headers,
                    label="查询视频任务",
                )
                video_url = _extract_video_url(data, base_origin=self.base_origin)
                status = (
                    str(data.get("status") or "").strip().lower()
                    if isinstance(data, dict)
                    else ""
                )

                if video_url and (not status or status in self._DONE):
                    logger.info(
                        "[Sora2Video] 成功: task_id=%s, 耗时=%.2fs, url=%s...",
                        task_id,
                        time.perf_counter() - t_start,
                        video_url[:80],
                    )
                    return video_url

                if status in self._FAILED:
                    raise RuntimeError(
                        f"Sora2 视频任务失败: {task_id}, {_format_upstream_error(data)}"
                    )

                logger.info(
                    "[Sora2Video] 等待任务: task_id=%s, status=%s, progress=%s",
                    task_id,
                    status or "unknown",
                    data.get("progress") if isinstance(data, dict) else "",
                )
