from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlsplit, urlunsplit

import aiohttp

from astrbot.api import logger

from .gitee_sizes import normalize_size_text
from .output_spec import OutputIntent


def _normalize_modelscope_base_url(raw: str) -> str:
    value = str(raw or "").strip().rstrip("/")
    if not value:
        return ""

    lower = value.lower()
    for suffix in ("/v1/images/generations", "/images/generations"):
        if lower.endswith(suffix):
            value = value[: -len(suffix)].rstrip("/")
            break

    if value.lower().endswith("/v1"):
        return value
    parts = urlsplit(value)
    if parts.scheme and parts.netloc:
        path = f"{(parts.path or '').rstrip('/')}/v1"
        return urlunsplit((parts.scheme, parts.netloc, path, "", "")).rstrip("/")
    return f"{value}/v1"


class ModelScopeAsyncImageBackend:
    """ModelScope image generation through its asynchronous task API."""

    def __init__(
        self,
        *,
        imgr,
        base_url: str,
        api_keys: list[str],
        timeout: int = 600,
        max_retries: int = 2,
        default_model: str = "",
        default_size: str = "1024x1024",
        supports_edit: bool = False,
        extra_body: dict | None = None,
        proxy_url: str | None = None,
        poll_interval: float = 2.0,
        poll_timeout: int = 600,
        output_format: str = "jpeg",
    ):
        self.imgr = imgr
        self.base_url = _normalize_modelscope_base_url(base_url)
        self.api_keys = [str(key).strip() for key in api_keys if str(key).strip()]
        self.timeout = max(10, min(int(timeout or 600), 600))
        self.max_retries = max(0, min(int(max_retries or 0), 5))
        self.default_model = str(default_model or "").strip()
        self.default_size = normalize_size_text(default_size) or "1024x1024"
        self.supports_edit = bool(supports_edit)
        self.extra_body = dict(extra_body or {})
        self.proxy_url = str(proxy_url or "").strip() or None
        self.poll_interval = max(0.5, min(float(poll_interval or 2.0), 30.0))
        self.output_format = str(output_format or "jpeg").strip().lower()
        self.poll_timeout = max(10, min(int(poll_timeout or 600), 1800))
        self._key_index = 0

    def _next_key(self) -> str:
        if not self.api_keys:
            raise RuntimeError("未配置 ModelScope API Key")
        key = self.api_keys[self._key_index % len(self.api_keys)]
        self._key_index = (self._key_index + 1) % len(self.api_keys)
        return key

    async def _request_json(
        self,
        session: aiohttp.ClientSession,
        method: str,
        url: str,
        *,
        headers: dict[str, str],
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        attempts = self.max_retries + 1
        for attempt in range(attempts):
            try:
                async with session.request(
                    method,
                    url,
                    headers=headers,
                    json=payload,
                    proxy=self.proxy_url,
                ) as response:
                    text = await response.text()
                    if response.status >= 400:
                        if (
                            response.status in {408, 429, 500, 502, 503, 504}
                            and attempt + 1 < attempts
                        ):
                            await asyncio.sleep(min(2**attempt, 5))
                            continue
                        raise RuntimeError(
                            f"ModelScope HTTP {response.status}: {text[:500]}"
                        )
                    try:
                        data = await response.json(content_type=None)
                    except Exception as exc:
                        raise RuntimeError(
                            f"ModelScope 返回了非 JSON 响应: {text[:500]}"
                        ) from exc
                    if not isinstance(data, dict):
                        raise RuntimeError(f"ModelScope 返回格式异常: {data!r}")
                    return data
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                if attempt + 1 >= attempts:
                    raise RuntimeError(f"ModelScope 请求失败: {exc}") from exc
                await asyncio.sleep(min(2**attempt, 5))
        raise RuntimeError("ModelScope 请求失败")

    async def _poll_task(
        self,
        session: aiohttp.ClientSession,
        *,
        task_id: str,
        api_key: str,
    ) -> str:
        task_url = f"{self.base_url}/tasks/{quote(task_id, safe='')}"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "X-ModelScope-Task-Type": "image_generation",
        }
        deadline = time.monotonic() + self.poll_timeout

        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"ModelScope 图片任务超时: task_id={task_id}, timeout={self.poll_timeout}s"
                )

            data = await self._request_json(
                session,
                "GET",
                task_url,
                headers=headers,
            )
            status = str(data.get("task_status") or data.get("status") or "").upper()
            if status in {"SUCCEED", "SUCCEEDED", "SUCCESS", "DONE"}:
                output_images = data.get("output_images") or []
                first = (
                    output_images[0]
                    if isinstance(output_images, list) and output_images
                    else None
                )
                if isinstance(first, dict):
                    first = first.get("url") or first.get("image_url")
                image_url = str(first or "").strip()
                if not image_url:
                    raise RuntimeError(
                        f"ModelScope 任务成功但未返回 output_images: task_id={task_id}"
                    )
                return image_url
            if status in {"FAILED", "ERROR", "CANCELED", "CANCELLED"}:
                errors = data.get("errors")
                detail = errors.get("message") if isinstance(errors, dict) else None
                detail = (
                    detail
                    or data.get("message")
                    or data.get("error")
                    or "unknown error"
                )
                raise RuntimeError(
                    f"ModelScope 图片任务失败: task_id={task_id}, error={detail}"
                )

            await asyncio.sleep(self.poll_interval)

    def resolve_output_intent(self, intent: OutputIntent) -> dict[str, str]:
        if intent.exact_size:
            return {"size": intent.exact_size}
        result: dict[str, str] = {}
        if intent.resolution:
            result["resolution"] = intent.resolution
        return result

    async def generate(
        self,
        prompt: str,
        *,
        model: str | None = None,
        size: str | None = None,
        resolution: str | None = None,
        extra_body: dict | None = None,
    ) -> Path:
        final_model = str(model or self.default_model or "").strip()
        if not final_model:
            raise RuntimeError("未配置 ModelScope model")

        final_size = normalize_size_text(size or resolution or self.default_size)
        payload: dict[str, Any] = dict(self.extra_body)
        payload.update(extra_body or {})
        payload.update({"model": final_model, "prompt": str(prompt or "").strip()})
        if final_size:
            payload["size"] = final_size

        api_key = self._next_key()
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-ModelScope-Async-Mode": "true",
        }
        submit_url = f"{self.base_url}/images/generations"
        timeout = aiohttp.ClientTimeout(total=float(self.timeout))
        started = time.monotonic()

        async with aiohttp.ClientSession(timeout=timeout) as session:
            submitted = await self._request_json(
                session,
                "POST",
                submit_url,
                headers=headers,
                payload=payload,
            )
            task_id = str(submitted.get("task_id") or submitted.get("id") or "").strip()
            if not task_id:
                raise RuntimeError(f"ModelScope 未返回 task_id: {submitted!r}")
            image_url = await self._poll_task(
                session,
                task_id=task_id,
                api_key=api_key,
            )

        logger.info(
            "[ModelScopeAsync][generate] task_id=%s completed in %.2fs",
            task_id,
            time.monotonic() - started,
        )
        return await self.imgr.download_image(
            image_url, output_format=self.output_format
        )

    async def edit(self, *args, **kwargs) -> Path:
        raise RuntimeError("ModelScope 当前模板不支持改图/图生图")
