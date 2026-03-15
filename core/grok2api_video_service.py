from __future__ import annotations

import asyncio
import random
import time
from typing import Any

import httpx

from astrbot.api import logger

from .image_format import guess_image_mime_and_ext


def _clamp_int(v: Any, default: int, min_value: int, max_value: int) -> int:
    try:
        val = int(v)
        return max(min_value, min(val, max_value))
    except (ValueError, TypeError):
        return default


class Grok2ApiVideoService:
    def __init__(self, *, settings: dict):
        self.settings = settings if isinstance(settings, dict) else {}

        self.base_url: str = str(
            self.settings.get("base_url", "https://api.x.ai")
        ).rstrip("/")

        base = self.base_url
        if not base.endswith("/v1"):
            if not base.endswith("/"):
                base += "/"
            base += "v1"
        self.api_url = f"{base}/videos"

        api_keys = self.settings.get("api_keys", [])
        if not api_keys and "api_key" in self.settings:
            api_keys = [self.settings["api_key"]]
        self.api_keys = [
            key
            for key in (api_keys if isinstance(api_keys, list) else [api_keys])
            if key
        ]
        self._key_index = 0
        self._key_lock = asyncio.Lock()

        self.model: str = (
            str(self.settings.get("model", "grok-imagine-1.0-video")).strip()
            or "grok-imagine-1.0-video"
        )
        self.timeout_seconds: int = _clamp_int(
            self.settings.get("timeout", 180)
            or self.settings.get("timeout_seconds", 180),
            default=180,
            min_value=1,
            max_value=3600,
        )
        self.max_retries: int = _clamp_int(
            self.settings.get("max_retries", 2),
            default=2,
            min_value=0,
            max_value=10,
        )

    async def _get_key(self) -> str:
        async with self._key_lock:
            if not self.api_keys:
                raise RuntimeError("未配置 API Key")
            key = self.api_keys[self._key_index]
            self._key_index = (self._key_index + 1) % len(self.api_keys)
            return key

    async def generate_video_url(
        self,
        prompt: str,
        image_bytes: bytes | None = None,
        *,
        preset: str | None = None,
    ) -> str:
        del preset

        api_key = await self._get_key()
        final_prompt = (prompt or "").strip()
        data_fields = {
            "model": self.model,
            "prompt": final_prompt,
            "n": "1",
        }
        headers = {"Authorization": f"Bearer {api_key}"}
        timeout = httpx.Timeout(
            connect=10.0,
            read=float(self.timeout_seconds),
            write=10.0,
            pool=float(self.timeout_seconds) + 10.0,
        )

        async def _request_once() -> Any:
            async with httpx.AsyncClient(
                timeout=timeout, follow_redirects=True
            ) as client:
                if image_bytes:
                    mime, ext = guess_image_mime_and_ext(image_bytes)
                    files = {"input_reference": (f"image.{ext}", image_bytes, mime)}
                    resp = await client.post(
                        self.api_url,
                        headers=headers,
                        data=data_fields,
                        files=files,
                    )
                else:
                    headers_json = dict(headers)
                    headers_json["Content-Type"] = "application/json"
                    payload = dict(data_fields)
                    payload["n"] = 1
                    resp = await client.post(
                        self.api_url,
                        headers=headers_json,
                        json=payload,
                    )

            if resp.status_code != 200:
                detail = resp.text[:500]
                if resp.status_code == 401:
                    raise RuntimeError("Grok API Key 无效或已过期 (401)")
                if resp.status_code == 403:
                    raise RuntimeError("Grok API 访问被拒绝 (403)")
                raise RuntimeError(
                    f"Grok API 请求失败 HTTP {resp.status_code}: {detail}"
                )

            try:
                return resp.json()
            except Exception as e:
                raise RuntimeError(
                    f"API 响应 JSON 解析失败: {e}, body={resp.text[:200]}"
                ) from e

        t_start = time.perf_counter()
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                data = await _request_once()
                video_url: str | None = None
                if isinstance(data, dict):
                    if "url" in data:
                        video_url = str(data["url"] or "").strip() or None
                    elif (
                        "data" in data
                        and isinstance(data["data"], list)
                        and data["data"]
                        and isinstance(data["data"][0], dict)
                    ):
                        video_url = (
                            str(data["data"][0].get("url") or "").strip() or None
                        )

                if video_url:
                    t_end = time.perf_counter()
                    logger.info(
                        "[Grok2ApiVideo] 成功: 耗时=%.2fs, url=%s...",
                        t_end - t_start,
                        video_url[:80],
                    )
                    return video_url

                raise RuntimeError(f"API 响应未包含视频 URL: {str(data)[:200]}")
            except Exception as e:
                last_exc = e
                if attempt >= self.max_retries:
                    break
                delay = 2 * (2**attempt) + random.uniform(0, 0.5)
                logger.warning(
                    "[Grok2ApiVideo] 请求失败: %s，%.1fs 后重试...",
                    e,
                    delay,
                )
                await asyncio.sleep(delay)

        raise last_exc or RuntimeError("Grok 视频生成失败")
