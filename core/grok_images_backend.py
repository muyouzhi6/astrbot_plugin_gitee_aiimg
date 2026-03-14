from __future__ import annotations

import asyncio
import base64
import json
import re
import time
from pathlib import Path
from typing import Any

import aiohttp

from astrbot.api import logger

from .image_format import guess_image_mime_and_ext
from .openai_compat_backend import _build_collage, resolution_to_size

_IMAGE_RESPONSE_FORMAT_CANDIDATES = ("b64_json", "url", None)
_RETRYABLE_HTTP_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}
_BASE64_PREFIX_RE = re.compile(r"^(?:b64|base64)\s*:\s*", re.IGNORECASE)


def _normalize_base_url(base_url: str) -> str:
    url = str(base_url or "").strip().rstrip("/")
    for suffix in (
        "/v1/images/generations",
        "/v1/images/edits",
        "/images/generations",
        "/images/edits",
        "/v1",
    ):
        if url.endswith(suffix):
            url = url[: -len(suffix)]
    return url.rstrip("/")


def _pick_first_api_key(api_keys: list[str]) -> str:
    keys = [str(k).strip() for k in (api_keys or []) if str(k).strip()]
    if not keys:
        raise RuntimeError("未配置 API Key")
    return keys[0]


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
    return b""


def _iter_strings(obj: object) -> list[str]:
    out: list[str] = []
    seen: set[int] = set()

    def walk(value: object) -> None:
        if value is None:
            return
        oid = id(value)
        if oid in seen:
            return
        seen.add(oid)
        if isinstance(value, str):
            out.append(value)
            return
        if isinstance(value, dict):
            for child in value.values():
                walk(child)
            return
        if isinstance(value, (list, tuple)):
            for child in value:
                walk(child)

    walk(obj)
    return out


def _extract_ref_from_string(text: str) -> tuple[str | None, bytes | None]:
    s = (text or "").strip().strip('"').strip("'")
    if not s:
        return None, None
    if s.startswith("data:image/") and "," in s:
        _header, b64_data = s.split(",", 1)
        raw = _decode_base64_bytes(b64_data)
        return (None, raw) if raw else (None, None)
    if s.startswith(("http://", "https://")):
        return s, None
    normalized = _BASE64_PREFIX_RE.sub("", s)
    if len(normalized) >= 128:
        raw = _decode_base64_bytes(normalized)
        if raw:
            return None, raw
    return None, None


def _parse_image_api_response(data: Any) -> list[tuple[str | None, bytes | None]]:
    results: list[tuple[str | None, bytes | None]] = []
    if isinstance(data, dict) and isinstance(data.get("data"), list):
        for item in data["data"]:
            if not isinstance(item, dict):
                continue
            url = item.get("url")
            if isinstance(url, str) and url.strip():
                results.append((url.strip(), None))
                continue
            b64_json = item.get("b64_json")
            if isinstance(b64_json, str) and b64_json.strip():
                raw = _decode_base64_bytes(b64_json)
                if raw:
                    results.append((None, raw))

    if results:
        return results

    for text in _iter_strings(data):
        url, raw = _extract_ref_from_string(text)
        if url or raw:
            results.append((url, raw))
            break
    return results


def _extract_api_error_message(raw_text: str) -> str:
    text = str(raw_text or "").strip()
    if not text:
        return ""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return text[:500]
    if not isinstance(data, dict):
        return text[:500]
    error_obj = data.get("error")
    if isinstance(error_obj, dict):
        message = str(error_obj.get("message") or "").strip()
        code = str(error_obj.get("code") or "").strip()
        param = str(error_obj.get("param") or "").strip()
        parts = [
            x
            for x in (
                message,
                f"code={code}" if code and code not in message else "",
                f"param={param}" if param and param not in message else "",
            )
            if x
        ]
        if parts:
            return " | ".join(parts)
    if isinstance(error_obj, str) and error_obj.strip():
        return error_obj.strip()
    for key in ("message", "detail", "error_description"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return text[:500]


def _is_response_format_related_error(error_message: str) -> bool:
    err = str(error_message or "").lower()
    if not err:
        return False
    if "response_format" in err:
        return True
    return "format" in err and (
        "invalid" in err or "unsupported" in err or "must be" in err
    )


def _is_size_related_error(error_message: str) -> bool:
    err = str(error_message or "").lower()
    if not err:
        return False
    if "invalid_size" in err or "size must be" in err:
        return True
    return "size" in err and (
        "invalid" in err or "unsupported" in err or "unknown" in err or "must be" in err
    )


class GrokImagesBackend:
    def __init__(
        self,
        *,
        imgr,
        base_url: str,
        api_keys: list[str],
        timeout: int = 120,
        max_retries: int = 2,
        default_model: str = "",
        default_size: str = "4096x4096",
        supports_edit: bool = True,
        extra_body: dict | None = None,
        proxy_url: str | None = None,
    ):
        self.imgr = imgr
        self.base_url = _normalize_base_url(base_url)
        self.api_key = _pick_first_api_key(api_keys)
        self.timeout = max(1, min(int(timeout or 120), 3600))
        self.max_retries = max(1, min(int(max_retries or 2), 10))
        self.default_model = str(default_model or "").strip()
        self.default_size = str(default_size or "4096x4096").strip()
        self.supports_edit = bool(supports_edit)
        self.extra_body = extra_body or {}
        self.proxy_url = str(proxy_url or "").strip() or None
        self._session: aiohttp.ClientSession | None = None
        self._session_lock = asyncio.Lock()

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
        self._session = None

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session and not self._session.closed:
            return self._session
        async with self._session_lock:
            if self._session and not self._session.closed:
                return self._session
            self._session = aiohttp.ClientSession()
            return self._session

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    @staticmethod
    def _retry_delay_seconds(attempt_index: int) -> float:
        return min(1.5 * (2**attempt_index), 4.0)

    def _coerce_form_value(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, (str, int, float, bool)):
            return str(value)
        try:
            return json.dumps(value, ensure_ascii=False)
        except Exception:
            return str(value)

    async def _save_first_result(
        self, results: list[tuple[str | None, bytes | None]]
    ) -> Path:
        if not results:
            raise RuntimeError("未能从响应中提取图片")
        ref, raw = results[0]
        if raw:
            return await self.imgr.save_image(raw)
        if ref:
            return await self.imgr.download_image(ref)
        raise RuntimeError("返回数据不包含图片")

    async def generate(
        self,
        prompt: str,
        *,
        model: str | None = None,
        size: str | None = None,
        resolution: str | None = None,
        extra_body: dict | None = None,
    ) -> Path:
        if not self.base_url:
            raise RuntimeError("未配置 base_url")

        final_model = str(model or self.default_model or "grok-imagine-1.0").strip()
        final_size = (
            str(size or "").strip()
            or (resolution_to_size(str(resolution or "")) or "").strip()
            or str(resolution or "").strip()
            or self.default_size
        )
        api_url = f"{self.base_url}/v1/images/generations"
        session = await self._ensure_session()
        last_error = ""

        for response_format in _IMAGE_RESPONSE_FORMAT_CANDIDATES:
            payload: dict[str, Any] = {
                "model": final_model,
                "prompt": (prompt or "").strip() or "a high quality image",
                "n": 1,
            }
            if response_format:
                payload["response_format"] = response_format
            if final_size:
                payload["size"] = final_size
            if isinstance(self.extra_body, dict) and self.extra_body:
                payload.update(self.extra_body)
            if isinstance(extra_body, dict) and extra_body:
                payload.update(extra_body)

            for attempt in range(self.max_retries):
                try:
                    t0 = time.perf_counter()
                    async with session.post(
                        api_url,
                        headers={**self._headers(), "Content-Type": "application/json"},
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=self.timeout),
                        proxy=self.proxy_url,
                    ) as resp:
                        raw_content = await resp.read()
                    if resp.status != 200:
                        text = raw_content.decode("utf-8", errors="replace")
                        detail = _extract_api_error_message(text)
                        last_error = detail or f"HTTP {resp.status}"
                        if response_format and _is_response_format_related_error(
                            detail
                        ):
                            logger.warning(
                                "[GrokImages][generate] response_format=%s rejected: %s",
                                response_format,
                                detail[:160],
                            )
                            break
                        if (
                            resp.status in _RETRYABLE_HTTP_STATUS_CODES
                            and attempt < self.max_retries - 1
                        ):
                            await asyncio.sleep(self._retry_delay_seconds(attempt))
                            continue
                        raise RuntimeError(last_error)
                    data = json.loads(raw_content.decode("utf-8"))
                    results = _parse_image_api_response(data)
                    if results:
                        logger.info(
                            "[GrokImages][generate] success in %.2fs format=%s",
                            time.perf_counter() - t0,
                            response_format or "default",
                        )
                        return await self._save_first_result(results)
                    last_error = "未能从响应中提取图片"
                    raise RuntimeError(last_error)
                except (asyncio.TimeoutError, aiohttp.ClientError) as e:
                    last_error = str(e) or "请求超时"
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(self._retry_delay_seconds(attempt))
                        continue
                except json.JSONDecodeError:
                    last_error = "API 响应格式异常"
                except Exception as e:
                    last_error = str(e)
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(self._retry_delay_seconds(attempt))
                        continue
                    if response_format and _is_response_format_related_error(
                        last_error
                    ):
                        break
                    raise

        raise RuntimeError(last_error or "Grok 文生图请求失败")

    async def edit(
        self,
        prompt: str,
        images: list[bytes],
        *,
        model: str | None = None,
        size: str | None = None,
        resolution: str | None = None,
        extra_body: dict | None = None,
    ) -> Path:
        if not self.supports_edit:
            raise RuntimeError("该后端不支持改图/图生图")
        if not images:
            raise ValueError("至少需要一张图片")
        if not self.base_url:
            raise RuntimeError("未配置 base_url")

        final_model = str(
            model or self.default_model or "grok-imagine-1.0-edit"
        ).strip()
        resolved_size = (
            str(size or "").strip()
            or (resolution_to_size(str(resolution or "")) or "").strip()
            or str(resolution or "").strip()
            or self.default_size
        )

        packed = _build_collage(images) if len(images) > 1 else images[0]
        mime, ext = guess_image_mime_and_ext(packed)
        api_url = f"{self.base_url}/v1/images/edits"
        session = await self._ensure_session()

        size_attempts: list[str | None] = [resolved_size] if resolved_size else [None]
        if resolved_size:
            size_attempts.append(None)
        last_error = ""

        for current_size in size_attempts:
            for response_format in _IMAGE_RESPONSE_FORMAT_CANDIDATES:
                for attempt in range(self.max_retries):
                    form = aiohttp.FormData()
                    form.add_field("model", final_model)
                    form.add_field(
                        "prompt", (prompt or "").strip() or "Edit this image"
                    )
                    form.add_field("n", "1")
                    if response_format:
                        form.add_field("response_format", response_format)
                    if current_size:
                        form.add_field("size", current_size)
                    form.add_field(
                        "image", packed, filename=f"image.{ext}", content_type=mime
                    )

                    for source in (self.extra_body, extra_body):
                        if not isinstance(source, dict):
                            continue
                        for key, value in source.items():
                            if key in {
                                "model",
                                "prompt",
                                "n",
                                "size",
                                "response_format",
                                "image",
                            }:
                                continue
                            form.add_field(str(key), self._coerce_form_value(value))

                    try:
                        t0 = time.perf_counter()
                        async with session.post(
                            api_url,
                            headers=self._headers(),
                            data=form,
                            timeout=aiohttp.ClientTimeout(total=self.timeout),
                            proxy=self.proxy_url,
                        ) as resp:
                            raw_content = await resp.read()
                        if resp.status != 200:
                            text = raw_content.decode("utf-8", errors="replace")
                            detail = _extract_api_error_message(text)
                            last_error = detail or f"HTTP {resp.status}"
                            if current_size and _is_size_related_error(detail):
                                logger.warning(
                                    "[GrokImages][edit] size=%s rejected: %s",
                                    current_size,
                                    detail[:160],
                                )
                                break
                            if response_format and _is_response_format_related_error(
                                detail
                            ):
                                logger.warning(
                                    "[GrokImages][edit] response_format=%s rejected: %s",
                                    response_format,
                                    detail[:160],
                                )
                                break
                            if (
                                resp.status in _RETRYABLE_HTTP_STATUS_CODES
                                and attempt < self.max_retries - 1
                            ):
                                await asyncio.sleep(self._retry_delay_seconds(attempt))
                                continue
                            raise RuntimeError(last_error)
                        data = json.loads(raw_content.decode("utf-8"))
                        results = _parse_image_api_response(data)
                        if results:
                            logger.info(
                                "[GrokImages][edit] success in %.2fs size=%s format=%s",
                                time.perf_counter() - t0,
                                current_size or "default",
                                response_format or "default",
                            )
                            return await self._save_first_result(results)
                        last_error = "未能从响应中提取图片"
                        raise RuntimeError(last_error)
                    except (asyncio.TimeoutError, aiohttp.ClientError) as e:
                        last_error = str(e) or "请求超时"
                        if attempt < self.max_retries - 1:
                            await asyncio.sleep(self._retry_delay_seconds(attempt))
                            continue
                    except json.JSONDecodeError:
                        last_error = "API 响应格式异常"
                    except Exception as e:
                        last_error = str(e)
                        if attempt < self.max_retries - 1:
                            await asyncio.sleep(self._retry_delay_seconds(attempt))
                            continue
                        if response_format and _is_response_format_related_error(
                            last_error
                        ):
                            break
                        raise

        raise RuntimeError(last_error or "Grok 改图请求失败")
