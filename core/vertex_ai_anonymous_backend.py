from __future__ import annotations

import asyncio
import base64
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiohttp

try:
    from curl_cffi import AsyncSession as CurlAsyncSession
except Exception:
    CurlAsyncSession = None

from astrbot.api import logger

from .image_format import guess_image_mime_and_ext
from .vertex_ai_anonymous_utils import (
    DEFAULT_OPERATION_NAME,
    MAX_OUTPUT_TOKENS,
    RECAPTCHA_CO,
    RECAPTCHA_HL,
    RECAPTCHA_SITE_KEY,
    RECAPTCHA_TOKEN_RETRIES,
    RECAPTCHA_V,
    RECAPTCHA_VH,
    TEMPERATURE,
    TOP_P,
    build_anchor_url,
    build_reload_url,
    extract_query_params,
    parse_anchor_token,
    parse_rresp,
    size_to_aspect_ratio,
)

_AIOHTTP_CONNECT_TIMEOUT_SECONDS = 30
_AIOHTTP_LIMIT = 10
_AIOHTTP_LIMIT_PER_HOST = 5
_AIOHTTP_DNS_CACHE_TTL_SECONDS = 300


@dataclass(frozen=True)
class VertexAIAnonymousSettings:
    model: str
    timeout_seconds: int
    max_retries: int
    proxy_url: str | None
    recaptcha_base_api: str
    vertex_base_api: str
    system_prompt: str | None
    query_signature: str
    graphql_api_key: str


class VertexAIAnonymousBackend:
    """Vertex AI Anonymous backend (recaptcha + GraphQL batchGraphql)."""

    def __init__(self, *, imgr, settings: VertexAIAnonymousSettings):
        self.imgr = imgr
        self.settings = settings
        self._session: object | None = None
        self._session_lock = asyncio.Lock()

    async def close(self) -> None:
        close = getattr(self._session, "close", None)
        if callable(close):
            result = close()
            if inspect.isawaitable(result):
                await result
        self._session = None

    @staticmethod
    def _session_closed(session: object | None) -> bool:
        if session is None:
            return True
        closed = getattr(session, "closed", None)
        if isinstance(closed, bool):
            return closed
        internal_closed = getattr(session, "_closed", None)
        if isinstance(internal_closed, bool):
            return internal_closed
        return False

    async def _get_session(self) -> object:
        if self._session is not None and not self._session_closed(self._session):
            return self._session
        async with self._session_lock:
            if self._session is not None and not self._session_closed(self._session):
                return self._session
            if CurlAsyncSession is not None:
                self._session = CurlAsyncSession(timeout=self.settings.timeout_seconds)
            else:
                timeout = aiohttp.ClientTimeout(
                    total=self.settings.timeout_seconds,
                    connect=_AIOHTTP_CONNECT_TIMEOUT_SECONDS,
                )
                connector = aiohttp.TCPConnector(
                    limit=_AIOHTTP_LIMIT,
                    limit_per_host=_AIOHTTP_LIMIT_PER_HOST,
                    ttl_dns_cache=_AIOHTTP_DNS_CACHE_TTL_SECONDS,
                )
                self._session = aiohttp.ClientSession(
                    timeout=timeout, connector=connector
                )
            return self._session

    @staticmethod
    def _ua_headers() -> dict[str, str]:
        return {
            "user-agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/131.0.0.0 Safari/537.36"
            )
        }

    async def generate(self, prompt: str, **kwargs) -> Path:
        images = await self._generate_images(prompt, image_bytes_list=None, **kwargs)
        if not images:
            raise RuntimeError("Vertex AI Anonymous did not return image data")
        mime, b64 = images[0]
        logger.info(
            "[VertexAIAnonymous] generate ok: images=%s mime=%s", len(images), mime
        )
        return await self.imgr.save_base64_image(b64)

    async def edit(self, prompt: str, images: list[bytes], **kwargs) -> Path:
        if not images:
            raise ValueError("At least one image is required")
        out = await self._generate_images(prompt, image_bytes_list=images, **kwargs)
        if not out:
            raise RuntimeError("Vertex AI Anonymous did not return image data")
        mime, b64 = out[0]
        logger.info("[VertexAIAnonymous] edit ok: images=%s mime=%s", len(out), mime)
        return await self.imgr.save_base64_image(b64)

    async def _generate_images(
        self,
        prompt: str,
        *,
        image_bytes_list: list[bytes] | None,
        size: str | None = None,
        resolution: str | None = None,
    ) -> list[tuple[str, str]]:
        recaptcha_token = await self._get_recaptcha_token()
        if not recaptcha_token:
            raise RuntimeError("Vertex AI Anonymous failed to get recaptcha_token")

        last_error_message: str | None = None
        captcha_try_count = 0
        body = self._build_body(
            prompt, image_bytes_list, size=size, resolution=resolution
        )
        for attempt in range(max(1, self.settings.max_retries)):
            body["variables"]["recaptchaToken"] = recaptcha_token
            result, status, err_msg = await self._call_api(body)
            if result is not None:
                return result

            last_error_message = err_msg or last_error_message

            if status == 3:
                if (
                    err_msg
                    and "Failed to verify action" in err_msg
                    and captcha_try_count < 1
                ):
                    captcha_try_count += 1
                    logger.info(
                        "[VertexAIAnonymous] retry once with same recaptcha token"
                    )
                    continue

                recaptcha_token = await self._get_recaptcha_token()
                if not recaptcha_token:
                    raise RuntimeError(
                        "Vertex AI Anonymous failed to refresh recaptcha_token"
                    )
                captcha_try_count = 0
                continue

            if status == 999:
                raise RuntimeError(err_msg or "Vertex AI Anonymous response rejected")

            logger.warning(
                "[VertexAIAnonymous] call failed attempt=%s/%s: %s",
                attempt + 1,
                self.settings.max_retries,
                err_msg or "unknown error",
            )

        raise RuntimeError(
            f"Vertex AI Anonymous request failed: {last_error_message or 'unknown error'}"
        )

    def _build_body(
        self,
        prompt: str,
        image_bytes_list: list[bytes] | None,
        *,
        size: str | None,
        resolution: str | None,
    ) -> dict[str, Any]:
        parts: list[dict[str, Any]] = []
        for img in image_bytes_list or []:
            mime, _ext = guess_image_mime_and_ext(img)
            parts.append(
                {
                    "inlineData": {
                        "mimeType": mime,
                        "data": base64.b64encode(img).decode(),
                    }
                }
            )

        context: dict[str, Any] = {
            "model": self.settings.model,
            "contents": [{"parts": [{"text": prompt}, *parts], "role": "user"}],
            "generationConfig": {
                "temperature": TEMPERATURE,
                "topP": TOP_P,
                "maxOutputTokens": MAX_OUTPUT_TOKENS,
                "responseModalities": ["IMAGE"],
                "imageConfig": {
                    "imageOutputOptions": {"mimeType": "image/png"},
                    "personGeneration": "ALLOW_ALL",
                },
            },
            "safetySettings": [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "OFF"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "OFF"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "OFF"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "OFF"},
            ],
            "region": "global",
        }

        image_config = dict(context["generationConfig"]["imageConfig"])
        aspect_ratio = size_to_aspect_ratio(size)
        if aspect_ratio:
            image_config["aspectRatio"] = aspect_ratio
        if resolution and str(resolution).strip().upper() in {"1K", "2K", "4K"}:
            if "gemini-3" in self.settings.model.lower():
                image_config["imageSize"] = str(resolution).strip().upper()
        context["generationConfig"]["imageConfig"] = image_config

        if self.settings.system_prompt:
            context["systemInstruction"] = {
                "parts": [{"text": self.settings.system_prompt}]
            }

        return {
            "querySignature": self.settings.query_signature,
            "operationName": DEFAULT_OPERATION_NAME,
            "variables": context,
        }

    async def _call_api(
        self, body: dict[str, Any]
    ) -> tuple[list[tuple[str, str]] | None, int | None, str | None]:
        session = await self._get_session()
        url = (
            f"{self.settings.vertex_base_api}/v3/entityServices/AiplatformEntityService/"
            f"schemas/AIPLATFORM_GRAPHQL:batchGraphql?key={self.settings.graphql_api_key}"
            "&prettyPrint=false"
        )
        headers = {
            **self._ua_headers(),
            "referer": "https://console.cloud.google.com/",
            "content-type": "application/json",
        }

        try:
            if CurlAsyncSession is not None and isinstance(session, CurlAsyncSession):
                resp = await session.post(
                    url,
                    headers=headers,
                    json=body,
                    timeout=self.settings.timeout_seconds,
                    impersonate="chrome131",
                    proxy=self.settings.proxy_url,
                )
                text = resp.text
                status_code = resp.status_code
                if status_code != 200:
                    logger.error(
                        "[VertexAIAnonymous] request failed: status=%s body=%s",
                        status_code,
                        text[:1024],
                    )
                    return None, None, f"HTTP {status_code}"
                try:
                    payload = resp.json()
                except Exception as exc:
                    logger.error("[VertexAIAnonymous] invalid JSON: %s", exc)
                    return None, None, f"Invalid JSON: {text[:1024]}"
            else:
                async with session.post(
                    url, headers=headers, json=body, proxy=self.settings.proxy_url
                ) as resp:
                    text = await resp.text()
                    status_code = resp.status
                    if status_code != 200:
                        logger.error(
                            "[VertexAIAnonymous] request failed: status=%s body=%s",
                            status_code,
                            text[:1024],
                        )
                        return None, None, f"HTTP {status_code}"
                    try:
                        payload = await resp.json()
                    except Exception as exc:
                        logger.error("[VertexAIAnonymous] invalid JSON: %s", exc)
                        return None, None, f"Invalid JSON: {text[:1024]}"
        except Exception as exc:
            logger.error("[VertexAIAnonymous] request error: %s", exc)
            return None, None, f"request error: {exc}"

        if not isinstance(payload, list):
            logger.warning(
                "[VertexAIAnonymous] unexpected response type: %s raw=%s",
                type(payload).__name__,
                text[:1024],
            )
            return None, 999, f"Unexpected response type: {type(payload).__name__}"

        out: list[tuple[str, str]] = []
        for elem in payload:
            if not isinstance(elem, dict):
                continue
            for item in elem.get("results", []):
                if not isinstance(item, dict):
                    continue

                errors = item.get("errors", [])
                for err in errors:
                    if not isinstance(err, dict):
                        continue
                    status = (
                        err.get("extensions", {}).get("status", {}).get("code", None)
                    )
                    err_msg = str(err.get("message") or "").strip()
                    if err_msg and "Failed to verify action" not in err_msg:
                        logger.warning(
                            "[VertexAIAnonymous] graphql error: status=%s msg=%s",
                            status,
                            err_msg,
                        )
                    return None, status, err_msg or "Vertex AI Anonymous error"

                for candidate in item.get("data", {}).get("candidates", []):
                    if not isinstance(candidate, dict):
                        continue
                    finish_reason = str(candidate.get("finishReason") or "").strip()
                    if finish_reason != "STOP":
                        logger.warning(
                            "[VertexAIAnonymous] response rejected: finishReason=%s raw=%s",
                            finish_reason,
                            text[:1024],
                        )
                        return (
                            None,
                            999,
                            f"Vertex AI Anonymous finishReason={finish_reason or 'UNKNOWN'}",
                        )

                    for part in candidate.get("content", {}).get("parts", []):
                        if not isinstance(part, dict):
                            continue
                        inline = part.get("inlineData")
                        if not isinstance(inline, dict):
                            continue
                        b64 = str(inline.get("data") or "").strip()
                        mime = str(inline.get("mimeType") or "").strip() or "image/png"
                        if b64:
                            out.append((mime, b64))

        if not out:
            logger.warning(
                "[VertexAIAnonymous] request succeeded but no images returned: %s",
                text[:1024],
            )
            return None, 999, "响应中未包含图片数据"
        return out, None, None

    async def _get_recaptcha_token(self) -> str | None:
        session = await self._get_session()
        anchor_url = build_anchor_url(self.settings.recaptcha_base_api)
        reload_url = build_reload_url(self.settings.recaptcha_base_api)

        for _ in range(RECAPTCHA_TOKEN_RETRIES):
            try:
                base_token = await self._fetch_anchor_token(session, anchor_url)
                if not base_token:
                    continue
                recaptcha_token = await self._fetch_reload_token(
                    session, reload_url, anchor_url, base_token
                )
                if recaptcha_token:
                    return recaptcha_token
            except Exception as exc:
                logger.warning("[VertexAIAnonymous] recaptcha attempt failed: %s", exc)
        return None

    async def _fetch_anchor_token(self, session: object, anchor_url: str) -> str | None:
        if CurlAsyncSession is not None and isinstance(session, CurlAsyncSession):
            resp = await session.get(
                anchor_url,
                headers=self._ua_headers(),
                proxy=self.settings.proxy_url,
                impersonate="chrome131",
            )
            html = resp.text
            if resp.status_code != 200:
                raise RuntimeError(
                    f"recaptcha anchor HTTP {resp.status_code}: {html[:512]}"
                )
        else:
            async with session.get(
                anchor_url, headers=self._ua_headers(), proxy=self.settings.proxy_url
            ) as resp:
                html = await resp.text()
                if resp.status != 200:
                    raise RuntimeError(
                        f"recaptcha anchor HTTP {resp.status}: {html[:512]}"
                    )
        return parse_anchor_token(html)

    async def _fetch_reload_token(
        self,
        session: object,
        reload_url: str,
        anchor_url: str,
        base_token: str,
    ) -> str | None:
        qp = extract_query_params(anchor_url)
        payload = {
            "v": qp.get("v", RECAPTCHA_V),
            "reason": "q",
            "k": qp.get("k", RECAPTCHA_SITE_KEY),
            "c": base_token,
            "co": qp.get("co", RECAPTCHA_CO),
            "hl": qp.get("hl", RECAPTCHA_HL),
            "size": "invisible",
            "vh": RECAPTCHA_VH,
            "chr": "",
            "bg": "",
        }
        headers = {
            **self._ua_headers(),
            "content-type": "application/x-www-form-urlencoded",
        }
        if CurlAsyncSession is not None and isinstance(session, CurlAsyncSession):
            resp = await session.post(
                reload_url,
                data=payload,
                headers=headers,
                proxy=self.settings.proxy_url,
                impersonate="chrome131",
            )
            text = resp.text
            if resp.status_code != 200:
                raise RuntimeError(
                    f"recaptcha reload HTTP {resp.status_code}: {text[:512]}"
                )
        else:
            async with session.post(
                reload_url,
                data=payload,
                headers=headers,
                proxy=self.settings.proxy_url,
            ) as resp:
                text = await resp.text()
                if resp.status != 200:
                    raise RuntimeError(
                        f"recaptcha reload HTTP {resp.status}: {text[:512]}"
                    )
        return parse_rresp(text)
