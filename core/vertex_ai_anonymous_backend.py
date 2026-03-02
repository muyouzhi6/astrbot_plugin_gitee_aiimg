from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiohttp

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
    RecaptchaExpiredError,
    NonRetryableError,
    build_anchor_url,
    build_reload_url,
    extract_images_from_graphql_payload,
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
        self._session: aiohttp.ClientSession | None = None
        self._session_lock = asyncio.Lock()

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
        self._session = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is not None and not self._session.closed:
            return self._session
        async with self._session_lock:
            if self._session is not None and not self._session.closed:
                return self._session
            timeout = aiohttp.ClientTimeout(
                total=self.settings.timeout_seconds,
                connect=_AIOHTTP_CONNECT_TIMEOUT_SECONDS,
            )
            connector = aiohttp.TCPConnector(
                limit=_AIOHTTP_LIMIT,
                limit_per_host=_AIOHTTP_LIMIT_PER_HOST,
                ttl_dns_cache=_AIOHTTP_DNS_CACHE_TTL_SECONDS,
            )
            self._session = aiohttp.ClientSession(timeout=timeout, connector=connector)
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
            raise RuntimeError("Vertex AI Anonymous 未返回图片数据")
        mime, b64 = images[0]
        logger.info("[VertexAIAnonymous] generate ok: images=%s mime=%s", len(images), mime)
        return await self.imgr.save_base64_image(b64)

    async def edit(self, prompt: str, images: list[bytes], **kwargs) -> Path:
        if not images:
            raise ValueError("At least one image is required")
        out = await self._generate_images(prompt, image_bytes_list=images, **kwargs)
        if not out:
            raise RuntimeError("Vertex AI Anonymous 未返回图片数据")
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
            raise RuntimeError("Vertex AI Anonymous 获取 recaptcha_token 失败")

        last_error: Exception | None = None
        body = self._build_body(prompt, image_bytes_list, size=size, resolution=resolution)
        for attempt in range(max(1, self.settings.max_retries)):
            try:
                body["variables"]["recaptchaToken"] = recaptcha_token
                return await self._call_api(body)
            except RecaptchaExpiredError:
                recaptcha_token = await self._get_recaptcha_token()
                if not recaptcha_token:
                    raise RuntimeError("Vertex AI Anonymous 刷新 recaptcha_token 失败")
            except NonRetryableError:
                raise
            except Exception as e:
                last_error = e
                logger.warning(
                    "[VertexAIAnonymous] call failed attempt=%s/%s: %s",
                    attempt + 1,
                    self.settings.max_retries,
                    e,
                )
        raise RuntimeError(f"Vertex AI Anonymous 请求失败: {last_error}") from last_error

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
                {"inlineData": {"mimeType": mime, "data": base64.b64encode(img).decode()}}
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
            context["systemInstruction"] = {"parts": [{"text": self.settings.system_prompt}]}

        return {
            "querySignature": self.settings.query_signature,
            "operationName": DEFAULT_OPERATION_NAME,
            "variables": context,
        }

    async def _call_api(self, body: dict[str, Any]) -> list[tuple[str, str]]:
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
        async with session.post(
            url, headers=headers, json=body, proxy=self.settings.proxy_url
        ) as resp:
            text = await resp.text()
            if resp.status != 200:
                raise RuntimeError(f"HTTP {resp.status}: {text[:1024]}")
            try:
                payload = await resp.json()
            except Exception as e:
                raise RuntimeError(f"Invalid JSON: {text[:1024]}") from e
        return extract_images_from_graphql_payload(payload)

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
            except Exception as e:
                logger.warning("[VertexAIAnonymous] recaptcha attempt failed: %s", e)
        return None

    async def _fetch_anchor_token(
        self, session: aiohttp.ClientSession, anchor_url: str
    ) -> str | None:
        async with session.get(
            anchor_url, headers=self._ua_headers(), proxy=self.settings.proxy_url
        ) as resp:
            html = await resp.text()
            if resp.status != 200:
                raise RuntimeError(f"recaptcha anchor HTTP {resp.status}: {html[:512]}")
        return parse_anchor_token(html)

    async def _fetch_reload_token(
        self,
        session: aiohttp.ClientSession,
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
        async with session.post(
            reload_url,
            data=payload,
            headers={**self._ua_headers(), "content-type": "application/x-www-form-urlencoded"},
            proxy=self.settings.proxy_url,
        ) as resp:
            text = await resp.text()
            if resp.status != 200:
                raise RuntimeError(f"recaptcha reload HTTP {resp.status}: {text[:512]}")
        return parse_rresp(text)
