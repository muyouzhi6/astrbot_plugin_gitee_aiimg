import asyncio
import base64
from pathlib import Path

import aiohttp

from astrbot.api import logger

from .image_manager import ImageManager


class NanoBananaService:
    def __init__(self, config: dict, imgr: ImageManager):
        self.config = config
        self.imgr = imgr

        nb_conf = config.get("nanobanana", {})
        self.api_url = nb_conf.get(
            "api_url", "https://generativelanguage.googleapis.com"
        )
        self.model = nb_conf.get("model", "gemini-2.0-flash-preview-image-generation")
        self.resolution = nb_conf.get("resolution", "4K")
        self.timeout = int(nb_conf.get("timeout", 120))
        self.use_proxy = bool(nb_conf.get("use_proxy", False))
        self.proxy_url = str(nb_conf.get("proxy_url", "")).strip()
        self.max_images = int(nb_conf.get("max_images", 8))
        self.max_concurrency = int(nb_conf.get("max_concurrency", 2))

        raw_keys = nb_conf.get("api_keys", [])
        self.api_keys = [str(k).strip() for k in raw_keys if str(k).strip()]
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
                    timeout = aiohttp.ClientTimeout(
                        total=self.timeout,
                        connect=30,
                        sock_read=self.timeout,
                    )
                    self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def _next_key(self) -> str:
        async with self._key_lock:
            if not self.api_keys:
                raise RuntimeError("NanoBanana API Key is not configured")
            key = self.api_keys[self._key_index]
            self._key_index = (self._key_index + 1) % len(self.api_keys)
            return key

    def _build_url(self) -> str:
        base = self.api_url.rstrip("/")
        if not base.endswith("v1beta"):
            base = f"{base}/v1beta"
        return f"{base}/models/{self.model}:generateContent"

    async def _generate_once(self, prompt: str, ratio: str | None = None) -> Path:
        api_key = await self._next_key()
        url = self._build_url()

        ratio_text = f" Aspect ratio: {ratio}." if ratio else ""
        final_prompt = (
            f"Generate an image based on this prompt: {prompt}."
            f"{ratio_text} Output the image directly."
        )

        payload = {
            "contents": [{"parts": [{"text": final_prompt}]}],
            "generationConfig": {
                "maxOutputTokens": 8192,
                "responseModalities": ["image", "text"],
                "imageConfig": {"imageSize": self.resolution},
            },
            "safetySettings": [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_NONE",
                },
            ],
        }

        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        }

        proxy = self.proxy_url if self.use_proxy and self.proxy_url else None
        session = await self._get_session()

        try:
            async with session.post(
                url, json=payload, headers=headers, proxy=proxy
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    raise RuntimeError(
                        f"NanoBanana API error ({resp.status}): {error_text[:200]}"
                    )
                data = await resp.json()
        except asyncio.TimeoutError as e:
            raise RuntimeError(f"NanoBanana request timeout (>{self.timeout}s)") from e
        except aiohttp.ClientError as e:
            raise RuntimeError(f"NanoBanana network error: {e}") from e

        if "error" in data:
            raise RuntimeError(f"NanoBanana API error: {data['error']}")

        images: list[bytes] = []
        for candidate in data.get("candidates", []):
            content = candidate.get("content", {})
            for part in content.get("parts", []):
                if "inlineData" in part:
                    b64_data = part["inlineData"]["data"]
                    images.append(base64.b64decode(b64_data))

        if not images:
            raise RuntimeError("NanoBanana returned no image data")

        return await self.imgr.save_image(images[-1])

    async def generate(
        self, prompt: str, count: int = 4, ratio: str | None = None
    ) -> list[Path]:
        if not prompt or not prompt.strip():
            raise ValueError("prompt is required")

        count = int(count)
        if count < 1:
            raise ValueError("count must be >= 1")
        if count > self.max_images:
            raise ValueError(f"count must be <= {self.max_images}")

        sem = asyncio.Semaphore(max(1, self.max_concurrency))

        async def run_one() -> Path:
            async with sem:
                return await self._generate_once(prompt, ratio=ratio)

        tasks = [asyncio.create_task(run_one()) for _ in range(count)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        paths: list[Path] = []
        first_error: Exception | None = None
        for r in results:
            if isinstance(r, Exception):
                first_error = first_error or r
            else:
                paths.append(r)

        if not paths and first_error:
            raise first_error

        if first_error:
            logger.warning(f"[NanoBanana] partial failure: {first_error}")

        return paths
