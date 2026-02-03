"""
Gemini 原生 API 改图后端

支持特性:
- gemini-3-pro-image-preview 模型
- 4K 高分辨率输出
- API Key 轮询
- 代理支持
- 详细日志
"""

import asyncio
import base64
import time
from pathlib import Path
from typing import TYPE_CHECKING

import aiohttp

from astrbot.api import logger

from .image_format import guess_image_mime_and_ext

if TYPE_CHECKING:
    from .image_manager import ImageManager


class GeminiEditBackend:
    """Gemini 原生 API 图像后端（文生图 + 改图）。"""

    name = "Gemini"

    def __init__(self, *, imgr: "ImageManager", settings: dict):
        self.imgr = imgr

        conf = settings if isinstance(settings, dict) else {}
        self.api_url = conf.get("api_url", "https://generativelanguage.googleapis.com")
        self.model = conf.get("model", "gemini-3-pro-image-preview")
        self.resolution = conf.get("resolution", "4K")
        self.timeout = conf.get("timeout", 120)
        self.use_proxy = conf.get("use_proxy", False)
        self.proxy_url = conf.get("proxy_url", "")

        raw_keys = conf.get("api_keys", [])
        self.api_keys = [str(k).strip() for k in raw_keys if str(k).strip()]
        self._key_index = 0
        self._key_lock = asyncio.Lock()

        # HTTP Session (带锁保护)
        self._session: aiohttp.ClientSession | None = None
        self._session_lock = asyncio.Lock()

    @staticmethod
    def _normalize_models_base_url(raw: str) -> str:
        """
        Normalize Gemini native api_url into ".../v1beta/models".

        Accepts:
        - https://generativelanguage.googleapis.com
        - https://generativelanguage.googleapis.com/v1beta
        - https://generativelanguage.googleapis.com/v1beta/models
        - https://proxy.example.com/v1/chat/completions (will be rewritten to v1beta/models)
        """
        s = str(raw or "").strip().rstrip("/")
        if not s:
            return ""

        lower = s.lower()
        for suffix in (
            "/v1/chat/completions",
            "/chat/completions",
            "/v1/images/generations",
            "/images/generations",
            "/v1/completions",
            "/completions",
        ):
            if lower.endswith(suffix):
                s = s[: -len(suffix)].rstrip("/")
                lower = s.lower()
                break

        if lower.endswith("/v1"):
            s = s[:-3].rstrip("/")
            lower = s.lower()

        if lower.endswith("/v1beta/models"):
            return s
        if lower.endswith("/v1beta"):
            return f"{s}/models"

        return f"{s}/v1beta/models"

    def _build_url(self) -> str:
        base = self._normalize_models_base_url(self.api_url)
        return f"{base}/{self.model}:generateContent"

    def _proxy(self) -> str | None:
        return self.proxy_url if self.use_proxy and self.proxy_url else None

    @staticmethod
    def _size_to_resolution(size: str | None) -> str | None:
        s = str(size or "").strip().lower().replace("×", "x")
        if not s:
            return None
        if s == "1024x1024":
            return "1K"
        if s == "2048x2048":
            return "2K"
        if s == "4096x4096":
            return "4K"
        return None

    async def _request(
        self, parts: list[dict], *, resolution: str | None = None
    ) -> dict:
        api_key = await self._next_key()
        url = self._build_url()
        image_size = str(resolution or self.resolution or "4K").strip() or "4K"

        payload = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "maxOutputTokens": 8192,
                "responseModalities": ["IMAGE"],
                "imageConfig": {"imageSize": image_size},
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
            "Authorization": f"Bearer {api_key}",
        }

        proxy = self._proxy()
        if proxy:
            logger.debug(f"[Gemini] 使用代理: {proxy}")

        session = await self._get_session()
        try:
            async with session.post(
                url,
                json=payload,
                headers=headers,
                proxy=proxy,
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    logger.error(
                        f"[Gemini] API 错误 ({resp.status}): {error_text[:500]}"
                    )
                    raise RuntimeError(
                        f"Gemini API 错误 ({resp.status}): {error_text[:200]}"
                    )
                data = await resp.json()
        except asyncio.TimeoutError:
            logger.error(f"[Gemini] 请求超时 (>{self.timeout}s)")
            raise RuntimeError(f"Gemini 请求超时 (>{self.timeout}s)")
        except aiohttp.ClientError as e:
            logger.error(f"[Gemini] 网络错误: {e}")
            raise RuntimeError(f"Gemini 网络错误: {e}")

        if "error" in data:
            error_msg = data["error"]
            logger.error(f"[Gemini] API 返回错误: {error_msg}")
            raise RuntimeError(f"Gemini API 错误: {error_msg}")

        return data

    @staticmethod
    def _extract_images(data: dict) -> list[bytes]:
        all_images: list[bytes] = []
        for candidate in data.get("candidates", []):
            content = candidate.get("content", {})
            for part in content.get("parts", []):
                if "inlineData" in part:
                    b64_data = part["inlineData"]["data"]
                    all_images.append(base64.b64decode(b64_data))
        return all_images

    async def generate(
        self, prompt: str, *, resolution: str | None = None, **_
    ) -> Path:
        t_start = time.perf_counter()
        parts = [
            {
                "text": (
                    f"Generate a high quality {resolution or self.resolution} resolution image. "
                    f"Follow this instruction: {prompt}. "
                    "Output the image directly."
                )
            }
        ]
        data = await self._request(parts, resolution=resolution)
        all_images = self._extract_images(data)
        if not all_images:
            raise RuntimeError("Gemini 未返回图片")

        result_bytes = all_images[-1]
        result_path = await self.imgr.save_image(result_bytes)
        t_end = time.perf_counter()
        logger.info(f"[Gemini] 生图完成: 耗时={t_end - t_start:.2f}s")
        return result_path

    async def close(self) -> None:
        """清理资源"""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """获取或创建 HTTP Session (线程安全)"""
        if self._session is None or self._session.closed:
            async with self._session_lock:
                # Double-check pattern
                if self._session is None or self._session.closed:
                    connector = aiohttp.TCPConnector(
                        limit=10,
                        limit_per_host=5,
                        ttl_dns_cache=300,
                        enable_cleanup_closed=True,
                    )
                    timeout = aiohttp.ClientTimeout(
                        total=self.timeout,
                        connect=30,
                        sock_read=self.timeout,
                    )
                    self._session = aiohttp.ClientSession(
                        connector=connector,
                        timeout=timeout,
                    )
        return self._session

    async def _next_key(self) -> str:
        """轮询获取下一个 API Key"""
        async with self._key_lock:
            if not self.api_keys:
                raise RuntimeError("Gemini API Key 未配置")
            key = self.api_keys[self._key_index]
            self._key_index = (self._key_index + 1) % len(self.api_keys)
            return key

    async def edit(
        self,
        prompt: str,
        images: list[bytes],
        *,
        size: str | None = None,
        resolution: str | None = None,
        **_,
    ) -> Path:
        """
        执行改图

        Args:
            prompt: 提示词
            images: 图片字节列表

        Returns:
            生成图片的本地路径
        """
        if not images:
            raise ValueError("至少需要一张图片")
        t_start = time.perf_counter()

        final_resolution = (
            str(
                resolution or self._size_to_resolution(size) or self.resolution or "4K"
            ).strip()
            or "4K"
        )
        logger.info(
            f"[Gemini] 开始改图: model={self.model}, "
            f"resolution={final_resolution}, images={len(images)}"
        )

        final_prompt = (
            f"Re-imagine the attached image based on this instruction: {prompt}. "
            f"Generate a high quality {final_resolution} resolution image. "
            f"Output the transformed image directly."
        )

        parts: list[dict] = [{"text": final_prompt}]
        for img_bytes in images:
            mime, _ = guess_image_mime_and_ext(img_bytes)
            parts.append(
                {
                    "inlineData": {
                        "mimeType": mime,
                        "data": base64.b64encode(img_bytes).decode(),
                    }
                }
            )

        data = await self._request(parts, resolution=final_resolution)
        try:
            all_images = self._extract_images(data)
        except Exception as e:
            logger.error(f"[Gemini] 解析响应失败: {e}")
            raise RuntimeError(f"Gemini 响应解析失败: {e}")

        if not all_images:
            raise RuntimeError("Gemini 未返回图片")

        # 取最后一张图（第一张可能是低分辨率预览）
        result_bytes = all_images[-1]
        logger.info(
            f"[Gemini] 收到 {len(all_images)} 张图片, "
            f"使用最后一张 ({len(result_bytes)} bytes)"
        )

        # 保存图片
        t_save = time.perf_counter()
        result_path = await self.imgr.save_image(result_bytes)
        t_end = time.perf_counter()

        logger.info(
            f"[Gemini] 改图完成: 总耗时={t_end - t_start:.2f}s, 保存={t_end - t_save:.2f}s"
        )

        return result_path
