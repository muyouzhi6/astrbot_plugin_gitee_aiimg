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

if TYPE_CHECKING:
    from .image_manager import ImageManager


class GeminiEditBackend:
    """Gemini 原生 API 改图后端"""

    name = "Gemini"

    def __init__(self, config: dict, imgr: "ImageManager"):
        self.config = config
        self.imgr = imgr

        # Gemini 配置
        gemini_conf = config.get("edit", {}).get("gemini", {})
        self.api_url = gemini_conf.get("api_url", "https://generativelanguage.googleapis.com")
        self.model = gemini_conf.get("model", "gemini-3-pro-image-preview")
        self.resolution = gemini_conf.get("resolution", "4K")
        self.timeout = gemini_conf.get("timeout", 120)
        self.use_proxy = gemini_conf.get("use_proxy", False)
        self.proxy_url = gemini_conf.get("proxy_url", "")

        # API Key 池
        raw_keys = gemini_conf.get("api_keys", [])
        self.api_keys = [str(k).strip() for k in raw_keys if str(k).strip()]
        self._key_index = 0
        self._key_lock = asyncio.Lock()

        # HTTP Session (带锁保护)
        self._session: aiohttp.ClientSession | None = None
        self._session_lock = asyncio.Lock()

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

    async def edit(self, prompt: str, images: list[bytes]) -> Path:
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

        api_key = await self._next_key()
        t_start = time.perf_counter()

        logger.info(
            f"[Gemini] 开始改图: model={self.model}, "
            f"resolution={self.resolution}, images={len(images)}"
        )

        # 构建请求 URL
        base = self.api_url.rstrip("/")
        if not base.endswith("v1beta"):
            base = f"{base}/v1beta"
        url = f"{base}/models/{self.model}:generateContent"

        # 构建请求体
        final_prompt = (
            f"Re-imagine the attached image based on this instruction: {prompt}. "
            f"Generate a high quality {self.resolution} resolution image. "
            f"Output the transformed image directly."
        )

        parts: list[dict] = [{"text": final_prompt}]
        for img_bytes in images:
            parts.append({
                "inlineData": {
                    "mimeType": "image/jpeg",
                    "data": base64.b64encode(img_bytes).decode(),
                }
            })

        payload = {
            "contents": [{"parts": parts}],
            "generationConfig": {
                "maxOutputTokens": 8192,
                "responseModalities": ["image", "text"],
                "imageConfig": {"imageSize": self.resolution},
            },
            "safetySettings": [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ],
        }

        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        }

        # 代理设置
        proxy = self.proxy_url if self.use_proxy and self.proxy_url else None
        if proxy:
            logger.debug(f"[Gemini] 使用代理: {proxy}")

        # 发送请求
        session = await self._get_session()
        try:
            async with session.post(
                url,
                json=payload,
                headers=headers,
                proxy=proxy,
            ) as resp:
                t_api = time.perf_counter()
                logger.debug(f"[Gemini] API 响应状态: {resp.status}, 耗时: {t_api - t_start:.2f}s")

                if resp.status != 200:
                    error_text = await resp.text()
                    logger.error(f"[Gemini] API 错误 ({resp.status}): {error_text[:500]}")
                    raise RuntimeError(f"Gemini API 错误 ({resp.status}): {error_text[:200]}")

                data = await resp.json()

        except asyncio.TimeoutError:
            logger.error(f"[Gemini] 请求超时 (>{self.timeout}s)")
            raise RuntimeError(f"Gemini 请求超时 (>{self.timeout}s)")
        except aiohttp.ClientError as e:
            logger.error(f"[Gemini] 网络错误: {e}")
            raise RuntimeError(f"Gemini 网络错误: {e}")

        # 检查错误响应
        if "error" in data:
            error_msg = data["error"]
            logger.error(f"[Gemini] API 返回错误: {error_msg}")
            raise RuntimeError(f"Gemini API 错误: {error_msg}")

        # 提取图片 - 取最后一张（高分辨率版本）
        all_images: list[bytes] = []
        try:
            for candidate in data.get("candidates", []):
                content = candidate.get("content", {})
                for part in content.get("parts", []):
                    if "inlineData" in part:
                        b64_data = part["inlineData"]["data"]
                        all_images.append(base64.b64decode(b64_data))
        except Exception as e:
            logger.error(f"[Gemini] 解析响应失败: {e}")
            raise RuntimeError(f"Gemini 响应解析失败: {e}")

        if not all_images:
            # 尝试获取文本响应用于调试
            text_parts = []
            for candidate in data.get("candidates", []):
                for part in candidate.get("content", {}).get("parts", []):
                    if "text" in part:
                        text_parts.append(part["text"][:200])
            logger.error(f"[Gemini] 未返回图片, 文本响应: {text_parts}")
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
            f"[Gemini] 改图完成: 总耗时={t_end - t_start:.2f}s, "
            f"API={t_api - t_start:.2f}s, 保存={t_end - t_save:.2f}s"
        )

        return result_path
