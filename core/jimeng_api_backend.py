from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import aiohttp

from astrbot.api import logger
from astrbot.api.message_components import Image


class JimengApiBackend:
    """即梦/豆包绘图 API（第三方聚合接口）后端。

    参考 data/plugins/doubao 的实现：GET api_url，返回 { code:200, image_url:[...] }。
    图生图需要 url 参数，因此会尝试把图片 bytes 暂存并注册到 AstrBot 文件服务。
    """

    def __init__(
        self,
        *,
        imgr,
        data_dir: Path,
        api_url: str,
        apikey: str,
        cookie_list: list[str] | None = None,
        default_style: str = "真实",
        default_ratio: str = "1:1",
        default_model: str = "Seedream 4.0",
        timeout: int = 120,
    ):
        self.imgr = imgr
        self.data_dir = Path(data_dir)
        self.api_url = str(api_url or "").strip()
        self.apikey = str(apikey or "").strip()
        self.cookie_list = [
            str(x).strip() for x in (cookie_list or []) if str(x).strip()
        ]
        self.default_style = str(default_style or "真实").strip()
        self.default_ratio = str(default_ratio or "1:1").strip()
        self.default_model = str(default_model or "Seedream 4.0").strip()
        self.timeout = int(timeout or 120)

        self._cookie_index = 0
        self._session: aiohttp.ClientSession | None = None
        self._lock = asyncio.Lock()

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
        self._session = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.timeout, connect=30)
            connector = aiohttp.TCPConnector(limit=10, limit_per_host=5)
            self._session = aiohttp.ClientSession(timeout=timeout, connector=connector)
        return self._session

    def _next_cookie_pair(self) -> tuple[str, str] | None:
        if not self.cookie_list:
            return None
        item = self.cookie_list[self._cookie_index]
        self._cookie_index = (self._cookie_index + 1) % len(self.cookie_list)
        if ":" not in item:
            return None
        conv_id, cookie = item.split(":", 1)
        conv_id = conv_id.strip()
        cookie = cookie.strip()
        if not conv_id or not cookie:
            return None
        return conv_id, cookie

    async def _call(
        self,
        *,
        desc: str,
        image_url: str | None = None,
        style: str | None = None,
        ratio: str | None = None,
        model: str | None = None,
    ) -> list[str]:
        if not self.api_url:
            raise RuntimeError("Jimeng api_url 未配置")
        if not self.apikey:
            raise RuntimeError("Jimeng apikey 未配置")

        cookie_pair = self._next_cookie_pair()
        if not cookie_pair:
            raise RuntimeError(
                "Jimeng cookie_list 未配置或格式错误（需 conversation_id:cookie）"
            )
        conv_id, cookie = cookie_pair

        params = {
            "description": desc,
            "type": (style or self.default_style),
            "ratio": (ratio or self.default_ratio),
            "model": (model or self.default_model),
            "conversation_id": conv_id,
            "Cookie": cookie,
            "apikey": self.apikey,
        }
        if image_url:
            params["url"] = image_url

        session = await self._get_session()
        t0 = time.time()
        async with session.get(
            self.api_url,
            params=params,
            headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"},
        ) as resp:
            text = await resp.text()
            if resp.status != 200:
                raise RuntimeError(f"Jimeng API HTTP {resp.status}")
            try:
                data = json.loads(text)
            except Exception:
                raise RuntimeError(f"Jimeng API 返回非 JSON: {text[:200]}")

        logger.info(f"[Jimeng] API 响应耗时: {time.time() - t0:.2f}s")

        if data.get("code") != 200 or "image_url" not in data:
            msg = data.get("message") or data.get("text") or str(data)[:200]
            raise RuntimeError(f"Jimeng API 业务错误: {msg}")

        urls = data.get("image_url")
        if isinstance(urls, list):
            return [str(u) for u in urls if str(u)]
        if isinstance(urls, str) and urls:
            return [urls]
        raise RuntimeError("Jimeng API 未返回 image_url")

    async def _bytes_to_public_url(self, image_bytes: bytes) -> str:
        """保存 bytes 并注册到 AstrBot 文件服务，得到可外网访问的 URL。"""
        if not image_bytes:
            raise RuntimeError("空图片数据")

        # 1) 保存到插件数据目录
        tmp_dir = self.data_dir / "jimeng_upload"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = tmp_dir / f"upload_{int(time.time() * 1000)}.jpg"
        await asyncio.to_thread(tmp_path.write_bytes, image_bytes)

        # 2) 注册到文件服务
        img_comp = Image.fromFileSystem(str(tmp_path))
        try:
            url = await img_comp.register_to_file_service()
        finally:
            # 仅清理本地文件，不影响 file service 已注册的 token
            try:
                await asyncio.to_thread(tmp_path.unlink)
            except Exception:
                pass
        return url

    async def generate(self, prompt: str, **kwargs) -> Path:
        urls = await self._call(desc=prompt)
        return await self.imgr.download_image(urls[0])

    async def edit(self, prompt: str, images: list[bytes], **kwargs) -> Path:
        if not images:
            raise ValueError("至少需要一张图片")
        # Jimeng API 只接受 url，因此把第一张图注册到文件服务
        image_url = await self._bytes_to_public_url(images[0])
        urls = await self._call(desc=prompt, image_url=image_url)
        return await self.imgr.download_image(urls[0])
