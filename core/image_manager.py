
import asyncio
import base64
import os
import time
from pathlib import Path

import aiofiles
import aiohttp

from astrbot.api import logger


class ImageManager:
    """
    图片管理器
    """

    def __init__(self, config: dict, data_dir: Path):
        self.config = config
        self.image_dir = data_dir / "images"
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.cleanup_batch_ratio = 0.5

        self._session: aiohttp.ClientSession | None = None
    async def _session_get(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def download_image(self, url: str) -> Path:
        """下载远程图片并保存到本地，返回文件路径"""
        session = await self._session_get()
        async with session.get(url) as resp:
            if resp.status != 200:
                raise RuntimeError(f"图片下载失败 HTTP {resp.status}")
            data = await resp.read()

        return await self.save_image(data)

    async def save_image(self, data: bytes) -> Path:
        """保存 bytes 图片到本地"""
        filename = f"{int(time.time())}_{id(data)}.jpg"
        path = self.image_dir / filename

        async with aiofiles.open(path, "wb") as f:
            await f.write(data)
        await self.cleanup_old_images()
        return path

    async def save_base64_image(self, b64: str) -> Path:
        """保存 base64 图片到本地"""
        data = base64.b64decode(b64)
        return await self.save_image(data)

    async def cleanup_old_images(self) -> None:
        """清理旧图片（按比例清理，默认清一半）"""
        try:
            max_keep: int = self.config["max_cached_images"]

            images: list[Path] = list(self.image_dir.iterdir())
            total = len(images)

            if total <= max_keep:
                return

            overflow = total - max_keep
            delete_count = max(1, int(overflow * self.cleanup_batch_ratio))

            # 获取 mtime（阻塞 IO → 线程池）
            stats = await asyncio.gather(
                *[asyncio.to_thread(p.stat) for p in images],
                return_exceptions=True,
            )

            valid: list[tuple[Path, float]] = []

            for p, st in zip(images, stats):
                if isinstance(st, os.stat_result):
                    valid.append((p, st.st_mtime))

            valid.sort(key=lambda x: x[1])  # 旧 → 新

            to_delete = valid[:delete_count]

            await asyncio.gather(
                *[asyncio.to_thread(p.unlink) for p, _ in to_delete],
                return_exceptions=True,
            )

        except Exception as e:
            logger.warning(f"清理旧图片时出错: {e}")
