"""
视频缓存管理器

用于在需要以本地文件方式发送时，下载 Grok 返回的视频并进行简单清理。
"""

from __future__ import annotations

import asyncio
import os
import time
import uuid
from pathlib import Path
from typing import Any

import aiofiles
import httpx

from astrbot.api import logger


def _clamp_int(value: Any, *, default: int, min_value: int, max_value: int) -> int:
    try:
        value_int = int(value)
    except (TypeError, ValueError):
        return default
    return max(min_value, min(max_value, value_int))


class VideoManager:
    def __init__(self, config: dict, data_dir: Path):
        self.config = config
        storage = config.get("storage", {}) if isinstance(config, dict) else {}

        self.video_dir = data_dir / "videos"
        self.video_dir.mkdir(parents=True, exist_ok=True)

        self.max_cached_videos: int = _clamp_int(
            (storage.get("max_cached_videos") if isinstance(storage, dict) else None)
            or config.get("max_cached_videos", 20),
            default=20,
            min_value=0,
            max_value=500,
        )
        self.cleanup_batch_ratio = 0.5

    async def download_video(self, url: str, *, timeout_seconds: int = 300) -> Path:
        if not url:
            raise ValueError("缺少视频 URL")

        timeout_seconds = max(1, min(int(timeout_seconds), 3600))
        filename = f"{int(time.time())}_{uuid.uuid4().hex[:8]}.mp4"
        path = self.video_dir / filename

        timeout = httpx.Timeout(
            connect=10.0,
            read=float(timeout_seconds),
            write=10.0,
            pool=float(timeout_seconds) + 10.0,
        )

        t0 = time.perf_counter()
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            async with client.stream("GET", url) as resp:
                resp.raise_for_status()
                async with aiofiles.open(path, "wb") as f:
                    async for chunk in resp.aiter_bytes(chunk_size=1024 * 256):
                        await f.write(chunk)

        logger.info(
            f"[VideoManager] 下载完成: path={path}, 耗时={time.perf_counter() - t0:.2f}s"
        )

        await self.cleanup_old_videos()
        return path

    async def cleanup_old_videos(self) -> None:
        if self.max_cached_videos <= 0:
            return

        try:
            videos: list[Path] = list(self.video_dir.iterdir())
            total = len(videos)
            if total <= self.max_cached_videos:
                return

            overflow = total - self.max_cached_videos
            delete_count = max(1, int(overflow * self.cleanup_batch_ratio))

            stats = await asyncio.gather(
                *[asyncio.to_thread(p.stat) for p in videos],
                return_exceptions=True,
            )

            valid: list[tuple[Path, float]] = []
            for p, st in zip(videos, stats):
                if isinstance(st, os.stat_result):
                    valid.append((p, st.st_mtime))

            valid.sort(key=lambda x: x[1])  # old -> new
            to_delete = valid[:delete_count]

            await asyncio.gather(
                *[asyncio.to_thread(p.unlink) for p, _ in to_delete],
                return_exceptions=True,
            )

            logger.debug(
                f"[VideoManager] 清理旧视频: 删除={len(to_delete)}, 当前={total - len(to_delete)}"
            )

        except Exception as e:
            logger.warning(f"[VideoManager] 清理旧视频失败: {e}")
