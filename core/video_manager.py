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

from .net_safety import URLFetchPolicy, collect_trusted_origins, ensure_url_allowed, read_network_policy


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

        net = read_network_policy(config)
        self._media_allow_private: bool = bool(net.get("media_allow_private", False))
        self._media_max_video_bytes: int = _clamp_int(
            net.get("max_video_bytes", 50 * 1024 * 1024),
            default=50 * 1024 * 1024,
            min_value=5 * 1024 * 1024,
            max_value=5 * 1024 * 1024 * 1024,
        )
        self._media_max_redirects: int = _clamp_int(
            net.get("max_redirects", 5), default=5, min_value=0, max_value=10
        )
        self._dns_timeout_seconds: int = _clamp_int(
            net.get("dns_resolve_timeout_seconds", 2),
            default=2,
            min_value=1,
            max_value=10,
        )
        self._trusted_origins: frozenset[str] = frozenset(collect_trusted_origins(config))

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
        tmp_path = self.video_dir / f"{filename}.part"

        timeout = httpx.Timeout(
            connect=10.0,
            read=float(timeout_seconds),
            write=10.0,
            pool=float(timeout_seconds) + 10.0,
        )

        policy = URLFetchPolicy(
            allow_private=self._media_allow_private,
            trusted_origins=self._trusted_origins,
            allowed_hosts=frozenset(),
            dns_timeout_seconds=float(self._dns_timeout_seconds),
        )

        t0 = time.perf_counter()
        current = str(url or "").strip()
        redirects = 0
        try:
            while True:
                await ensure_url_allowed(current, policy=policy)
                async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client:
                    async with client.stream("GET", current) as resp:
                        if resp.status_code in {301, 302, 303, 307, 308}:
                            if redirects >= self._media_max_redirects:
                                raise RuntimeError("Too many redirects")
                            loc = (resp.headers.get("location") or "").strip()
                            if not loc:
                                raise RuntimeError("Redirect without location")
                            current = str(httpx.URL(current).join(loc))
                            redirects += 1
                            continue

                        resp.raise_for_status()

                        total = 0
                        async with aiofiles.open(tmp_path, "wb") as f:
                            async for chunk in resp.aiter_bytes(chunk_size=1024 * 256):
                                if not chunk:
                                    continue
                                total += len(chunk)
                                if total > self._media_max_video_bytes:
                                    raise RuntimeError("Video too large")
                                await f.write(chunk)

                break
        except Exception:
            try:
                if tmp_path.exists():
                    await asyncio.to_thread(tmp_path.unlink)
            except Exception:
                pass
            raise

        try:
            await asyncio.to_thread(tmp_path.replace, path)
        except Exception:
            # fallback copy if replace fails
            await asyncio.to_thread(tmp_path.rename, path)

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
