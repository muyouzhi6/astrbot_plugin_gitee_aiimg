import asyncio
import base64
import os
import time
from pathlib import Path

import aiofiles
import aiohttp

from astrbot.api import logger

from .image_format import guess_image_mime_and_ext
from .net_safety import URLFetchPolicy, collect_trusted_origins, ensure_url_allowed, read_network_policy


class ImageManager:
    """
    图片管理器
    """

    def __init__(self, config: dict, data_dir: Path):
        self.config = config
        self.image_dir = data_dir / "images"
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.cleanup_batch_ratio = 0.5
        self._session_lock = asyncio.Lock()

        self._timeout_seconds = self._clamp_int(
            config.get("timeout", 120) if isinstance(config, dict) else 120,
            default=120,
            min_value=10,
            max_value=3600,
        )

        net = read_network_policy(config)
        self._media_allow_private: bool = bool(net.get("media_allow_private", False))
        self._media_max_image_bytes: int = self._clamp_int(
            net.get("max_image_bytes", 50 * 1024 * 1024),
            default=50 * 1024 * 1024,
            min_value=256 * 1024,
            max_value=200 * 1024 * 1024,
        )
        self._media_max_redirects: int = self._clamp_int(
            net.get("max_redirects", 5), default=5, min_value=0, max_value=10
        )
        self._dns_timeout_seconds: int = self._clamp_int(
            net.get("dns_resolve_timeout_seconds", 2),
            default=2,
            min_value=1,
            max_value=10,
        )
        self._trusted_origins: frozenset[str] = frozenset(collect_trusted_origins(config))

        self._session: aiohttp.ClientSession | None = None

    @staticmethod
    def _clamp_int(value, *, default: int, min_value: int, max_value: int) -> int:
        try:
            value_int = int(value)
        except (TypeError, ValueError):
            return default
        return max(min_value, min(max_value, value_int))

    async def _session_get(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            async with self._session_lock:
                if self._session is None or self._session.closed:
                    timeout = aiohttp.ClientTimeout(
                        total=float(self._timeout_seconds),
                        connect=min(30.0, float(self._timeout_seconds)),
                        sock_read=float(self._timeout_seconds),
                    )
                    connector = aiohttp.TCPConnector(limit=10, limit_per_host=5)
                    self._session = aiohttp.ClientSession(
                        timeout=timeout,
                        connector=connector,
                    )
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def download_image(self, url: str) -> Path:
        """下载远程图片并保存到本地，返回文件路径"""
        t0 = time.time()
        session = await self._session_get()

        policy = URLFetchPolicy(
            allow_private=self._media_allow_private,
            trusted_origins=self._trusted_origins,
            allowed_hosts=frozenset(),
            dns_timeout_seconds=float(self._dns_timeout_seconds),
        )

        current = str(url or "").strip()
        redirects = 0
        while True:
            await ensure_url_allowed(current, policy=policy)
            async with session.get(current, allow_redirects=False) as resp:
                if resp.status in {301, 302, 303, 307, 308}:
                    if redirects >= self._media_max_redirects:
                        raise RuntimeError("Too many redirects")
                    loc = (resp.headers.get("location") or "").strip()
                    if not loc:
                        raise RuntimeError("Redirect without location")
                    current = (
                        aiohttp.client.URL(current)
                        .join(aiohttp.client.URL(loc))
                        .human_repr()
                    )
                    redirects += 1
                    continue

                if resp.status != 200:
                    raise RuntimeError(f"图片下载失败 HTTP {resp.status}")

                total = 0
                chunks: list[bytes] = []
                async for chunk in resp.content.iter_chunked(1024 * 256):
                    if not chunk:
                        continue
                    total += len(chunk)
                    if total > self._media_max_image_bytes:
                        raise RuntimeError("Image too large")
                    chunks.append(chunk)
                data = b"".join(chunks)
                break

        logger.info(
            f"[ImageManager] 网络下载耗时: {time.time() - t0:.2f}s, 大小: {len(data)} bytes"
        )

        return await self.save_image(data)

    async def save_image(self, data: bytes) -> Path:
        """保存 bytes 图片到本地"""
        t0 = time.time()
        _, ext = guess_image_mime_and_ext(data)
        filename = f"{int(time.time())}_{id(data)}.{ext}"
        path = self.image_dir / filename

        async with aiofiles.open(path, "wb") as f:
            await f.write(data)

        t1 = time.time()
        await self.cleanup_old_images()
        logger.info(
            f"[ImageManager] 保存耗时: {t1 - t0:.2f}s, 清理耗时: {time.time() - t1:.2f}s"
        )

        return path

    async def save_base64_image(self, b64: str) -> Path:
        """保存 base64 图片到本地"""
        data = base64.b64decode(b64)
        return await self.save_image(data)

    async def cleanup_old_images(self) -> None:
        """清理旧图片（按比例清理，默认清一半）"""
        try:
            storage = (
                self.config.get("storage", {}) if isinstance(self.config, dict) else {}
            )
            max_keep: int = int(
                (
                    storage.get("max_cached_images")
                    if isinstance(storage, dict)
                    else None
                )
                or self.config.get("max_cached_images", 50)
            )

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
