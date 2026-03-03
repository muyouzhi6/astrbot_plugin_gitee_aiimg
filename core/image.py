import asyncio
import base64
import os
import time
from pathlib import Path

import aiofiles
import aiohttp

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent
from astrbot.api.message_components import Image
from astrbot.core.message.components import Reply
from astrbot.core.utils.io import download_image_by_url

from .net_safety import (
    URLFetchPolicy,
    collect_trusted_origins,
    ensure_url_allowed,
    read_network_policy,
)


class ImageManager:
    """
    图片管理器
    """

    def __init__(self, config: dict, data_dir: Path):
        self.config = config
        self.image_dir = data_dir / "images"
        self.image_dir.mkdir(parents=True, exist_ok=True)
        timeout_seconds = 60
        try:
            timeout_seconds = int(config.get("timeout", 60) or 60)
        except Exception:
            timeout_seconds = 60

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

        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=float(max(1, min(timeout_seconds, 3600))))
        )
        self.cleanup_batch_ratio = 0.5

    @staticmethod
    def _clamp_int(value, *, default: int, min_value: int, max_value: int) -> int:
        try:
            value_int = int(value)
        except (TypeError, ValueError):
            return default
        return max(min_value, min(max_value, value_int))

    async def close(self) -> None:
        await self._session.close()

    async def download_image(self, url: str) -> Path:
        """下载远程图片并保存到本地，返回文件路径"""
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
            async with self._session.get(current, allow_redirects=False) as resp:
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
            max_keep: int = int(self.config.get("max_cached_images", 50) or 50)

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

    async def download_image_bytes(self, url: str, timeout: int = 30) -> bytes | None:
        """下载图片并返回二进制数据，失败返回 None"""
        if not url or not url.startswith(("http://", "https://")):
            return None

        try:
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
                async with self._session.get(
                    current,
                    timeout=aiohttp.ClientTimeout(total=float(max(1, min(timeout, 3600)))),
                    allow_redirects=False,
                ) as resp:
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
                        logger.warning(
                            f"下载图片失败: HTTP {resp.status}, URL: {current[:60]}..."
                        )
                        return None

                    total = 0
                    chunks: list[bytes] = []
                    async for chunk in resp.content.iter_chunked(1024 * 256):
                        if not chunk:
                            continue
                        total += len(chunk)
                        if total > self._media_max_image_bytes:
                            raise RuntimeError("Image too large")
                        chunks.append(chunk)
                    return b"".join(chunks)
        except Exception as e:
            logger.warning(f"下载图片异常: {type(e).__name__}, URL: {url[:60]}...")
        return None

    async def extract_images_from_event(self, event: AstrMessageEvent) -> list[bytes]:
        """从消息事件中提取图片二进制数据列表

        支持：
        1. 回复/引用消息中的图片（优先）
        2. 当前消息中的图片
        3. 多图输入
        4. base64 格式图片
        """
        images: list[bytes] = []

        message_obj = event.message_obj
        chain = message_obj.message

        logger.debug(f"[extract_images] 开始提取图片, 消息链长度: {len(chain)}")

        # 1. 从回复/引用消息中提取图片（优先）
        for seg in chain:
            if isinstance(seg, Reply):
                logger.debug(
                    f"[extract_images] 发现 Reply 段, chain={getattr(seg, 'chain', None)}"
                )
                if hasattr(seg, "chain") and seg.chain:
                    for chain_item in seg.chain:
                        if isinstance(chain_item, Image):
                            img_data = await self._load_image_data(chain_item)
                            if img_data:
                                images.append(img_data)
                                logger.debug(
                                    f"[extract_images] 从回复链提取图片: {len(img_data)} bytes"
                                )

        # 2. 从当前消息中提取图片
        for seg in chain:
            if isinstance(seg, Image):
                img_data = await self._load_image_data(seg)
                if img_data:
                    images.append(img_data)
                    logger.debug(
                        f"[extract_images] 从当前消息提取图片: {len(img_data)} bytes"
                    )

        logger.info(f"[extract_images] 共提取到 {len(images)} 张图片")
        return images

    async def _load_image_data(self, img: Image) -> bytes | None:
        """从 Image 对象加载图片二进制数据

        优先级：本地文件 > base64 > URL下载
        """
        # 1. 尝试从本地文件读取（NapCat/LLOneBot 会缓存图片到本地）
        file_path = getattr(img, "file", None)
        if file_path and not file_path.startswith(("http://", "https://")):
            local_path = Path(file_path)
            # 尝试绝对路径
            if local_path.is_file():
                try:
                    logger.debug(f"[_load_image_data] 从本地绝对路径读取: {local_path}")
                    return local_path.read_bytes()
                except Exception as e:
                    logger.debug(f"读取本地文件失败: {e}")
            # 尝试 NapCat 缓存目录（常见路径）
            else:
                # NapCat 通常缓存在 data/Cache/Image 或类似目录
                possible_dirs = [
                    Path("data/Cache/Image"),
                    Path("data/image_cache"),
                    Path("cache/images"),
                ]
                for cache_dir in possible_dirs:
                    cached_path = cache_dir / file_path
                    if cached_path.is_file():
                        try:
                            logger.debug(
                                f"[_load_image_data] 从缓存目录读取: {cached_path}"
                            )
                            return cached_path.read_bytes()
                        except Exception as e:
                            logger.debug(f"读取缓存文件失败: {e}")

        # 2. 尝试 base64
        b64 = getattr(img, "base64", None)
        if b64:
            try:
                logger.debug("[_load_image_data] 从 base64 解码")
                return base64.b64decode(b64)
            except Exception:
                pass

        # 3. 尝试从 URL 下载（使用 AstrBot 内置函数，支持 SSL fallback）
        url = getattr(img, "url", None)
        if url:
            logger.debug(f"[_load_image_data] 尝试从 URL 下载: {url[:60]}...")
            try:
                # 使用 AstrBot 内置的下载函数，有更好的 SSL 处理
                downloaded_path = await download_image_by_url(url)
                if downloaded_path:
                    img_bytes = Path(downloaded_path).read_bytes()
                    logger.debug(f"[_load_image_data] 下载成功: {len(img_bytes)} bytes")
                    return img_bytes
            except Exception as e:
                logger.warning(f"[_load_image_data] 使用 AstrBot 下载失败: {e}")
                # fallback 到自己的下载方法
                result = await self.download_image_bytes(url, timeout=60)
                if result:
                    return result

        return None
