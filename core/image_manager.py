import asyncio
import base64
import io
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import aiofiles
import aiohttp

from astrbot.api import logger

from .image_format import guess_image_mime_and_ext, normalize_output_format
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
        self.cleanup_batch_ratio = 0.5
        self._session_lock = asyncio.Lock()
        self._encode_executor: ThreadPoolExecutor | None = ThreadPoolExecutor(
            max_workers=max(1, min(2, os.cpu_count() or 1)),
            thread_name_prefix="gitee-image-encode",
        )

        encoding = config.get("image_encoding", {}) if isinstance(config, dict) else {}
        if not isinstance(encoding, dict):
            encoding = {}
        self._jpeg_quality = self._clamp_int(
            encoding.get("jpeg_quality", 95),
            default=95,
            min_value=85,
            max_value=100,
        )
        raw_subsampling = str(
            encoding.get("jpeg_subsampling", "4:4:4") or "4:4:4"
        ).strip()
        self._jpeg_subsampling = {
            "444": 0,
            "4:4:4": 0,
            "422": 1,
            "4:2:2": 1,
            "420": 2,
            "4:2:0": 2,
        }.get(raw_subsampling, 0)
        self._webp_quality = self._clamp_int(
            encoding.get("webp_quality", 97),
            default=97,
            min_value=85,
            max_value=100,
        )
        self._webp_lossless_effort = self._clamp_int(
            encoding.get("webp_lossless_effort", 80),
            default=80,
            min_value=0,
            max_value=100,
        )
        self._webp_method = self._clamp_int(
            encoding.get("webp_method", 4),
            default=4,
            min_value=0,
            max_value=6,
        )
        self._png_compress_level = self._clamp_int(
            encoding.get("png_compress_level", 9),
            default=9,
            min_value=0,
            max_value=9,
        )

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
        self._trusted_origins: frozenset[str] = frozenset(
            collect_trusted_origins(config)
        )

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

    async def close(self) -> None:
        """Close network and image-encoding resources."""

        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
        executor = self._encode_executor
        if executor is not None:
            self._encode_executor = None
            await asyncio.to_thread(
                executor.shutdown,
                wait=True,
                cancel_futures=True,
            )

    async def download_image(self, url: str, *, output_format: str = "jpeg") -> Path:
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

        return await self.save_image(data, output_format=output_format)

    def _convert_image(self, data: bytes, fmt: str) -> tuple[bytes, str]:
        """Convert image bytes without touching the asyncio event loop.

        Args:
            data: Source image bytes.
            fmt: Normalized target image format.

        Returns:
            Converted bytes and a human-readable encoding label.
        """

        from PIL import Image as PILImage
        from PIL import ImageOps

        original_data = data
        conversion_label = ""
        with PILImage.open(io.BytesIO(data)) as opened:
            im = ImageOps.exif_transpose(opened)
            im.load()
            metadata: dict[str, object] = {}
            for key in ("icc_profile", "exif", "xmp"):
                value = im.info.get(key)
                if value:
                    metadata[key] = value

            buf = io.BytesIO()
            if fmt == "jpeg":
                if im.mode in {"RGBA", "LA"} or (
                    im.mode == "P" and "transparency" in im.info
                ):
                    rgba = im.convert("RGBA")
                    bg = PILImage.new("RGB", rgba.size, (255, 255, 255))
                    bg.paste(rgba, mask=rgba.split()[-1])
                    im = bg
                elif im.mode != "RGB":
                    im = im.convert("RGB")
                im.save(
                    buf,
                    format="JPEG",
                    quality=self._jpeg_quality,
                    subsampling=self._jpeg_subsampling,
                    optimize=True,
                    progressive=True,
                    **{
                        key: value
                        for key, value in metadata.items()
                        if key in {"icc_profile", "exif"}
                    },
                )
                conversion_label = (
                    f"JPEG q={self._jpeg_quality} subsampling={self._jpeg_subsampling}"
                )
            elif fmt in {"webp", "webp_lossless"}:
                if im.mode == "P":
                    im = im.convert("RGBA" if "transparency" in im.info else "RGB")
                elif im.mode not in {"RGB", "RGBA"}:
                    im = im.convert("RGBA" if "A" in im.getbands() else "RGB")
                lossless = fmt == "webp_lossless"
                quality = self._webp_lossless_effort if lossless else self._webp_quality
                im.save(
                    buf,
                    format="WEBP",
                    lossless=lossless,
                    quality=quality,
                    method=self._webp_method,
                    alpha_quality=100,
                    exact=True,
                    **metadata,
                )
                conversion_label = (
                    f"WebP lossless effort={quality} method={self._webp_method}"
                    if lossless
                    else f"WebP q={quality} method={self._webp_method}"
                )
            elif fmt == "png":
                im.save(
                    buf,
                    format="PNG",
                    optimize=self._png_compress_level == 9,
                    compress_level=self._png_compress_level,
                    **{
                        key: value
                        for key, value in metadata.items()
                        if key in {"icc_profile", "exif"}
                    },
                )
                conversion_label = f"PNG compress_level={self._png_compress_level}"

            converted_data = buf.getvalue()
            if converted_data:
                data = converted_data
                logger.debug(
                    "[ImageManager] 格式转换: %s bytes -> %s bytes (%s)",
                    len(original_data),
                    len(data),
                    conversion_label,
                )
        return data, conversion_label

    async def save_image(self, data: bytes, *, output_format: str = "jpeg") -> Path:
        """Persist image bytes after optional bounded background encoding.

        Args:
            data: Source image bytes.
            output_format: Target format: jpeg, webp, webp_lossless, png, or auto.

        Returns:
            Path to the persisted image.
        """

        t0 = time.time()
        fmt = normalize_output_format(output_format)
        original_data = data
        conversion_label = ""
        source_mime, _ = guess_image_mime_and_ext(data)
        preserve_existing = (fmt == "jpeg" and source_mime == "image/jpeg") or (
            fmt == "webp" and source_mime == "image/webp"
        )

        if fmt != "auto" and not preserve_existing:
            try:
                executor = self._encode_executor
                if executor is None:
                    raise RuntimeError("Image manager is closed")
                (
                    data,
                    conversion_label,
                ) = await asyncio.get_running_loop().run_in_executor(
                    executor,
                    self._convert_image,
                    data,
                    fmt,
                )
            except Exception as e:
                logger.warning("[ImageManager] 格式转换失败，使用原图: %s", e)
                data = original_data
                conversion_label = ""

        _, ext = guess_image_mime_and_ext(data)
        filename = f"{int(time.time())}_{id(data)}.{ext}"
        path = self.image_dir / filename

        async with aiofiles.open(path, "wb") as f:
            await f.write(data)

        t1 = time.time()
        await self.cleanup_old_images()
        logger.info(
            f"[ImageManager] 保存耗时: {t1 - t0:.2f}s, 清理耗时: {time.time() - t1:.2f}s"
            + (f", 已转换为 {conversion_label}" if conversion_label else "")
        )

        return path

    async def save_base64_image(self, b64: str, *, output_format: str = "jpeg") -> Path:
        """保存 base64 图片到本地"""
        data = base64.b64decode(b64)
        return await self.save_image(data, output_format=output_format)

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
