"""
视频缓存管理器

用于在需要以本地文件方式发送时，下载 Grok 返回的视频并进行简单清理。

注意：部分后端可能返回的是一个页面 URL（text/html），页面里再包含真正的 mp4 链接。
本管理器会在下载前尝试把 HTML/JSON 解析成“直链 mp4”，避免最终发送的只是一个网页链接。
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import aiofiles
import httpx

from astrbot.api import logger

from .net_safety import (
    URLFetchPolicy,
    collect_trusted_origins,
    ensure_url_allowed,
    read_network_policy,
)


def _clamp_int(value: Any, *, default: int, min_value: int, max_value: int) -> int:
    try:
        value_int = int(value)
    except (TypeError, ValueError):
        return default
    return max(min_value, min(max_value, value_int))


def _normalized_origin(url: str) -> tuple[str, str, int | None]:
    parts = urlsplit(url)
    scheme = parts.scheme.lower()
    port = parts.port
    if port is None:
        port = 443 if scheme == "https" else 80 if scheme == "http" else None
    return scheme, (parts.hostname or "").lower(), port


def _looks_like_video_bytes(prefix: bytes) -> bool:
    return bool(
        (len(prefix) >= 12 and prefix[4:8] == b"ftyp")
        or prefix.startswith(b"\x1a\x45\xdf\xa3")
        or prefix.startswith(b"OggS")
        or (prefix.startswith(b"RIFF") and prefix[8:12] == b"AVI ")
        or prefix.startswith(b"\x00\x00\x01\xba")
    )


def _validate_video_payload(
    *, content_type: str, prefix: bytes, total_bytes: int
) -> None:
    if total_bytes <= 0:
        raise RuntimeError("Downloaded video is empty")

    media_type = content_type.split(";", 1)[0].strip().lower()
    if (
        media_type.startswith("text/")
        or media_type
        in {
            "application/json",
            "application/problem+json",
        }
        or media_type.endswith("+json")
    ):
        raise RuntimeError(f"Upstream returned non-video content: {media_type}")

    if media_type.startswith("video/") or media_type == "application/mp4":
        return
    if _looks_like_video_bytes(prefix):
        return
    raise RuntimeError(
        f"Upstream response is not a recognized video: {media_type or 'unknown type'}"
    )


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
        self._trusted_origins: frozenset[str] = frozenset(
            collect_trusted_origins(config)
        )
        self._video_range_download_enabled: bool = bool(
            net.get("video_range_download", False)
        )
        self._video_range_chunk_bytes: int = _clamp_int(
            net.get("video_range_chunk_bytes", 512 * 1024),
            default=512 * 1024,
            min_value=64 * 1024,
            max_value=8 * 1024 * 1024,
        )
        self._video_range_concurrency: int = _clamp_int(
            net.get("video_range_concurrency", 4),
            default=4,
            min_value=1,
            max_value=8,
        )

        self.max_cached_videos: int = _clamp_int(
            (storage.get("max_cached_videos") if isinstance(storage, dict) else None)
            or config.get("max_cached_videos", 20),
            default=20,
            min_value=0,
            max_value=500,
        )
        self.cleanup_batch_ratio = 0.5

    async def _download_video_ranges(
        self,
        url: str,
        *,
        tmp_path: Path,
        timeout: httpx.Timeout,
        headers: dict[str, str],
        policy: URLFetchPolicy,
    ) -> bool:
        """Download a same-origin authenticated video using parallel byte ranges.

        A few OpenAI-compatible gateways advertise a valid MP4 but deliver a full
        response so slowly that a single streaming request is not practical.  A
        206 probe lets us split that response into bounded requests while keeping
        the bearer token on the trusted origin.
        """
        if not headers:
            return False

        await ensure_url_allowed(url, policy=policy)
        range_dir = tmp_path.with_name(f"{tmp_path.name}.ranges")
        try:
            await asyncio.to_thread(shutil.rmtree, range_dir, True)
            if tmp_path.exists():
                await asyncio.to_thread(tmp_path.unlink)
            range_dir.mkdir(parents=True, exist_ok=True)

            probe_headers = dict(headers)
            probe_headers["Range"] = "bytes=0-0"
            async with httpx.AsyncClient(
                timeout=timeout, follow_redirects=False
            ) as client:
                async with client.stream("GET", url, headers=probe_headers) as resp:
                    if resp.status_code in {301, 302, 303, 307, 308}:
                        return False
                    if resp.status_code == 200:
                        return False
                    resp.raise_for_status()
                    if resp.status_code != 206:
                        return False
                    content_range = resp.headers.get("content-range", "")
                    probe_match = re.fullmatch(
                        r"bytes\s+0-0/(\d+)", content_range, re.IGNORECASE
                    )
                    if not probe_match:
                        return False
                    total_size = int(probe_match.group(1))
                    if total_size <= 0 or total_size > self._media_max_video_bytes:
                        raise RuntimeError("Video too large")
                    probe_body = bytearray()
                    async for chunk in resp.aiter_bytes(chunk_size=64 * 1024):
                        probe_body.extend(chunk)
                    if len(probe_body) != 1:
                        return False
                    content_type = resp.headers.get("content-type") or ""

                ranges = [
                    (
                        start,
                        min(total_size - 1, start + self._video_range_chunk_bytes - 1),
                    )
                    for start in range(0, total_size, self._video_range_chunk_bytes)
                ]
                next_index = 0
                next_lock = asyncio.Lock()

                async def fetch_one(index: int, start: int, end: int) -> None:
                    part_path = range_dir / f"{index:08d}.part"
                    expected_size = end - start + 1
                    request_headers = dict(headers)
                    request_headers["Range"] = f"bytes={start}-{end}"
                    last_error: Exception | None = None
                    for attempt in range(3):
                        try:
                            body = bytearray()
                            async with client.stream(
                                "GET", url, headers=request_headers
                            ) as part_resp:
                                part_resp.raise_for_status()
                                part_range = part_resp.headers.get("content-range", "")
                                match = re.fullmatch(
                                    r"bytes\s+(\d+)-(\d+)/(\d+)",
                                    part_range,
                                    re.IGNORECASE,
                                )
                                if (
                                    part_resp.status_code != 206
                                    or not match
                                    or int(match.group(1)) != start
                                    or int(match.group(2)) != end
                                    or int(match.group(3)) != total_size
                                ):
                                    raise RuntimeError(
                                        "Upstream returned an unexpected video byte range"
                                    )
                                async for chunk in part_resp.aiter_bytes(
                                    chunk_size=64 * 1024
                                ):
                                    if chunk:
                                        body.extend(chunk)
                            if len(body) != expected_size:
                                raise httpx.ReadError(
                                    "Upstream closed the video range before completion"
                                )
                            await asyncio.to_thread(part_path.write_bytes, bytes(body))
                            return
                        except (
                            httpx.ReadError,
                            httpx.RemoteProtocolError,
                            httpx.ConnectError,
                            httpx.ReadTimeout,
                        ) as exc:
                            last_error = exc
                            if attempt + 1 < 3:
                                await asyncio.sleep(min(2**attempt, 8))
                    raise last_error or RuntimeError("Video range download failed")

                async def worker() -> None:
                    nonlocal next_index
                    while True:
                        async with next_lock:
                            if next_index >= len(ranges):
                                return
                            index = next_index
                            next_index += 1
                        start, end = ranges[index]
                        await fetch_one(index, start, end)

                workers = [
                    asyncio.create_task(worker())
                    for _ in range(min(self._video_range_concurrency, len(ranges)))
                ]
                try:
                    await asyncio.gather(*workers)
                finally:
                    for task in workers:
                        if not task.done():
                            task.cancel()
                    if workers:
                        await asyncio.gather(*workers, return_exceptions=True)

            prefix = bytearray()
            async with aiofiles.open(tmp_path, "wb") as output:
                for index, _ in enumerate(ranges):
                    part_path = range_dir / f"{index:08d}.part"
                    part_bytes = await asyncio.to_thread(part_path.read_bytes)
                    if len(prefix) < 32:
                        prefix.extend(part_bytes[: 32 - len(prefix)])
                    await output.write(part_bytes)
            _validate_video_payload(
                content_type=content_type,
                prefix=bytes(prefix),
                total_bytes=total_size,
            )
            return True
        finally:
            await asyncio.to_thread(shutil.rmtree, range_dir, True)

    async def _resolve_video_url(
        self,
        url: str,
        *,
        timeout: httpx.Timeout,
        policy: URLFetchPolicy,
    ) -> str:
        """Resolve a possibly indirect URL into a direct mp4 URL.

        Some providers return an HTML page or JSON wrapper that contains the real mp4 link.
        """
        u = str(url or "").strip()
        if not u:
            return ""
        if urlsplit(u).path.lower().endswith((".mp4", ".webm", ".mov")):
            return u

        current = u
        redirects = 0
        while True:
            await ensure_url_allowed(current, policy=policy)
            async with httpx.AsyncClient(
                timeout=timeout, follow_redirects=False
            ) as client:
                try:
                    async with client.stream(
                        "GET",
                        current,
                        headers={"Accept": "text/html,application/json,video/*"},
                    ) as resp:
                        if resp.status_code in {301, 302, 303, 307, 308}:
                            if redirects >= self._media_max_redirects:
                                raise RuntimeError("Too many redirects")
                            loc = (resp.headers.get("location") or "").strip()
                            if not loc:
                                raise RuntimeError("Redirect without location")
                            current = str(httpx.URL(current).join(loc))
                            redirects += 1
                            continue

                        if resp.status_code >= 400:
                            return current

                        ct = (resp.headers.get("content-type") or "").lower()
                        if ct.startswith("video/") or ct.startswith(
                            "application/octet-stream"
                        ):
                            return current

                        body = bytearray()
                        async for chunk in resp.aiter_bytes(chunk_size=64 * 1024):
                            body.extend(chunk)
                            if len(body) > 1024 * 1024:
                                return current
                except (httpx.HTTPError, OSError):
                    return current

            text = bytes(body).decode("utf-8", errors="replace")

            if "application/json" in ct:
                try:
                    data = json.loads(text)
                    if isinstance(data, dict):
                        candidates = [
                            data.get(k) for k in ("url", "video_url", "download_url")
                        ]
                        nested = data.get("data")
                        if (
                            isinstance(nested, list)
                            and nested
                            and isinstance(nested[0], dict)
                        ):
                            candidates.append(nested[0].get("url"))
                        for candidate in candidates:
                            value = str(candidate or "").strip()
                            resolved = (
                                str(httpx.URL(current).join(value)) if value else ""
                            )
                            if (
                                urlsplit(resolved)
                                .path.lower()
                                .endswith((".mp4", ".webm", ".mov"))
                            ):
                                return resolved
                except (TypeError, ValueError, json.JSONDecodeError):
                    pass

            if "text/html" in ct or text.lstrip().lower().startswith("<!doctype"):
                match = re.search(
                    r"https?://[^\s\"']+?\.(?:mp4|webm|mov)(?:\?[^\s\"']*)?",
                    text,
                    re.IGNORECASE,
                )
                if match:
                    return match.group(0)

            return current

    async def download_video(
        self,
        url: str,
        *,
        timeout_seconds: int = 300,
        headers: dict[str, str] | None = None,
    ) -> Path:
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
        original_url = str(url or "").strip()
        auth_headers = dict(headers or {})
        auth_origin = _normalized_origin(original_url)
        current = (
            original_url
            if auth_headers
            else await self._resolve_video_url(
                original_url,
                timeout=timeout,
                policy=policy,
            )
        )
        redirects = 0
        max_download_attempts = 3
        try:
            segmented = False
            if (
                self._video_range_download_enabled
                and auth_headers
                and _normalized_origin(current) == auth_origin
            ):
                segmented = await self._download_video_ranges(
                    current,
                    tmp_path=tmp_path,
                    timeout=timeout,
                    headers=auth_headers,
                    policy=policy,
                )

            if not segmented:
                for attempt in range(max_download_attempts):
                    try:
                        while True:
                            await ensure_url_allowed(current, policy=policy)
                            current_origin = _normalized_origin(current)
                            request_headers = dict(
                                auth_headers
                                if auth_headers and current_origin == auth_origin
                                else {}
                            )
                            resume_from = (
                                tmp_path.stat().st_size if tmp_path.exists() else 0
                            )
                            if resume_from:
                                request_headers["Range"] = f"bytes={resume_from}-"

                            async with httpx.AsyncClient(
                                timeout=timeout, follow_redirects=False
                            ) as client:
                                async with client.stream(
                                    "GET", current, headers=request_headers or None
                                ) as resp:
                                    if resp.status_code in {301, 302, 303, 307, 308}:
                                        if redirects >= self._media_max_redirects:
                                            raise RuntimeError("Too many redirects")
                                        loc = (
                                            resp.headers.get("location") or ""
                                        ).strip()
                                        if not loc:
                                            raise RuntimeError(
                                                "Redirect without location"
                                            )
                                        current = str(httpx.URL(current).join(loc))
                                        redirects += 1
                                        continue

                                    resp.raise_for_status()
                                    content_range = resp.headers.get(
                                        "content-range", ""
                                    )
                                    range_match = re.match(
                                        r"bytes\s+(\d+)-(\d+)/(\d+|\*)",
                                        content_range,
                                        re.IGNORECASE,
                                    )
                                    if resume_from and resp.status_code == 206:
                                        if (
                                            not range_match
                                            or int(range_match.group(1)) != resume_from
                                        ):
                                            raise RuntimeError(
                                                "Upstream returned an unexpected video byte range"
                                            )
                                        write_mode = "ab"
                                    else:
                                        # A server that ignores Range must restart from byte 0.
                                        resume_from = 0
                                        write_mode = "wb"

                                    total = resume_from
                                    prefix = bytearray()
                                    content_type = (
                                        resp.headers.get("content-type") or ""
                                    )
                                    content_length = _clamp_int(
                                        resp.headers.get("content-length"),
                                        default=0,
                                        min_value=0,
                                        max_value=self._media_max_video_bytes,
                                    )
                                    expected_total = (
                                        int(range_match.group(3))
                                        if range_match and range_match.group(3) != "*"
                                        else (
                                            resume_from + content_length
                                            if content_length
                                            else 0
                                        )
                                    )
                                    async with aiofiles.open(tmp_path, write_mode) as f:
                                        async for chunk in resp.aiter_bytes(
                                            chunk_size=1024 * 256
                                        ):
                                            if not chunk:
                                                continue
                                            total += len(chunk)
                                            if len(prefix) < 32:
                                                prefix.extend(chunk[: 32 - len(prefix)])
                                            if total > self._media_max_video_bytes:
                                                raise RuntimeError("Video too large")
                                            await f.write(chunk)
                                    if expected_total and total < expected_total:
                                        raise httpx.ReadError(
                                            "Upstream closed the video response before completion"
                                        )
                                    _validate_video_payload(
                                        content_type=content_type,
                                        prefix=bytes(prefix),
                                        total_bytes=total,
                                    )

                            break
                        break
                    except (
                        httpx.ReadError,
                        httpx.RemoteProtocolError,
                        httpx.ConnectError,
                        httpx.ReadTimeout,
                    ) as exc:
                        if attempt + 1 >= max_download_attempts:
                            raise
                        logger.warning(
                            "[VideoManager] 视频下载中断，将从已下载字节续传: attempt=%s/%s error=%r",
                            attempt + 1,
                            max_download_attempts,
                            exc,
                        )
                        await asyncio.sleep(min(2**attempt, 8))
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