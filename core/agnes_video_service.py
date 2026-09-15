"""Agnes Video's JSON creation and video_id-based asynchronous retrieval."""

from __future__ import annotations

import asyncio
import base64
import copy
import hashlib
import sqlite3
import time
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from astrbot.api import logger

from .image_format import guess_image_mime_and_ext
from .sora2_video_service import _parse_retry_after_seconds


class AgnesVideoError(RuntimeError):
    def __init__(self, message: str, *, stop_provider_chain: bool = False):
        super().__init__(message)
        self.stop_provider_chain = stop_provider_chain


class AgnesVideoService:
    """Use one durable rate budget for creation and polling, including reloads."""

    def __init__(self, *, settings: dict, data_dir: Path):
        self.settings = copy.deepcopy(settings)
        base = str(settings.get("base_url") or "https://apihub.agnes-ai.com/v1").rstrip(
            "/"
        )
        parts = urlsplit(base)
        if (
            parts.scheme not in {"http", "https"}
            or not parts.netloc
            or parts.username
            or parts.password
            or parts.query
            or parts.fragment
        ):
            raise ValueError("Agnes 接口地址无效")
        base = base.removesuffix("/videos")
        if not base.endswith("/v1"):
            base += "/v1"
        self.api_url = base + "/videos"
        self.status_url = base[:-3] + "/agnesapi"
        self.api_key = str(settings.get("api_key") or "").strip()
        self.interval = max(61.0, float(settings.get("request_interval_seconds", 61)))
        self.timeout = max(120, min(3600, int(settings.get("timeout_seconds", 1800))))
        self.request_timeout = max(
            10, min(300, int(settings.get("request_timeout_seconds", 120)))
        )
        self.poll_interval = max(1, float(settings.get("poll_interval_seconds", 61)))
        self.poll_retries = max(0, min(5, int(settings.get("max_retries", 3))))
        self.rate_path = Path(data_dir) / "agnes_video" / "requests.sqlite3"
        self.rate_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.rate_path) as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS rate_limit ("
                "identity TEXT PRIMARY KEY, next_at REAL NOT NULL)"
            )
        self.rate_identity = hashlib.sha256(
            (parts.netloc + "\0" + self.api_key).encode()
        ).hexdigest()

    def _reserve(self) -> float:
        with sqlite3.connect(self.rate_path, timeout=5) as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT next_at FROM rate_limit WHERE identity=?", (self.rate_identity,)
            ).fetchone()
            now = time.time()
            wait = max(0.0, row[0] - now) if row else 0.0
            if not wait:
                db.execute(
                    "INSERT OR REPLACE INTO rate_limit VALUES (?,?)",
                    (self.rate_identity, now + self.interval),
                )
            return wait

    async def _wait_request(self, deadline: float) -> None:
        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError("Agnes 请求等待超时")
            wait = await asyncio.to_thread(self._reserve)
            if not wait:
                return
            if time.monotonic() + wait >= deadline:
                raise TimeoutError("Agnes 请求等待超时")
            await asyncio.sleep(wait)

    def _error_detail(self, data) -> str:
        detail = (
            data.get("error")
            or data.get("detail")
            or data.get("message")
            or "未提供错误详情"
        )
        if isinstance(detail, dict):
            detail = detail.get("message") or detail.get("code") or "上游任务失败"
        text = str(detail)
        if self.api_key:
            text = text.replace(self.api_key, "[redacted]")
        return text[:300]

    def build_payload(self, prompt: str, image_bytes: bytes | None = None) -> dict:
        extra = self.settings.get("extra_body") or {}
        if not isinstance(extra, dict):
            raise ValueError("Agnes 额外请求体必须为 JSON 对象")
        payload = {
            "model": self.settings.get("model") or "agnes-video-2.5-flash",
            "seconds": str(self.settings.get("seconds", "5")),
            "size": self.settings.get("size") or "720P",
            "aspect_ratio": self.settings.get("aspect_ratio") or "16:9",
            "n": 1,
            **copy.deepcopy(extra),
        }
        payload["prompt"] = str(prompt or "").strip()
        if not payload["prompt"]:
            raise ValueError("请填写视频提示词")
        if payload["model"] != "agnes-video-2.5-flash":
            raise ValueError("Agnes 视频模型不受支持")
        mode = payload.get("mode") or self.settings.get("mode") or "auto"
        if mode == "auto":
            mode = (
                "keyframe"
                if image_bytes
                or payload.get("first_frame")
                or payload.get("last_frame")
                else "reference"
                if payload.get("images") or payload.get("audios")
                else "text"
            )
        payload["mode"] = mode
        if image_bytes:
            if len(image_bytes) >= 15 * 1024 * 1024:
                raise ValueError("Agnes 图片需小于 15 MB")
            mime, _ = guess_image_mime_and_ext(image_bytes)
            reference = (
                "data:"
                + mime
                + ";base64,"
                + base64.b64encode(image_bytes).decode("ascii")
            )
            if mode == "keyframe":
                if payload.get("first_frame"):
                    raise ValueError("消息图片与已配置首帧冲突, 请移除固定首帧")
                payload["first_frame"] = reference
            elif mode == "reference":
                images = payload.get("images", [])
                if not isinstance(images, list):
                    raise ValueError("Agnes images 必须为列表")
                payload["images"] = [reference, *images]
            else:
                raise ValueError("收到图片时不能使用纯文本模式")
        self.validate_payload(payload)
        return payload

    @staticmethod
    def validate_payload(payload: dict) -> None:
        if payload["size"] != "720P":
            raise ValueError("Agnes Flash 仅支持 720P")
        if payload["seconds"] not in {str(i) for i in range(4, 13)}:
            raise ValueError("Agnes seconds 必须为字符串 4 至 12")
        if type(payload["n"]) is not int or payload["n"] != 1:
            raise ValueError("Agnes 每个任务仅支持 n=1")
        if payload["aspect_ratio"] not in {"21:9", "16:9", "4:3", "1:1", "3:4", "9:16"}:
            raise ValueError("Agnes 画幅不受支持")
        if "seed" in payload and type(payload["seed"]) is not int:
            raise ValueError("Agnes seed 必须为整数")
        mode = payload["mode"]
        allowed = {
            "text": set(),
            "keyframe": {"first_frame", "last_frame"},
            "reference": {"images", "audios"},
        }
        if mode not in allowed:
            raise ValueError("Agnes mode 需为 text, keyframe 或 reference")
        media = {
            k
            for k in ("first_frame", "last_frame", "images", "audios", "videos")
            if payload.get(k)
        }
        if media - allowed[mode] or (mode != "text" and not media):
            raise ValueError("Agnes 媒体与生成模式不匹配, Flash 不接受视频参考")
        for name, limit in (("images", 5), ("audios", 3)):
            if name in payload and (
                not isinstance(payload[name], list) or len(payload[name]) > limit
            ):
                raise ValueError(f"Agnes {name} 必须为列表且最多 {limit} 项")
        urls = [payload[k] for k in ("first_frame", "last_frame") if payload.get(k)]
        urls += payload.get("images", []) + payload.get("audios", [])
        if any(
            not isinstance(u, str)
            or not u.startswith(("https://", "http://", "data:image/"))
            for u in urls
        ):
            raise ValueError("Agnes 媒体需为有效 URL")

    async def _request(self, client, method, url, *, deadline, **kwargs):
        await self._wait_request(deadline)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Agnes 请求超时")
        response = await asyncio.wait_for(
            client.request(
                method,
                url,
                headers={"Authorization": "Bearer " + self.api_key},
                **kwargs,
            ),
            timeout=remaining,
        )
        if len(response.content) > 2 * 1024 * 1024:
            raise AgnesVideoError("Agnes 响应过大", stop_provider_chain=True)
        try:
            data = response.json()
        except ValueError:
            data = {}
        if not isinstance(data, dict):
            data = {}
        if not 200 <= response.status_code < 300:
            raise httpx.HTTPStatusError(
                f"Agnes HTTP {response.status_code}: {self._error_detail(data)}",
                request=response.request,
                response=response,
            )
        if not data:
            raise AgnesVideoError("Agnes 返回无效任务响应", stop_provider_chain=True)
        return data

    async def poll(self, client, video_id: str, model: str, *, deadline: float):
        failures = 0
        while True:
            await asyncio.sleep(
                min(self.poll_interval, max(0, deadline - time.monotonic()))
            )
            try:
                data = await self._request(
                    client,
                    "GET",
                    self.status_url,
                    deadline=deadline,
                    params={"video_id": video_id, "model_name": model},
                )
            except (httpx.TransportError, httpx.HTTPStatusError) as exc:
                status = (
                    exc.response.status_code
                    if isinstance(exc, httpx.HTTPStatusError)
                    else 0
                )
                if (
                    status
                    and status not in {408, 429, 500, 502, 503, 504}
                    or failures >= self.poll_retries
                ):
                    raise AgnesVideoError(
                        f"Agnes 查询失败, video_id={video_id}; 不会重新创建任务",
                        stop_provider_chain=True,
                    ) from exc
                failures += 1
                delay = (
                    _parse_retry_after_seconds(exc.response.headers.get("retry-after"))
                    if isinstance(exc, httpx.HTTPStatusError)
                    else None
                )
                if delay and time.monotonic() + delay < deadline:
                    await asyncio.sleep(delay)
                continue
            failures = 0
            status = data.get("status")
            logger.info(
                "[AgnesVideo] video_id=%s status=%s progress=%s",
                video_id,
                status,
                data.get("progress"),
            )
            if status == "failed":
                raise AgnesVideoError(f"Agnes 任务失败: {self._error_detail(data)}")
            if status == "completed":
                metadata = data.get("metadata")
                url = (
                    metadata.get("url") if isinstance(metadata, dict) else None
                ) or data.get("url")
                if not isinstance(url, str) or not url.startswith(
                    ("https://", "http://")
                ):
                    raise AgnesVideoError(
                        f"Agnes 已完成但未返回视频地址, video_id={video_id}",
                        stop_provider_chain=True,
                    )
                return url

    async def generate_video_url(
        self,
        prompt: str,
        image_bytes: bytes | None = None,
        *,
        preset: str | None = None,
    ) -> str:
        payload = self.build_payload(prompt, image_bytes)
        if not self.api_key:
            raise ValueError("请配置 Agnes API Key")
        deadline = time.monotonic() + self.timeout
        async with httpx.AsyncClient(
            timeout=self.request_timeout, follow_redirects=False
        ) as client:
            try:
                data = await self._request(
                    client, "POST", self.api_url, deadline=deadline, json=payload
                )
            except httpx.HTTPStatusError as exc:
                raise AgnesVideoError(
                    str(exc),
                    stop_provider_chain=(
                        exc.response.status_code >= 500
                        or exc.response.status_code == 408
                    ),
                ) from exc
            except (httpx.TransportError, TimeoutError, asyncio.TimeoutError) as exc:
                raise AgnesVideoError(
                    "Agnes 创建请求结果不明, 不会自动重建任务, 请先到平台检查",
                    stop_provider_chain=True,
                ) from exc
            video_id = data.get("video_id")
            if not isinstance(video_id, str) or not video_id.strip():
                raise AgnesVideoError(
                    "Agnes 创建响应缺少 video_id, 不会使用 task_id 重建任务",
                    stop_provider_chain=True,
                )
            logger.info(
                "[AgnesVideo] created video_id=%s model=%s", video_id, payload["model"]
            )
            try:
                return await self.poll(
                    client, video_id, payload["model"], deadline=deadline
                )
            except (TimeoutError, asyncio.TimeoutError) as exc:
                raise AgnesVideoError(
                    f"Agnes 等待超时, video_id={video_id}; 请到平台查询, "
                    "不会重新创建任务",
                    stop_provider_chain=True,
                ) from exc
