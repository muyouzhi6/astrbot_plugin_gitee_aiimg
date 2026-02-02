from __future__ import annotations

import io
import re
import time
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from openai import AsyncOpenAI
from openai.types.images_response import ImagesResponse

from astrbot.api import logger

from .image_format import guess_image_mime_and_ext


def _looks_like_size(s: str) -> bool:
    return bool(re.fullmatch(r"\d{2,5}x\d{2,5}", (s or "").strip()))


def resolution_to_size(resolution: str) -> str | None:
    r = (resolution or "").strip().upper()
    if not r or r == "AUTO":
        return None
    if r in {"1K", "1024"}:
        return "1024x1024"
    if r in {"2K", "2048"}:
        return "2048x2048"
    if r in {"4K", "4096"}:
        return "4096x4096"
    if _looks_like_size(r.lower()):
        return r.lower()
    return None


def _bytes_to_upload_file(image_bytes: bytes, filename: str) -> io.BytesIO:
    bio = io.BytesIO(image_bytes)
    bio.name = filename
    return bio


def normalize_openai_compat_base_url(raw: str) -> str:
    """Normalize OpenAI-compatible base_url.

    Users may paste either:
    - https://api.x.ai
    - https://api.x.ai/v1
    - https://ai.gitee.com
    - https://ai.gitee.com/v1
    - https://proxy.example.com/openai/v1

    The OpenAI client expects base_url to include /v1 (unless the path already contains /v1/).
    """
    s = str(raw or "").strip()
    if not s:
        return ""
    s = s.rstrip("/")

    lower = s.lower()
    for suffix in (
        "/v1/chat/completions",
        "/chat/completions",
        "/v1/images",
        "/images",
    ):
        if lower.endswith(suffix):
            s = s[: -len(suffix)].rstrip("/")
            break

    # If user already included /v1 in the path anywhere (e.g. /openai/v1), keep as-is.
    if re.search(r"/v1($|/)", s):
        return s

    try:
        parts = urlsplit(s)
        if parts.scheme and parts.netloc:
            path = (parts.path or "").rstrip("/") + "/v1"
            return urlunsplit((parts.scheme, parts.netloc, path, "", "")).rstrip("/")
    except Exception:
        pass

    return f"{s}/v1"


def _build_collage(images: list[bytes]) -> bytes:
    """Combine multiple reference images into a single image for backends that only accept 1 input image.

    Uses Pillow if available; otherwise falls back to the first image.
    """
    if not images:
        return b""
    if len(images) == 1:
        return images[0]

    try:
        from PIL import Image as PILImage
    except Exception:
        return images[0]

    pil_images: list[PILImage.Image] = []
    for b in images:
        try:
            pil_images.append(PILImage.open(io.BytesIO(b)).convert("RGB"))
        except Exception:
            continue

    if not pil_images:
        return images[0]

    target_h = 768
    resized: list[PILImage.Image] = []
    for im in pil_images:
        w, h = im.size
        if h <= 0:
            continue
        new_w = max(1, int(w * (target_h / h)))
        resized.append(im.resize((new_w, target_h)))

    if not resized:
        return images[0]

    total_w = sum(im.size[0] for im in resized)
    canvas = PILImage.new("RGB", (total_w, target_h), color=(0, 0, 0))
    x = 0
    for im in resized:
        canvas.paste(im, (x, 0))
        x += im.size[0]

    out = io.BytesIO()
    canvas.save(out, format="JPEG", quality=90)
    return out.getvalue()


class OpenAICompatBackend:
    """OpenAI-compatible Images API backend (generate/edit)."""

    def __init__(
        self,
        *,
        imgr,
        base_url: str,
        api_keys: list[str],
        timeout: int = 120,
        max_retries: int = 2,
        default_model: str = "",
        default_size: str = "4096x4096",
        supports_edit: bool = True,
        extra_body: dict | None = None,
    ):
        self.imgr = imgr
        self.base_url = normalize_openai_compat_base_url(base_url)
        self.api_keys = [str(k).strip() for k in (api_keys or []) if str(k).strip()]
        self.timeout = int(timeout or 120)
        self.max_retries = int(max_retries or 2)
        self.default_model = str(default_model or "").strip()
        self.default_size = str(default_size or "4096x4096").strip()
        self.supports_edit = bool(supports_edit)
        self.extra_body = extra_body or {}

        self._key_index = 0
        self._clients: dict[str, AsyncOpenAI] = {}
        self._images_generate_disabled_until = 0.0
        self._images_edit_disabled_until = 0.0

    @staticmethod
    def _image_support_cooldown_seconds() -> int:
        # Some third-party gateways route different backends; a 404 may be transient.
        # Use a cooldown instead of permanent disable to avoid "worked once then never again".
        return 600

    def _is_generate_temporarily_disabled(self) -> bool:
        return time.time() < self._images_generate_disabled_until

    def _is_edit_temporarily_disabled(self) -> bool:
        return time.time() < self._images_edit_disabled_until

    def _disable_generate_temporarily(self) -> None:
        self._images_generate_disabled_until = (
            time.time() + self._image_support_cooldown_seconds()
        )

    def _disable_edit_temporarily(self) -> None:
        self._images_edit_disabled_until = (
            time.time() + self._image_support_cooldown_seconds()
        )

    @staticmethod
    def _try_get_image_size(path: Path) -> tuple[int, int] | None:
        try:
            from PIL import Image as PILImage
        except Exception:
            return None
        try:
            with PILImage.open(path) as im:
                return im.size
        except Exception:
            return None

    @staticmethod
    def _is_invalid_size_error(exc: Exception) -> bool:
        msg = str(exc).lower()
        if "size" not in msg:
            return False
        return any(
            k in msg
            for k in (
                "invalid",
                "unsupported",
                "not supported",
                "allowed",
                "must be",
                "not one of",
            )
        )

    async def close(self) -> None:
        for client in self._clients.values():
            try:
                await client.close()
            except Exception:
                pass
        self._clients.clear()

    def _next_key(self) -> str:
        if not self.api_keys:
            raise RuntimeError("未配置 API Key")
        key = self.api_keys[self._key_index]
        self._key_index = (self._key_index + 1) % len(self.api_keys)
        return key

    def _get_client(self, key: str) -> AsyncOpenAI:
        client = self._clients.get(key)
        if client is None:
            client = AsyncOpenAI(
                base_url=self.base_url,
                api_key=key,
                timeout=self.timeout,
                max_retries=self.max_retries,
            )
            self._clients[key] = client
        return client

    async def _recreate_client(self, key: str) -> AsyncOpenAI:
        old = self._clients.pop(key, None)
        if old is not None:
            try:
                await old.close()
            except Exception:
                pass
        client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=key,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )
        self._clients[key] = client
        return client

    async def _save_images_response(self, resp: ImagesResponse) -> Path:
        if not resp.data:
            raise RuntimeError("未返回图片数据")

        img = resp.data[0]
        if getattr(img, "url", None):
            return await self.imgr.download_image(img.url)
        if getattr(img, "b64_json", None):
            return await self.imgr.save_base64_image(img.b64_json)
        raise RuntimeError("返回数据不包含图片")

    async def generate(
        self,
        prompt: str,
        *,
        model: str | None = None,
        size: str | None = None,
        resolution: str | None = None,
        extra_body: dict | None = None,
    ) -> Path:
        key = self._next_key()
        client = self._get_client(key)

        final_model = str(model or self.default_model or "").strip()
        if not final_model:
            raise RuntimeError("未配置 model")

        final_size = (
            (size or "").strip()
            or resolution_to_size(resolution or "")  # may be None
            or self.default_size
        )

        kwargs: dict = {
            "model": final_model,
            "prompt": prompt,
            "size": final_size,
        }
        eb = {}
        eb.update(self.extra_body)
        eb.update(extra_body or {})
        if eb:
            kwargs["extra_body"] = eb

        t0 = time.time()
        try:
            if self._is_generate_temporarily_disabled():
                raise RuntimeError(
                    "该后端 images.generate 暂时不可用（此前返回 404，已进入冷却期）"
                )
            resp: ImagesResponse = await client.images.generate(**kwargs)
        except Exception as e:
            if "client has been closed" in str(e).lower():
                logger.warning(
                    "[OpenAICompat][generate] client 已关闭，重建 client 后重试一次"
                )
                client = await self._recreate_client(key)
                resp = await client.images.generate(**kwargs)
            elif final_size == "4096x4096" and self._is_invalid_size_error(e):
                logger.warning(
                    f"[OpenAICompat][generate] 4096x4096 可能不受该后端支持，尝试降级到 2048x2048: {e}"
                )
                kwargs["size"] = "2048x2048"
                resp = await client.images.generate(**kwargs)
            else:
                if "404" in str(e):
                    self._disable_generate_temporarily()
                    logger.error(
                        "[OpenAICompat][generate] 404 通常表示 base_url 填错或该服务不支持 Images API；"
                        "请确认 base_url 指向包含 /v1/images 的 OpenAI 兼容入口。"
                    )
                logger.error(
                    f"[OpenAICompat][generate] API 调用失败，base_url={self.base_url}，耗时: {time.time() - t0:.2f}s: {e}"
                )
                raise

        logger.info(f"[OpenAICompat][generate] API 响应耗时: {time.time() - t0:.2f}s")
        out_path = await self._save_images_response(resp)
        self._images_generate_disabled_until = 0.0
        if _looks_like_size(final_size):
            got = self._try_get_image_size(out_path)
            if got is not None and f"{got[0]}x{got[1]}" != final_size:
                logger.warning(
                    "[OpenAICompat][generate] 输出尺寸与请求不一致: requested=%s got=%sx%s (服务商可能忽略 size)",
                    final_size,
                    got[0],
                    got[1],
                )
        return out_path

    async def edit(
        self,
        prompt: str,
        images: list[bytes],
        *,
        model: str | None = None,
        size: str | None = None,
        resolution: str | None = None,
        extra_body: dict | None = None,
    ) -> Path:
        if not self.supports_edit:
            raise RuntimeError("该后端不支持改图/图生图")

        if not images:
            raise ValueError("至少需要一张图片")

        key = self._next_key()
        client = self._get_client(key)

        final_model = str(model or self.default_model or "").strip()
        if not final_model:
            raise RuntimeError("未配置 model")

        final_size = (
            (size or "").strip()
            or resolution_to_size(resolution or "")
            or self.default_size
        )

        # Some providers only accept a single input image for edits.
        packed = _build_collage(images) if len(images) > 1 else images[0]
        mime, ext = guess_image_mime_and_ext(packed)
        upload = _bytes_to_upload_file(packed, f"input.{ext}")

        kwargs: dict = {
            "model": final_model,
            "prompt": prompt,
            "image": upload,
            "size": final_size,
        }
        eb = {}
        eb.update(self.extra_body)
        eb.update(extra_body or {})
        if eb:
            kwargs["extra_body"] = eb

        t0 = time.time()
        try:
            if self._is_edit_temporarily_disabled():
                raise RuntimeError(
                    "该后端 images.edit 暂时不可用（此前返回 404，已进入冷却期）"
                )
            resp: ImagesResponse = await client.images.edit(**kwargs)
        except Exception as e:
            if "client has been closed" in str(e).lower():
                logger.warning(
                    "[OpenAICompat][edit] client 已关闭，重建 client 后重试一次"
                )
                client = await self._recreate_client(key)
                resp = await client.images.edit(**kwargs)
            elif final_size == "4096x4096" and self._is_invalid_size_error(e):
                logger.warning(
                    f"[OpenAICompat][edit] 4096x4096 可能不受该后端支持，尝试降级到 2048x2048: {e}"
                )
                kwargs["size"] = "2048x2048"
                resp = await client.images.edit(**kwargs)
            else:
                if "404" in str(e):
                    self._disable_edit_temporarily()
                    logger.error(
                        "[OpenAICompat][edit] 404 通常表示 base_url 填错或该服务不支持 images.edit；"
                        "请确认 base_url 指向包含 /v1/images 的 OpenAI 兼容入口，并且该服务支持改图。"
                    )
                logger.error(
                    f"[OpenAICompat][edit] API 调用失败，base_url={self.base_url}，耗时: {time.time() - t0:.2f}s: {e}"
                )
                raise

        logger.info(f"[OpenAICompat][edit] API 响应耗时: {time.time() - t0:.2f}s")
        out_path = await self._save_images_response(resp)
        self._images_edit_disabled_until = 0.0
        if _looks_like_size(final_size):
            got = self._try_get_image_size(out_path)
            if got is not None and f"{got[0]}x{got[1]}" != final_size:
                logger.warning(
                    "[OpenAICompat][edit] 输出尺寸与请求不一致: requested=%s got=%sx%s (服务商可能忽略 size)",
                    final_size,
                    got[0],
                    got[1],
                )
        return out_path
