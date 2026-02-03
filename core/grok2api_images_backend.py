"""
Grok2API Images 后端（/v1/images/generations）

根据你贴的 Grok2API 文档：
- POST /v1/images/generations：图像接口，支持图像生成、图像编辑

很多 Grok2API 部署会在 chat.completions + grok-imagine-0.9 的“带图输入”场景优先输出 video(mp4)。
为避免混淆，本后端强制走 images 接口生成/编辑图片。
"""

from __future__ import annotations

import base64
import time
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlsplit

import httpx

from astrbot.api import logger

from .image_format import guess_image_mime_and_ext
from .openai_compat_backend import (
    _build_collage,
    normalize_openai_compat_base_url,
    resolution_to_size,
)


def _origin(url: str) -> str:
    try:
        u = urlsplit(url)
        if u.scheme and u.netloc:
            return f"{u.scheme}://{u.netloc}"
    except Exception:
        pass
    return ""


def _normalize_images_generations_url(base_url: str) -> str:
    # normalize_openai_compat_base_url ensures it contains /v1
    b = normalize_openai_compat_base_url(base_url).rstrip("/")
    if not b:
        return ""
    return f"{b}/images/generations"


def _pick_first_api_key(api_keys: list[str]) -> str:
    keys = [str(k).strip() for k in (api_keys or []) if str(k).strip()]
    if not keys:
        raise RuntimeError("未配置 API Key")
    return keys[0]


def _extract_image_ref(data: Any) -> str | None:
    # OpenAI-like images response: {"data":[{"url":"..." }]} or {"data":[{"b64_json":"..."}]}
    if not isinstance(data, dict):
        return None
    items = data.get("data")
    if not isinstance(items, list) or not items:
        return None
    item0 = items[0]
    if not isinstance(item0, dict):
        return None
    url = item0.get("url")
    if isinstance(url, str) and url.strip():
        return url.strip()
    b64 = item0.get("b64_json")
    if isinstance(b64, str) and b64.strip():
        return f"data:image/png;base64,{b64.strip()}"
    return None


def _looks_like_video_url(url: str) -> bool:
    u = (url or "").strip().lower()
    if not u:
        return False
    if any(ext in u for ext in (".mp4", ".webm", ".mov")):
        return True
    if "generated_video" in u:
        return True
    return False


class Grok2ApiImagesBackend:
    """Grok2API 的 /v1/images/generations 后端（generate + edit）。"""

    def __init__(
        self,
        *,
        imgr,
        base_url: str,
        api_keys: list[str],
        timeout: int = 120,
        default_model: str = "",
        default_size: str = "4096x4096",
        extra_body: dict | None = None,
    ):
        self.imgr = imgr
        self.base_url = str(base_url or "").strip()
        self.api_key = _pick_first_api_key(api_keys)
        self.timeout = max(1, min(int(timeout or 120), 3600))
        self.default_model = str(default_model or "").strip()
        self.default_size = str(default_size or "4096x4096").strip()
        self.extra_body = extra_body or {}

        self._endpoint = _normalize_images_generations_url(self.base_url)
        self._origin = _origin(self._endpoint)

    async def close(self) -> None:
        return None

    def _headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def _merge_extra(self, payload: dict) -> dict:
        eb = self.extra_body if isinstance(self.extra_body, dict) else {}
        if eb:
            # Shallow merge; user can override defaults if needed.
            out = dict(payload)
            out.update(eb)
            return out
        return payload

    @staticmethod
    def _coerce_form_value(v: Any) -> str:
        if v is None:
            return ""
        if isinstance(v, (str, int, float, bool)):
            return str(v)
        # For dict/list, stringify to JSON-ish representation to avoid multipart failure.
        try:
            import json

            return json.dumps(v, ensure_ascii=False)
        except Exception:
            return str(v)

    async def generate(
        self,
        prompt: str,
        *,
        model: str | None = None,
        size: str | None = None,
        resolution: str | None = None,
        extra_body: dict | None = None,
    ) -> Path:
        if not self._endpoint:
            raise RuntimeError("未配置 base_url")

        final_model = str(model or self.default_model or "").strip()
        if not final_model:
            raise RuntimeError("未配置 model")

        final_size = (
            str(size or "").strip()
            or (resolution_to_size(str(resolution or "")) or "").strip()
            or str(resolution or "").strip()
            or self.default_size
        )

        payload: dict[str, Any] = {
            "model": final_model,
            "prompt": (prompt or "").strip() or "a high quality image",
            "n": 1,
        }
        # Grok2API 文档未明确 size 字段，但很多兼容实现支持；传了也不会影响不支持的实现（通常会忽略/报错）
        if final_size:
            payload["size"] = final_size

        payload = self._merge_extra(payload)
        if isinstance(extra_body, dict) and extra_body:
            payload.update(extra_body)

        t0 = time.perf_counter()
        async with httpx.AsyncClient(
            timeout=float(self.timeout), follow_redirects=True
        ) as client:
            resp = await client.post(
                self._endpoint, headers=self._headers(), json=payload
            )
        if resp.status_code != 200:
            raise RuntimeError(
                f"Grok2API images.generate 失败 HTTP {resp.status_code}: {resp.text[:300]}"
            )

        data = resp.json()
        ref = _extract_image_ref(data)
        if ref and _looks_like_video_url(ref):
            raise RuntimeError(f"Grok2API images.generate 返回了视频而不是图片: {ref}")
        if not ref:
            raise RuntimeError(
                f"Grok2API images.generate 未返回图片: {str(data)[:200]}"
            )

        logger.info("[Grok2APIImages][generate] 耗时: %.2fs", time.perf_counter() - t0)
        return await self._save_ref(ref)

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
        if not images:
            raise ValueError("至少需要一张图片")
        if not self._endpoint:
            raise RuntimeError("未配置 base_url")

        final_model = str(model or self.default_model or "").strip()
        if not final_model:
            raise RuntimeError("未配置 model")

        final_size = (
            str(size or "").strip()
            or (resolution_to_size(str(resolution or "")) or "").strip()
            or str(resolution or "").strip()
            or self.default_size
        )

        # Grok2API 文档说明 /v1/images/generations 同时支持“图像生成/编辑”，但不同实现对编辑入参并不一致：
        # - 有的实现接受 JSON（image=data:... 或 images=[data:...] 等）
        # - 有的实现沿用 OpenAI 官方图片编辑接口，要求 multipart 上传文件
        # 因此这里做“多形态兜底”：JSON 多种字段尝试 -> multipart 兜底。
        merged_img = _build_collage(images)
        mime, ext = guess_image_mime_and_ext(merged_img)

        t0 = time.perf_counter()
        async with httpx.AsyncClient(
            timeout=float(self.timeout), follow_redirects=True
        ) as client:
            image_b64 = base64.b64encode(merged_img).decode("utf-8")
            image_data_url = f"data:{mime};base64,{image_b64}"

            base_payload: dict[str, Any] = {
                "model": final_model,
                "prompt": (prompt or "").strip() or "Edit this image",
                "n": 1,
            }
            if final_size:
                base_payload["size"] = final_size

            base_payload = self._merge_extra(base_payload)
            if isinstance(extra_body, dict) and extra_body:
                base_payload.update(extra_body)

            last_resp: httpx.Response | None = None

            # 1) JSON variants
            json_payloads: list[dict[str, Any]] = [
                dict(base_payload, image=image_data_url),
                dict(base_payload, images=[image_data_url]),
                dict(base_payload, image_url=image_data_url),
            ]
            for p in json_payloads:
                resp = await client.post(
                    self._endpoint, headers=self._headers(), json=p
                )
                last_resp = resp
                if resp.status_code == 200:
                    break
                # 仅在“参数/格式”类错误下继续尝试；其余错误直接终止，留给外层降级。
                if resp.status_code not in {400, 415, 422}:
                    break

            resp = last_resp or resp

            # 2) multipart fallback (image / images)
            if resp.status_code in {400, 415, 422}:
                data_fields: dict[str, str] = {
                    "model": final_model,
                    "prompt": (prompt or "").strip() or "Edit this image",
                    "n": "1",
                }
                if final_size:
                    data_fields["size"] = final_size

                eb = self.extra_body if isinstance(self.extra_body, dict) else {}
                for k, v in eb.items():
                    if k not in data_fields:
                        data_fields[str(k)] = self._coerce_form_value(v)
                if isinstance(extra_body, dict) and extra_body:
                    for k, v in extra_body.items():
                        data_fields[str(k)] = self._coerce_form_value(v)

                headers = {"Authorization": f"Bearer {self.api_key}"}
                for field_name in ("image", "images"):
                    files = {
                        field_name: (f"image.{ext}", merged_img, mime),
                    }
                    resp = await client.post(
                        self._endpoint, headers=headers, data=data_fields, files=files
                    )
                    if resp.status_code == 200:
                        break
        if resp.status_code != 200:
            raise RuntimeError(
                f"Grok2API images.edit 失败 HTTP {resp.status_code}: {resp.text[:300]}"
            )

        data = resp.json()
        ref = _extract_image_ref(data)
        if ref and _looks_like_video_url(ref):
            raise RuntimeError(f"Grok2API images.edit 返回了视频而不是图片: {ref}")
        if not ref:
            raise RuntimeError(f"Grok2API images.edit 未返回图片: {str(data)[:200]}")

        logger.info("[Grok2APIImages][edit] 耗时: %.2fs", time.perf_counter() - t0)
        return await self._save_ref(ref)

    async def _save_ref(self, ref: str) -> Path:
        ref = (ref or "").strip()
        if not ref:
            raise RuntimeError("空图片引用")

        if ref.startswith("data:image/"):
            try:
                _header, b64_data = ref.split(",", 1)
            except ValueError:
                raise RuntimeError("data:image 缺少 base64 数据") from None
            image_bytes = base64.b64decode((b64_data or "").strip())
            return await self.imgr.save_image(image_bytes)

        if ref.startswith(("http://", "https://")):
            return await self.imgr.download_image(ref)

        # Relative URL like "/images/xxx.png"
        if self._origin and ref.startswith("/"):
            return await self.imgr.download_image(
                urljoin(self._origin + "/", ref.lstrip("/"))
            )

        # Other relative forms
        if self._origin:
            return await self.imgr.download_image(urljoin(self._origin + "/", ref))

        raise RuntimeError(f"不支持的图片 URL: {ref}")
