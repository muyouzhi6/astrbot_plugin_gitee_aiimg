"""
Grok2API Images 鍚庣锛?v1/images/generations锛?
鏍规嵁浣犺创鐨?Grok2API 鏂囨。锛?- POST /v1/images/generations锛氬浘鍍忔帴鍙ｏ紝鏀寔鍥惧儚鐢熸垚銆佸浘鍍忕紪杈?
寰堝 Grok2API 閮ㄧ讲浼氬湪 chat.completions + grok-imagine-0.9 鐨勨€滃甫鍥捐緭鍏モ€濆満鏅紭鍏堣緭鍑?video(mp4)銆?涓洪伩鍏嶆贩娣嗭紝鏈悗绔己鍒惰蛋 images 鎺ュ彛鐢熸垚/缂栬緫鍥剧墖銆?"""

from __future__ import annotations

import base64
import json
import re
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


def _normalize_images_edits_url(base_url: str) -> str:
    b = normalize_openai_compat_base_url(base_url).rstrip("/")
    if not b:
        return ""
    return f"{b}/images/edits"


def _pick_first_api_key(api_keys: list[str]) -> str:
    keys = [str(k).strip() for k in (api_keys or []) if str(k).strip()]
    if not keys:
        raise RuntimeError("鏈厤缃?API Key")
    return keys[0]


_MD_IMAGE_RE = re.compile(r"!\[.*?\]\((.*?)\)")
_DATA_IMAGE_RE = re.compile(r"(data:image/[^\s)]+)")
_JSON_URL_FIELD_RE = re.compile(
    r'"(?:image_url|imageUrl|url|image|src|uri|link|href|fifeUrl|fife_url|final_image_url|origin_image_url)"\s*:\s*"([^"]+)"'
)


def _strip_markdown_target(target: str) -> str | None:
    s = (target or "").strip()
    if not s:
        return None
    if s.startswith("<") and ">" in s:
        right = s.find(">")
        if right > 1:
            s = s[1:right].strip()
    m = re.match(r'^(?P<url>\S+)(?:\s+(?:"[^"]*"|\'[^\']*\'))?\s*$', s)
    if m:
        s = m.group("url")
    s = s.strip().strip('"').strip("'")
    return s or None


def _decode_base64_bytes(text: str) -> bytes:
    s = re.sub(r"\s+", "", str(text or "").strip())
    if not s:
        return b""
    candidates = [s, s.replace("-", "+").replace("_", "/")]
    for cand in candidates:
        pad = "=" * ((4 - len(cand) % 4) % 4)
        try:
            raw = base64.b64decode(cand + pad, validate=False)
            if raw:
                return raw
        except Exception:
            continue
    try:
        raw = base64.urlsafe_b64decode(s + ("=" * ((4 - len(s) % 4) % 4)))
        if raw:
            return raw
    except Exception:
        pass
    return b""


def _is_valid_data_image_ref(ref: str) -> bool:
    s = str(ref or "").strip()
    if not s.startswith("data:image/"):
        return False
    if "," not in s:
        return False
    _header, b64 = s.split(",", 1)
    b64 = re.sub(r"\s+", "", (b64 or "").strip())
    if len(b64) < 64:
        return False
    return bool(re.fullmatch(r"[A-Za-z0-9+/=_-]+", b64[:2048]))


def _extract_ref_from_text(text: str) -> str | None:
    s = (text or "").strip()
    if not s:
        return None
    if s.startswith("data:image/"):
        compact = re.sub(r"\s+", "", s)
        if _is_valid_data_image_ref(compact):
            return compact

    m = _MD_IMAGE_RE.search(s)
    if m:
        cand = _strip_markdown_target(m.group(1))
        if cand:
            if cand.startswith("data:image/"):
                cand = re.sub(r"\s+", "", cand)
                if _is_valid_data_image_ref(cand):
                    return cand
            if cand.startswith(("http://", "https://", "/")):
                return cand

    for m in _DATA_IMAGE_RE.finditer(s):
        cand = re.sub(r"\s+", "", m.group(1).strip())
        if _is_valid_data_image_ref(cand):
            return cand

    for m in _JSON_URL_FIELD_RE.finditer(s):
        cand = (m.group(1) or "").strip().replace("\\/", "/")
        cand = _strip_markdown_target(cand) or cand
        if cand.startswith("data:image/"):
            cand = re.sub(r"\s+", "", cand)
            if _is_valid_data_image_ref(cand):
                return cand
        if cand.startswith(("http://", "https://", "/")):
            return cand

    if s.startswith(("http://", "https://", "/")):
        return s

    if (s.startswith("{") and s.endswith("}")) or (
        s.startswith("[") and s.endswith("]")
    ):
        try:
            parsed = json.loads(s)
        except Exception:
            parsed = None
        if parsed is not None:
            ref = _extract_image_ref(parsed)
            if ref:
                return ref
    return None


def _extract_image_ref(data: Any) -> str | None:
    # OpenAI-like images response: {"data":[{"url":"..." }]} or {"data":[{"b64_json":"..."}]}
    if isinstance(data, dict):
        items = data.get("data")
        if isinstance(items, list):
            for item in items:
                ref = _extract_image_ref(item)
                if ref:
                    return ref

        b64 = data.get("b64_json")
        if isinstance(b64, str) and b64.strip():
            return f"data:image/png;base64,{b64.strip()}"

        for key in (
            "url",
            "image_url",
            "image",
            "data",
            "src",
            "uri",
            "link",
            "href",
            "final_image_url",
            "origin_image_url",
            "fifeUrl",
            "fife_url",
            "thumbnail",
        ):
            value = data.get(key)
            if isinstance(value, str):
                ref = _extract_ref_from_text(value)
                if ref:
                    return ref
            ref = _extract_image_ref(value)
            if ref:
                return ref

        for key in (
            "images",
            "image_urls",
            "attachments",
            "media",
            "result",
            "response",
        ):
            ref = _extract_image_ref(data.get(key))
            if ref:
                return ref
        return None
    if isinstance(data, list):
        for item in data:
            ref = _extract_image_ref(item)
            if ref:
                return ref
        return None
    if isinstance(data, str):
        return _extract_ref_from_text(data)
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
    """Grok2API images backend (generate + edit)."""

    def __init__(
        self,
        *,
        imgr,
        base_url: str,
        api_keys: list[str],
        timeout: int = 120,
        default_model: str = "",
        default_size: str = "1024x1024",
        extra_body: dict | None = None,
    ):
        self.imgr = imgr
        self.base_url = str(base_url or "").strip()
        self.api_key = _pick_first_api_key(api_keys)
        self.timeout = max(1, min(int(timeout or 120), 3600))
        self.default_model = str(default_model or "").strip()
        self.default_size = str(default_size or "4096x4096").strip()
        self.extra_body = extra_body or {}

        self._endpoint_generate = _normalize_images_generations_url(self.base_url)
        self._endpoint_edit = _normalize_images_edits_url(self.base_url)
        self._origin = _origin(self._endpoint_generate or self._endpoint_edit)

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
        if not self._endpoint_generate:
            raise RuntimeError("鏈厤缃?base_url")

        final_model = str(model or self.default_model or "").strip()
        if not final_model:
            raise RuntimeError("鏈厤缃?model")

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
                self._endpoint_generate, headers=self._headers(), json=payload
            )
        if resp.status_code != 200:
            raise RuntimeError(
                f"Grok2API images.generate 澶辫触 HTTP {resp.status_code}: {resp.text[:300]}"
            )

        data = resp.json()
        ref = _extract_image_ref(data)
        if ref and _looks_like_video_url(ref):
            raise RuntimeError(
                f"Grok2API images.generate 杩斿洖浜嗚棰戣€屼笉鏄浘鐗? {ref}"
            )
        if not ref:
            raise RuntimeError(
                f"Grok2API images.generate 鏈繑鍥炲浘鐗? {str(data)[:200]}"
            )

        logger.info(
            "[Grok2APIImages][generate] 鑰楁椂: %.2fs", time.perf_counter() - t0
        )
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
            raise ValueError("At least one image is required")
        if not self._endpoint_edit:
            raise RuntimeError("鏈厤缃?base_url")

        final_model = str(model or self.default_model or "").strip()
        if not final_model:
            raise RuntimeError("鏈厤缃?model")

        final_size = (
            str(size or "").strip()
            or (resolution_to_size(str(resolution or "")) or "").strip()
            or str(resolution or "").strip()
            or self.default_size
        )

        # Grok2API 鏂囨。璇存槑 /v1/images/generations 鍚屾椂鏀寔鈥滃浘鍍忕敓鎴?缂栬緫鈥濓紝浣嗕笉鍚屽疄鐜板缂栬緫鍏ュ弬骞朵笉涓€鑷达細
        # - 鏈夌殑瀹炵幇鎺ュ彈 JSON锛坕mage=data:... 鎴?images=[data:...] 绛夛級
        # - 鏈夌殑瀹炵幇娌跨敤 OpenAI 瀹樻柟鍥剧墖缂栬緫鎺ュ彛锛岃姹?multipart 涓婁紶鏂囦欢
        # 鍥犳杩欓噷鍋氣€滃褰㈡€佸厹搴曗€濓細JSON 澶氱瀛楁灏濊瘯 -> multipart 鍏滃簳銆?        merged_img = _build_collage(images)
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

            # Prefer multipart first: /v1/images/edits (newer Grok2API) only supports multipart.
            resp: httpx.Response | None = None

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
                for endpoint in (self._endpoint_edit, self._endpoint_generate):
                    if not endpoint:
                        continue
                    resp = await client.post(
                        endpoint, headers=headers, data=data_fields, files=files
                    )
                    if resp.status_code == 200:
                        break
                if resp is not None and resp.status_code == 200:
                    break

            # Fallback for older forks that (incorrectly) accept JSON edit payloads.
            if resp is None or resp.status_code in {400, 415, 422}:
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

                # 1) JSON variants (try generations only)
                if self._endpoint_generate:
                    json_payloads: list[dict[str, Any]] = [
                        dict(base_payload, image=image_data_url),
                        dict(base_payload, images=[image_data_url]),
                        dict(base_payload, image_url=image_data_url),
                    ]
                    for p in json_payloads:
                        resp = await client.post(
                            self._endpoint_generate,
                            headers=self._headers(),
                            json=p,
                        )
                        if resp.status_code == 200:
                            break
                        if resp.status_code not in {400, 415, 422}:
                            break
        if resp is None or resp.status_code != 200:
            status = resp.status_code if resp is not None else 0
            text = resp.text[:300] if resp is not None else "no response"
            raise RuntimeError(f"Grok2API images.edit 澶辫触 HTTP {status}: {text}")

        data = resp.json()
        ref = _extract_image_ref(data)
        if ref and _looks_like_video_url(ref):
            raise RuntimeError(
                f"Grok2API images.edit 杩斿洖浜嗚棰戣€屼笉鏄浘鐗? {ref}"
            )
        if not ref:
            raise RuntimeError(f"Grok2API images.edit 鏈繑鍥炲浘鐗? {str(data)[:200]}")

        logger.info("[Grok2APIImages][edit] 鑰楁椂: %.2fs", time.perf_counter() - t0)
        return await self._save_ref(ref)

    async def _save_ref(self, ref: str) -> Path:
        ref = (ref or "").strip()
        if not ref:
            raise RuntimeError("Empty image reference")

        if ref.startswith("data:image/"):
            ref = re.sub(r"\s+", "", ref)
            try:
                _header, b64_data = ref.split(",", 1)
            except ValueError:
                raise RuntimeError("data:image 缂哄皯 base64 鏁版嵁") from None
            image_bytes = _decode_base64_bytes((b64_data or "").strip())
            if not image_bytes:
                raise RuntimeError("data:image base64 瑙ｇ爜澶辫触")
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

        raise RuntimeError(f"涓嶆敮鎸佺殑鍥剧墖 URL: {ref}")
