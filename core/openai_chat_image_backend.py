from __future__ import annotations

import base64
import inspect
import json
import re
import time
from pathlib import Path

from openai import AsyncOpenAI

from astrbot.api import logger

from .image_format import guess_image_mime_and_ext
from .openai_compat_backend import (
    build_proxy_http_client,
    normalize_openai_compat_base_url,
)

_MARKDOWN_IMAGE_RE = re.compile(r"!\[.*?\]\((.*?)\)")
_DATA_IMAGE_RE = re.compile(r"(data:image/[^\s)]+)")
_HTML_IMG_RE = re.compile(r'<img[^>]*src=["\']([^"\'>]+)["\']', re.IGNORECASE)
_IMAGE_URL_RE = re.compile(
    r"(https?://[^\s<>\"')\]]+?\.(?:png|jpg|jpeg|webp|gif)(?:\?[^\s<>\"')\]]*)?)",
    re.IGNORECASE,
)
_JSON_URL_FIELD_RE = re.compile(
    r'"(?:image_url|imageUrl|url|image|src|uri|link|href|fifeUrl|fife_url|final_image_url|origin_image_url)"\s*:\s*"([^"]+)"'
)

_HTML_VIDEO_RE = re.compile(r'<video[^>]*src=["\']([^"\'>]+)["\']', re.IGNORECASE)
_VIDEO_URL_RE = re.compile(
    r"(https?://[^\s<>\"')\]]+?\.(?:mp4|webm|mov)(?:\?[^\s<>\"')\]]*)?)",
    re.IGNORECASE,
)

_BASE64_PREFIX_RE = re.compile(r"^(?:b64|base64)\s*:\s*", re.IGNORECASE)


def _strip_markdown_target(target: str) -> str | None:
    s = (target or "").strip()
    if not s:
        return None
    if s.startswith("<") and ">" in s:
        right = s.find(">")
        if right > 1:
            s = s[1:right].strip()
    # markdown may include optional title: (url "title")
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


def _looks_like_video_url(url: str) -> bool:
    u = (url or "").strip().lower()
    if not u.startswith(("http://", "https://")):
        return False
    if any(ext in u for ext in (".mp4", ".webm", ".mov")):
        return True
    if "generated_video" in u:
        return True
    return False


def _is_valid_data_image_ref(ref: str) -> bool:
    s = str(ref or "").strip()
    if not s.startswith("data:image/"):
        return False
    if "," not in s:
        return False
    _header, b64 = s.split(",", 1)
    b64 = re.sub(r"\s+", "", (b64 or "").strip())
    if not b64 or b64 == "...":
        return False
    if len(b64) < 16:
        return False
    # lightweight charset sanity check (prefix only)
    try:
        import re as _re

        if not _re.fullmatch(r"[A-Za-z0-9+/=_-]+", b64[:2048]):
            return False
    except Exception:
        pass
    # short payloads may still be valid (small png/jpg); verify by decode + magic.
    if len(b64) < 128:
        raw = _decode_base64_bytes(b64)
        if not raw:
            return False
        if not _guess_mime_from_magic(raw):
            return False
    return True


def _guess_mime_from_magic(image_bytes: bytes) -> str | None:
    if len(image_bytes) >= 3 and image_bytes[0:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if len(image_bytes) >= 8 and image_bytes[0:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if len(image_bytes) >= 6 and (
        image_bytes[0:6] == b"GIF87a" or image_bytes[0:6] == b"GIF89a"
    ):
        return "image/gif"
    if (
        len(image_bytes) >= 12
        and image_bytes[0:4] == b"RIFF"
        and image_bytes[8:12] == b"WEBP"
    ):
        return "image/webp"
    return None


def _base64_to_data_image_ref(text: str) -> str | None:
    s = (text or "").strip().strip('"').strip("'")
    s = _BASE64_PREFIX_RE.sub("", s).strip()
    s = re.sub(r"\s+", "", s)
    if len(s) < 128:
        return None
    raw = _decode_base64_bytes(s)
    if not raw:
        return None
    mime = _guess_mime_from_magic(raw)
    if not mime:
        return None
    std_b64 = base64.b64encode(raw).decode()
    return f"data:{mime};base64,{std_b64}"


def _extract_first_image_ref(text: str) -> str | None:
    s = (text or "").strip()
    if not s:
        return None
    if s.startswith("data:image/"):
        compact = re.sub(r"\s+", "", s)
        if _is_valid_data_image_ref(compact):
            return compact
    m = _MARKDOWN_IMAGE_RE.search(s)
    if m:
        cand = _strip_markdown_target(m.group(1))
        if cand:
            if cand.startswith("data:image/"):
                cand = re.sub(r"\s+", "", cand)
                if _is_valid_data_image_ref(cand):
                    return cand
            elif not _looks_like_video_url(cand):
                return cand

    # data:image refs may be huge and occasionally truncated; only accept well-formed ones.
    for m in _DATA_IMAGE_RE.finditer(s):
        cand = re.sub(r"\s+", "", m.group(1).strip())
        if _is_valid_data_image_ref(cand):
            return cand

    m = _HTML_IMG_RE.search(s)
    if m:
        url = m.group(1).strip()
        if url and not _looks_like_video_url(url):
            return url
    m = _IMAGE_URL_RE.search(s)
    if m:
        url = m.group(1).strip()
        if url and not _looks_like_video_url(url):
            return url
    if s.startswith("http://") or s.startswith("https://"):
        if _looks_like_video_url(s):
            return None
        return s

    # JSON-like snippets: {"image_url":"..."} / {"url":"..."} etc.
    for m in _JSON_URL_FIELD_RE.finditer(s):
        cand = m.group(1).strip().replace("\\/", "/")
        cand = _strip_markdown_target(cand) or cand
        if cand.startswith("data:image/"):
            cand = re.sub(r"\s+", "", cand)
            if _is_valid_data_image_ref(cand):
                return cand
        if cand.startswith(("http://", "https://")) and not _looks_like_video_url(cand):
            return cand

    # Some gateways wrap full payload into JSON string.
    if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
        try:
            parsed = json.loads(s)
        except Exception:
            parsed = None
        if parsed is not None:
            for v in _iter_strings(parsed):
                ref = _extract_first_image_ref(v)
                if ref:
                    return ref

    # Some gateways/models return raw base64 without data:image prefix.
    ref = _base64_to_data_image_ref(s)
    if ref:
        return ref
    return None


def _extract_first_video_url(text: str) -> str | None:
    s = (text or "").strip()
    if not s:
        return None
    m = _HTML_VIDEO_RE.search(s)
    if m:
        url = m.group(1).strip()
        return url if _looks_like_video_url(url) else None
    m = _VIDEO_URL_RE.search(s)
    if m:
        url = m.group(1).strip()
        return url if _looks_like_video_url(url) else None
    if _looks_like_video_url(s):
        return s
    return None


def _is_client_closed_error(exc: Exception) -> bool:
    msg = f"{exc!r} {exc}".lower()
    if "client has been closed" in msg:
        return True
    cur: Exception | None = exc
    for _ in range(3):
        nxt = getattr(cur, "__cause__", None) or getattr(cur, "__context__", None)
        if not isinstance(nxt, Exception):
            break
        cur = nxt
        if "client has been closed" in f"{cur!r} {cur}".lower():
            return True
    return False


async def _resolve_awaitable(value: object) -> object:
    while inspect.isawaitable(value):
        value = await value
    return value


def _iter_strings(obj: object) -> list[str]:
    out: list[str] = []
    seen: set[int] = set()

    def walk(x: object) -> None:
        if x is None:
            return
        oid = id(x)
        if oid in seen:
            return
        seen.add(oid)
        if isinstance(x, str):
            out.append(x)
            return
        if isinstance(x, dict):
            for v in x.values():
                walk(v)
            return
        if isinstance(x, (list, tuple)):
            for v in x:
                walk(v)
            return
        model_dump = getattr(x, "model_dump", None)
        if callable(model_dump):
            try:
                walk(model_dump())
                return
            except Exception:
                pass
        as_dict = getattr(x, "dict", None)
        if callable(as_dict):
            try:
                walk(as_dict())
                return
            except Exception:
                pass
        obj_dict = getattr(x, "__dict__", None)
        if isinstance(obj_dict, dict):
            walk(obj_dict)
            return

    walk(obj)
    return out


def _extract_image_ref_from_content(content: object) -> str | None:
    if content is None:
        return None

    if isinstance(content, str):
        return _extract_first_image_ref(content)

    # OpenAI-style multimodal content: [{"type":"text","text":...}, {"type":"image_url","image_url":{"url":"..."}}]
    if isinstance(content, list):
        for part in content:
            ref = _extract_image_ref_from_content(part)
            if ref:
                return ref
        return None

    if isinstance(content, dict):
        # Common patterns:
        # - {"type":"image_url","image_url":{"url":"https://..."}} (or data:...)
        # - {"type":"text","text":"..."}
        if str(content.get("type") or "").lower() == "image_url":
            image_url = content.get("image_url")
            if isinstance(image_url, dict):
                url = image_url.get("url")
                if isinstance(url, str):
                    return url.strip() or None
            if isinstance(image_url, str):
                return image_url.strip() or None

        if str(content.get("type") or "").lower() == "text":
            text = content.get("text")
            if isinstance(text, str):
                ref = _extract_first_image_ref(text)
                if ref:
                    return ref

        # Some gateways return explicit base64 fields.
        for k in ("b64_json", "b64", "base64", "image_b64", "image_base64", "imageB64"):
            v = content.get(k)
            if isinstance(v, str):
                ref = _base64_to_data_image_ref(v)
                if ref:
                    return ref

        # Vertex-style inlineData.
        inline = content.get("inlineData")
        if isinstance(inline, dict):
            b64 = inline.get("data")
            if isinstance(b64, str):
                ref = _base64_to_data_image_ref(b64)
                if ref:
                    return ref

        # Some gateways return {"url": "..."} / {"image": "..."} with various key names.
        for k in (
            "url",
            "image",
            "image_url",
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
            v = content.get(k)
            if isinstance(v, str):
                ref = _extract_first_image_ref(v)
                if ref:
                    return ref
            ref = _extract_image_ref_from_content(v)
            if ref:
                return ref

        # Common container fields.
        for k in ("images", "image_urls", "attachments", "media", "result", "response"):
            ref = _extract_image_ref_from_content(content.get(k))
            if ref:
                return ref

        # Last resort: scan all nested strings.
        for s in _iter_strings(content):
            ref = _extract_first_image_ref(s)
            if ref:
                return ref
        return None

    # Unknown type: attempt to scan its string fields if any.
    for s in _iter_strings(content):
        ref = _extract_first_image_ref(s)
        if ref:
            return ref
    return None


def _extract_video_ref_from_content(content: object) -> str | None:
    if content is None:
        return None
    if isinstance(content, str):
        return _extract_first_video_url(content)
    for s in _iter_strings(content):
        url = _extract_first_video_url(s)
        if url:
            return url
    return None


class OpenAIChatImageBackend:
    """Image generation/edit via chat.completions (gateway-style).

    Many third-party gateways do NOT implement /v1/images/* at all, but will return images via chat content,
    e.g. markdown: ![](data:image/png;base64,...)
    """

    def __init__(
        self,
        *,
        imgr,
        base_url: str,
        api_keys: list[str],
        timeout: int = 120,
        max_retries: int = 2,
        default_model: str = "",
        supports_edit: bool = True,
        extra_body: dict | None = None,
        proxy_url: str | None = None,
    ):
        self.imgr = imgr
        self.base_url = normalize_openai_compat_base_url(base_url)
        self.api_keys = [str(k).strip() for k in (api_keys or []) if str(k).strip()]
        self.timeout = int(timeout or 120)
        self.max_retries = int(max_retries or 2)
        self.default_model = str(default_model or "").strip()
        self.supports_edit = bool(supports_edit)
        self.extra_body = extra_body or {}
        self.proxy_url = str(proxy_url or "").strip() or None

        self._key_index = 0
        self._clients: dict[str, AsyncOpenAI] = {}
        self._http_client = None

    @staticmethod
    def _supports_http_client_param() -> bool:
        try:
            sig = inspect.signature(AsyncOpenAI)
        except Exception:
            try:
                sig = inspect.signature(AsyncOpenAI.__init__)  # type: ignore[misc]
            except Exception:
                return False
        return "http_client" in sig.parameters

    def _get_http_client(self):
        if not self.proxy_url:
            return None
        if self._http_client is not None:
            return self._http_client
        self._http_client = build_proxy_http_client(self.proxy_url)
        return self._http_client

    async def close(self) -> None:
        for client in self._clients.values():
            try:
                await client.close()
            except Exception:
                pass
        self._clients.clear()
        if self._http_client is not None:
            try:
                await self._http_client.aclose()
            except Exception:
                pass
            self._http_client = None

    def _next_key(self) -> str:
        if not self.api_keys:
            raise RuntimeError("未配置 API Key")
        key = self.api_keys[self._key_index]
        self._key_index = (self._key_index + 1) % len(self.api_keys)
        return key

    def _get_client(self, key: str) -> AsyncOpenAI:
        client = self._clients.get(key)
        if client is None:
            kwargs: dict = {
                "base_url": self.base_url,
                "api_key": key,
                "timeout": self.timeout,
                "max_retries": self.max_retries,
            }
            if self.proxy_url and self._supports_http_client_param():
                http_client = self._get_http_client()
                if http_client is not None:
                    kwargs["http_client"] = http_client
            client = AsyncOpenAI(**kwargs)
            self._clients[key] = client
        return client

    async def _recreate_client(self, key: str) -> AsyncOpenAI:
        old = self._clients.pop(key, None)
        if old is not None:
            try:
                await old.close()
            except Exception:
                pass
        kwargs: dict = {
            "base_url": self.base_url,
            "api_key": key,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
        }
        if self.proxy_url and self._supports_http_client_param():
            http_client = self._get_http_client()
            if http_client is not None:
                kwargs["http_client"] = http_client
        client = AsyncOpenAI(**kwargs)
        self._clients[key] = client
        return client

    @staticmethod
    def _normalize_ref_candidate(value: object) -> str | None:
        if not isinstance(value, str):
            return None
        s = value.strip().strip('"').strip("'")
        if not s:
            return None
        if s.startswith("data:image/"):
            return re.sub(r"\s+", "", s)
        if s.startswith("http://") or s.startswith("https://"):
            return s
        ref = _extract_first_image_ref(s)
        if not ref:
            return None
        if ref.startswith("data:image/"):
            return re.sub(r"\s+", "", ref)
        return ref

    async def _extract_image_refs_from_response(self, resp: object) -> list[str]:
        refs: list[str] = []
        seen: set[str] = set()

        def add_ref(value: object) -> None:
            ref = self._normalize_ref_candidate(value)
            if not ref:
                return
            if ref.startswith(("http://", "https://")) and _looks_like_video_url(ref):
                return
            if ref in seen:
                return
            seen.add(ref)
            refs.append(ref)

        async def collect(content: object) -> None:
            content = await _resolve_awaitable(content)
            if content is None:
                return
            add_ref(_extract_image_ref_from_content(content))
            for s in _iter_strings(content):
                add_ref(s)

        # 1) Preferred: all choices message blocks.
        try:
            choices_raw = await _resolve_awaitable(getattr(resp, "choices", []))  # type: ignore[attr-defined]
            if choices_raw is None:
                choices = []
            else:
                try:
                    choices = list(choices_raw)
                except TypeError:
                    choices = [choices_raw]
            for choice in choices[:4]:
                choice = await _resolve_awaitable(choice)
                msg = await _resolve_awaitable(getattr(choice, "message", None))
                if msg is None:
                    continue
                await collect(getattr(msg, "images", None))
                await collect(getattr(msg, "content", None))
                await collect(getattr(msg, "tool_calls", None))
        except Exception:
            pass

        # 2) Fallback: scan model dump (dict/list) for data:image / markdown / url.
        try:
            model_dump = getattr(resp, "model_dump", None)  # type: ignore[attr-defined]
            dumped = await _resolve_awaitable(model_dump()) if callable(model_dump) else None
        except Exception:
            dumped = None
        if dumped is not None:
            await collect(dumped)

        return refs

    async def _extract_image_ref_from_response(self, resp: object) -> str | None:
        refs = await self._extract_image_refs_from_response(resp)
        return refs[0] if refs else None

    async def _extract_video_ref_from_response(self, resp: object) -> str | None:
        try:
            choices_raw = await _resolve_awaitable(getattr(resp, "choices", []))  # type: ignore[attr-defined]
            if choices_raw is None:
                choices = []
            else:
                try:
                    choices = list(choices_raw)
                except TypeError:
                    choices = [choices_raw]
            for choice in choices[:4]:
                choice = await _resolve_awaitable(choice)
                msg = await _resolve_awaitable(getattr(choice, "message", None))
                if msg is None:
                    continue
                content = await _resolve_awaitable(getattr(msg, "content", None))
                url = _extract_video_ref_from_content(content)
                if url:
                    return url
        except Exception:
            pass

        try:
            model_dump = getattr(resp, "model_dump", None)  # type: ignore[attr-defined]
            dumped = await _resolve_awaitable(model_dump()) if callable(model_dump) else None
        except Exception:
            dumped = None
        if dumped is not None:
            return _extract_video_ref_from_content(dumped)
        return None

    async def _save_single_ref(self, ref: str, *, debug_snippet: str = "") -> Path:
        if not ref:
            raise RuntimeError(
                f"chat 返回未包含图片（需 markdown/data:image/url）：{debug_snippet}"
            )

        if ref.startswith("data:image/"):
            compact = re.sub(r"\s+", "", ref)
            try:
                _header, b64_data = compact.split(",", 1)
            except ValueError:
                raise RuntimeError(
                    "chat 返回 data:image 但缺少 base64 数据"
                    f"（len={len(compact)} head={compact[:64]!r} tail={compact[-32:]!r}）：{debug_snippet}"
                ) from None
            try:
                image_bytes = _decode_base64_bytes((b64_data or "").strip())
            except Exception:
                image_bytes = b""
            if not image_bytes:
                raise RuntimeError(
                    "chat 返回 data:image 但 base64 解码失败"
                    f"（len={len(b64_data or '')} head={str(b64_data)[:48]!r}）：{debug_snippet}"
                )
            return await self.imgr.save_image(image_bytes)

        if ref.startswith("http://") or ref.startswith("https://"):
            if _looks_like_video_url(ref):
                raise RuntimeError(
                    f"chat 返回了视频而不是图片：{ref}（如果想要视频请用 /视频；如果想要图片请换模型或改用 images 接口）"
                )
            return await self.imgr.download_image(ref)

        raise RuntimeError("chat 返回的图片引用格式不支持")

    async def _save_from_ref(
        self, ref: str, *, debug_snippet: str = "", fallback_refs: list[str] | None = None
    ) -> Path:
        candidates: list[str] = [str(ref or "").strip()]
        for extra in fallback_refs or []:
            s = str(extra or "").strip()
            if s and s not in candidates:
                candidates.append(s)

        last_error: Exception | None = None
        for idx, cand in enumerate(candidates):
            try:
                return await self._save_single_ref(cand, debug_snippet=debug_snippet)
            except Exception as e:
                last_error = e
                if idx + 1 < len(candidates):
                    logger.warning(
                        "[OpenAIChatImage] 图片引用解析失败，尝试回退候选 %s/%s: %s",
                        idx + 1,
                        len(candidates),
                        e,
                    )
                    continue
                raise
        raise RuntimeError(f"chat 图片保存失败: {last_error}")

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

        size_hint = ""
        if size:
            size_hint = f" Output size target: {size}."
        elif resolution:
            size_hint = f" Output resolution target: {resolution}."

        user_text = (
            f"{prompt}\n\n"
            "Return ONLY one image. Do NOT return video/mp4, HTML, or explanations.\n"
            "Output format MUST be exactly one markdown image:\n"
            "![](data:image/png;base64,...)"
            f"{size_hint}"
        )

        eb = {}
        eb.update(self.extra_body)
        eb.update(extra_body or {})

        t0 = time.time()
        try:
            resp = await client.chat.completions.create(
                model=final_model,
                messages=[{"role": "user", "content": user_text}],
                extra_body=eb or None,
            )
        except Exception as e:
            if _is_client_closed_error(e):
                logger.warning(
                    "[OpenAIChatImage][generate] client 已关闭，重建后重试一次"
                )
                client = await self._recreate_client(key)
                resp = await client.chat.completions.create(
                    model=final_model,
                    messages=[{"role": "user", "content": user_text}],
                    extra_body=eb or None,
                )
            else:
                logger.error(
                    "[OpenAIChatImage][generate] API 调用失败，base_url=%s，耗时: %.2fs: %s",
                    self.base_url,
                    time.time() - t0,
                    e,
                )
                raise

        refs = await self._extract_image_refs_from_response(resp)
        ref = refs[0] if refs else None
        debug_snippet = ""
        try:
            debug_snippet = (
                str(getattr(resp.choices[0].message, "content", ""))
                .strip()
                .replace("\n", " ")[:200]  # type: ignore[attr-defined]
            )
        except Exception:
            pass

        logger.info("[OpenAIChatImage][generate] API 响应耗时: %.2fs", time.time() - t0)
        if not ref:
            video_url = await self._extract_video_ref_from_response(resp)
            if video_url:
                raise RuntimeError(
                    f"chat 返回了视频而不是图片：{video_url}（如果想要视频请用 /视频；如果想要图片请换模型或改用 images 接口）"
                )
        return await self._save_from_ref(
            ref or "", debug_snippet=debug_snippet, fallback_refs=refs[1:]
        )

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
            raise RuntimeError("该后端不支持改图/图生图（chat 模式）")
        if not images:
            raise ValueError("至少需要一张图片")

        key = self._next_key()
        client = self._get_client(key)

        final_model = str(model or self.default_model or "").strip()
        if not final_model:
            raise RuntimeError("未配置 model")

        size_hint = ""
        if size:
            size_hint = f" Output size target: {size}."
        elif resolution:
            size_hint = f" Output resolution target: {resolution}."

        text = (
            f"{prompt}\n\n"
            "Edit the attached image(s). Return ONLY one image.\n"
            "Do NOT return video/mp4, HTML, or explanations.\n"
            "Output format MUST be exactly one markdown image:\n"
            "![](data:image/png;base64,...)"
            f"{size_hint}"
        )

        parts: list[dict] = [{"type": "text", "text": text}]
        for img_bytes in images:
            mime, _ext = guess_image_mime_and_ext(img_bytes)
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime};base64,{base64.b64encode(img_bytes).decode()}",
                    },
                }
            )

        eb = {}
        eb.update(self.extra_body)
        eb.update(extra_body or {})

        t0 = time.time()
        try:
            resp = await client.chat.completions.create(
                model=final_model,
                messages=[{"role": "user", "content": parts}],
                extra_body=eb or None,
            )
        except Exception as e:
            if _is_client_closed_error(e):
                logger.warning("[OpenAIChatImage][edit] client 已关闭，重建后重试一次")
                client = await self._recreate_client(key)
                resp = await client.chat.completions.create(
                    model=final_model,
                    messages=[{"role": "user", "content": parts}],
                    extra_body=eb or None,
                )
            else:
                logger.error(
                    "[OpenAIChatImage][edit] API 调用失败，base_url=%s，耗时: %.2fs: %s",
                    self.base_url,
                    time.time() - t0,
                    e,
                )
                raise

        refs = await self._extract_image_refs_from_response(resp)
        ref = refs[0] if refs else None
        debug_snippet = ""
        try:
            debug_snippet = (
                str(getattr(resp.choices[0].message, "content", ""))
                .strip()
                .replace("\n", " ")[:200]  # type: ignore[attr-defined]
            )
        except Exception:
            pass

        logger.info("[OpenAIChatImage][edit] API 响应耗时: %.2fs", time.time() - t0)
        if not ref:
            video_url = await self._extract_video_ref_from_response(resp)
            if video_url:
                raise RuntimeError(
                    f"chat 返回了视频而不是图片：{video_url}（如果想要视频请用 /视频；如果想要图片请换模型或改用 images 接口）"
                )
        return await self._save_from_ref(
            ref or "", debug_snippet=debug_snippet, fallback_refs=refs[1:]
        )
