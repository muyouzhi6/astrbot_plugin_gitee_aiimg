from __future__ import annotations

import asyncio
import base64
import ipaddress
import inspect
import json
import re
import time
from pathlib import Path
from urllib.parse import urljoin, urlsplit, urlunsplit

import httpx
from openai import AsyncOpenAI

from astrbot.api import logger

from .image_format import guess_image_mime_and_ext
from .net_safety import URLFetchPolicy, ensure_url_allowed
from .openai_compat_backend import (
    build_proxy_http_client,
    normalize_openai_compat_base_url,
)
from .openai_full_url_backend import OpenAIFullURLBackend

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
_LOCAL_MEDIA_HOSTS = {"0.0.0.0", "127.0.0.1", "localhost", "host.docker.internal"}

_KNOWN_TRUSTED_RESULT_ORIGINS: dict[str, set[str]] = {
    "api.bltcy.ai": {
        "https://files.closeai.fans",
    },
}

_REQUEST_MODE_ALIASES = {
    "auto": "auto",
    "stream": "stream",
    "non_stream": "non_stream",
    "non-stream": "non_stream",
    "nonstream": "non_stream",
    "non stream": "non_stream",
}


def _parse_png_size(image_bytes: bytes) -> tuple[int, int] | None:
    if len(image_bytes) < 24:
        return None
    if image_bytes[0:8] != b"\x89PNG\r\n\x1a\n":
        return None
    try:
        width = int.from_bytes(image_bytes[16:20], "big")
        height = int.from_bytes(image_bytes[20:24], "big")
    except Exception:
        return None
    if width <= 0 or height <= 0:
        return None
    return width, height


def _looks_like_placeholder_image_bytes(image_bytes: bytes) -> bool:
    if not image_bytes:
        return True
    # 某些网关在被强制要求输出 data:image 时，会秒回一个 1x1 占位 PNG，
    # 解析上看似成功，实际等于没出图。
    if len(image_bytes) <= 128:
        return True

    png_size = _parse_png_size(image_bytes)
    if png_size == (1, 1):
        return True

    return False


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


def _looks_like_relative_media_ref(ref: str) -> bool:
    s = str(ref or "").strip()
    if not s:
        return False
    if s.startswith(("data:image/", "http://", "https://", "file://")):
        return False
    if "://" in s:
        return False
    lowered = s.lower()
    return any(ext in lowered for ext in (".png", ".jpg", ".jpeg", ".webp", ".gif"))


def _is_local_media_host(host: str) -> bool:
    h = str(host or "").strip().lower()
    if not h:
        return False
    if h in _LOCAL_MEDIA_HOSTS or h.endswith(".localhost") or h.endswith(".local"):
        return True
    try:
        ip = ipaddress.ip_address(h)
    except ValueError:
        return False
    return bool(ip.is_loopback or ip.is_private or ip.is_link_local)


def _rewrite_local_media_url(ref: str, *, base_url: str) -> str:
    s = str(ref or "").strip()
    if not s:
        return ""
    try:
        parts = urlsplit(s)
    except Exception:
        return s
    if not parts.scheme or not parts.netloc or not _is_local_media_host(parts.hostname or ""):
        return s

    base_parts = urlsplit(str(base_url or "").strip())
    if not base_parts.hostname:
        return s

    host = base_parts.hostname
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    netloc = f"{host}:{parts.port}" if parts.port is not None else host
    return urlunsplit(
        (
            parts.scheme or base_parts.scheme or "http",
            netloc,
            parts.path or "/",
            parts.query,
            parts.fragment,
        )
    )


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
    if s.startswith("//"):
        return s
    if s.startswith("/") and _looks_like_relative_media_ref(s):
        return s
    if _looks_like_relative_media_ref(s):
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
        if cand.startswith("//"):
            return cand
        if cand.startswith("/") and _looks_like_relative_media_ref(cand):
            return cand
        if _looks_like_relative_media_ref(cand):
            return cand

    # Some gateways wrap full payload into JSON string.
    if (s.startswith("{") and s.endswith("}")) or (
        s.startswith("[") and s.endswith("]")
    ):
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


def _extract_media_refs_from_sse_text(text: str) -> tuple[list[str], list[str]]:
    image_refs: list[str] = []
    video_refs: list[str] = []
    full_text = ""

    def add_image(ref: str | None) -> None:
        if not ref or ref in image_refs:
            return
        if ref.startswith(("http://", "https://")) and _looks_like_video_url(ref):
            return
        image_refs.append(ref)

    def add_video(ref: str | None) -> None:
        if not ref or ref in video_refs:
            return
        video_refs.append(ref)

    def content_to_text(value: object) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            return "".join(content_to_text(item) for item in value)
        if isinstance(value, dict):
            text_value = value.get("text")
            if isinstance(text_value, str) and text_value:
                return text_value
            image_url = value.get("image_url")
            if isinstance(image_url, dict):
                url = image_url.get("url")
                if isinstance(url, str) and url:
                    return url
            url = value.get("url")
            if isinstance(url, str) and url:
                return url
            return str(value)
        return str(value)

    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("```") or not line.startswith("data:"):
            continue
        data_str = line[5:].strip()
        if not data_str or data_str == "[DONE]":
            continue
        try:
            obj = json.loads(data_str)
        except Exception:
            continue

        add_image(_extract_image_ref_from_content(obj))
        add_video(_extract_video_ref_from_content(obj))

        choice0 = (obj.get("choices") or [{}])[0] if isinstance(obj, dict) else {}
        if not isinstance(choice0, dict):
            choice0 = {}
        delta = choice0.get("delta") or {}
        message = choice0.get("message") or {}
        delta_content = (
            delta.get("content") if "content" in delta else message.get("content")
        )
        if delta_content is None and "reasoning_content" in delta:
            delta_content = delta.get("reasoning_content")
        if delta_content is None and "reasoning_content" in message:
            delta_content = message.get("reasoning_content")

        full_text += content_to_text(delta_content)

    add_image(_extract_first_image_ref(full_text))
    add_video(_extract_first_video_url(full_text))
    return image_refs, video_refs


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
        generate_request_mode: str | None = None,
        edit_request_mode: str | None = None,
        enable_stream_generate: bool | None = None,
        enable_stream_edit: bool | None = None,
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
        self.generate_request_mode = self._resolve_request_mode(
            generate_request_mode,
            legacy_stream_enabled=enable_stream_generate,
        )
        self.edit_request_mode = self._resolve_request_mode(
            edit_request_mode,
            legacy_stream_enabled=enable_stream_edit,
        )
        self.enable_stream_generate = self._should_try_stream("generate")
        self.enable_stream_edit = self._should_try_stream("edit")

        self._key_index = 0
        self._clients: dict[str, AsyncOpenAI] = {}
        self._http_client = None
        self._prefer_file_service_url_input = False
        self._trusted_result_origins = self._build_trusted_result_origins()

    @staticmethod
    def _normalize_request_mode(value: object) -> str:
        if isinstance(value, bool):
            return "stream" if value else "non_stream"
        text = str(value or "").strip().lower()
        if not text:
            return ""
        return _REQUEST_MODE_ALIASES.get(text, "")

    @classmethod
    def _resolve_request_mode(
        cls,
        mode_value: object,
        *,
        legacy_stream_enabled: bool | None = None,
    ) -> str:
        normalized = cls._normalize_request_mode(mode_value)
        if normalized and normalized != "auto":
            return normalized
        if legacy_stream_enabled is not None:
            return "stream" if legacy_stream_enabled else "non_stream"
        if normalized == "auto":
            return "auto"
        return "auto"

    def _should_try_stream(self, operation: str) -> bool:
        request_mode = (
            self.generate_request_mode
            if operation == "generate"
            else self.edit_request_mode
        )
        if request_mode == "stream":
            return True
        if request_mode == "non_stream":
            return False
        return operation == "generate"

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

    def _chat_completions_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/chat/completions"

    def _build_trusted_result_origins(self) -> frozenset[str]:
        parts = urlsplit(self.base_url)
        host = (parts.hostname or "").strip().lower()
        trusted: set[str] = set()
        for known_host, origins in _KNOWN_TRUSTED_RESULT_ORIGINS.items():
            if host == known_host or host.endswith(f".{known_host}"):
                trusted.update(origins)
        return frozenset(trusted)

    @staticmethod
    def _is_gemini_chat_image_model(model: str | None) -> bool:
        return "gemini" in str(model or "").strip().lower()

    def _trusted_result_url_policy(self) -> URLFetchPolicy:
        trusted_origins: set[str] = set(self._trusted_result_origins)
        raw_imgr_trusted = getattr(self.imgr, "_trusted_origins", frozenset())
        if isinstance(raw_imgr_trusted, (set, frozenset, list, tuple)):
            for item in raw_imgr_trusted:
                text = str(item or "").strip().rstrip("/")
                if text:
                    trusted_origins.add(text)
        return URLFetchPolicy(
            allow_private=bool(getattr(self.imgr, "_media_allow_private", False)),
            trusted_origins=frozenset(trusted_origins),
            allowed_hosts=frozenset(),
            dns_timeout_seconds=float(
                getattr(self.imgr, "_dns_timeout_seconds", 2) or 2
            ),
        )

    @staticmethod
    def _sse_debug_snippet(text: str) -> str:
        snippet = re.sub(r"\s+", " ", str(text or "").strip())
        return snippet[:200]

    @staticmethod
    def _build_generate_prompt(
        prompt: str,
        *,
        size: str | None = None,
        resolution: str | None = None,
        strict_format: bool,
    ) -> str:
        size_hint = ""
        if size:
            size_hint = f" Output size target: {size}."
        elif resolution:
            size_hint = f" Output resolution target: {resolution}."

        if strict_format:
            return (
                f"{prompt}\n\n"
                "Return ONLY one image. Do NOT return video/mp4, HTML, or explanations.\n"
                "Output format MUST be exactly one markdown image and nothing else."
                f"{size_hint}"
            )

        return (
            f"{prompt}\n\n"
            "Generate exactly one image. Do NOT return video/mp4, HTML, or explanations."
            f"{size_hint}"
        )

    @staticmethod
    def _normalize_image_size_hint(
        *,
        model: str | None,
        size: str | None,
        resolution: str | None,
    ) -> str | None:
        raw = str(size or "").strip()
        if not raw:
            raw = str(resolution or "").strip()
        if not raw:
            model_name = str(model or "").lower()
            if "4k" in model_name:
                raw = "4K"
            elif "2k" in model_name:
                raw = "2K"
            elif "1k" in model_name:
                raw = "1K"
            elif "512px" in model_name:
                raw = "512px"
        if not raw:
            return None

        normalized = raw.strip().upper().replace("×", "X")
        size_map = {
            "1024X1024": "1K",
            "2048X2048": "2K",
            "4096X4096": "4K",
        }
        return size_map.get(normalized, raw.strip())

    @classmethod
    def _apply_gemini_image_config(
        cls,
        extra_body: dict | None,
        *,
        model: str | None,
        size: str | None,
        resolution: str | None,
    ) -> dict:
        merged = dict(extra_body or {})
        model_name = str(model or "").lower()
        if not cls._is_gemini_chat_image_model(model):
            return merged

        image_size = cls._normalize_image_size_hint(
            model=model,
            size=size,
            resolution=resolution,
        )
        if not image_size:
            image_size = None

        merged_modalities = merged.get("modalities")
        normalized_modalities: list[str] = []
        raw_modalities: list[object]
        if isinstance(merged_modalities, (list, tuple, set)):
            raw_modalities = list(merged_modalities)
        elif isinstance(merged_modalities, str) and merged_modalities.strip():
            raw_modalities = [merged_modalities.strip()]
        else:
            raw_modalities = []
        for value in raw_modalities:
            text = str(value or "").strip().lower()
            if text and text not in normalized_modalities:
                normalized_modalities.append(text)
        for required in ("image", "text"):
            if required not in normalized_modalities:
                normalized_modalities.append(required)
        if normalized_modalities:
            merged["modalities"] = normalized_modalities

        if not image_size:
            return merged

        request_image_size = image_size
        if (
            "gemini-3.1-flash-image-preview-4k" in model_name
            and str(image_size).strip().upper() in {"4K", "4096X4096", "2K", "2048X2048"}
        ):
            # 该网关的 chat 兼容层对 4K 规格经常直接 503；
            # 实测把 image_config 压到 1K 仍会回 3840x4380 这类高分辨率图，
            # 因此这里只降请求规格，不动 prompt 里的 4K 目标提示。
            request_image_size = "1K"

        raw_image_config = merged.get("image_config")
        image_config = (
            dict(raw_image_config) if isinstance(raw_image_config, dict) else {}
        )
        image_config.setdefault("image_size", request_image_size)
        image_config.setdefault("imageSize", request_image_size)
        merged["image_config"] = image_config

        merged.setdefault("image_size", request_image_size)
        merged.setdefault("imageSize", request_image_size)
        merged.setdefault("size", request_image_size)

        raw_generation_config = merged.get("generation_config")
        generation_config = (
            dict(raw_generation_config)
            if isinstance(raw_generation_config, dict)
            else {}
        )
        generation_config.setdefault("image_size", request_image_size)
        generation_config.setdefault("imageSize", request_image_size)
        if generation_config:
            merged["generation_config"] = generation_config

        raw_generation_config_camel = merged.get("generationConfig")
        generation_config_camel = (
            dict(raw_generation_config_camel)
            if isinstance(raw_generation_config_camel, dict)
            else {}
        )
        raw_image_config_camel = generation_config_camel.get("imageConfig")
        image_config_camel = (
            dict(raw_image_config_camel)
            if isinstance(raw_image_config_camel, dict)
            else {}
        )
        image_config_camel.setdefault("image_size", request_image_size)
        image_config_camel.setdefault("imageSize", request_image_size)
        if image_config_camel:
            generation_config_camel["imageConfig"] = image_config_camel
        if generation_config_camel:
            merged["generationConfig"] = generation_config_camel
        return merged

    @staticmethod
    def _is_retryable_chat_image_error(exc: Exception, *, model: str | None) -> bool:
        text = f"{exc!r} {exc}".lower()
        if not OpenAIChatImageBackend._is_gemini_chat_image_model(model):
            return False
        markers = (
            "no provider for",
            "503",
            "readtimeout",
            "timed out",
            "timeout",
            "unsupported protocol scheme",
            "unsupported url scheme",
            "failed to download",
            "get file base64 from url",
            "invalid image_url",
            "unable to process input image",
        )
        return any(marker in text for marker in markers)

    @classmethod
    def _chat_request_attempts(cls, *, model: str | None) -> int:
        if cls._is_gemini_chat_image_model(model):
            return 2
        return 1

    async def _run_chat_request_with_retries(
        self,
        request_factory,
        *,
        model: str | None,
        log_tag: str,
        input_mode: str,
    ):
        attempts = self._chat_request_attempts(model=model)
        for attempt in range(1, attempts + 1):
            try:
                return await request_factory()
            except Exception as exc:
                if (
                    attempt >= attempts
                    or not self._is_retryable_chat_image_error(exc, model=model)
                ):
                    raise
                logger.warning(
                    "[OpenAIChatImage][%s] chat 请求失败，准备重试 %s/%s，input_mode=%s: %s",
                    log_tag,
                    attempt + 1,
                    attempts,
                    input_mode,
                    exc,
                )
                await asyncio.sleep(min(1.5 * attempt, 3.0))

    @staticmethod
    def _build_edit_text(
        prompt: str,
        *,
        size: str | None = None,
        resolution: str | None = None,
        strict_format: bool,
    ) -> str:
        size_hint = ""
        if size:
            size_hint = f" Output size target: {size}."
        elif resolution:
            size_hint = f" Output resolution target: {resolution}."

        if strict_format:
            return (
                f"{prompt}\n\n"
                "Edit the attached image(s). Return ONLY one image.\n"
                "Do NOT return video/mp4, HTML, or explanations.\n"
                "Output format MUST be exactly one markdown image and nothing else."
                f"{size_hint}"
            )

        return (
            f"{prompt}\n\n"
            "Edit the attached image(s) and return exactly one image."
            " Do NOT return video/mp4, HTML, or explanations."
            f"{size_hint}"
        )

    @staticmethod
    def _build_edit_parts(
        text: str,
        images: list[bytes],
        *,
        image_urls: list[str] | None = None,
    ) -> list[dict]:
        parts: list[dict] = [{"type": "text", "text": text}]
        for idx, img_bytes in enumerate(images):
            if image_urls is not None:
                if idx >= len(image_urls):
                    raise RuntimeError("image_urls 数量少于输入图片数量")
                image_url = str(image_urls[idx] or "").strip()
            else:
                mime, _ext = guess_image_mime_and_ext(img_bytes)
                image_url = f"data:{mime};base64,{base64.b64encode(img_bytes).decode()}"
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url,
                    },
                }
            )
        return parts

    async def _stream_chat_completion(
        self,
        *,
        key: str,
        model: str,
        messages: list[dict],
        extra_body: dict | None,
        log_tag: str,
    ) -> tuple[list[str], list[str], str]:
        payload: dict = {
            "model": model,
            "messages": messages,
            "stream": True,
        }
        if extra_body:
            payload.update(extra_body)
            payload["stream"] = True

        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }

        client = self._get_http_client()
        close_client = False
        if client is None:
            client = httpx.AsyncClient(
                timeout=httpx.Timeout(float(self.timeout)),
                follow_redirects=True,
            )
            close_client = True

        refs: list[str] = []
        videos: list[str] = []
        seen_refs: set[str] = set()
        seen_videos: set[str] = set()
        debug_pieces: list[str] = []
        first_media_hit_logged = False

        def add_ref(value: str | None) -> None:
            if not value or value in seen_refs:
                return
            seen_refs.add(value)
            refs.append(value)

        def add_video(value: str | None) -> None:
            if not value or value in seen_videos:
                return
            seen_videos.add(value)
            videos.append(value)

        async def consume_json_body(resp: httpx.Response) -> tuple[list[str], list[str], str]:
            raw = await resp.aread()
            body_text = raw.decode("utf-8", errors="ignore")
            try:
                obj = json.loads(body_text)
            except Exception:
                return [], [], self._sse_debug_snippet(body_text)

            add_ref(_extract_image_ref_from_content(obj))
            add_video(_extract_video_ref_from_content(obj))
            for s in _iter_strings(obj):
                image_refs, video_refs = _extract_media_refs_from_sse_text(s)
                for ref in image_refs:
                    add_ref(ref)
                for video in video_refs:
                    add_video(video)
                if len(debug_pieces) < 8 and s.strip():
                    debug_pieces.append(self._sse_debug_snippet(s))
            return refs, videos, self._sse_debug_snippet(" ".join(debug_pieces) or body_text)

        t0 = time.time()
        try:
            async with client.stream(
                "POST",
                self._chat_completions_url(),
                headers=headers,
                json=payload,
            ) as resp:
                if resp.status_code != 200:
                    body_text = (await resp.aread()).decode("utf-8", errors="ignore")
                    raise RuntimeError(
                        f"chat stream request failed HTTP {resp.status_code}: {body_text[:300]}"
                    )

                content_type = (resp.headers.get("content-type") or "").lower()
                if "text/event-stream" not in content_type:
                    refs_out, videos_out, debug_snippet = await consume_json_body(resp)
                    logger.info(
                        "[OpenAIChatImage][%s][stream] API response time: %.2fs",
                        log_tag,
                        time.time() - t0,
                    )
                    return refs_out, videos_out, debug_snippet

                buffer = ""
                async for chunk in resp.aiter_text():
                    if not chunk:
                        continue
                    buffer += chunk
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        line = line.strip()
                        if not line or not line.startswith("data:"):
                            continue
                        data_str = line[5:].strip()
                        if not data_str or data_str == "[DONE]":
                            continue
                        try:
                            obj = json.loads(data_str)
                        except Exception:
                            if len(debug_pieces) < 8:
                                debug_pieces.append(self._sse_debug_snippet(data_str))
                            continue

                        add_ref(_extract_image_ref_from_content(obj))
                        add_video(_extract_video_ref_from_content(obj))

                        for s in _iter_strings(obj):
                            image_refs, video_refs = _extract_media_refs_from_sse_text(s)
                            for ref in image_refs:
                                add_ref(ref)
                            for video in video_refs:
                                add_video(video)
                            if len(debug_pieces) < 8 and s.strip():
                                debug_pieces.append(self._sse_debug_snippet(s))

                        if (refs or videos) and not first_media_hit_logged:
                            first_media_hit_logged = True
                            logger.info(
                                "[OpenAIChatImage][%s][stream] first media ref hit in %.2fs, waiting for final output",
                                log_tag,
                                time.time() - t0,
                            )

                logger.info(
                    "[OpenAIChatImage][%s][stream] API response time: %.2fs",
                    log_tag,
                    time.time() - t0,
                )
                return refs, videos, self._sse_debug_snippet(" ".join(debug_pieces))
        finally:
            if close_client:
                await client.aclose()

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
        if s.startswith("//"):
            return s
        if s.startswith("/") and _looks_like_relative_media_ref(s):
            return s
        if _looks_like_relative_media_ref(s):
            return s
        ref = _extract_first_image_ref(s)
        if not ref:
            return None
        if ref.startswith("data:image/"):
            return re.sub(r"\s+", "", ref)
        return ref

    def _absolutize_media_ref(self, ref: str) -> str:
        s = str(ref or "").strip()
        if not s:
            return ""
        if s.startswith(("data:image/", "http://", "https://")):
            rewritten = _rewrite_local_media_url(s, base_url=self.base_url)
            if rewritten != s:
                logger.info(
                    "[OpenAIChatImage] 重写本地结果图地址: %s -> %s",
                    s,
                    rewritten,
                )
            return rewritten

        split = urlsplit(self.base_url)
        origin = f"{split.scheme}://{split.netloc}" if split.scheme and split.netloc else ""
        if s.startswith("//"):
            return f"{split.scheme}:{s}" if split.scheme else f"https:{s}"
        if s.startswith("/"):
            return urljoin(origin + "/", s.lstrip("/")) if origin else s
        if _looks_like_relative_media_ref(s):
            if origin:
                return urljoin(origin + "/", s.lstrip("/"))
            return urljoin(self.base_url.rstrip("/") + "/", s)
        return s

    @staticmethod
    def _needs_file_service_url_retry(exc: Exception) -> bool:
        text = f"{exc!r} {exc}".lower()
        if "data:image" not in text:
            return False
        markers = (
            "unsupported protocol scheme",
            "unsupported url scheme",
            "failed to download",
            "get file base64 from url",
            "invalid image_url",
            "unable to process input image",
        )
        return any(marker in text for marker in markers)

    @staticmethod
    def _needs_images_api_fallback(exc: Exception) -> bool:
        text = f"{exc!r} {exc}".lower()
        markers = (
            "unsupported protocol scheme",
            "unsupported url scheme",
            "get file base64 from url",
            "image_url is required",
            "image_url is required for image edits",
            "missing_image",
            "image edits",
            "input_image",
            "use images api",
            "images/generations",
            "images/edits",
            "does not support chat completions",
            "chat completions not supported",
            "not a chat model",
            "image generation model",
        )
        return any(marker in text for marker in markers)

    def _build_images_api_endpoints(self) -> tuple[str, str]:
        base = normalize_openai_compat_base_url(self.base_url).rstrip("/")
        return f"{base}/images/edits", f"{base}/images/generations"

    async def _edit_via_images_api(
        self,
        *,
        key: str,
        prompt: str,
        images: list[bytes],
        model: str,
        size: str | None,
        resolution: str | None,
        extra_body: dict | None,
    ) -> Path:
        full_edit_url, full_generate_url = self._build_images_api_endpoints()
        helper = OpenAIFullURLBackend(
            imgr=self.imgr,
            full_generate_url=full_generate_url,
            full_edit_url=full_edit_url,
            api_keys=[key],
            default_model=model,
            timeout=self.timeout,
            max_retries=self.max_retries,
            supports_edit=True,
            extra_body=extra_body or None,
        )
        try:
            return await helper.edit(
                prompt,
                images,
                model=model,
                size=size,
                resolution=resolution,
            )
        finally:
            await helper.close()

    async def _register_input_image_urls(self, images: list[bytes]) -> list[str]:
        try:
            from astrbot.api.message_components import Image as AstrImage
        except Exception as e:
            raise RuntimeError("当前 AstrBot 环境缺少图片文件服务能力，无法回退为 URL 输入") from e

        urls: list[str] = []
        for idx, image_bytes in enumerate(images):
            saved_path = await self.imgr.save_image(image_bytes)
            img_comp = AstrImage.fromFileSystem(str(saved_path))
            register = getattr(img_comp, "register_to_file_service", None)
            if not callable(register):
                raise RuntimeError("当前 AstrBot Image 组件不支持 register_to_file_service")
            url = await _resolve_awaitable(register())
            url_text = str(url or "").strip()
            if not url_text:
                raise RuntimeError(f"第 {idx + 1} 张输入图注册到文件服务失败")
            urls.append(url_text)
        return urls

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
                image_refs, _video_refs = _extract_media_refs_from_sse_text(s)
                for ref in image_refs:
                    add_ref(ref)
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
            dumped = (
                await _resolve_awaitable(model_dump()) if callable(model_dump) else None
            )
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
                for s in _iter_strings(content):
                    _image_refs, video_refs = _extract_media_refs_from_sse_text(s)
                    if video_refs:
                        return video_refs[0]
        except Exception:
            pass

        try:
            model_dump = getattr(resp, "model_dump", None)  # type: ignore[attr-defined]
            dumped = (
                await _resolve_awaitable(model_dump()) if callable(model_dump) else None
            )
        except Exception:
            dumped = None
        if dumped is not None:
            url = _extract_video_ref_from_content(dumped)
            if url:
                return url
            for s in _iter_strings(dumped):
                _image_refs, video_refs = _extract_media_refs_from_sse_text(s)
                if video_refs:
                    return video_refs[0]
        return None

    async def _save_single_ref(self, ref: str, *, debug_snippet: str = "") -> Path:
        if not ref:
            raise RuntimeError(
                f"chat 返回未包含图片（需 markdown/data:image/url）：{debug_snippet}"
            )
        ref = self._absolutize_media_ref(ref)

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
            if _looks_like_placeholder_image_bytes(image_bytes):
                raise RuntimeError(
                    "chat 返回了疑似占位图片（通常是网关被强制输出 data:image 时伪造的 1x1/极小图）"
                    f"（bytes={len(image_bytes)}）：{debug_snippet}"
                )
            return await self.imgr.save_image(image_bytes)

        if ref.startswith("http://") or ref.startswith("https://"):
            if _looks_like_video_url(ref):
                raise RuntimeError(
                    f"chat 返回了视频而不是图片：{ref}（如果想要视频请用 /视频；如果想要图片请换模型或改用 images 接口）"
                )
            try:
                return await self.imgr.download_image(ref)
            except Exception as exc:
                if not self._should_trust_result_url_download(ref, exc):
                    raise
                logger.warning(
                    "[OpenAIChatImage] 结果图 URL 被安全策略拦截，改用受限直连下载: %s",
                    exc,
                )
                return await self._download_trusted_result_url(ref)

        raise RuntimeError("chat 返回的图片引用格式不支持")

    def _should_trust_result_url_download(self, ref: str, exc: Exception) -> bool:
        text = f"{exc!r} {exc}".lower()
        if "disallowed resolved ip address" not in text:
            return False
        try:
            origin = self._origin_of_url(ref)
        except Exception:
            origin = ""
        return bool(origin and origin in self._trusted_result_origins)

    @staticmethod
    def _origin_of_url(url: str) -> str:
        parts = urlsplit(str(url or "").strip())
        if not parts.scheme or not parts.netloc:
            return ""
        return f"{parts.scheme}://{parts.netloc}".rstrip("/")

    async def _download_trusted_result_url(self, ref: str) -> Path:
        origin = self._origin_of_url(ref)
        if origin not in self._trusted_result_origins:
            raise RuntimeError(f"result image URL is not trusted: {origin or ref}")

        client = self._get_http_client()
        close_client = False
        if client is None:
            client = httpx.AsyncClient(
                timeout=httpx.Timeout(float(self.timeout)),
            )
            close_client = True

        max_bytes = 50 * 1024 * 1024
        max_redirects = 5
        policy = self._trusted_result_url_policy()
        try:
            try:
                max_bytes = max(
                    256 * 1024,
                    min(
                        int(getattr(self.imgr, "_media_max_image_bytes", max_bytes) or max_bytes),
                        200 * 1024 * 1024,
                    ),
                )
            except Exception:
                max_bytes = 50 * 1024 * 1024
            try:
                max_redirects = max(
                    0,
                    min(
                        int(
                            getattr(self.imgr, "_media_max_redirects", max_redirects)
                            or max_redirects
                        ),
                        10,
                    ),
                )
            except Exception:
                max_redirects = 5

            current = ref
            redirects = 0
            while True:
                await ensure_url_allowed(current, policy=policy)
                async with client.stream(
                    "GET",
                    current,
                    headers={"Accept": "image/*,*/*;q=0.8"},
                    follow_redirects=False,
                ) as resp:
                    if resp.status_code in {301, 302, 303, 307, 308}:
                        if redirects >= max_redirects:
                            raise RuntimeError("too many redirects while downloading trusted image result")
                        location = (resp.headers.get("location") or "").strip()
                        if not location:
                            raise RuntimeError("redirect without location while downloading trusted image result")
                        current = urljoin(current, location)
                        redirects += 1
                        continue
                    if resp.status_code != 200:
                        body_text = (await resp.aread()).decode("utf-8", errors="ignore")
                        raise RuntimeError(
                            f"trusted image result download failed HTTP {resp.status_code}: {body_text[:300]}"
                        )
                    content_type = (resp.headers.get("content-type") or "").lower()
                    if content_type and not content_type.startswith("image/"):
                        raise RuntimeError(
                            f"trusted image result response is not an image: content-type={content_type or 'unknown'}"
                        )
                    total = 0
                    chunks: list[bytes] = []
                    async for chunk in resp.aiter_bytes():
                        if not chunk:
                            continue
                        total += len(chunk)
                        if total > max_bytes:
                            raise RuntimeError("trusted image result is too large to download")
                        chunks.append(chunk)
                    return await self.imgr.save_image(b"".join(chunks))
        finally:
            if close_client:
                await client.aclose()

    async def _save_from_ref(
        self,
        ref: str,
        *,
        debug_snippet: str = "",
        fallback_refs: list[str] | None = None,
    ) -> Path:
        candidates: list[str] = [str(ref or "").strip()]
        for extra in fallback_refs or []:
            s = str(extra or "").strip()
            if s and s not in candidates:
                candidates.append(s)

        if len(candidates) > 1:
            preferred: list[str] = []
            seen_preferred: set[str] = set()
            for cand in reversed(candidates):
                if not cand or cand in seen_preferred:
                    continue
                seen_preferred.add(cand)
                preferred.append(cand)
            candidates = preferred

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

        eb = {}
        eb.update(self.extra_body)
        eb.update(extra_body or {})
        eb = self._apply_gemini_image_config(
            eb,
            model=final_model,
            size=size,
            resolution=resolution,
        )

        stream_error: Exception | None = None
        if self._should_try_stream("generate"):
            stream_messages = [
                {
                    "role": "user",
                    "content": self._build_generate_prompt(
                        prompt,
                        size=size,
                        resolution=resolution,
                        strict_format=False,
                    ),
                }
            ]
            try:
                refs, videos, debug_snippet = await self._run_chat_request_with_retries(
                    lambda: self._stream_chat_completion(
                        key=key,
                        model=final_model,
                        messages=stream_messages,
                        extra_body=eb or None,
                        log_tag="generate",
                    ),
                    model=final_model,
                    log_tag="generate",
                    input_mode="prompt_only",
                )
                if refs:
                    return await self._save_from_ref(
                        refs[0], debug_snippet=debug_snippet, fallback_refs=refs[1:]
                    )
                if videos:
                    raise RuntimeError(
                        f"chat 返回了视频而不是图片：{videos[0]}（如果想要视频请用 /视频；如果想要图片请换模型或改用 images 接口）"
                    )
                stream_error = RuntimeError("stream 未解析到图片引用")
                logger.warning(
                    "[OpenAIChatImage][generate] 流式模式未解析到图片，回退非流式"
                )
            except Exception as e:
                stream_error = e
                logger.warning(
                    "[OpenAIChatImage][generate] 流式模式失败，回退非流式: %s", e
                )

        user_text = self._build_generate_prompt(
            prompt,
            size=size,
            resolution=resolution,
            strict_format=True,
        )

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
        try:
            return await self._save_from_ref(
                ref or "", debug_snippet=debug_snippet, fallback_refs=refs[1:]
            )
        except Exception as e:
            if stream_error is not None:
                raise RuntimeError(
                    f"{e}；且此前流式兜底也失败：{stream_error}"
                ) from e
            raise

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

        eb = {}
        eb.update(self.extra_body)
        eb.update(extra_body or {})
        eb = self._apply_gemini_image_config(
            eb,
            model=final_model,
            size=size,
            resolution=resolution,
        )

        prefetched_image_urls: list[str] | None = None
        if self._prefer_file_service_url_input:
            prefetched_image_urls = await self._register_input_image_urls(images)

        stream_error: Exception | None = None
        if self._should_try_stream("edit"):
            stream_parts = self._build_edit_parts(
                self._build_edit_text(
                    prompt,
                    size=size,
                    resolution=resolution,
                    strict_format=False,
                ),
                images,
                image_urls=prefetched_image_urls,
            )
            try:
                refs, videos, debug_snippet = await self._run_chat_request_with_retries(
                    lambda: self._stream_chat_completion(
                        key=key,
                        model=final_model,
                        messages=[{"role": "user", "content": stream_parts}],
                        extra_body=eb or None,
                        log_tag="edit",
                    ),
                    model=final_model,
                    log_tag="edit",
                    input_mode="data_url"
                    if prefetched_image_urls is None
                    else "file_service_url",
                )
                if refs:
                    return await self._save_from_ref(
                        refs[0], debug_snippet=debug_snippet, fallback_refs=refs[1:]
                    )
                if videos:
                    raise RuntimeError(
                        f"chat 返回了视频而不是图片：{videos[0]}（如果想要视频请用 /视频；如果想要图片请换模型或改用 images 接口）"
                    )
                stream_error = RuntimeError("stream 未解析到图片引用")
                logger.warning("[OpenAIChatImage][edit] 流式模式未解析到图片，回退非流式")
            except Exception as e:
                stream_error = e
                logger.warning("[OpenAIChatImage][edit] 流式模式失败，回退非流式: %s", e)

        parts = self._build_edit_parts(
            self._build_edit_text(
                prompt,
                size=size,
                resolution=resolution,
                strict_format=True,
            ),
            images,
            image_urls=prefetched_image_urls,
        )

        async def request_edit(parts_payload: list[dict], *, input_mode: str) -> object:
            nonlocal client
            t0 = time.time()
            try:
                response = await client.chat.completions.create(
                    model=final_model,
                    messages=[{"role": "user", "content": parts_payload}],
                    extra_body=eb or None,
                )
            except Exception as e:
                if _is_client_closed_error(e):
                    logger.warning("[OpenAIChatImage][edit] client 已关闭，重建后重试一次")
                    client = await self._recreate_client(key)
                    response = await client.chat.completions.create(
                        model=final_model,
                        messages=[{"role": "user", "content": parts_payload}],
                        extra_body=eb or None,
                    )
                else:
                    logger.error(
                        "[OpenAIChatImage][edit] API 调用失败，base_url=%s，input_mode=%s，耗时: %.2fs: %s",
                        self.base_url,
                        input_mode,
                        time.time() - t0,
                        e,
                    )
                    raise
            logger.info(
                "[OpenAIChatImage][edit] API 响应耗时: %.2fs, input_mode=%s",
                time.time() - t0,
                input_mode,
            )
            return response

        input_mode = "data_url"
        try:
            resp = await self._run_chat_request_with_retries(
                lambda: request_edit(parts, input_mode=input_mode),
                model=final_model,
                log_tag="edit",
                input_mode=input_mode,
            )
        except Exception as e:
            images_api_error: Exception | None = None
            if self._needs_images_api_fallback(e):
                logger.warning(
                    "[OpenAIChatImage][edit] chat 改图不兼容当前网关，改用 Images API 兜底: %s",
                    e,
                )
                try:
                    return await self._edit_via_images_api(
                        key=key,
                        prompt=prompt,
                        images=images,
                        model=final_model,
                        size=size,
                        resolution=resolution,
                        extra_body=eb,
                    )
                except Exception as images_exc:
                    images_api_error = images_exc
                    logger.warning(
                        "[OpenAIChatImage][edit] Images API 兜底失败，继续尝试文件服务 URL: %s",
                        images_exc,
                    )

            if not self._needs_file_service_url_retry(e):
                if images_api_error is not None:
                    raise RuntimeError(
                        f"{e}；且 Images API 兜底也失败：{images_api_error}"
                    ) from e
                raise
            self._prefer_file_service_url_input = True
            logger.warning(
                "[OpenAIChatImage][edit] data:image 输入不被目标网关接受，改用文件服务 URL 重试: %s",
                e,
            )
            input_mode = "file_service_url"
            try:
                image_urls = prefetched_image_urls or await self._register_input_image_urls(images)
            except Exception as file_service_exc:
                if images_api_error is not None:
                    raise RuntimeError(
                        f"{file_service_exc}；且 Images API 兜底也失败：{images_api_error}"
                    ) from file_service_exc
                raise
            parts = self._build_edit_parts(
                self._build_edit_text(
                    prompt,
                    size=size,
                    resolution=resolution,
                    strict_format=True,
                ),
                images,
                image_urls=image_urls,
            )
            resp = await self._run_chat_request_with_retries(
                lambda: request_edit(parts, input_mode=input_mode),
                model=final_model,
                log_tag="edit",
                input_mode=input_mode,
            )

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

        if not ref:
            video_url = await self._extract_video_ref_from_response(resp)
            if video_url:
                raise RuntimeError(
                    f"chat 返回了视频而不是图片：{video_url}（如果想要视频请用 /视频；如果想要图片请换模型或改用 images 接口）"
                )
        try:
            return await self._save_from_ref(
                ref or "", debug_snippet=debug_snippet, fallback_refs=refs[1:]
            )
        except Exception as e:
            if stream_error is not None:
                raise RuntimeError(
                    f"{e}；且此前流式兜底也失败：{stream_error}"
                ) from e
            raise
