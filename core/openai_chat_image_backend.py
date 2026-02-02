from __future__ import annotations

import base64
import re
import time
from pathlib import Path

from openai import AsyncOpenAI

from astrbot.api import logger

from .image_format import guess_image_mime_and_ext
from .openai_compat_backend import normalize_openai_compat_base_url

_MARKDOWN_IMAGE_RE = re.compile(r"!\[.*?\]\((.*?)\)")
_DATA_IMAGE_RE = re.compile(r"(data:image/[^\\s)]+)")


def _extract_first_image_ref(text: str) -> str | None:
    s = (text or "").strip()
    if not s:
        return None
    m = _MARKDOWN_IMAGE_RE.search(s)
    if m:
        return m.group(1).strip()
    m = _DATA_IMAGE_RE.search(s)
    if m:
        return m.group(1).strip()
    if s.startswith("http://") or s.startswith("https://"):
        return s
    return None


def _iter_strings(obj: object) -> list[str]:
    out: list[str] = []

    def walk(x: object) -> None:
        if x is None:
            return
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

        # Some gateways return {"url": "..."} / {"image": "data:image/..."} etc.
        for k in ("url", "image", "image_url", "data"):
            v = content.get(k)
            if isinstance(v, str):
                ref = _extract_first_image_ref(v)
                if ref:
                    return ref
            ref = _extract_image_ref_from_content(v)
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
    ):
        self.imgr = imgr
        self.base_url = normalize_openai_compat_base_url(base_url)
        self.api_keys = [str(k).strip() for k in (api_keys or []) if str(k).strip()]
        self.timeout = int(timeout or 120)
        self.max_retries = int(max_retries or 2)
        self.default_model = str(default_model or "").strip()
        self.supports_edit = bool(supports_edit)
        self.extra_body = extra_body or {}

        self._key_index = 0
        self._clients: dict[str, AsyncOpenAI] = {}

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

    def _extract_image_ref_from_response(self, resp: object) -> str | None:
        # 1) Preferred: first choice message content.
        try:
            choice0 = resp.choices[0]  # type: ignore[attr-defined]
            msg = getattr(choice0, "message", None)
            if msg is not None:
                ref = _extract_image_ref_from_content(getattr(msg, "content", None))
                if ref:
                    return ref

                tool_calls = getattr(msg, "tool_calls", None)
                ref = _extract_image_ref_from_content(tool_calls)
                if ref:
                    return ref
        except Exception:
            pass

        # 2) Fallback: scan model dump (dict/list) for any data:image / markdown / url.
        try:
            dumped = resp.model_dump()  # type: ignore[attr-defined]
        except Exception:
            dumped = None
        if dumped is not None:
            ref = _extract_image_ref_from_content(dumped)
            if ref:
                return ref

        return None

    async def _save_from_ref(self, ref: str, *, debug_snippet: str = "") -> Path:
        if not ref:
            raise RuntimeError(
                f"chat 返回未包含图片（需 markdown/data:image/url）：{debug_snippet}"
            )

        if ref.startswith("data:image/"):
            try:
                _header, b64_data = ref.split(",", 1)
            except ValueError:
                raise RuntimeError("chat 返回 data:image 但缺少 base64 数据") from None
            image_bytes = base64.b64decode((b64_data or "").strip())
            return await self.imgr.save_image(image_bytes)

        if ref.startswith("http://") or ref.startswith("https://"):
            return await self.imgr.download_image(ref)

        raise RuntimeError("chat 返回的图片引用格式不支持")

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
            "Return ONLY one markdown image in the format:\n"
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
            if "client has been closed" in str(e).lower():
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

        ref = self._extract_image_ref_from_response(resp)
        debug_snippet = ""
        try:
            debug_snippet = (
                str(getattr(resp.choices[0].message, "content", "")).strip().replace("\n", " ")[:200]  # type: ignore[attr-defined]
            )
        except Exception:
            pass

        logger.info("[OpenAIChatImage][generate] API 响应耗时: %.2fs", time.time() - t0)
        return await self._save_from_ref(ref or "", debug_snippet=debug_snippet)

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
            "Edit the attached image(s). Return ONLY one markdown image in the format:\n"
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
            if "client has been closed" in str(e).lower():
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

        ref = self._extract_image_ref_from_response(resp)
        debug_snippet = ""
        try:
            debug_snippet = (
                str(getattr(resp.choices[0].message, "content", "")).strip().replace("\n", " ")[:200]  # type: ignore[attr-defined]
            )
        except Exception:
            pass

        logger.info("[OpenAIChatImage][edit] API 响应耗时: %.2fs", time.time() - t0)
        return await self._save_from_ref(ref or "", debug_snippet=debug_snippet)
