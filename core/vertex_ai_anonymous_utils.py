from __future__ import annotations

import random
import re
from typing import Any
from urllib.parse import parse_qs, urlparse

from .gitee_sizes import size_to_ratio

DEFAULT_OPERATION_NAME = "StreamGenerateContentAnonymous"

RECAPTCHA_SITE_KEY = "6LdCjtspAAAAAMcV4TGdWLJqRTEk1TfpdLqEnKdj"
RECAPTCHA_CO = "aHR0cHM6Ly9jb25zb2xlLmNsb3VkLmdvb2dsZS5jb206NDQz"
RECAPTCHA_HL = "zh-CN"
RECAPTCHA_V = "jdMmXeCQEkPbnFDy9T04NbgJ"
RECAPTCHA_VH = "6581054572"
RECAPTCHA_TOKEN_RETRIES = 3

RANDOM_CB_LEN = 10
ANCHOR_MS = 20000
EXECUTE_MS = 15000

TEMPERATURE = 1
TOP_P = 0.95
MAX_OUTPUT_TOKENS = 32768

ANCHOR_TOKEN_RE = re.compile(r'id="recaptcha-token"[^>]*value="([^"]+)"')
RRESP_RE = re.compile(r'rresp","(.*?)"')


class RecaptchaExpiredError(RuntimeError):
    pass


class NonRetryableError(RuntimeError):
    pass


def _as_str(value: Any) -> str:
    return str(value or "").strip()


def _looks_like_px_size(value: str) -> bool:
    return bool(re.fullmatch(r"\d{2,5}x\d{2,5}", (value or "").strip().lower()))


def size_to_aspect_ratio(size: str | None) -> str | None:
    if not size:
        return None
    s = (size or "").strip().lower()
    if not _looks_like_px_size(s):
        return None
    ratio = size_to_ratio(s)
    return ratio or None


def build_anchor_url(recaptcha_base_api: str) -> str:
    cb = "".join(
        random.choice("abcdefghijklmnopqrstuvwxyz0123456789") for _ in range(RANDOM_CB_LEN)
    )
    return (
        f"{recaptcha_base_api}/recaptcha/enterprise/anchor"
        f"?ar=1&k={RECAPTCHA_SITE_KEY}&co={RECAPTCHA_CO}&hl={RECAPTCHA_HL}"
        f"&v={RECAPTCHA_V}&size=invisible&anchor-ms={ANCHOR_MS}&execute-ms={EXECUTE_MS}"
        f"&cb={cb}"
    )


def build_reload_url(recaptcha_base_api: str) -> str:
    return f"{recaptcha_base_api}/recaptcha/enterprise/reload?k={RECAPTCHA_SITE_KEY}"


def parse_anchor_token(html: str) -> str | None:
    m = ANCHOR_TOKEN_RE.search(html or "")
    return m.group(1) if m else None


def parse_rresp(text: str) -> str | None:
    m = RRESP_RE.search(text or "")
    return m.group(1) if m else None


def extract_query_params(url: str) -> dict[str, str]:
    parsed = urlparse(url)
    qs = parse_qs(parsed.query)
    out: dict[str, str] = {}
    for k in ("v", "k", "co", "hl"):
        if k in qs and qs[k]:
            out[k] = str(qs[k][0])
    return out


def extract_images_from_graphql_payload(payload: Any) -> list[tuple[str, str]]:
    if not isinstance(payload, list):
        raise RuntimeError(f"Unexpected response type: {type(payload).__name__}")

    out: list[tuple[str, str]] = []
    for elem in payload:
        if not isinstance(elem, dict):
            continue
        for item in (elem.get("results") or []):
            if not isinstance(item, dict):
                continue
            errors = item.get("errors") or []
            if isinstance(errors, list) and errors:
                err = errors[0] if isinstance(errors[0], dict) else {}
                status = (
                    (err.get("extensions") or {}).get("status") or {}
                ).get("code")
                msg = _as_str(err.get("message"))
                if status == 3:
                    raise RecaptchaExpiredError(msg or "recaptcha expired")
                raise RuntimeError(f"Vertex AI Anonymous error {status}: {msg}")

            candidates = (item.get("data") or {}).get("candidates") or []
            for cand in candidates:
                if not isinstance(cand, dict):
                    continue
                if _as_str(cand.get("finishReason")) != "STOP":
                    reason = _as_str(cand.get("finishReason"))
                    raise NonRetryableError(f"Vertex AI Anonymous finishReason={reason}")
                parts = ((cand.get("content") or {}).get("parts")) or []
                for part in parts:
                    if not isinstance(part, dict):
                        continue
                    inline = part.get("inlineData")
                    if not isinstance(inline, dict):
                        continue
                    b64 = _as_str(inline.get("data"))
                    mime = _as_str(inline.get("mimeType")) or "image/png"
                    if b64:
                        out.append((mime, b64))

    if not out:
        raise RuntimeError("Vertex AI Anonymous 响应中未包含图片数据")
    return out
