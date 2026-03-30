from __future__ import annotations

import base64
import re


def guess_image_mime_and_ext(image_bytes: bytes) -> tuple[str, str]:
    """Best-effort guess for image mime/ext using magic bytes.

    Returns:
        (mime, ext) where ext does not include the leading dot.
    """
    if not image_bytes:
        return "image/jpeg", "jpg"

    b = image_bytes

    # JPEG
    if len(b) >= 3 and b[0:3] == b"\xff\xd8\xff":
        return "image/jpeg", "jpg"

    # PNG
    if len(b) >= 8 and b[0:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png", "png"

    # GIF
    if len(b) >= 6 and (b[0:6] == b"GIF87a" or b[0:6] == b"GIF89a"):
        return "image/gif", "gif"

    # WEBP (RIFF....WEBP)
    if len(b) >= 12 and b[0:4] == b"RIFF" and b[8:12] == b"WEBP":
        return "image/webp", "webp"

    return "image/jpeg", "jpg"


def guess_image_mime_and_ext_strict(image_bytes: bytes) -> tuple[str, str] | None:
    """Strictly guess image mime/ext using magic bytes.

    Returns None when the payload does not look like a supported image.
    """
    if not image_bytes:
        return None

    b = image_bytes

    if len(b) >= 3 and b[0:3] == b"\xff\xd8\xff":
        return "image/jpeg", "jpg"

    if len(b) >= 8 and b[0:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png", "png"

    if len(b) >= 6 and (b[0:6] == b"GIF87a" or b[0:6] == b"GIF89a"):
        return "image/gif", "gif"

    if len(b) >= 12 and b[0:4] == b"RIFF" and b[8:12] == b"WEBP":
        return "image/webp", "webp"

    return None


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


def decode_base64_image_payload(payload: str) -> bytes:
    """Decode raw/data-url/base64:// image payload into verified image bytes."""
    s = str(payload or "").strip()
    if not s:
        raise ValueError("empty image payload")

    if s.startswith("data:image/"):
        _header, sep, body = s.partition(",")
        if not sep or not body.strip():
            raise ValueError("data:image payload missing base64 body")
        s = body
    elif s.startswith("base64://"):
        s = s.removeprefix("base64://")

    raw = _decode_base64_bytes(s)
    if not raw:
        raise ValueError("base64 decode failed")

    if guess_image_mime_and_ext_strict(raw) is None:
        raise ValueError("decoded payload is not a supported image")

    return raw
