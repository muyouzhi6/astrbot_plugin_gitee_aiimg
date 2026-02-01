from __future__ import annotations


def guess_image_mime_and_ext(image_bytes: bytes) -> tuple[str, str]:
    """Best-effort guess for image mime/ext using magic bytes.

    Returns:
        (mime, ext) where ext does not include the leading dot.
    """
    if not image_bytes:
        return "image/jpeg", "jpg"

    b = image_bytes

    # JPEG
    if len(b) >= 3 and b[0:3] == b"\xFF\xD8\xFF":
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

