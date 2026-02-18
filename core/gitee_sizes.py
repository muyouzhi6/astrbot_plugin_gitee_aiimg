from __future__ import annotations

import re
from collections.abc import Iterable
from math import gcd


def normalize_size_text(size: str | None) -> str:
    return str(size or "").strip().lower().replace("×", "x")


GITEE_SUPPORTED_RATIOS: dict[str, list[str]] = {
    "1:1": ["256x256", "512x512", "1024x1024", "2048x2048"],
    "4:3": ["1152x896", "2048x1536"],
    "3:4": ["768x1024", "1536x2048"],
    "3:2": ["2048x1360"],
    "2:3": ["1360x2048"],
    "16:9": ["1024x576", "2048x1152"],
    "9:16": ["576x1024", "1152x2048"],
}

_CANONICAL_RATIO_VALUES = {
    ratio: (int(ratio.split(":")[0]) / int(ratio.split(":")[1]))
    for ratio in GITEE_SUPPORTED_RATIOS
}


def build_supported_sizes() -> list[str]:
    sizes: list[str] = []
    for size_list in GITEE_SUPPORTED_RATIOS.values():
        for size in size_list:
            s = normalize_size_text(size)
            if s and s not in sizes:
                sizes.append(s)
    return sizes


GITEE_SUPPORTED_SIZES = build_supported_sizes()


def normalize_ratio_default_sizes(
    raw: dict | None,
    *,
    supported_ratios: dict[str, list[str]] | None = None,
) -> dict[str, str]:
    """Normalize ratio->size overrides and drop unsupported entries."""
    if not isinstance(raw, dict):
        return {}
    supported = supported_ratios or GITEE_SUPPORTED_RATIOS
    out: dict[str, str] = {}
    for ratio, size in raw.items():
        r = str(ratio or "").strip()
        s = normalize_size_text(size)
        if not r or not s:
            continue
        sizes = [
            normalize_size_text(item) for item in supported.get(r, []) if item is not None
        ]
        if sizes and s in sizes:
            out[r] = s
    return out


def resolve_ratio_size(
    ratio: str | None,
    *,
    overrides: dict[str, str] | None = None,
    supported_ratios: dict[str, list[str]] | None = None,
) -> tuple[str, str | None]:
    """Resolve ratio to size with optional overrides.

    Returns (size, warning). warning is None when no fallback was needed.
    """
    ratio_text = str(ratio or "").strip()
    supported = supported_ratios or GITEE_SUPPORTED_RATIOS
    sizes = [
        normalize_size_text(item)
        for item in supported.get(ratio_text, [])
        if normalize_size_text(item)
    ]
    if not sizes:
        return "", f"unsupported ratio '{ratio_text}'"

    override = normalize_size_text((overrides or {}).get(ratio_text, ""))
    if override:
        if override in sizes:
            return override, None
        return sizes[0], f"ratio_default_sizes[{ratio_text}]='{override}' not supported"

    return sizes[0], None


def _canonicalize_ratio_text(ratio: str | None) -> str | None:
    if not ratio:
        return None
    m = re.fullmatch(r"(\d{1,4}):(\d{1,4})", str(ratio).strip())
    if not m:
        return None
    a = int(m.group(1))
    b = int(m.group(2))
    if a <= 0 or b <= 0:
        return None
    value = a / b

    # 优先精确命中
    exact = f"{a}:{b}"
    if exact in _CANONICAL_RATIO_VALUES:
        return exact

    # 近似命中官方比例（如 2048x1360 ≈ 3:2）
    best_key = None
    best_diff = 10.0
    for key, v in _CANONICAL_RATIO_VALUES.items():
        d = abs(v - value)
        if d < best_diff:
            best_key = key
            best_diff = d

    if best_key is not None and best_diff <= 0.03:
        return best_key
    return exact


def size_to_ratio(size: str | None) -> str | None:
    s = normalize_size_text(size)
    if not s:
        return None
    m = re.fullmatch(r"(\d{2,5})x(\d{2,5})", s)
    if not m:
        return None
    w = int(m.group(1))
    h = int(m.group(2))
    if w <= 0 or h <= 0:
        return None
    g = gcd(w, h)
    raw = f"{w // g}:{h // g}"
    return _canonicalize_ratio_text(raw)


def ratio_defaults_from_sizes(sizes: Iterable[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for size in sizes:
        ratio = size_to_ratio(size)
        if ratio and ratio not in out:
            out[ratio] = size
    return out
