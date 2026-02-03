from __future__ import annotations


def parse_output(output: str | None) -> tuple[str | None, str | None]:
    """Parse user output into (size, resolution).

    size: "2048x2048"
    resolution: "4K" / "2K" / "1K"
    """
    s = str(output or "").strip()
    if not s:
        return None, None
    if "x" in s.lower():
        return s, None
    return None, s
