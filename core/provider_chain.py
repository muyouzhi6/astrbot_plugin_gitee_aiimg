from __future__ import annotations

from typing import Any


def as_dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def as_list(value: Any) -> list:
    return value if isinstance(value, list) else []


def parse_chain_item(item: object) -> tuple[str, str] | None:
    if isinstance(item, str):
        pid = item.strip()
        return (pid, "") if pid else None
    if not isinstance(item, dict):
        return None

    nested_provider = item.get("provider")
    nested = nested_provider if isinstance(nested_provider, dict) else None
    source = nested or item

    pid = str(
        source.get("provider_id")
        or source.get("id")
        or source.get("provider")
        or source.get("backend")
        or source.get("value")
        or ""
    ).strip()
    if not pid:
        return None

    out_override = str(
        item.get("output")
        or source.get("output")
        or item.get("default_output")
        or source.get("default_output")
        or ""
    ).strip()
    return pid, out_override


def candidates_from_chain(raw_chain: list) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for item in raw_chain:
        parsed = parse_chain_item(item)
        if not parsed:
            continue
        pid, out_override = parsed
        if pid in seen:
            continue
        seen.add(pid)
        out.append((pid, out_override))
    return out
