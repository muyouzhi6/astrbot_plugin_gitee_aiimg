from __future__ import annotations

import asyncio
import ipaddress
import socket
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

from astrbot.api import logger


def _origin(url: str) -> str:
    try:
        parts = urlsplit(str(url or "").strip())
    except Exception:
        return ""
    if not parts.scheme or not parts.netloc:
        return ""
    return f"{parts.scheme}://{parts.netloc}".rstrip("/")


def _hostname(url: str) -> str:
    try:
        parts = urlsplit(str(url or "").strip())
    except Exception:
        return ""
    return (parts.hostname or "").strip().lower()


def _is_ip_literal(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
        return True
    except Exception:
        return False


def _is_public_ip(ip: ipaddress._BaseAddress) -> bool:  # type: ignore[attr-defined]
    # ipaddress uses is_global for "publicly routable" in most cases.
    try:
        return bool(getattr(ip, "is_global"))
    except Exception:
        return False


async def _resolve_host_ips(host: str, *, timeout_seconds: float = 2.0) -> list[str]:
    host = (host or "").strip()
    if not host:
        return []

    def _call() -> list[str]:
        infos = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
        out: list[str] = []
        for _family, _socktype, _proto, _canonname, sockaddr in infos:
            try:
                ip = sockaddr[0]
            except Exception:
                continue
            if ip and ip not in out:
                out.append(ip)
        return out

    try:
        return await asyncio.wait_for(asyncio.to_thread(_call), timeout=timeout_seconds)
    except Exception as e:
        logger.debug("[net_safety] dns resolve failed: host=%s err=%s", host, e)
        return []


@dataclass(frozen=True)
class URLFetchPolicy:
    allow_private: bool = False
    trusted_origins: frozenset[str] = frozenset()
    allowed_hosts: frozenset[str] = frozenset()
    dns_timeout_seconds: float = 2.0


def collect_trusted_origins(config: dict) -> set[str]:
    out: set[str] = set()
    if not isinstance(config, dict):
        return out
    providers = config.get("providers", [])
    if not isinstance(providers, list):
        return out

    for p in providers:
        if not isinstance(p, dict):
            continue
        for key in (
            "base_url",
            "api_url",
            "server_url",
            "full_generate_url",
            "full_edit_url",
        ):
            val = str(p.get(key) or "").strip()
            if not val:
                continue
            o = _origin(val)
            if o:
                out.add(o)
    return out


async def ensure_url_allowed(url: str, *, policy: URLFetchPolicy) -> None:
    """Raise RuntimeError if url is not allowed to fetch under policy."""
    s = str(url or "").strip()
    if not s:
        raise RuntimeError("Empty URL")

    parts = urlsplit(s)
    if parts.scheme not in {"http", "https"}:
        raise RuntimeError("Unsupported URL scheme")
    if not parts.netloc:
        raise RuntimeError("Invalid URL (missing host)")

    origin = _origin(s)
    if origin and origin in policy.trusted_origins:
        return

    host = (parts.hostname or "").strip().lower()
    if not host:
        raise RuntimeError("Invalid URL (missing hostname)")

    if host in policy.allowed_hosts:
        return

    if host in {"localhost"} or host.endswith(".localhost") or host.endswith(".local"):
        raise RuntimeError("Disallowed hostname")

    if policy.allow_private:
        return

    # IP literal: check directly
    if _is_ip_literal(host):
        ip = ipaddress.ip_address(host)
        if _is_public_ip(ip):
            return
        raise RuntimeError("Disallowed IP address")

    # Hostname: resolve and ensure all IPs are public
    ips = await _resolve_host_ips(
        host, timeout_seconds=float(policy.dns_timeout_seconds)
    )
    if not ips:
        # Fail-closed for SSRF safety
        raise RuntimeError("DNS resolve failed")

    for ip_s in ips:
        try:
            ip = ipaddress.ip_address(ip_s)
        except Exception:
            raise RuntimeError("DNS returned invalid IP") from None
        if not _is_public_ip(ip):
            raise RuntimeError("Disallowed resolved IP address")


def read_network_policy(config: dict) -> dict[str, Any]:
    if not isinstance(config, dict):
        return {}
    net = config.get("network", {})
    return net if isinstance(net, dict) else {}
