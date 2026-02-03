from __future__ import annotations

import time
from pathlib import Path

from astrbot.api import logger

from .output_spec import parse_output
from .provider_registry import ProviderRegistry


def _as_dict(value: object) -> dict:
    return value if isinstance(value, dict) else {}


def _as_list(value: object) -> list:
    return value if isinstance(value, list) else []


class ImageDrawService:
    """Text-to-Image router for v4 config (provider chain)."""

    def __init__(
        self,
        config: dict,
        imgr,
        data_dir: Path,
        *,
        registry: ProviderRegistry | None = None,
    ):
        self.config = config if isinstance(config, dict) else {}
        self.imgr = imgr
        self.data_dir = Path(data_dir)
        self.registry = registry or ProviderRegistry(
            self.config, imgr=self.imgr, data_dir=self.data_dir
        )

    def _feature_conf(self) -> dict:
        feats = _as_dict(self.config.get("features"))
        return _as_dict(feats.get("draw"))

    def _default_output(self) -> str:
        return str(self._feature_conf().get("default_output") or "").strip()

    def _chain(self) -> list[dict]:
        return [
            x
            for x in _as_list(self._feature_conf().get("chain"))
            if isinstance(x, dict)
        ]

    def _candidate_ids(self) -> list[str]:
        out: list[str] = []
        for item in self._chain():
            pid = str(item.get("provider_id") or "").strip()
            if pid and pid not in out:
                out.append(pid)
        return out

    async def close(self) -> None:
        await self.registry.close()

    async def generate(
        self,
        prompt: str,
        *,
        size: str | None = None,
        resolution: str | None = None,
        provider_id: str | None = None,
    ) -> Path:
        feature = self._feature_conf()
        if not bool(feature.get("enabled", True)):
            raise RuntimeError(
                "Text-to-image is disabled (features.draw.enabled=false)"
            )

        candidates: list[tuple[str, str]] = []
        if provider_id:
            candidates = [(str(provider_id).strip(), "")]
        else:
            for item in self._chain():
                pid = str(item.get("provider_id") or "").strip()
                out_override = str(item.get("output") or "").strip()
                if pid:
                    candidates.append((pid, out_override))

        if not candidates:
            raise RuntimeError(
                "No draw providers configured. Please add providers and set features.draw.chain."
            )

        default_output = self._default_output()

        last_error: Exception | None = None
        for pid, out_override in candidates:
            try:
                backend = self.registry.get_backend(pid)
            except Exception as e:
                last_error = e
                logger.warning("[draw] Provider build failed: %s: %s", pid, e)
                continue

            output = out_override or default_output
            if size or resolution:
                final_size = size
                final_res = resolution
            else:
                out_size, out_res = parse_output(output)
                final_size = out_size
                final_res = out_res

            t0 = time.perf_counter()
            try:
                gen = getattr(backend, "generate", None)
                if not callable(gen):
                    raise RuntimeError("Provider does not support generate()")
                result = await gen(prompt, size=final_size, resolution=final_res)
                logger.info(
                    "[draw] Provider=%s success in %.2fs", pid, time.perf_counter() - t0
                )
                return result
            except Exception as e:
                last_error = e
                logger.warning("[draw] Provider=%s failed: %s", pid, e)

        raise RuntimeError(f"Draw failed: {last_error}") from last_error
