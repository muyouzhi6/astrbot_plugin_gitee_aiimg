from __future__ import annotations

import time
from pathlib import Path

from astrbot.api import logger

from .output_spec import (
    OutputIntent,
    extract_output_intent_from_prompt,
    merge_output_intents,
    output_intent_from_legacy,
    parse_output_intent,
    resolve_backend_output,
)
from .provider_chain import as_dict, as_list, candidates_from_chain
from .provider_registry import ProviderRegistry


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
        feats = as_dict(self.config.get("features"))
        return as_dict(feats.get("draw"))

    def _default_output(self) -> str:
        return str(self._feature_conf().get("default_output") or "").strip()

    def _chain(self) -> list:
        return as_list(self._feature_conf().get("chain"))

    def _candidate_ids(self) -> list[str]:
        return [pid for pid, _ in candidates_from_chain(self._chain())]

    async def close(self) -> None:
        await self.registry.close()

    async def generate(
        self,
        prompt: str,
        *,
        size: str | None = None,
        resolution: str | None = None,
        output_intent: OutputIntent | None = None,
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
            candidates = candidates_from_chain(self._chain())

        if not candidates:
            raise RuntimeError(
                "No draw providers configured. Please add providers and set features.draw.chain."
            )
        logger.debug("[draw] candidates=%s", [pid for pid, _ in candidates])

        default_output = self._default_output()
        request_intent = merge_output_intents(
            output_intent,
            output_intent_from_legacy(size, resolution),
            extract_output_intent_from_prompt(prompt),
        )

        last_error: Exception | None = None
        for pid, out_override in candidates:
            try:
                backend = self.registry.get_backend(pid)
            except Exception as e:
                last_error = e
                logger.warning("[draw] Provider build failed: %s: %s", pid, e)
                continue

            t0 = time.perf_counter()
            try:
                candidate_intent = merge_output_intents(
                    request_intent,
                    parse_output_intent(out_override) if out_override else None,
                    parse_output_intent(default_output) if default_output else None,
                )
                output_kwargs = resolve_backend_output(backend, candidate_intent)
                logger.debug(
                    "[draw] Provider=%s output_intent=%s kwargs=%s",
                    pid,
                    candidate_intent,
                    output_kwargs,
                )
                gen = getattr(backend, "generate", None)
                if not callable(gen):
                    raise RuntimeError("Provider does not support generate()")
                result = await gen(prompt, **output_kwargs)
                if not result:
                    raise RuntimeError("Provider returned empty generate result")
                logger.info(
                    "[draw] Provider=%s success in %.2fs", pid, time.perf_counter() - t0
                )
                return result
            except Exception as e:
                last_error = e
                logger.warning("[draw] Provider=%s failed: %s", pid, e)

        raise RuntimeError(f"Draw failed: {last_error}") from last_error
