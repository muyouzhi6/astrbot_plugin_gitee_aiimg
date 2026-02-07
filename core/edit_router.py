from __future__ import annotations

import asyncio
import time
from collections.abc import Iterable
from pathlib import Path

from astrbot.api import logger

from .gitee_edit import GiteeEditBackend
from .output_spec import parse_output
from .provider_registry import ProviderRegistry


def _as_dict(value: object) -> dict:
    return value if isinstance(value, dict) else {}


def _as_list(value: object) -> list:
    return value if isinstance(value, list) else []


class EditRouter:
    """Image-to-Image router for v4 config (provider chain + presets)."""

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

        self.presets = self._load_presets()

        logger.info(
            "[EditRouter] Initialized: presets=%s, providers=%s",
            len(self.presets),
            len(self.registry.provider_ids()),
        )

    def _feature_conf(self) -> dict:
        feats = _as_dict(self.config.get("features"))
        return _as_dict(feats.get("edit"))

    def _default_output(self) -> str:
        return str(self._feature_conf().get("default_output") or "").strip()

    def _chain(self) -> list[dict]:
        return [
            x
            for x in _as_list(self._feature_conf().get("chain"))
            if isinstance(x, dict)
        ]

    def _load_presets(self) -> dict[str, str]:
        presets: dict[str, str] = {}
        for item in _as_list(self._feature_conf().get("presets")):
            if isinstance(item, str) and ":" in item:
                key, val = item.split(":", 1)
                key = key.strip()
                val = val.strip()
                if key and val:
                    presets[key] = val
        return presets

    def get_preset_names(self) -> list[str]:
        return list(self.presets.keys())

    def get_available_backends(self) -> list[str]:
        out: list[str] = []
        for pid in self.registry.provider_ids():
            try:
                backend = self.registry.get_backend(pid)
                edit_fn = getattr(backend, "edit", None)
                if callable(edit_fn):
                    out.append(pid)
            except Exception:
                continue
        return out

    async def close(self) -> None:
        await self.registry.close()

    @staticmethod
    def _candidates_from_chain(raw_chain: list[dict]) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        for item in raw_chain:
            if not isinstance(item, dict):
                continue
            pid = str(item.get("provider_id") or "").strip()
            out_override = str(item.get("output") or "").strip()
            if pid:
                out.append((pid, out_override))
        return out

    def _candidate_chain(
        self, backend: str | None, chain_override: list[dict] | None
    ) -> list[tuple[str, str]]:
        if backend:
            return [(str(backend).strip(), "")]
        if chain_override is not None:
            return self._candidates_from_chain(chain_override)
        return self._candidates_from_chain(self._chain())

    def _default_gitee_task_types(self) -> list[str]:
        raw = _as_list(self._feature_conf().get("gitee_task_types"))
        out = [str(x).strip() for x in raw if str(x).strip()]
        return out or ["id", "background", "style"]

    async def edit(
        self,
        prompt: str,
        images: list[bytes],
        backend: str | None = None,
        preset: str | None = None,
        task_types: Iterable[str] = ("id",),
        *,
        size: str | None = None,
        resolution: str | None = None,
        default_output: str | None = None,
        chain_override: list[dict] | None = None,
    ) -> Path:
        feature = self._feature_conf()
        if not bool(feature.get("enabled", True)):
            raise RuntimeError("Image edit is disabled (features.edit.enabled=false)")
        if not images:
            raise ValueError("At least one image is required")

        if preset and preset in self.presets:
            p = self.presets[preset]
            prompt = f"{p}, {prompt}" if prompt else p
        if not prompt:
            prompt = "Transform this image with artistic style"

        candidates = self._candidate_chain(backend, chain_override)
        if not candidates:
            raise RuntimeError(
                "No edit providers configured. Please add providers and set features.edit.chain."
            )
        logger.debug(
            "[edit] candidates=%s backend_override=%s chain_override=%s",
            [pid for pid, _ in candidates],
            str(backend or ""),
            chain_override is not None,
        )
        if backend:
            logger.debug("[edit] backend override active, chain fallback disabled")

        effective_default_output = (
            str(default_output).strip()
            if default_output is not None and str(default_output).strip()
            else self._default_output()
        )
        gitee_default_task_types = self._default_gitee_task_types()
        final_task_types = (
            list(task_types) if list(task_types) else gitee_default_task_types
        )

        max_attempts = 1

        last_error: Exception | None = None
        t_start = time.perf_counter()

        for pid, out_override in candidates:
            try:
                backend_obj = self.registry.get_backend(pid)
            except Exception as e:
                last_error = e
                logger.warning("[edit] Provider build failed: %s: %s", pid, e)
                continue

            if size or resolution:
                final_size = size
                final_res = resolution
            else:
                output = out_override or effective_default_output
                out_size, out_res = parse_output(output)
                final_size = out_size
                final_res = out_res

            for attempt in range(max_attempts):
                try:
                    logger.info(
                        "[edit] Provider=%s attempt=%s/%s",
                        pid,
                        attempt + 1,
                        max_attempts,
                    )
                    edit_fn = getattr(backend_obj, "edit", None)
                    if not callable(edit_fn):
                        raise RuntimeError("Provider does not support edit()")
                    if isinstance(backend_obj, GiteeEditBackend):
                        result = await backend_obj.edit(
                            prompt, images, task_types=final_task_types
                        )
                    else:
                        result = await edit_fn(
                            prompt,
                            images,
                            size=final_size,
                            resolution=final_res,
                        )
                    logger.info(
                        "[edit] Provider=%s success in %.2fs",
                        pid,
                        time.perf_counter() - t_start,
                    )
                    return result
                except Exception as e:
                    last_error = e
                    logger.warning("[edit] Provider=%s failed: %s", pid, e)
                    if attempt + 1 < max_attempts:
                        await asyncio.sleep(0.5 * (2**attempt))

        raise RuntimeError(f"Edit failed: {last_error}") from last_error
