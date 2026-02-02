"""
改图路由器（基于新的配置界面）

配置要点：
- edit.provider 选择默认服务商
- edit.<provider> 填该服务商的参数
- edit.fallback_providers 可选降级链路
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Iterable
from pathlib import Path

from astrbot.api import logger

from .gemini_edit import GeminiEditBackend
from .gitee_edit import GiteeEditBackend
from .jimeng_api_backend import JimengApiBackend
from .openai_chat_image_backend import OpenAIChatImageBackend
from .openai_compat_backend import OpenAICompatBackend

BUILTIN_PRESETS: dict[str, str] = {}


class EditRouter:
    def __init__(self, config: dict, imgr, data_dir: Path):
        self.config = config or {}
        self.imgr = imgr
        self.data_dir = Path(data_dir)

        self.edit_conf: dict = (
            (self.config.get("edit") or {}) if isinstance(self.config, dict) else {}
        )

        self.presets = self._load_presets()
        self._backends: dict[str, object] = {}

        logger.info(
            "[EditRouter] 初始化完成: provider=%s, presets=%s",
            self._default_provider(),
            len(self.presets),
        )

    def _load_presets(self) -> dict[str, str]:
        presets = dict(BUILTIN_PRESETS)
        for item in self.edit_conf.get("presets") or []:
            if isinstance(item, str) and ":" in item:
                key, val = item.split(":", 1)
                if key.strip() and val.strip():
                    presets[key.strip()] = val.strip()
        return presets

    def get_preset_names(self) -> list[str]:
        return list(self.presets.keys())

    def get_available_backends(self) -> list[str]:
        return [
            k
            for k in (
                "grok",
                "gemini_native",
                "gemini_openai",
                "openai_compat",
                "jimeng",
                "gitee_async",
            )
            if self._can_build_backend(k)
        ]

    def _default_provider(self) -> str:
        return (
            str(self.edit_conf.get("provider") or "gemini_native").strip()
            or "gemini_native"
        )

    def _fallback_provider_ids(self) -> list[str]:
        allowed = {
            "grok",
            "gemini_native",
            "gemini_openai",
            "openai_compat",
            "jimeng",
            "gitee_async",
        }

        out: list[str] = []

        # New UI: explicit dropdown fields.
        for key in ("fallback_1", "fallback_2", "fallback_3"):
            v = str(self.edit_conf.get(key) or "").strip()
            if v and v in allowed and v not in out:
                out.append(v)

        # Backward: accept legacy list configs.
        raw = self.edit_conf.get("fallback_providers")
        if isinstance(raw, list):
            for x in raw:
                v = str(x).strip()
                if v and v in allowed and v not in out:
                    out.append(v)

        return out

    def _normalize_alias(self, backend: str | None) -> str | None:
        if backend is None:
            return None
        b = str(backend).strip()
        if not b or b.lower() == "auto":
            return None
        aliases = {"gemini": "gemini_native", "gitee": "gitee_async"}
        return aliases.get(b, b)

    def _can_build_backend(self, backend: str) -> bool:
        try:
            self._build_backend(backend)
            return True
        except Exception:
            return False

    def _get_backend(self, backend: str) -> object:
        if backend in self._backends:
            return self._backends[backend]
        obj = self._build_backend(backend)
        self._backends[backend] = obj
        return obj

    def _build_openai_compat(
        self, settings: dict, *, supports_edit: bool = True
    ) -> OpenAICompatBackend:
        return OpenAICompatBackend(
            imgr=self.imgr,
            base_url=str(settings.get("base_url") or "").strip(),
            api_keys=[
                str(x).strip()
                for x in (settings.get("api_keys") or [])
                if str(x).strip()
            ],
            timeout=int(settings.get("timeout") or 120),
            max_retries=int(settings.get("max_retries") or 2),
            default_model=str(settings.get("model") or "").strip(),
            default_size=str(settings.get("size") or "4096x4096").strip(),
            supports_edit=supports_edit,
            extra_body=settings.get("extra_body")
            if isinstance(settings.get("extra_body"), dict)
            else None,
        )

    def _build_backend(self, backend: str) -> object:
        b = str(backend).strip()
        if b == "grok":
            settings = dict(self.edit_conf.get("grok") or {})
            settings.setdefault("base_url", "https://api.x.ai/v1")
            api_mode = str(settings.get("api_mode") or "chat").strip().lower()
            if api_mode == "chat":
                return OpenAIChatImageBackend(
                    imgr=self.imgr,
                    base_url=str(settings.get("base_url") or "").strip(),
                    api_keys=[
                        str(x).strip()
                        for x in (settings.get("api_keys") or [])
                        if str(x).strip()
                    ],
                    timeout=int(settings.get("timeout") or 120),
                    max_retries=int(settings.get("max_retries") or 2),
                    default_model=str(settings.get("model") or "").strip(),
                    supports_edit=bool(settings.get("supports_edit", True)),
                    extra_body=settings.get("extra_body")
                    if isinstance(settings.get("extra_body"), dict)
                    else None,
                )
            return self._build_openai_compat(
                settings,
                supports_edit=bool(settings.get("supports_edit", True)),
            )
        if b == "gemini_openai":
            settings = dict(self.edit_conf.get("gemini_openai") or {})
            api_mode = str(settings.get("api_mode") or "images").strip().lower()
            if api_mode == "chat":
                return OpenAIChatImageBackend(
                    imgr=self.imgr,
                    base_url=str(settings.get("base_url") or "").strip(),
                    api_keys=[
                        str(x).strip()
                        for x in (settings.get("api_keys") or [])
                        if str(x).strip()
                    ],
                    timeout=int(settings.get("timeout") or 120),
                    max_retries=int(settings.get("max_retries") or 2),
                    default_model=str(settings.get("model") or "").strip(),
                    supports_edit=bool(settings.get("supports_edit", True)),
                    extra_body=settings.get("extra_body")
                    if isinstance(settings.get("extra_body"), dict)
                    else None,
                )
            return self._build_openai_compat(
                settings,
                supports_edit=bool(settings.get("supports_edit", True)),
            )
        if b == "openai_compat":
            settings = dict(self.edit_conf.get("openai_compat") or {})
            api_mode = str(settings.get("api_mode") or "images").strip().lower()
            if api_mode == "chat":
                return OpenAIChatImageBackend(
                    imgr=self.imgr,
                    base_url=str(settings.get("base_url") or "").strip(),
                    api_keys=[
                        str(x).strip()
                        for x in (settings.get("api_keys") or [])
                        if str(x).strip()
                    ],
                    timeout=int(settings.get("timeout") or 120),
                    max_retries=int(settings.get("max_retries") or 2),
                    default_model=str(settings.get("model") or "").strip(),
                    supports_edit=bool(settings.get("supports_edit", True)),
                    extra_body=settings.get("extra_body")
                    if isinstance(settings.get("extra_body"), dict)
                    else None,
                )
            return self._build_openai_compat(
                settings,
                supports_edit=bool(settings.get("supports_edit", True)),
            )
        if b == "gemini_native":
            return GeminiEditBackend(
                imgr=self.imgr, settings=dict(self.edit_conf.get("gemini_native") or {})
            )
        if b == "gitee_async":
            return GiteeEditBackend(
                imgr=self.imgr, settings=dict(self.edit_conf.get("gitee_async") or {})
            )
        if b == "jimeng":
            conf = dict(self.edit_conf.get("jimeng") or {})
            return JimengApiBackend(
                imgr=self.imgr,
                data_dir=self.data_dir,
                api_url=str(conf.get("api_url") or "").strip(),
                apikey=str(conf.get("apikey") or "").strip(),
                cookie_list=conf.get("cookie_list")
                if isinstance(conf.get("cookie_list"), list)
                else [],
                default_style=str(conf.get("default_style") or "真实").strip(),
                default_ratio=str(conf.get("default_ratio") or "1:1").strip(),
                default_model=str(conf.get("default_model") or "Seedream 4.0").strip(),
                timeout=int(conf.get("timeout") or 120),
            )
        raise RuntimeError(f"未知后端: {backend}")

    async def close(self) -> None:
        for backend in self._backends.values():
            close = getattr(backend, "close", None)
            if callable(close):
                try:
                    await close()
                except Exception:
                    pass
        self._backends.clear()

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
    ) -> Path:
        if not images:
            raise ValueError("至少需要一张图片")

        if preset and preset in self.presets:
            p = self.presets[preset]
            prompt = f"{p}, {prompt}" if prompt else p
        if not prompt:
            prompt = "Transform this image with artistic style"

        target = self._normalize_alias(backend) or self._default_provider()

        fallback_cfg = self.edit_conf.get("fallback") or {}
        max_retries = int(fallback_cfg.get("max_retries", 2) or 2)
        retry_delay = int(fallback_cfg.get("retry_delay", 2) or 2)
        fallback_enabled = bool(fallback_cfg.get("enabled", True))

        candidates = [target]
        if fallback_enabled:
            candidates.extend(self._fallback_provider_ids())

        last_error: Exception | None = None
        t_start = time.perf_counter()

        for backend_id in candidates:
            try:
                backend_obj = self._get_backend(backend_id)
            except Exception as e:
                last_error = e
                logger.warning("[EditRouter] 后端=%s 构建失败: %s", backend_id, e)
                continue

            for attempt in range(max_retries + 1):
                try:
                    logger.info(
                        "[EditRouter] 后端=%s 尝试=%s prompt=%s",
                        backend_id,
                        attempt + 1,
                        prompt[:80],
                    )
                    if isinstance(backend_obj, GiteeEditBackend):
                        result = await backend_obj.edit(
                            prompt, images, task_types=task_types
                        )
                    elif isinstance(backend_obj, OpenAICompatBackend):
                        result = await backend_obj.edit(
                            prompt, images, size=size, resolution=resolution
                        )
                    elif isinstance(backend_obj, GeminiEditBackend):
                        result = await backend_obj.edit(
                            prompt, images, resolution=resolution
                        )
                    elif isinstance(backend_obj, JimengApiBackend):
                        result = await backend_obj.edit(prompt, images)
                    else:
                        edit_fn = getattr(backend_obj, "edit", None)
                        if not callable(edit_fn):
                            raise RuntimeError(f"后端 {backend_id} 不支持改图")
                        result = await edit_fn(prompt, images)

                    logger.info(
                        "[EditRouter] 后端=%s 成功, 总耗时=%.2fs",
                        backend_id,
                        time.perf_counter() - t_start,
                    )
                    return result
                except Exception as e:
                    last_error = e
                    logger.warning(
                        "[EditRouter] 后端=%s 尝试=%s 失败: %s",
                        backend_id,
                        attempt + 1,
                        e,
                    )
                    if "404" in str(e):
                        break
                    if attempt < max_retries:
                        await asyncio.sleep(retry_delay * (2**attempt))

        raise RuntimeError(f"改图失败: {last_error}") from last_error
