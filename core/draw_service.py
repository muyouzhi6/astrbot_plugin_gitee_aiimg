from __future__ import annotations

import time
from pathlib import Path

from astrbot.api import logger

from .gemini_edit import GeminiEditBackend
from .jimeng_api_backend import JimengApiBackend
from .openai_chat_image_backend import OpenAIChatImageBackend
from .openai_compat_backend import OpenAICompatBackend

MODELS_WITHOUT_NEGATIVE_PROMPT = frozenset(
    {
        "z-image-turbo",
        "z-image-base",
        "flux.1-dev",
        "flux.1-schnell",
    }
)


class ImageDrawService:
    """生图服务（基于新的配置界面）。"""

    def __init__(self, config: dict, imgr, data_dir: Path):
        self.config = config or {}
        self.imgr = imgr
        self.data_dir = Path(data_dir)

        self.draw_conf: dict = (
            (self.config.get("draw") or {}) if isinstance(self.config, dict) else {}
        )
        self._backends: dict[str, object] = {}

    def _default_provider(self) -> str:
        return str(self.draw_conf.get("provider") or "grok").strip() or "grok"

    def _fallback_provider_ids(self) -> list[str]:
        allowed = {
            "grok",
            "gemini_native",
            "gemini_openai",
            "openai_compat",
            "jimeng",
            "gitee",
        }

        out: list[str] = []

        # New UI: explicit dropdown fields.
        for key in ("fallback_1", "fallback_2", "fallback_3"):
            v = str(self.draw_conf.get(key) or "").strip()
            if v and v in allowed and v not in out:
                out.append(v)

        # Backward: accept legacy list configs.
        raw = self.draw_conf.get("fallback_providers")
        if isinstance(raw, list):
            for x in raw:
                v = str(x).strip()
                if v and v in allowed and v not in out:
                    out.append(v)

        return out

    def _normalize_alias(self, provider: str | None) -> str | None:
        if provider is None:
            return None
        p = str(provider).strip()
        if not p or p.lower() == "auto":
            return None
        aliases = {"gemini": "gemini_native", "gitee": "gitee"}
        return aliases.get(p, p)

    def _get_backend(self, provider: str) -> object:
        if provider in self._backends:
            return self._backends[provider]
        obj = self._build_backend(provider)
        self._backends[provider] = obj
        return obj

    def _build_openai_compat(
        self, settings: dict, *, supports_edit: bool
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

    def _build_backend(self, provider: str) -> object:
        p = str(provider).strip()
        if p == "grok":
            settings = dict(self.draw_conf.get("grok") or {})
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
                    supports_edit=False,
                    extra_body=settings.get("extra_body")
                    if isinstance(settings.get("extra_body"), dict)
                    else None,
                )
            return self._build_openai_compat(settings, supports_edit=False)

        if p == "gemini_openai":
            settings = dict(self.draw_conf.get("gemini_openai") or {})
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
                    supports_edit=False,
                    extra_body=settings.get("extra_body")
                    if isinstance(settings.get("extra_body"), dict)
                    else None,
                )
            return self._build_openai_compat(settings, supports_edit=False)

        if p == "openai_compat":
            settings = dict(self.draw_conf.get("openai_compat") or {})
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
                    supports_edit=False,
                    extra_body=settings.get("extra_body")
                    if isinstance(settings.get("extra_body"), dict)
                    else None,
                )
            return self._build_openai_compat(settings, supports_edit=False)

        if p == "gemini_native":
            return GeminiEditBackend(
                imgr=self.imgr, settings=dict(self.draw_conf.get("gemini_native") or {})
            )

        if p == "jimeng":
            conf = dict(self.draw_conf.get("jimeng") or {})
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

        if p == "gitee":
            conf = dict(self.draw_conf.get("gitee") or {})
            model = str(conf.get("model") or "z-image-turbo").strip()
            extra_body: dict = {}
            if conf.get("num_inference_steps") is not None:
                extra_body["num_inference_steps"] = conf.get("num_inference_steps")
            if conf.get("negative_prompt"):
                if model.lower() not in MODELS_WITHOUT_NEGATIVE_PROMPT:
                    extra_body["negative_prompt"] = conf.get("negative_prompt")

            return OpenAICompatBackend(
                imgr=self.imgr,
                base_url=str(conf.get("base_url") or "https://ai.gitee.com/v1").strip(),
                api_keys=[
                    str(x).strip()
                    for x in (conf.get("api_keys") or [])
                    if str(x).strip()
                ],
                timeout=int(conf.get("timeout") or 300),
                max_retries=int(conf.get("max_retries") or 2),
                default_model=model,
                default_size=str(conf.get("size") or "1024x1024").strip(),
                supports_edit=False,
                extra_body=extra_body,
            )

        raise RuntimeError(f"未知生图服务商: {provider}")

    async def close(self) -> None:
        for backend in self._backends.values():
            close = getattr(backend, "close", None)
            if callable(close):
                try:
                    await close()
                except Exception:
                    pass
        self._backends.clear()

    async def generate(
        self,
        prompt: str,
        *,
        size: str | None = None,
        resolution: str | None = None,
        provider_id: str | None = None,
    ) -> Path:
        provider = self._normalize_alias(provider_id) or self._default_provider()
        candidates = [provider, *self._fallback_provider_ids()]

        last_error: Exception | None = None
        for p in candidates:
            try:
                backend = self._get_backend(p)
            except Exception as e:
                last_error = e
                logger.warning("[生图] 服务商=%s 构建失败: %s", p, e)
                continue

            t0 = time.perf_counter()
            try:
                if hasattr(backend, "generate"):
                    result = await backend.generate(  # type: ignore[attr-defined]
                        prompt,
                        size=size,
                        resolution=resolution,
                    )
                    logger.info(
                        "[生图] 服务商=%s 成功, 耗时=%.2fs", p, time.perf_counter() - t0
                    )
                    return result
                raise RuntimeError(f"服务商 {p} 不支持生图")
            except Exception as e:
                last_error = e
                logger.warning("[生图] 服务商=%s 失败: %s", p, e)

        raise RuntimeError(f"生图失败: {last_error}") from last_error
