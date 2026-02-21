from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from astrbot.api import logger

from .gemini_edit import GeminiEditBackend
from .gemini_flow2api import GeminiFlow2ApiBackend
from .gitee_edit import GiteeEditBackend
from .gitee_sizes import GITEE_SUPPORTED_SIZES, normalize_size_text
from .grok2api_images_backend import Grok2ApiImagesBackend
from .grok_video_service import GrokVideoService
from .jimeng_api_backend import JimengApiBackend
from .openai_chat_image_backend import OpenAIChatImageBackend
from .openai_compat_backend import OpenAICompatBackend
from .openai_full_url_backend import OpenAIFullURLBackend


@dataclass(frozen=True)
class ProviderRef:
    provider_id: str
    output: str = ""


def _as_list(value: Any) -> list:
    return value if isinstance(value, list) else []


def _as_dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _is_http_url(value: Any) -> bool:
    s = str(value or "").strip().lower()
    return s.startswith("http://") or s.startswith("https://")


class ProviderRegistry:
    """Build and cache provider backends from v4 config."""

    def __init__(self, config: dict, *, imgr, data_dir: Path):
        self._config = config if isinstance(config, dict) else {}
        self._imgr = imgr
        self._data_dir = Path(data_dir)

        self._providers: dict[str, dict] = {}
        self._backends: dict[str, object] = {}
        self._video_backends: dict[str, GrokVideoService] = {}

        self._load_providers()

    def _load_providers(self) -> None:
        raw = _as_list(self._config.get("providers"))
        for item in raw:
            if not isinstance(item, dict):
                continue
            provider_id = str(item.get("id") or "").strip()
            if not provider_id:
                continue
            if provider_id in self._providers:
                logger.warning(
                    "[ProviderRegistry] Duplicate provider id detected: %s", provider_id
                )
                continue
            self._providers[provider_id] = item

    def validate(self) -> list[str]:
        """Return human-readable validation errors. Never raises."""
        errors: list[str] = []

        raw = self._config.get("providers")
        if raw is None:
            errors.append("Missing config key: providers")
            return errors
        if not isinstance(raw, list):
            errors.append("Config providers must be a list")
            return errors

        seen: set[str] = set()
        for idx, item in enumerate(raw):
            if not isinstance(item, dict):
                errors.append(f"providers[{idx}] must be an object")
                continue
            provider_id = str(item.get("id") or "").strip()
            if not provider_id:
                errors.append(f"providers[{idx}].id is required")
                continue
            if provider_id in seen:
                errors.append(f"providers[{idx}].id duplicated: {provider_id}")
                continue
            seen.add(provider_id)

            template_key = str(item.get("__template_key") or "").strip()
            if not template_key:
                errors.append(f"providers[{idx}].__template_key is required")
                continue

            # Minimal required fields per provider type.
            if template_key in {
                "openai_images",
                "grok_images",
                "gitee_images",
                "gemini_openai_images",
                "modelscope_openai_images",
            }:
                if not str(item.get("base_url") or "").strip():
                    errors.append(f"provider '{provider_id}' missing base_url")
                if not str(item.get("model") or "").strip():
                    errors.append(f"provider '{provider_id}' missing model")
            if template_key in {"openai_chat", "grok_chat", "gemini_openai_chat"}:
                if not str(item.get("base_url") or "").strip():
                    errors.append(f"provider '{provider_id}' missing base_url")
                if not str(item.get("model") or "").strip():
                    errors.append(f"provider '{provider_id}' missing model")
            if template_key in {"gemini_native"}:
                if not str(item.get("api_url") or "").strip():
                    errors.append(f"provider '{provider_id}' missing api_url")
                if not str(item.get("model") or "").strip():
                    errors.append(f"provider '{provider_id}' missing model")
            if template_key in {"flow2api"}:
                if not str(item.get("api_url") or "").strip():
                    errors.append(f"provider '{provider_id}' missing api_url")
                if not str(item.get("model") or "").strip():
                    errors.append(f"provider '{provider_id}' missing model")
            if template_key in {"grok2api_images"}:
                if not str(item.get("base_url") or "").strip():
                    errors.append(f"provider '{provider_id}' missing base_url")
                if not str(item.get("model") or "").strip():
                    errors.append(f"provider '{provider_id}' missing model")
            if template_key in {"gitee_async"}:
                if not str(item.get("base_url") or "").strip():
                    errors.append(f"provider '{provider_id}' missing base_url")
            if template_key in {"jimeng"}:
                if not str(item.get("api_url") or "").strip():
                    errors.append(f"provider '{provider_id}' missing api_url")
                if not str(item.get("apikey") or "").strip():
                    errors.append(f"provider '{provider_id}' missing apikey")
            if template_key in {"grok_video"}:
                if not str(item.get("server_url") or "").strip():
                    errors.append(f"provider '{provider_id}' missing server_url")
                if not str(item.get("api_key") or "").strip():
                    errors.append(f"provider '{provider_id}' missing api_key")
            if template_key in {"openai_full_url_images"}:
                full_generate_url = str(item.get("full_generate_url") or "").strip()
                if not full_generate_url:
                    errors.append(
                        f"provider '{provider_id}' 缺少 full_generate_url（完整文生图 URL）"
                    )
                elif not _is_http_url(full_generate_url):
                    errors.append(
                        f"provider '{provider_id}' 的 full_generate_url 必须以 http:// 或 https:// 开头"
                    )

                full_edit_url = str(item.get("full_edit_url") or "").strip()
                if full_edit_url and not _is_http_url(full_edit_url):
                    errors.append(
                        f"provider '{provider_id}' 的 full_edit_url 必须是完整 URL（http/https）"
                    )

                if not str(item.get("model") or "").strip():
                    errors.append(f"provider '{provider_id}' missing model")

        return errors

    def provider_ids(self) -> list[str]:
        return list(self._providers.keys())

    def get(self, provider_id: str) -> dict | None:
        return self._providers.get(str(provider_id or "").strip())

    def _get_draw_ratio_default_sizes(self) -> dict[str, str]:
        feats = _as_dict(self._config.get("features"))
        draw = _as_dict(feats.get("draw"))
        raw = draw.get("ratio_default_sizes", {})
        if not isinstance(raw, dict):
            return {}
        out: dict[str, str] = {}
        for ratio, size in raw.items():
            r = str(ratio or "").strip()
            s = normalize_size_text(size)
            if r and s:
                out[r] = s
        return out

    def get_backend(self, provider_id: str) -> object:
        pid = str(provider_id or "").strip()
        if not pid:
            raise RuntimeError("Empty provider_id")
        if pid in self._backends:
            return self._backends[pid]

        p = self.get(pid)
        if not p:
            raise RuntimeError(f"Unknown provider_id: {pid}")

        template_key = str(p.get("__template_key") or "").strip()
        if not template_key:
            raise RuntimeError(f"Provider '{pid}' missing __template_key")

        backend = self._build_backend(pid, template_key, p)
        self._backends[pid] = backend
        return backend

    def _build_backend(self, pid: str, template_key: str, conf: dict) -> object:
        if template_key == "gemini_native":
            settings = {
                "api_url": conf.get("api_url"),
                "api_keys": _as_list(conf.get("api_keys")),
                "model": conf.get("model"),
                "resolution": conf.get("default_resolution", "4K"),
                "timeout": conf.get("timeout", 120),
                "use_proxy": bool(conf.get("use_proxy", False)),
                "proxy_url": conf.get("proxy_url", ""),
            }
            return GeminiEditBackend(imgr=self._imgr, settings=settings)

        if template_key == "flow2api":
            settings = {
                "api_url": conf.get("api_url"),
                "api_keys": _as_list(conf.get("api_keys")),
                "model": conf.get("model"),
                "timeout": conf.get("timeout", 120),
                "use_proxy": bool(conf.get("use_proxy", False)),
                "proxy_url": conf.get("proxy_url", ""),
            }
            return GeminiFlow2ApiBackend(imgr=self._imgr, settings=settings)

        if template_key in {"openai_images", "grok_images", "gemini_openai_images"}:
            return OpenAICompatBackend(
                imgr=self._imgr,
                base_url=str(conf.get("base_url") or "").strip(),
                api_keys=[
                    str(x).strip()
                    for x in _as_list(conf.get("api_keys"))
                    if str(x).strip()
                ],
                timeout=int(conf.get("timeout") or 120),
                max_retries=int(conf.get("max_retries") or 2),
                default_model=str(conf.get("model") or "").strip(),
                default_size=str(conf.get("default_size") or "4096x4096").strip(),
                supports_edit=bool(conf.get("supports_edit", True)),
                extra_body=_as_dict(conf.get("extra_body")) or None,
                proxy_url=str(conf.get("proxy_url") or "").strip() or None,
            )

        if template_key == "openai_full_url_images":
            return OpenAIFullURLBackend(
                imgr=self._imgr,
                full_generate_url=str(conf.get("full_generate_url") or "").strip(),
                full_edit_url=str(conf.get("full_edit_url") or "").strip(),
                api_keys=[
                    str(x).strip()
                    for x in _as_list(conf.get("api_keys"))
                    if str(x).strip()
                ],
                timeout=int(conf.get("timeout") or 120),
                max_retries=int(conf.get("max_retries") or 2),
                default_model=str(conf.get("model") or "").strip(),
                default_size=str(conf.get("default_size") or "4096x4096").strip(),
                supports_edit=bool(conf.get("supports_edit", True)),
                extra_body=_as_dict(conf.get("extra_body")) or None,
            )

        if template_key == "modelscope_openai_images":
            return OpenAICompatBackend(
                imgr=self._imgr,
                base_url=str(conf.get("base_url") or "").strip(),
                api_keys=[
                    str(x).strip()
                    for x in _as_list(conf.get("api_keys"))
                    if str(x).strip()
                ],
                timeout=int(conf.get("timeout") or 120),
                max_retries=int(conf.get("max_retries") or 2),
                default_model=str(conf.get("model") or "").strip(),
                default_size=str(conf.get("default_size") or "1024x1024").strip(),
                supports_edit=bool(conf.get("supports_edit", False)),
                extra_body=_as_dict(conf.get("extra_body")) or None,
                proxy_url=str(conf.get("proxy_url") or "").strip() or None,
            )

        if template_key in {"openai_chat", "grok_chat", "gemini_openai_chat"}:
            return OpenAIChatImageBackend(
                imgr=self._imgr,
                base_url=str(conf.get("base_url") or "").strip(),
                api_keys=[
                    str(x).strip()
                    for x in _as_list(conf.get("api_keys"))
                    if str(x).strip()
                ],
                timeout=int(conf.get("timeout") or 120),
                max_retries=int(conf.get("max_retries") or 2),
                default_model=str(conf.get("model") or "").strip(),
                supports_edit=bool(conf.get("supports_edit", True)),
                extra_body=_as_dict(conf.get("extra_body")) or None,
                proxy_url=str(conf.get("proxy_url") or "").strip() or None,
            )

        if template_key == "grok2api_images":
            return Grok2ApiImagesBackend(
                imgr=self._imgr,
                base_url=str(conf.get("base_url") or "").strip(),
                api_keys=[
                    str(x).strip()
                    for x in _as_list(conf.get("api_keys"))
                    if str(x).strip()
                ],
                timeout=int(conf.get("timeout") or 120),
                default_model=str(conf.get("model") or "").strip(),
                default_size=str(conf.get("default_size") or "4096x4096").strip(),
                extra_body=_as_dict(conf.get("extra_body")) or None,
            )

        if template_key == "gitee_images":
            model = str(conf.get("model") or "z-image-turbo").strip()
            extra_body: dict[str, Any] = {}
            if conf.get("num_inference_steps") is not None:
                extra_body["num_inference_steps"] = conf.get("num_inference_steps")
            if str(conf.get("negative_prompt") or "").strip():
                extra_body["negative_prompt"] = str(
                    conf.get("negative_prompt") or ""
                ).strip()
            return OpenAICompatBackend(
                imgr=self._imgr,
                base_url=str(conf.get("base_url") or "https://ai.gitee.com/v1").strip(),
                api_keys=[
                    str(x).strip()
                    for x in _as_list(conf.get("api_keys"))
                    if str(x).strip()
                ],
                timeout=int(conf.get("timeout") or 300),
                max_retries=int(conf.get("max_retries") or 2),
                default_model=model,
                default_size=str(conf.get("default_size") or "1024x1024").strip(),
                supports_edit=False,
                extra_body=extra_body or None,
                allowed_sizes=GITEE_SUPPORTED_SIZES,
                ratio_default_sizes=self._get_draw_ratio_default_sizes(),
            )

        if template_key == "gitee_async":
            settings = dict(conf)
            settings.setdefault("base_url", "https://ai.gitee.com/v1")
            return GiteeEditBackend(imgr=self._imgr, settings=settings)

        if template_key == "jimeng":
            return JimengApiBackend(
                imgr=self._imgr,
                data_dir=self._data_dir,
                api_url=str(conf.get("api_url") or "").strip(),
                apikey=str(conf.get("apikey") or "").strip(),
                cookie_list=_as_list(conf.get("cookie_list")),
                default_style=str(conf.get("default_style") or "真实").strip(),
                default_ratio=str(conf.get("default_ratio") or "1:1").strip(),
                default_model=str(conf.get("default_model") or "Seedream 4.0").strip(),
                timeout=int(conf.get("timeout") or 120),
            )

        raise RuntimeError(f"Unsupported provider type: {template_key} ({pid})")

    def get_video_backend(self, provider_id: str) -> GrokVideoService:
        pid = str(provider_id or "").strip()
        if not pid:
            raise RuntimeError("Empty provider_id")
        if pid in self._video_backends:
            return self._video_backends[pid]
        p = self.get(pid)
        if not p:
            raise RuntimeError(f"Unknown provider_id: {pid}")
        template_key = str(p.get("__template_key") or "").strip()
        if template_key != "grok_video":
            raise RuntimeError(f"Provider '{pid}' is not a video provider")
        backend = GrokVideoService(settings=p)
        self._video_backends[pid] = backend
        return backend

    async def close(self) -> None:
        for backend in list(self._backends.values()):
            close = getattr(backend, "close", None)
            if callable(close):
                try:
                    await close()
                except Exception:
                    pass
        self._backends.clear()
        self._video_backends.clear()
