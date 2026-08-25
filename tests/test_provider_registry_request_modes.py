import importlib.util
import json
import sys
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "provider_registry_testpkg"
CORE_PACKAGE_NAME = f"{PACKAGE_NAME}.core"
MODULE_NAME = f"{CORE_PACKAGE_NAME}.provider_registry"


class _Logger:
    def debug(self, *args, **kwargs):
        return None

    def info(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None

    def error(self, *args, **kwargs):
        return None


class _StubBackend:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class _StubVertexSettings:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class Sora2VideoService(_StubBackend):
    pass


def _clear_modules():
    for name in list(sys.modules):
        if name.startswith(PACKAGE_NAME) or name in {"astrbot", "astrbot.api"}:
            sys.modules.pop(name, None)


def _install_stub_module(name: str, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module


def _load_module():
    _clear_modules()

    pkg = types.ModuleType(PACKAGE_NAME)
    pkg.__path__ = [str(ROOT)]
    sys.modules[PACKAGE_NAME] = pkg

    core_pkg = types.ModuleType(CORE_PACKAGE_NAME)
    core_pkg.__path__ = [str(ROOT / "core")]
    sys.modules[CORE_PACKAGE_NAME] = core_pkg

    astrbot_mod = types.ModuleType("astrbot")
    sys.modules["astrbot"] = astrbot_mod

    api_mod = types.ModuleType("astrbot.api")
    api_mod.logger = _Logger()
    sys.modules["astrbot.api"] = api_mod

    _install_stub_module(
        f"{CORE_PACKAGE_NAME}.gemini_edit",
        GeminiEditBackend=_StubBackend,
    )
    _install_stub_module(
        f"{CORE_PACKAGE_NAME}.gemini_flow2api",
        Flow2ApiVideoBackend=_StubBackend,
        GeminiFlow2ApiBackend=_StubBackend,
    )
    _install_stub_module(
        f"{CORE_PACKAGE_NAME}.gitee_edit",
        GiteeEditBackend=_StubBackend,
    )
    _install_stub_module(
        f"{CORE_PACKAGE_NAME}.gitee_sizes",
        GITEE_SUPPORTED_SIZES=["1024x1024"],
        normalize_size_text=lambda value: str(value or "").strip(),
    )
    _install_stub_module(
        f"{CORE_PACKAGE_NAME}.grok2api_images_backend",
        Grok2ApiImagesBackend=_StubBackend,
    )
    _install_stub_module(
        f"{CORE_PACKAGE_NAME}.grok_images_backend",
        GrokImagesBackend=_StubBackend,
    )
    _install_stub_module(
        f"{CORE_PACKAGE_NAME}.grok_video_service",
        GrokVideoService=_StubBackend,
    )
    _install_stub_module(
        f"{CORE_PACKAGE_NAME}.jimeng_api_backend",
        JimengApiBackend=_StubBackend,
    )
    _install_stub_module(
        f"{CORE_PACKAGE_NAME}.modelscope_async_backend",
        ModelScopeAsyncImageBackend=_StubBackend,
    )
    _install_stub_module(
        f"{CORE_PACKAGE_NAME}.openai_chat_image_backend",
        OpenAIChatImageBackend=_StubBackend,
    )
    _install_stub_module(
        f"{CORE_PACKAGE_NAME}.openai_compat_backend",
        OpenAICompatBackend=_StubBackend,
    )
    _install_stub_module(
        f"{CORE_PACKAGE_NAME}.openai_full_url_backend",
        OpenAIFullURLBackend=_StubBackend,
    )
    _install_stub_module(
        f"{CORE_PACKAGE_NAME}.sora2_video_service",
        Sora2VideoService=Sora2VideoService,
    )
    _install_stub_module(
        f"{CORE_PACKAGE_NAME}.vertex_ai_anonymous_backend",
        VertexAIAnonymousBackend=_StubBackend,
        VertexAIAnonymousSettings=_StubVertexSettings,
    )

    spec = importlib.util.spec_from_file_location(
        MODULE_NAME,
        ROOT / "core" / "provider_registry.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[MODULE_NAME] = module
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


class ProviderRegistryRequestModeTests(unittest.TestCase):
    def test_provider_schema_uses_600_second_timeout_defaults(self):
        schema = json.loads((ROOT / "_conf_schema.json").read_text(encoding="utf-8"))
        templates = schema["providers"]["templates"]
        timeout_defaults = {
            key: template["items"]["timeout"]["default"]
            for key, template in templates.items()
            if "timeout" in template.get("items", {})
        }

        self.assertTrue(timeout_defaults)
        self.assertEqual(set(timeout_defaults.values()), {600})
        self.assertEqual(
            templates["gemini_native"]["items"]["max_retries"]["default"], 2
        )

    def test_gemini_native_legacy_config_uses_new_missing_field_defaults(self):
        mod = _load_module()
        legacy_provider = {
            "id": "gemini-old",
            "__template_key": "gemini_native",
            "api_url": "https://example.invalid",
            "api_keys": ["test-key"],
            "model": "gemini-test",
            "output_format": "png",
        }
        original_provider = dict(legacy_provider)
        registry = mod.ProviderRegistry(
            config={"providers": [legacy_provider]},
            imgr=object(),
            data_dir=Path("/tmp"),
        )

        backend = registry.get_backend("gemini-old")

        self.assertEqual(backend.kwargs["settings"]["timeout"], 600)
        self.assertEqual(backend.kwargs["settings"]["max_retries"], 2)
        self.assertEqual(legacy_provider, original_provider)

    def test_gemini_native_preserves_explicit_legacy_values(self):
        mod = _load_module()
        registry = mod.ProviderRegistry(
            config={
                "providers": [
                    {
                        "id": "gemini-custom",
                        "__template_key": "gemini_native",
                        "api_url": "https://example.invalid",
                        "api_keys": ["test-key"],
                        "model": "gemini-test",
                        "timeout": 120,
                        "max_retries": 0,
                    }
                ]
            },
            imgr=object(),
            data_dir=Path("/tmp"),
        )

        backend = registry.get_backend("gemini-custom")

        self.assertEqual(backend.kwargs["settings"]["timeout"], 120)
        self.assertEqual(backend.kwargs["settings"]["max_retries"], 0)

    def test_registry_uses_modelscope_async_backend(self):
        mod = _load_module()
        registry = mod.ProviderRegistry(
            config={
                "providers": [
                    {
                        "id": "modelscope",
                        "__template_key": "modelscope_openai_images",
                        "base_url": "https://api-inference.modelscope.cn/v1",
                        "api_keys": ["test-key"],
                        "model": "Qwen/Qwen-Image",
                        "poll_interval": 3,
                        "poll_timeout": 480,
                    }
                ]
            },
            imgr=object(),
            data_dir=Path("/tmp"),
        )

        backend = registry.get_backend("modelscope")

        self.assertEqual(backend.kwargs["poll_interval"], 3.0)
        self.assertEqual(backend.kwargs["poll_timeout"], 480)

    def test_registry_forwards_jimeng_timeout_and_output_format(self):
        mod = _load_module()
        registry = mod.ProviderRegistry(
            config={
                "providers": [
                    {
                        "id": "jimeng-test",
                        "__template_key": "jimeng",
                        "api_url": "https://example.invalid/jimeng",
                        "apikey": "test-key",
                        "cookie_list": ["conversation:cookie"],
                        "timeout": 240,
                        "output_format": "webp_lossless",
                    }
                ]
            },
            imgr=object(),
            data_dir=Path("/tmp"),
        )

        backend = registry.get_backend("jimeng-test")

        self.assertEqual(backend.kwargs["timeout"], 240)
        self.assertEqual(backend.kwargs["output_format"], "webp_lossless")

    def test_registry_resolves_x666_sora2_video_provider(self):
        mod = _load_module()
        registry = mod.ProviderRegistry(
            config={
                "providers": [
                    {
                        "id": "x666_sora2",
                        "base_url": "https://x666.me",
                        "api_keys": ["test-key"],
                        "model": "sora-2",
                    }
                ]
            },
            imgr=object(),
            data_dir=Path("/tmp"),
        )

        backend = registry.get_video_backend("x666_sora2")

        self.assertEqual(backend.__class__.__name__, "Sora2VideoService")

    def test_registry_resolves_openai_video_provider_alias(self):
        mod = _load_module()
        registry = mod.ProviderRegistry(
            config={
                "providers": [
                    {
                        "id": "openai_video",
                        "base_url": "https://gateway.example/v1",
                        "api_keys": ["test-key"],
                        "model": "video-model",
                    }
                ]
            },
            imgr=object(),
            data_dir=Path("/tmp"),
        )

        backend = registry.get_video_backend("openai_video")

        self.assertEqual(backend.__class__.__name__, "Sora2VideoService")
        self.assertEqual(
            backend.kwargs["settings"]["base_url"],
            "https://gateway.example/v1",
        )

    def test_registry_resolves_3365_video_provider(self):
        mod = _load_module()
        registry = mod.ProviderRegistry(
            config={
                "providers": [
                    {
                        "id": "3365_video",
                        "__template_key": "3365_video",
                        "server_url": "https://api.3365api.cn",
                        "api_key": "test-key",
                        "model": "grok-imagine-video-1.5",
                    }
                ]
            },
            imgr=object(),
            data_dir=Path("/tmp"),
        )

        backend = registry.get_video_backend("3365_video")

        self.assertEqual(backend.__class__.__name__, "_StubBackend")
        self.assertEqual(
            backend.kwargs["settings"]["server_url"], "https://api.3365api.cn"
        )

    def test_validate_requires_3365_video_credentials(self):
        mod = _load_module()
        registry = mod.ProviderRegistry(
            config={
                "providers": [
                    {
                        "id": "3365_video",
                        "__template_key": "3365_video",
                        "server_url": "https://api.3365api.cn",
                        "model": "grok-imagine-video-1.5",
                    }
                ]
            },
            imgr=object(),
            data_dir=Path("/tmp"),
        )

        self.assertEqual(registry.validate(), ["provider '3365_video' missing api_key"])

    def test_validate_requires_sora2_api_key_source(self):
        mod = _load_module()
        registry = mod.ProviderRegistry(
            config={
                "providers": [
                    {
                        "id": "x666_sora2",
                        "__template_key": "sora2_video",
                        "base_url": "https://x666.me",
                        "model": "sora-2",
                    }
                ]
            },
            imgr=object(),
            data_dir=Path("/tmp"),
        )

        errors = registry.validate()

        self.assertEqual(errors, ["provider 'x666_sora2' missing api_keys"])

    def test_validate_requires_grok2api_video_api_key_source(self):
        mod = _load_module()
        registry = mod.ProviderRegistry(
            config={
                "providers": [
                    {
                        "id": "grok2-video",
                        "__template_key": "grok2api_video",
                    }
                ]
            },
            imgr=object(),
            data_dir=Path("/tmp"),
        )

        self.assertEqual(
            registry.validate(), ["provider 'grok2-video' missing api_keys"]
        )

        registry = mod.ProviderRegistry(
            config={
                "providers": [
                    {
                        "id": "grok2-video",
                        "__template_key": "grok2api_video",
                        "api_key": "test-key",
                    }
                ]
            },
            imgr=object(),
            data_dir=Path("/tmp"),
        )

        self.assertEqual(registry.validate(), [])

    def test_validate_requires_vertex_graphql_api_key(self):
        mod = _load_module()
        registry = mod.ProviderRegistry(
            config={
                "providers": [
                    {
                        "id": "vertex",
                        "__template_key": "vertex_ai_anonymous",
                        "model": "gemini-3-pro-image-preview",
                    }
                ]
            },
            imgr=object(),
            data_dir=Path("/tmp"),
        )

        self.assertEqual(
            registry.validate(), ["provider 'vertex' missing graphql_api_key"]
        )

        with self.assertRaisesRegex(RuntimeError, "missing graphql_api_key"):
            registry.get_backend("vertex")

    def test_registry_passes_explicit_vertex_graphql_api_key(self):
        mod = _load_module()
        registry = mod.ProviderRegistry(
            config={
                "providers": [
                    {
                        "id": "vertex",
                        "__template_key": "vertex_ai_anonymous",
                        "model": "gemini-3-pro-image-preview",
                        "graphql_api_key": "user-configured-key",
                    }
                ]
            },
            imgr=object(),
            data_dir=Path("/tmp"),
        )

        backend = registry.get_backend("vertex")

        self.assertEqual(
            backend.kwargs["settings"].kwargs["graphql_api_key"],
            "user-configured-key",
        )

    def test_registry_keeps_legacy_generate_flag_when_new_mode_is_auto(self):
        mod = _load_module()
        registry = mod.ProviderRegistry(
            config={
                "providers": [
                    {
                        "id": "chat-provider",
                        "__template_key": "openai_chat",
                        "base_url": "https://api.example.com/v1",
                        "api_keys": ["test-key"],
                        "model": "gpt-image",
                        "generate_request_mode": "auto",
                        "enable_stream_generate": False,
                    }
                ]
            },
            imgr=object(),
            data_dir=Path("/tmp"),
        )

        backend = registry.get_backend("chat-provider")

        self.assertEqual(backend.kwargs["generate_request_mode"], "non_stream")
        self.assertFalse(backend.kwargs["enable_stream_generate"])

    def test_registry_keeps_legacy_edit_flag_when_new_mode_is_auto(self):
        mod = _load_module()
        registry = mod.ProviderRegistry(
            config={
                "providers": [
                    {
                        "id": "chat-provider",
                        "__template_key": "openai_chat",
                        "base_url": "https://api.example.com/v1",
                        "api_keys": ["test-key"],
                        "model": "gpt-image",
                        "edit_request_mode": "auto",
                        "enable_stream_edit": True,
                    }
                ]
            },
            imgr=object(),
            data_dir=Path("/tmp"),
        )

        backend = registry.get_backend("chat-provider")

        self.assertEqual(backend.kwargs["edit_request_mode"], "stream")
        self.assertTrue(backend.kwargs["enable_stream_edit"])

    def test_registry_passes_generic_request_modes_to_chat_backend(self):
        mod = _load_module()
        registry = mod.ProviderRegistry(
            config={
                "providers": [
                    {
                        "id": "chat-provider",
                        "__template_key": "openai_chat",
                        "base_url": "https://api.example.com/v1",
                        "api_keys": ["test-key"],
                        "model": "gpt-image",
                        "generate_request_mode": "non_stream",
                        "edit_request_mode": "stream",
                    }
                ]
            },
            imgr=object(),
            data_dir=Path("/tmp"),
        )

        backend = registry.get_backend("chat-provider")

        self.assertEqual(backend.kwargs["generate_request_mode"], "non_stream")
        self.assertEqual(backend.kwargs["edit_request_mode"], "stream")
        self.assertIsNone(backend.kwargs["enable_stream_generate"])
        self.assertIsNone(backend.kwargs["enable_stream_edit"])

    def test_registry_falls_back_to_legacy_stream_flags(self):
        mod = _load_module()
        registry = mod.ProviderRegistry(
            config={
                "providers": [
                    {
                        "id": "chat-provider",
                        "__template_key": "openai_chat",
                        "base_url": "https://api.example.com/v1",
                        "api_keys": ["test-key"],
                        "model": "gpt-image",
                        "enable_stream_generate": False,
                        "enable_stream_edit": True,
                    }
                ]
            },
            imgr=object(),
            data_dir=Path("/tmp"),
        )

        backend = registry.get_backend("chat-provider")

        self.assertEqual(backend.kwargs["generate_request_mode"], "non_stream")
        self.assertEqual(backend.kwargs["edit_request_mode"], "stream")
        self.assertFalse(backend.kwargs["enable_stream_generate"])
        self.assertTrue(backend.kwargs["enable_stream_edit"])

    def test_validate_rejects_invalid_request_mode(self):
        mod = _load_module()
        registry = mod.ProviderRegistry(
            config={
                "providers": [
                    {
                        "id": "images-provider",
                        "__template_key": "openai_images",
                        "base_url": "https://api.example.com/v1",
                        "api_keys": ["test-key"],
                        "model": "gpt-image",
                        "generate_request_mode": "sometimes",
                    }
                ]
            },
            imgr=object(),
            data_dir=Path("/tmp"),
        )

        errors = registry.validate()

        self.assertEqual(
            errors,
            [
                "provider 'images-provider' invalid generate_request_mode: sometimes; runtime will fallback to auto"
            ],
        )

    def test_validate_reports_single_path_provider_ignores_request_mode(self):
        mod = _load_module()
        registry = mod.ProviderRegistry(
            config={
                "providers": [
                    {
                        "id": "images-provider",
                        "__template_key": "openai_images",
                        "base_url": "https://api.example.com/v1",
                        "api_keys": ["test-key"],
                        "model": "gpt-image",
                        "generate_request_mode": "stream",
                    }
                ]
            },
            imgr=object(),
            data_dir=Path("/tmp"),
        )

        errors = registry.validate()

        self.assertEqual(
            errors,
            [
                "provider 'images-provider' set generate_request_mode=stream, but template 'openai_images' currently ignores request_mode (single-path backend)"
            ],
        )


if __name__ == "__main__":
    unittest.main()
