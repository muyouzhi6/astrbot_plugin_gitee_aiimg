import importlib.util
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
