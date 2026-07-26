import asyncio
import importlib.util
import io
import sys
import types
from pathlib import Path

import pytest
from PIL import Image as PILImage

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "output_routing_testpkg"
CORE_PACKAGE_NAME = f"{PACKAGE_NAME}.core"


class _Logger:
    def debug(self, *args, **kwargs):
        return None

    def info(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None

    def error(self, *args, **kwargs):
        return None


class _Registry:
    def __init__(self, backends):
        self.backends = backends

    def get_backend(self, provider_id):
        return self.backends[provider_id]

    def provider_ids(self):
        return list(self.backends)

    async def close(self):
        return None


def _load_module(name: str):
    module_name = f"{CORE_PACKAGE_NAME}.{name}"
    spec = importlib.util.spec_from_file_location(
        module_name,
        ROOT / "core" / f"{name}.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def modules():
    for name in list(sys.modules):
        if name.startswith(PACKAGE_NAME) or name in {"astrbot", "astrbot.api"}:
            sys.modules.pop(name, None)

    package = types.ModuleType(PACKAGE_NAME)
    package.__path__ = [str(ROOT)]
    sys.modules[PACKAGE_NAME] = package

    core_package = types.ModuleType(CORE_PACKAGE_NAME)
    core_package.__path__ = [str(ROOT / "core")]
    sys.modules[CORE_PACKAGE_NAME] = core_package

    astrbot_module = types.ModuleType("astrbot")
    sys.modules["astrbot"] = astrbot_module
    api_module = types.ModuleType("astrbot.api")
    api_module.logger = _Logger()
    sys.modules["astrbot.api"] = api_module

    provider_registry_module = types.ModuleType(
        f"{CORE_PACKAGE_NAME}.provider_registry"
    )
    provider_registry_module.ProviderRegistry = _Registry
    sys.modules[provider_registry_module.__name__] = provider_registry_module

    gitee_edit_module = types.ModuleType(f"{CORE_PACKAGE_NAME}.gitee_edit")
    gitee_edit_module.GiteeEditBackend = type("GiteeEditBackend", (), {})
    sys.modules[gitee_edit_module.__name__] = gitee_edit_module

    output_spec = _load_module("output_spec")
    draw_service = _load_module("draw_service")
    edit_router = _load_module("edit_router")
    return output_spec, draw_service, edit_router


class _AdaptiveDrawBackend:
    def __init__(self, *, fail: bool = False):
        self.fail = fail
        self.calls = []

    @staticmethod
    def resolve_output_intent(intent):
        return {
            "aspect_ratio": intent.aspect_ratio,
            "resolution": intent.resolution,
        }

    async def generate(self, prompt, *, aspect_ratio=None, resolution=None):
        self.calls.append((prompt, aspect_ratio, resolution))
        if self.fail:
            raise RuntimeError("provider unavailable")
        return Path("/tmp/generated.png")


class _AdaptiveEditBackend:
    def __init__(self):
        self.calls = []

    @staticmethod
    def resolve_output_intent(intent):
        return {
            "aspect_ratio": intent.aspect_ratio,
            "resolution": intent.resolution,
        }

    async def edit(self, prompt, images, *, aspect_ratio=None, resolution=None):
        self.calls.append((prompt, len(images), aspect_ratio, resolution))
        return Path("/tmp/edited.png")


class _StrictLegacyDrawBackend:
    def __init__(self):
        self.calls = []

    async def generate(self, prompt, *, resolution=None):
        self.calls.append((prompt, resolution))
        return Path("/tmp/legacy.png")


def test_draw_fallback_resolves_each_provider_override(modules):
    output_spec, draw_service, _ = modules
    first = _AdaptiveDrawBackend(fail=True)
    second = _AdaptiveDrawBackend()
    registry = _Registry({"first": first, "second": second})
    config = {
        "features": {
            "draw": {
                "chain": [
                    {"provider_id": "first", "output": "1:1 2K"},
                    {"provider_id": "second", "output": "16:9 2K"},
                ],
                "default_output": "4:3 1K",
            }
        }
    }
    service = draw_service.ImageDrawService(
        config,
        imgr=object(),
        data_dir=Path("/tmp"),
        registry=registry,
    )

    result = asyncio.run(
        service.generate(
            "draw",
            output_intent=output_spec.OutputIntent(resolution="4K"),
        )
    )

    assert result == Path("/tmp/generated.png")
    assert first.calls == [("draw", "1:1", "4K")]
    assert second.calls == [("draw", "16:9", "4K")]


def test_draw_does_not_send_aspect_ratio_to_strict_legacy_backend(modules):
    output_spec, draw_service, _ = modules
    backend = _StrictLegacyDrawBackend()
    service = draw_service.ImageDrawService(
        {"features": {"draw": {"chain": ["legacy"]}}},
        imgr=object(),
        data_dir=Path("/tmp"),
        registry=_Registry({"legacy": backend}),
    )

    result = asyncio.run(
        service.generate(
            "draw",
            output_intent=output_spec.OutputIntent(aspect_ratio="16:9"),
        )
    )

    assert result == Path("/tmp/legacy.png")
    assert backend.calls == [("draw", None)]


def test_draw_extracts_prompt_controls_before_exact_default(modules):
    _, draw_service, _ = modules
    backend = _AdaptiveDrawBackend()
    service = draw_service.ImageDrawService(
        {
            "features": {
                "draw": {
                    "chain": ["adaptive"],
                    "default_output": "1024x1024",
                }
            }
        },
        imgr=object(),
        data_dir=Path("/tmp"),
        registry=_Registry({"adaptive": backend}),
    )

    result = asyncio.run(service.generate("电影海报, 16:9, 4K"))

    assert result == Path("/tmp/generated.png")
    assert backend.calls == [("电影海报, 16:9, 4K", "16:9", "4K")]


def _png_bytes(size: tuple[int, int]) -> bytes:
    output = io.BytesIO()
    PILImage.new("RGB", size, "white").save(output, format="PNG")
    return output.getvalue()


def test_edit_infers_single_source_aspect_ratio(modules):
    _, _, edit_router = modules
    backend = _AdaptiveEditBackend()
    router = edit_router.EditRouter(
        {"features": {"edit": {"chain": ["adaptive"]}}},
        imgr=object(),
        data_dir=Path("/tmp"),
        registry=_Registry({"adaptive": backend}),
    )

    result = asyncio.run(router.edit("edit", [_png_bytes((1600, 900))]))

    assert result == Path("/tmp/edited.png")
    assert backend.calls == [("edit", 1, "16:9", None)]


def test_edit_can_disable_source_aspect_inference_for_selfie(modules):
    _, _, edit_router = modules
    backend = _AdaptiveEditBackend()
    router = edit_router.EditRouter(
        {"features": {"edit": {"chain": ["adaptive"]}}},
        imgr=object(),
        data_dir=Path("/tmp"),
        registry=_Registry({"adaptive": backend}),
    )

    asyncio.run(
        router.edit(
            "selfie",
            [_png_bytes((900, 1600))],
            infer_source_aspect=False,
        )
    )

    assert backend.calls == [("selfie", 1, None, None)]
