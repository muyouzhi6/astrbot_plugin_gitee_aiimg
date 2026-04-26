import sys
import types
import unittest
from dataclasses import dataclass
from pathlib import Path
import importlib.util


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "batch_result_delivery_testpkg"
CORE_PACKAGE_NAME = f"{PACKAGE_NAME}.core"
PROVIDER_REGISTRY_MODULE_NAME = f"{CORE_PACKAGE_NAME}.provider_registry"
MAIN_MODULE_NAME = f"{PACKAGE_NAME}.main"


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


class _StubService:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class _StubRouter(_StubService):
    def get_available_backends(self):
        return []

    def get_preset_names(self):
        return []


class _StubStore(_StubService):
    pass


class _StubVideoManager(_StubService):
    pass


class _StubVertexSettings:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


@dataclass
class _StubImageTaskSpec:
    mode: str = ""
    preset_name: str | None = None
    effective_prompt: str = ""
    user_prompt: str = ""
    variant_title: str | None = None


@dataclass
class _StubParsedImageRequest:
    spec: object | None = None


@dataclass
class _StubPlannedPromptItem:
    title: str = ""
    prompt: str = ""
    variation_focus: str = ""


class _DummyPlain:
    def __init__(self, text: str = "", **kwargs):
        self.text = text
        self.kwargs = kwargs


class _DummyImage:
    def __init__(self, path: str = "", **kwargs):
        self.path = path
        self.kwargs = kwargs

    @staticmethod
    def fromFileSystem(path: str):
        return _DummyImage(path=path)


class _DummyNode:
    def __init__(self, content=None, **kwargs):
        self.content = list(content or [])
        self.kwargs = kwargs
        self.uin = kwargs.get("uin")
        self.name = kwargs.get("name")


class _DummyNodes:
    def __init__(self, nodes=None, **kwargs):
        self.nodes = list(nodes or [])
        self.kwargs = kwargs


class _DummyMessageComponent:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    @staticmethod
    def fromFileSystem(path: str):
        return _DummyMessageComponent(path=path)


class _DummyStar:
    def __init__(self, context):
        self.context = context


class _DummyStarTools:
    @staticmethod
    def get_data_dir(name: str):
        return Path("/tmp") / name


class _DummyFilter:
    def __getattr__(self, name):
        def decorator_factory(*args, **kwargs):
            def decorator(func):
                return func

            return decorator

        return decorator_factory


class _SubscriptableType:
    @classmethod
    def __class_getitem__(cls, item):
        return cls


@dataclass
class _Result:
    index: int
    success: bool
    value: object | None = None
    error: Exception | None = None


class _DummyEvent:
    def __init__(self):
        self.sent: list[tuple[str, object]] = []

    def plain_result(self, text: str):
        return ("plain", text)

    def chain_result(self, chain):
        return ("chain", chain)

    async def send(self, payload):
        self.sent.append(payload)

    def get_self_id(self):
        return "123456"


def _clear_modules():
    for name in list(sys.modules):
        if name.startswith(PACKAGE_NAME) or name in {
            "astrbot",
            "astrbot.api",
            "astrbot.api.event",
            "astrbot.api.message_components",
            "astrbot.api.star",
            "astrbot.core",
            "astrbot.core.utils",
            "astrbot.core.utils.astrbot_path",
            "mcp",
        }:
            sys.modules.pop(name, None)


def _install_stub_module(name: str, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module


def _load_module():
    _clear_modules()
    logger = _Logger()

    pkg = types.ModuleType(PACKAGE_NAME)
    pkg.__path__ = [str(ROOT)]
    sys.modules[PACKAGE_NAME] = pkg

    core_pkg = types.ModuleType(CORE_PACKAGE_NAME)
    core_pkg.__path__ = [str(ROOT / "core")]
    sys.modules[CORE_PACKAGE_NAME] = core_pkg

    mcp_mod = types.ModuleType("mcp")
    mcp_mod.types = types.SimpleNamespace(
        CallToolResult=type("CallToolResult", (), {})
    )
    sys.modules["mcp"] = mcp_mod

    astrbot_mod = types.ModuleType("astrbot")
    sys.modules["astrbot"] = astrbot_mod

    api_mod = types.ModuleType("astrbot.api")
    api_mod.logger = logger
    sys.modules["astrbot.api"] = api_mod

    _install_stub_module(
        "astrbot.api.event",
        AstrMessageEvent=type("AstrMessageEvent", (), {}),
        filter=_DummyFilter(),
    )
    _install_stub_module(
        "astrbot.api.message_components",
        At=_DummyMessageComponent,
        AtAll=_DummyMessageComponent,
        File=_DummyMessageComponent,
        Image=_DummyImage,
        Node=_DummyNode,
        Nodes=_DummyNodes,
        Plain=_DummyPlain,
        Reply=_DummyMessageComponent,
        Video=_DummyMessageComponent,
    )
    _install_stub_module(
        "astrbot.api.star",
        Context=type("Context", (), {}),
        Star=_DummyStar,
        StarTools=_DummyStarTools,
    )
    _install_stub_module(
        "astrbot.core.utils.astrbot_path",
        get_astrbot_temp_path=lambda: Path("/tmp"),
    )

    _install_stub_module(f"{CORE_PACKAGE_NAME}.gemini_edit", GeminiEditBackend=_StubBackend)
    _install_stub_module(
        f"{CORE_PACKAGE_NAME}.gemini_flow2api",
        Flow2ApiVideoBackend=_StubBackend,
        GeminiFlow2ApiBackend=_StubBackend,
    )
    _install_stub_module(f"{CORE_PACKAGE_NAME}.gitee_edit", GiteeEditBackend=_StubBackend)
    _install_stub_module(
        f"{CORE_PACKAGE_NAME}.gitee_sizes",
        GITEE_SUPPORTED_SIZES=["1024x1024"],
        GITEE_SUPPORTED_RATIOS={"1:1": ["1024x1024"]},
        normalize_size_text=lambda value: str(value or "").strip(),
        resolve_ratio_size=lambda *args, **kwargs: ("1024x1024", None),
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

    provider_registry_spec = importlib.util.spec_from_file_location(
        PROVIDER_REGISTRY_MODULE_NAME,
        ROOT / "core" / "provider_registry.py",
    )
    provider_registry_module = importlib.util.module_from_spec(provider_registry_spec)
    sys.modules[PROVIDER_REGISTRY_MODULE_NAME] = provider_registry_module
    assert provider_registry_spec and provider_registry_spec.loader
    provider_registry_spec.loader.exec_module(provider_registry_module)

    _install_stub_module(
        f"{CORE_PACKAGE_NAME}.batch_executor",
        BatchRunResult=type("BatchRunResult", (_SubscriptableType,), {}),
        run_batch=lambda *args, **kwargs: None,
    )
    _install_stub_module(f"{CORE_PACKAGE_NAME}.debouncer", Debouncer=_StubService)
    _install_stub_module(f"{CORE_PACKAGE_NAME}.draw_service", ImageDrawService=_StubService)
    _install_stub_module(f"{CORE_PACKAGE_NAME}.edit_router", EditRouter=_StubRouter)
    _install_stub_module(
        f"{CORE_PACKAGE_NAME}.emoji_feedback",
        mark_failed=lambda *args, **kwargs: None,
        mark_processing=lambda *args, **kwargs: None,
        mark_success=lambda *args, **kwargs: None,
    )
    _install_stub_module(
        f"{CORE_PACKAGE_NAME}.image_task_parser",
        ImageTaskSpec=_StubImageTaskSpec,
        ParsedImageRequest=_StubParsedImageRequest,
        parse_image_request=lambda *args, **kwargs: _StubParsedImageRequest(),
    )
    _install_stub_module(
        f"{CORE_PACKAGE_NAME}.llm_batch_planner",
        PlannedPromptItem=_StubPlannedPromptItem,
        build_batch_planning_prompt=lambda *args, **kwargs: "",
        parse_planned_prompt_items=lambda *args, **kwargs: [],
        validate_planned_prompt_items=lambda *args, **kwargs: [],
    )
    _install_stub_module(
        f"{CORE_PACKAGE_NAME}.image_format",
        decode_base64_image_payload=lambda *args, **kwargs: b"",
        guess_image_mime_and_ext=lambda *args, **kwargs: ("image/png", ".png"),
    )
    _install_stub_module(f"{CORE_PACKAGE_NAME}.image_manager", ImageManager=_StubService)
    _install_stub_module(f"{CORE_PACKAGE_NAME}.nanobanana", NanoBananaService=_StubService)
    _install_stub_module(f"{CORE_PACKAGE_NAME}.ref_store", ReferenceStore=_StubStore)
    _install_stub_module(
        f"{CORE_PACKAGE_NAME}.utils",
        close_session=lambda *args, **kwargs: None,
        get_images_from_event=lambda *args, **kwargs: [],
    )
    _install_stub_module(
        f"{CORE_PACKAGE_NAME}.video_manager",
        VideoManager=_StubVideoManager,
    )

    spec = importlib.util.spec_from_file_location(MAIN_MODULE_NAME, ROOT / "main.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[MAIN_MODULE_NAME] = module
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def _make_success_result(mod, index: int, image_name: str, mode: str = "selfie_ref"):
    spec = types.SimpleNamespace(
        mode=mode,
        preset_name=None,
        effective_prompt="ignored",
        user_prompt="ignored",
        variant_title="Window Lean",
    )
    value = mod.ExecutedImageTask(
        spec=spec,
        image_path=Path("/tmp") / image_name,
        task_meta={},
    )
    return _Result(index=index, success=True, value=value)


class BatchResultDeliveryTests(unittest.IsolatedAsyncioTestCase):
    async def test_batch_results_ignore_legacy_merge_setting_and_send_images_only(self):
        mod = _load_module()
        plugin = mod.GiteeAIImagePlugin(
            context=types.SimpleNamespace(),
            config={"features": {"batch": {"result_send_mode": "merge_forward"}}},
        )
        event = _DummyEvent()
        results = [
            _make_success_result(mod, 0, "one.png"),
            _make_success_result(mod, 1, "two.png"),
        ]
        image_paths: list[Path] = []

        async def _fake_send_image(evt, path):
            image_paths.append(Path(path))
            return True

        plugin._send_image_with_fallback = _fake_send_image

        await plugin._send_batch_results(event, results, title="LLM 批量自拍 x2")

        self.assertEqual(event.sent, [])
        self.assertEqual(image_paths, [Path("/tmp/one.png"), Path("/tmp/two.png")])

    async def test_batch_results_send_only_images_without_plain_messages(self):
        mod = _load_module()
        plugin = mod.GiteeAIImagePlugin(context=types.SimpleNamespace(), config={})
        event = _DummyEvent()
        results = [
            _make_success_result(mod, 0, "one.png"),
            _make_success_result(mod, 1, "two.png"),
        ]
        image_paths: list[Path] = []

        async def _fake_send_image(evt, path):
            image_paths.append(Path(path))
            return True

        plugin._send_image_with_fallback = _fake_send_image

        await plugin._send_batch_results(event, results, title="LLM 批量自拍 x2")

        self.assertEqual(event.sent, [])
        self.assertEqual(image_paths, [Path("/tmp/one.png"), Path("/tmp/two.png")])

    async def test_batch_results_stay_silent_on_partial_failures(self):
        mod = _load_module()
        plugin = mod.GiteeAIImagePlugin(context=types.SimpleNamespace(), config={})
        event = _DummyEvent()
        results = [
            _make_success_result(mod, 0, "one.png"),
            _Result(index=1, success=False, error=RuntimeError("boom")),
        ]
        image_paths: list[Path] = []

        async def _fake_send_image(evt, path):
            image_paths.append(Path(path))
            return True

        plugin._send_image_with_fallback = _fake_send_image

        await plugin._send_batch_results(event, results, title="LLM 批量自拍 x2")

        self.assertEqual(event.sent, [])
        self.assertEqual(image_paths, [Path("/tmp/one.png")])


if __name__ == "__main__":
    unittest.main()
