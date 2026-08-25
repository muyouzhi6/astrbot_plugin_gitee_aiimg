import sys
import types
import unittest
import os
import time
from tempfile import TemporaryDirectory
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
    provider_id: str | None = None
    preset_name: str | None = None
    effective_prompt: str = ""
    user_prompt: str = ""
    source_command: str = ""
    variant_title: str | None = None
    output: str = ""


@dataclass
class _StubParsedImageRequest:
    spec: object | None = None


@dataclass
class _StubPlannedPromptItem:
    title: str = ""
    prompt: str = ""
    variation_focus: str = ""
    aspect_ratio: str = "3:4"


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

    @staticmethod
    def fromBytes(data: bytes):
        return _DummyImage(path=f"bytes:{len(data)}")


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

    @staticmethod
    def fromURL(url: str):
        return _DummyMessageComponent(url=url)


class _DummyStar:
    def __init__(self, context):
        self.context = context


class _DummyStarTools:
    @staticmethod
    def get_data_dir(name: str):
        return Path("/tmp") / name

    @staticmethod
    async def create_message(**kwargs):
        return types.SimpleNamespace(**kwargs)


@dataclass
class _DummyMessageMember:
    user_id: str
    nickname: str | None = None


class _DummyCustomFilter:
    def __init__(self, raise_error=True, **kwargs):
        self.raise_error = raise_error


class _DummyFilter:
    CustomFilter = _DummyCustomFilter
    EventMessageType = types.SimpleNamespace(ALL="all")

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


class _McpValue:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


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

    def get_platform_name(self):
        return "aiocqhttp"


def _clear_modules():
    for name in list(sys.modules):
        if name.startswith(PACKAGE_NAME) or name in {
            "astrbot",
            "astrbot.api",
            "astrbot.api.event",
            "astrbot.api.message_components",
            "astrbot.api.platform",
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
        CallToolResult=_McpValue,
        TextContent=_McpValue,
        ImageContent=_McpValue,
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
        "astrbot.api.platform",
        MessageMember=_DummyMessageMember,
    )
    _install_stub_module(
        "astrbot.core.utils.astrbot_path",
        get_astrbot_temp_path=lambda: Path("/tmp"),
    )

    _install_stub_module(
        f"{CORE_PACKAGE_NAME}.gemini_edit", GeminiEditBackend=_StubBackend
    )
    _install_stub_module(
        f"{CORE_PACKAGE_NAME}.gemini_flow2api",
        Flow2ApiVideoBackend=_StubBackend,
        GeminiFlow2ApiBackend=_StubBackend,
    )
    _install_stub_module(
        f"{CORE_PACKAGE_NAME}.gitee_edit", GiteeEditBackend=_StubBackend
    )
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
        f"{CORE_PACKAGE_NAME}.sora2_video_service",
        Sora2VideoService=_StubBackend,
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
    _install_stub_module(
        f"{CORE_PACKAGE_NAME}.draw_service", ImageDrawService=_StubService
    )
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
    _install_stub_module(
        f"{CORE_PACKAGE_NAME}.image_manager", ImageManager=_StubService
    )
    _install_stub_module(
        f"{CORE_PACKAGE_NAME}.nanobanana", NanoBananaService=_StubService
    )
    _install_stub_module(f"{CORE_PACKAGE_NAME}.ref_store", ReferenceStore=_StubStore)
    _install_stub_module(
        f"{CORE_PACKAGE_NAME}.utils",
        close_session=lambda *args, **kwargs: None,
        collect_at_user_ids=lambda *args, **kwargs: [],
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
    async def test_video_image_snapshot_survives_event_temp_file_cleanup(self):
        mod = _load_module()
        plugin = mod.GiteeAIImagePlugin(
            context=types.SimpleNamespace(),
            config={"features": {"video": {"chain": ["video"]}}},
        )
        raw_image = b"\x89PNG\r\n\x1a\nvideo-reference"

        with TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "context-aware-compressed-test.jpg"
            image_path.write_bytes(raw_image)

            class _ImageSegment:
                async def convert_to_base64(self):
                    image_path.read_bytes()
                    image_path.unlink()
                    return "ignored"

            async def _get_images_from_event(*args, **kwargs):
                return [_ImageSegment()]

            mod.get_images_from_event = _get_images_from_event
            mod.decode_base64_image_payload = lambda payload: raw_image

            snapshot = await plugin._capture_video_image_snapshot(_DummyEvent())
            self.assertEqual(snapshot, (True, raw_image))
            self.assertFalse(image_path.exists())

            calls: list[bytes | None] = []

            class _Backend:
                async def generate_video_url(self, *, prompt, image_bytes):
                    calls.append(image_bytes)
                    return "https://cdn.example/video.mp4"

            plugin.registry = types.SimpleNamespace(
                get_video_backend=lambda provider_id: _Backend()
            )
            plugin._get_video_chain = lambda: ["video"]

            async def _send_video_result(*args, **kwargs):
                return None

            plugin._send_video_result = _send_video_result
            plugin._video_end = _send_video_result

            async def _noop(*args, **kwargs):
                return None

            mod.mark_success = _noop
            mod.mark_failed = _noop

            await plugin._async_generate_video(
                _DummyEvent(),
                "animate the reference",
                "user-1",
                image_snapshot=snapshot,
            )

            self.assertEqual(calls, [raw_image])

    async def test_video_chain_falls_back_after_download_or_send_failure(self):
        mod = _load_module()
        plugin = mod.GiteeAIImagePlugin(
            context=types.SimpleNamespace(),
            config={"features": {"video": {"chain": ["first", "second"]}}},
        )
        calls: list[str] = []

        class _Backend:
            def __init__(self, provider_id):
                self.provider_id = provider_id

            async def generate_video_url(self, *, prompt, image_bytes):
                calls.append(f"generate:{self.provider_id}")
                return f"https://cdn.example/{self.provider_id}.mp4"

        plugin.registry = types.SimpleNamespace(
            get_video_backend=lambda provider_id: _Backend(provider_id)
        )

        async def _send_video_result(event, url, **kwargs):
            calls.append(f"send:{url.rsplit('/', 1)[-1]}")
            if url.endswith("first.mp4"):
                raise RuntimeError("first content download failed")

        plugin._send_video_result = _send_video_result

        async def _video_end(*args, **kwargs):
            return None

        plugin._video_end = _video_end

        async def _noop(*args, **kwargs):
            return None

        mod.mark_success = _noop
        mod.mark_failed = _noop

        await plugin._async_generate_video(
            _DummyEvent(),
            "animate the reference",
            "user-1",
            image_snapshot=(False, None),
        )

        self.assertEqual(
            calls,
            [
                "generate:first",
                "send:first.mp4",
                "generate:second",
                "send:second.mp4",
            ],
        )

    async def test_video_chain_forwards_single_request_download_hint(self):
        mod = _load_module()
        plugin = mod.GiteeAIImagePlugin(
            context=types.SimpleNamespace(),
            config={"features": {"video": {"chain": ["3365"]}}},
        )
        sent_kwargs = {}

        class _Backend:
            async def generate_video_url(self, *, prompt, image_bytes):
                return types.SimpleNamespace(
                    url="https://api.3365api.cn/v1/videos/task/content",
                    download_headers={"Authorization": "Bearer test-key"},
                    single_request_download=True,
                )

        plugin.registry = types.SimpleNamespace(
            get_video_backend=lambda provider_id: _Backend()
        )

        async def _send_video_result(event, url, **kwargs):
            sent_kwargs.update(kwargs)

        async def _noop(*args, **kwargs):
            return None

        plugin._send_video_result = _send_video_result
        plugin._video_end = _noop
        mod.mark_success = _noop
        mod.mark_failed = _noop

        await plugin._async_generate_video(
            _DummyEvent(),
            "a paper airplane",
            "user-1",
            image_snapshot=(False, None),
        )

        self.assertEqual(
            sent_kwargs,
            {
                "download_headers": {"Authorization": "Bearer test-key"},
                "single_request_download": True,
            },
        )

    async def test_auto_video_send_prefers_url_before_file_download(self):
        mod = _load_module()
        plugin = mod.GiteeAIImagePlugin(
            context=types.SimpleNamespace(),
            config={"features": {"video": {"send_mode": "auto"}}},
        )
        calls: list[str] = []

        class _Video:
            @staticmethod
            def fromURL(url: str):
                calls.append("url_component")
                return ("video_url", url)

            @staticmethod
            def fromFileSystem(path: str):
                calls.append("file_component")
                return ("video_file", path)

        class _Event(_DummyEvent):
            async def send(self, payload):
                calls.append("send")
                await super().send(payload)

        async def _download_video(*args, **kwargs):
            calls.append("download")
            raise AssertionError("auto mode should not download when URL send succeeds")

        mod.Video = _Video
        plugin.videomgr = types.SimpleNamespace(download_video=_download_video)
        event = _Event()

        await plugin._send_video_result(event, "https://cdn.example/video.mp4")

        self.assertEqual(calls, ["url_component", "send"])
        self.assertEqual(
            event.sent,
            [("chain", [("video_url", "https://cdn.example/video.mp4")])],
        )

    async def test_auto_video_send_falls_back_to_file_after_url_failure(self):
        mod = _load_module()
        plugin = mod.GiteeAIImagePlugin(
            context=types.SimpleNamespace(),
            config={"features": {"video": {"send_mode": "auto"}}},
        )
        calls: list[str] = []
        video_path = Path("/tmp/video.mp4")

        class _Video:
            @staticmethod
            def fromURL(url: str):
                calls.append("url_component")
                return ("video_url", url)

            @staticmethod
            def fromFileSystem(path: str):
                calls.append("file_component")
                return ("video_file", path)

        class _Event(_DummyEvent):
            async def send(self, payload):
                kind = payload[1][0][0]
                calls.append(f"send_{kind}")
                if kind == "video_url":
                    raise RuntimeError("URL send failed")
                await super().send(payload)

        async def _download_video(*args, **kwargs):
            calls.append("download")
            return video_path

        mod.Video = _Video
        plugin.videomgr = types.SimpleNamespace(download_video=_download_video)
        event = _Event()

        await plugin._send_video_result(event, "https://cdn.example/video.mp4")

        self.assertEqual(
            calls,
            [
                "url_component",
                "send_video_url",
                "download",
                "file_component",
                "send_video_file",
            ],
        )
        self.assertEqual(
            event.sent,
            [("chain", [("video_file", str(video_path))])],
        )

    async def test_auto_video_send_raises_when_url_and_file_both_fail(self):
        mod = _load_module()
        plugin = mod.GiteeAIImagePlugin(
            context=types.SimpleNamespace(),
            config={"features": {"video": {"send_mode": "auto"}}},
        )

        class _Video:
            @staticmethod
            def fromURL(url: str):
                return ("video_url", url)

            @staticmethod
            def fromFileSystem(path: str):
                return ("video_file", path)

        class _Event(_DummyEvent):
            async def send(self, payload):
                raise RuntimeError("QQ rejected video")

        async def _download_video(*args, **kwargs):
            raise RuntimeError("download failed")

        mod.Video = _Video
        plugin.videomgr = types.SimpleNamespace(download_video=_download_video)

        with self.assertRaisesRegex(RuntimeError, "URL 和本地文件发送均失败"):
            await plugin._send_video_result(_Event(), "https://cdn.example/video.mp4")

    async def test_authenticated_video_skips_url_send_and_downloads_with_headers(self):
        mod = _load_module()
        plugin = mod.GiteeAIImagePlugin(
            context=types.SimpleNamespace(),
            config={"features": {"video": {"send_mode": "auto"}}},
        )
        calls: list[object] = []
        video_path = Path("/tmp/video.mp4")

        class _Video:
            @staticmethod
            def fromURL(url: str):
                calls.append("url_component")
                return ("video_url", url)

            @staticmethod
            def fromFileSystem(path: str):
                calls.append("file_component")
                return ("video_file", path)

        async def _download_video(url: str, **kwargs):
            calls.append(("download", url, kwargs))
            return video_path

        mod.Video = _Video
        plugin.videomgr = types.SimpleNamespace(download_video=_download_video)
        event = _DummyEvent()

        await plugin._send_video_result(
            event,
            "https://gateway.example/v1/videos/task/content",
            download_headers={"Authorization": "Bearer test-key"},
            single_request_download=True,
        )

        self.assertEqual(
            calls,
            [
                (
                    "download",
                    "https://gateway.example/v1/videos/task/content",
                    {
                        "timeout_seconds": 300,
                        "headers": {"Authorization": "Bearer test-key"},
                        "single_request_download": True,
                    },
                ),
                "file_component",
            ],
        )
        self.assertEqual(
            event.sent,
            [("chain", [("video_file", str(video_path))])],
        )

    async def test_video_send_ignores_invalid_timeout_config_values(self):
        mod = _load_module()
        plugin = mod.GiteeAIImagePlugin(
            context=types.SimpleNamespace(),
            config={
                "features": {
                    "video": {
                        "send_mode": "url",
                        "send_timeout_seconds": "not-a-number",
                        "download_timeout_seconds": "not-a-number",
                    }
                }
            },
        )

        class _Video:
            @staticmethod
            def fromURL(url: str):
                return ("video_url", url)

        mod.Video = _Video
        event = _DummyEvent()

        await plugin._send_video_result(event, "https://cdn.example/video.mp4")

        self.assertEqual(
            event.sent,
            [("chain", [("video_url", "https://cdn.example/video.mp4")])],
        )

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

    async def test_batch_item_ratio_fills_missing_group_ratio(self):
        mod = _load_module()
        plugin = mod.GiteeAIImagePlugin(context=types.SimpleNamespace(), config={})
        calls = []

        class _Draw:
            async def generate(self, prompt, **kwargs):
                calls.append((prompt, kwargs))
                return Path("/tmp/planned.png")

        plugin.draw = _Draw()
        spec = mod.ImageTaskSpec(
            mode="draw",
            provider_id=None,
            preset_name=None,
            effective_prompt="cinematic portrait",
            user_prompt="cinematic portrait",
            source_command="llm_batch",
            variant_title="portrait",
            output="16:9",
        )

        result = await plugin._execute_image_task_spec(
            _DummyEvent(),
            spec,
            output_intent=mod.OutputIntent(resolution="4K"),
        )

        self.assertEqual(result.image_path, Path("/tmp/planned.png"))
        self.assertEqual(len(calls), 1)
        self.assertEqual(
            calls[0][1]["output_intent"],
            mod.OutputIntent(aspect_ratio="16:9", resolution="4K"),
        )

    def test_selfie_default_output_always_contains_an_aspect_ratio(self):
        mod = _load_module()
        fallback = mod.GiteeAIImagePlugin(
            context=types.SimpleNamespace(),
            config={"features": {"selfie": {"default_output": "4K"}}},
        )
        configured = mod.GiteeAIImagePlugin(
            context=types.SimpleNamespace(),
            config={
                "features": {
                    "selfie": {
                        "default_output": "4K",
                        "default_aspect_ratio": "16:9",
                    }
                }
            },
        )

        self.assertEqual(fallback._get_selfie_default_output(), "3:4 4K")
        self.assertEqual(configured._get_selfie_default_output(), "16:9 4K")

    def test_selfie_default_prompt_preserves_user_capture_intent(self):
        mod = _load_module()
        plugin = mod.GiteeAIImagePlugin(
            context=types.SimpleNamespace(),
            config={"features": {"selfie": {"prompt_prefix": ""}}},
        )

        prompt = plugin._build_selfie_prompt(
            "手持前置摄像头自拍，窗边自然光",
            extra_refs=0,
        )

        self.assertIn("请根据参考图创作一张符合用户要求的人像图片", prompt)
        self.assertIn("用户指定的图像类型、拍摄视角", prompt)
        self.assertIn("普通手持自拍或前置摄像头自拍", prompt)
        self.assertIn("拍摄设备位于画面外", prompt)
        self.assertIn("以上规则只用于保持拍摄逻辑一致", prompt)
        self.assertIn(
            "用户要求（最高优先级）：手持前置摄像头自拍，窗边自然光",
            prompt,
        )
        self.assertNotIn("专业人像摄影质感", prompt)
        self.assertNotIn("中性日光白平衡", prompt)
        self.assertNotIn("固定机位", prompt)
        self.assertNotIn("双手自然且不持物", prompt)
        self.assertLess(prompt.index("用户要求"), prompt.index("拍摄设备一致性"))

    async def test_selfie_prompt_uses_cached_life_context_without_generation(self):
        mod = _load_module()
        calls = []

        class _LifeScheduler:
            async def get_life_context(self, **kwargs):
                calls.append(kwargs)
                return {
                    "outfit": "白色衬衫搭配蓝色半裙，赤足",
                    "schedule": "上午去咖啡店，下午在家看书",
                }

        context = types.SimpleNamespace(
            get_registered_star=lambda name: (
                types.SimpleNamespace(star_cls=_LifeScheduler())
                if name == "astrbot_plugin_life_scheduler"
                else None
            )
        )
        plugin = mod.GiteeAIImagePlugin(
            context=context,
            config={"features": {"selfie": {"prompt_prefix": ""}}},
        )

        life_context = await plugin._get_life_context_without_llm()
        prompt = plugin._build_selfie_prompt(
            "窗边自然光自拍",
            extra_refs=0,
            life_context=life_context,
        )

        self.assertEqual(calls, [{"allow_generate": False}])
        self.assertIn("今日穿搭：白色衬衫搭配蓝色半裙，赤足", prompt)
        self.assertIn("今日日程：上午去咖啡店，下午在家看书", prompt)
        self.assertIn("用户要求（最高优先级）：窗边自然光自拍", prompt)

    async def test_selfie_life_context_degrades_without_optional_plugin(self):
        mod = _load_module()
        plugin = mod.GiteeAIImagePlugin(
            context=types.SimpleNamespace(),
            config={"features": {"selfie": {"prompt_prefix": ""}}},
        )

        self.assertEqual(await plugin._get_life_context_without_llm(), {})
        prompt = plugin._build_selfie_prompt("自然自拍", extra_refs=0)
        self.assertNotIn("今日生活状态", prompt)

    async def test_selfie_life_context_does_not_fallback_to_legacy_generating_api(self):
        mod = _load_module()
        called = False

        class _LegacyLifeScheduler:
            async def get_life_context(self):
                nonlocal called
                called = True
                return {"outfit": "不应读取"}

        plugin = mod.GiteeAIImagePlugin(
            context=types.SimpleNamespace(
                get_registered_star=lambda name: types.SimpleNamespace(
                    star_cls=_LegacyLifeScheduler()
                )
            ),
            config={},
        )

        self.assertEqual(await plugin._get_life_context_without_llm(), {})
        self.assertFalse(called)

    def test_selfie_prompt_allows_explicit_mirror_selfie_and_visible_phone(self):
        mod = _load_module()
        plugin = mod.GiteeAIImagePlugin(
            context=types.SimpleNamespace(),
            config={"features": {"selfie": {"prompt_prefix": ""}}},
        )

        prompt = plugin._build_selfie_prompt(
            "对镜自拍，手机自然入镜",
            extra_refs=0,
        )

        self.assertIn("明确要求对镜自拍、手机入镜或展示拍摄设备", prompt)
        self.assertIn("才让相应设备自然出现", prompt)
        self.assertIn("数量、持握位置和镜面反射合理", prompt)
        self.assertIn("不得覆盖或改写用户明确要求", prompt)
        self.assertIn("用户要求（最高优先级）：对镜自拍，手机自然入镜", prompt)
        self.assertNotIn("不是镜面自拍", prompt)
        self.assertNotIn("画面不得出现手机", prompt)

    def test_selfie_prompt_preserves_third_person_actions_and_held_objects(self):
        mod = _load_module()
        plugin = mod.GiteeAIImagePlugin(
            context=types.SimpleNamespace(),
            config={"features": {"selfie": {"prompt_prefix": ""}}},
        )

        prompt = plugin._build_selfie_prompt(
            "恋人视角随手拍，手里拿着咖啡杯",
            extra_refs=0,
        )

        self.assertIn("第三人拍摄、恋人视角或定时拍摄", prompt)
        self.assertIn("人物手势和手持日常物品遵循用户要求", prompt)
        self.assertIn(
            "用户要求（最高优先级）：恋人视角随手拍，手里拿着咖啡杯",
            prompt,
        )
        self.assertNotIn("双手自然且不持物", prompt)

    def test_selfie_custom_prefix_keeps_capture_policy_without_overriding_user(self):
        mod = _load_module()
        plugin = mod.GiteeAIImagePlugin(
            context=types.SimpleNamespace(),
            config={
                "features": {"selfie": {"prompt_prefix": "固定角色外貌，电影感照片。"}}
            },
        )

        prompt = plugin._build_selfie_prompt("", extra_refs=2)

        self.assertIn("固定角色外貌，电影感照片。", prompt)
        self.assertIn("用户要求（最高优先级）：自然真实的人像照片", prompt)
        self.assertIn("额外参考图数量：2", prompt)
        self.assertIn("拍摄设备一致性", prompt)
        self.assertIn("明确要求对镜自拍、手机入镜或展示拍摄设备", prompt)
        self.assertLess(prompt.index("用户要求"), prompt.index("拍摄设备一致性"))

    async def test_weixin_send_temp_file_is_removed_after_send(self):
        mod = _load_module()
        with TemporaryDirectory() as td:
            data_dir = Path(td)
            original = data_dir / "original.jpg"
            original.write_bytes(b"original")
            temp_dir = data_dir / "Temp"
            temp_dir.mkdir()
            temp = temp_dir / "weixin_send_test.jpg"
            temp.write_bytes(b"optimized")

            plugin = mod.GiteeAIImagePlugin(context=types.SimpleNamespace(), config={})
            plugin.data_dir = data_dir

            async def _prepare_image_for_send(event, path):
                return temp

            plugin._prepare_image_for_send = _prepare_image_for_send
            event = _DummyEvent()

            result = await plugin._send_image_with_fallback(event, original)

            self.assertTrue(result.ok)
            self.assertFalse(temp.exists())
            self.assertTrue(original.exists())
            self.assertEqual(result.cached_path, original)

    async def test_ambiguous_send_error_is_not_retried(self):
        mod = _load_module()
        with TemporaryDirectory() as td:
            image_path = Path(td) / "image.png"
            image_path.write_bytes(b"image")
            plugin = mod.GiteeAIImagePlugin(
                context=types.SimpleNamespace(),
                config={},
            )
            plugin.data_dir = Path(td)

            class _TimeoutEvent(_DummyEvent):
                def __init__(self):
                    super().__init__()
                    self.send_calls = 0

                async def send(self, payload):
                    self.send_calls += 1
                    raise TimeoutError("adapter response timed out")

            event = _TimeoutEvent()
            result = await plugin._send_image_with_fallback(
                event,
                image_path,
                max_attempts=5,
            )

            self.assertFalse(result.ok)
            self.assertEqual(result.reason, "delivery_unknown")
            self.assertEqual(event.send_calls, 1)

    def test_weixin_send_temp_cleanup_removes_only_stale_and_overflow_files(self):
        mod = _load_module()
        with TemporaryDirectory() as td:
            data_dir = Path(td)
            temp_dir = data_dir / "Temp"
            temp_dir.mkdir()

            plugin = mod.GiteeAIImagePlugin(context=types.SimpleNamespace(), config={})
            plugin.data_dir = data_dir
            plugin.WEIXIN_SEND_TEMP_MAX_FILES = 2
            plugin.WEIXIN_SEND_TEMP_TTL_SECONDS = 60

            old_keep = temp_dir / "not_weixin_send_old.jpg"
            old_keep.write_bytes(b"keep")
            stale = temp_dir / "weixin_send_stale.jpg"
            oldest = temp_dir / "weixin_send_oldest.jpg"
            middle = temp_dir / "weixin_send_middle.jpg"
            newest = temp_dir / "weixin_send_newest.jpg"
            for p in (stale, oldest, middle, newest):
                p.write_bytes(b"x")

            now = time.time()
            os.utime(old_keep, (now - 3600, now - 3600))
            os.utime(stale, (now - 3600, now - 3600))
            os.utime(oldest, (now - 30, now - 30))
            os.utime(middle, (now - 15, now - 15))
            os.utime(newest, (now, now))

            plugin._cleanup_weixin_send_temp_images_sync()

            self.assertTrue(old_keep.exists())
            self.assertFalse(stale.exists())
            self.assertFalse(oldest.exists())
            self.assertTrue(middle.exists())
            self.assertTrue(newest.exists())

    def test_weixin_send_temp_cleanup_does_not_overdelete_after_stale_removal(self):
        mod = _load_module()
        with TemporaryDirectory() as td:
            data_dir = Path(td)
            temp_dir = data_dir / "Temp"
            temp_dir.mkdir()

            plugin = mod.GiteeAIImagePlugin(context=types.SimpleNamespace(), config={})
            plugin.data_dir = data_dir
            plugin.WEIXIN_SEND_TEMP_MAX_FILES = 2
            plugin.WEIXIN_SEND_TEMP_TTL_SECONDS = 60

            stale = [
                temp_dir / "weixin_send_stale_1.jpg",
                temp_dir / "weixin_send_stale_2.jpg",
                temp_dir / "weixin_send_stale_3.jpg",
            ]
            keep = [
                temp_dir / "weixin_send_keep_1.jpg",
                temp_dir / "weixin_send_keep_2.jpg",
            ]
            for p in [*stale, *keep]:
                p.write_bytes(b"x")

            now = time.time()
            for p in stale:
                os.utime(p, (now - 3600, now - 3600))
            for index, p in enumerate(keep):
                os.utime(p, (now - index, now - index))

            plugin._cleanup_weixin_send_temp_images_sync()

            self.assertTrue(all(not p.exists() for p in stale))
            self.assertTrue(all(p.exists() for p in keep))


if __name__ == "__main__":
    unittest.main()
