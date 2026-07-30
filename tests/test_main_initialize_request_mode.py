import importlib.util
import sys
import types
import unittest
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "main_init_request_mode_testpkg"
CORE_PACKAGE_NAME = f"{PACKAGE_NAME}.core"
PROVIDER_REGISTRY_MODULE_NAME = f"{CORE_PACKAGE_NAME}.provider_registry"
MAIN_MODULE_NAME = f"{PACKAGE_NAME}.main"


class _Logger:
    def __init__(self):
        self.warning_messages: list[str] = []

    def debug(self, *args, **kwargs):
        return None

    def info(self, *args, **kwargs):
        return None

    def warning(self, msg, *args, **kwargs):
        if args:
            try:
                msg = msg % args
            except Exception:
                msg = f"{msg} {' '.join(str(x) for x in args)}"
        self.warning_messages.append(str(msg))
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


class _DummyMessageComponent:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    @staticmethod
    def fromFileSystem(path: str):
        return _DummyMessageComponent(path=path)


class _DummyPlain:
    def __init__(self, text: str = "", **kwargs):
        self.text = text
        self.kwargs = kwargs


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
        Image=_DummyMessageComponent,
        Node=_DummyMessageComponent,
        Nodes=_DummyMessageComponent,
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
        GITEE_SUPPORTED_RATIOS={"1:1": ["1024x1024"]},
        normalize_size_text=lambda value: str(value or "").strip(),
        resolve_ratio_size=lambda *args, **kwargs: "1024x1024",
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
    _install_stub_module(
        f"{CORE_PACKAGE_NAME}.debouncer",
        Debouncer=_StubService,
    )
    _install_stub_module(
        f"{CORE_PACKAGE_NAME}.draw_service",
        ImageDrawService=_StubService,
    )
    _install_stub_module(
        f"{CORE_PACKAGE_NAME}.edit_router",
        EditRouter=_StubRouter,
    )
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
        f"{CORE_PACKAGE_NAME}.image_manager",
        ImageManager=_StubService,
    )
    _install_stub_module(
        f"{CORE_PACKAGE_NAME}.nanobanana",
        NanoBananaService=_StubService,
    )
    _install_stub_module(
        f"{CORE_PACKAGE_NAME}.ref_store",
        ReferenceStore=_StubStore,
    )
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

    spec = importlib.util.spec_from_file_location(
        MAIN_MODULE_NAME,
        ROOT / "main.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[MAIN_MODULE_NAME] = module
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module, logger


class MainInitializeRequestModeTests(unittest.IsolatedAsyncioTestCase):
    async def test_initialize_logs_fallback_warning_and_builds_consistent_backend(self):
        mod, logger = _load_module()
        plugin = mod.GiteeAIImagePlugin(
            context=types.SimpleNamespace(
                get_config=lambda: {"wake_prefix": ["."]},
            ),
            config={
                "providers": [
                    {
                        "id": "chat-provider",
                        "__template_key": "openai_chat",
                        "base_url": "https://api.example.com/v1",
                        "api_keys": ["test-key"],
                        "model": "gpt-image",
                        "generate_request_mode": "bogus",
                        "enable_stream_generate": False,
                    }
                ]
            },
        )
        plugin._patch_tool_image_cache_runtime = lambda: None
        plugin._register_preset_commands = lambda: None

        await plugin.initialize()

        self.assertEqual(plugin._wake_prefixes, (".",))
        backend = plugin.registry.get_backend("chat-provider")

        self.assertEqual(backend.kwargs["generate_request_mode"], "non_stream")
        self.assertFalse(backend.kwargs["enable_stream_generate"])
        self.assertTrue(
            any(
                "invalid generate_request_mode: bogus; runtime will fallback to non_stream via enable_stream_generate=false"
                in msg
                for msg in logger.warning_messages
            )
        )

    async def test_selfie_regex_fallback_handles_direct_slash_command(self):
        mod, _ = _load_module()
        plugin = mod.GiteeAIImagePlugin(
            context=types.SimpleNamespace(),
            config={},
        )
        plugin._is_selfie_enabled = lambda: True

        calls = []

        async def fake_do_selfie(event, prompt, backend=None):
            calls.append((event, prompt, backend))

        plugin._do_selfie = fake_do_selfie

        plain = mod.Plain()
        plain.text = "/自拍 窗边自然光"

        class DummyEvent:
            message_str = "/自拍 窗边自然光"

            def __init__(self):
                self.call_llm = False
                self.stopped = False

            def get_messages(self):
                return [plain]

            def should_call_llm(self, value):
                self.call_llm = value

            def stop_event(self):
                self.stopped = True

        event = DummyEvent()

        await plugin.selfie_regex_fallback(event)

        self.assertEqual(calls, [(event, "窗边自然光", None)])
        self.assertTrue(event.call_llm)
        self.assertTrue(event.stopped)

    async def test_selfie_regex_fallback_ignores_wake_stripped_bare_command(self):
        mod, _ = _load_module()
        plugin = mod.GiteeAIImagePlugin(
            context=types.SimpleNamespace(),
            config={},
        )
        plugin._is_selfie_enabled = lambda: True

        calls = []

        async def fake_do_selfie(event, prompt, backend=None):
            calls.append((event, prompt, backend))

        plugin._do_selfie = fake_do_selfie

        class DummyEvent:
            message_str = "自拍 窗边自然光"
            is_at_or_wake_command = True

            def __init__(self):
                self.call_llm = False
                self.stopped = False

            def get_extra(self, key, default=None):
                return default

            def should_call_llm(self, value):
                self.call_llm = value

            def stop_event(self):
                self.stopped = True

        event = DummyEvent()

        await plugin.selfie_regex_fallback(event)

        self.assertEqual(calls, [])
        self.assertFalse(event.call_llm)
        self.assertFalse(event.stopped)

    async def test_selfie_regex_fallback_skips_when_command_handler_active(self):
        mod, _ = _load_module()
        plugin = mod.GiteeAIImagePlugin(
            context=types.SimpleNamespace(),
            config={},
        )
        plugin._is_selfie_enabled = lambda: True

        calls = []

        async def fake_do_selfie(event, prompt, backend=None):
            calls.append((event, prompt, backend))

        plugin._do_selfie = fake_do_selfie

        class DummyHandler:
            handler_name = "selfie_command"

        class DummyEvent:
            message_str = "自拍 窗边自然光"
            is_at_or_wake_command = True

            def get_extra(self, key, default=None):
                if key == "activated_handlers":
                    return [DummyHandler()]
                return default

        await plugin.selfie_regex_fallback(DummyEvent())

        self.assertEqual(calls, [])

    async def test_selfie_regex_fallback_ignores_unwoken_bare_text(self):
        mod, _ = _load_module()
        plugin = mod.GiteeAIImagePlugin(
            context=types.SimpleNamespace(),
            config={},
        )
        plugin._is_selfie_enabled = lambda: True

        calls = []

        async def fake_do_selfie(event, prompt, backend=None):
            calls.append((event, prompt, backend))

        plugin._do_selfie = fake_do_selfie

        class DummyEvent:
            message_str = "自拍 窗边自然光"
            is_at_or_wake_command = False

            def get_extra(self, key, default=None):
                return default

        await plugin.selfie_regex_fallback(DummyEvent())

        self.assertEqual(calls, [])

    async def test_selfie_reference_regex_fallback_handles_image_prefixed_command(self):
        mod, _ = _load_module()
        plugin = mod.GiteeAIImagePlugin(
            context=types.SimpleNamespace(),
            config={},
        )
        plugin._wake_prefixes = ("/",)
        plugin._is_selfie_enabled = lambda: True

        calls = []

        async def fake_set_selfie_reference(event):
            calls.append(event)

        plugin._set_selfie_reference = fake_set_selfie_reference

        image = mod.Image()
        plain = mod.Plain()
        plain.text = "/自拍参考 设置"

        class DummyEvent:
            message_str = "图片 /自拍参考 设置"
            is_at_or_wake_command = False

            def __init__(self):
                self.call_llm = False
                self.stopped = False

            def get_extra(self, key, default=None):
                return default

            def get_messages(self):
                return [image, plain]

            def should_call_llm(self, value):
                self.call_llm = value

            def stop_event(self):
                self.stopped = True

        event = DummyEvent()
        yielded = []

        async for result in plugin.selfie_reference_regex_fallback(event):
            yielded.append(result)

        self.assertEqual(calls, [event])
        self.assertEqual(yielded, [])
        self.assertTrue(event.call_llm)
        self.assertTrue(event.stopped)

    async def test_group_image_command_filter_requires_configured_raw_prefix(self):
        mod, _ = _load_module()
        gate = mod.ImageCommandWakePrefixFilter()
        cfg = {"wake_prefix": ["."]}

        class DummyEvent:
            is_at_or_wake_command = True

            def __init__(self, texts, *, private=False, woken=True):
                self._private = private
                self.is_at_or_wake_command = woken
                self._messages = []
                for text in texts:
                    plain = mod.Plain()
                    plain.text = text
                    self._messages.append(plain)

            def is_private_chat(self):
                return self._private

            def get_messages(self):
                return self._messages

        self.assertFalse(gate.filter(DummyEvent(["绘图 一只猫"]), cfg))
        self.assertFalse(gate.filter(DummyEvent(["/绘图 一只猫"]), cfg))
        self.assertFalse(gate.filter(DummyEvent(["，绘图 一只猫"]), cfg))
        self.assertTrue(gate.filter(DummyEvent([".绘图 一只猫"]), cfg))
        self.assertTrue(gate.filter(DummyEvent(["", ".改图 加点光影"]), cfg))
        self.assertTrue(gate.filter(DummyEvent([".自拍 窗边自然光"]), cfg))
        self.assertTrue(gate.filter(DummyEvent([".批量2 aiimg 一只猫"]), cfg))
        self.assertTrue(gate.filter(DummyEvent([".表情包 加字"]), cfg))
        self.assertFalse(gate.filter(DummyEvent([". 表情包 加字"]), cfg))
        self.assertTrue(gate.filter(DummyEvent(["绘图 一只猫"], private=True), cfg))
        self.assertFalse(
            gate.filter(
                DummyEvent(["/绘图 一只猫"], private=True, woken=False),
                cfg,
            )
        )

    async def test_batch_fragment_requires_configured_prefix_at_segment_start(self):
        mod, _ = _load_module()
        plugin = mod.GiteeAIImagePlugin(
            context=types.SimpleNamespace(),
            config={},
        )
        plugin._wake_prefixes = (".",)

        class DummyEvent:
            def __init__(self, texts):
                self._messages = []
                for text in texts:
                    if text is None:
                        self._messages.append(mod.Image())
                        continue
                    plain = mod.Plain()
                    plain.text = text
                    self._messages.append(plain)

            def get_messages(self):
                return self._messages

        self.assertEqual(
            plugin._extract_batch_command_fragment(
                DummyEvent([None, ".批量2 aiimg 一只猫"])
            ),
            ".批量2 aiimg 一只猫",
        )
        self.assertEqual(
            plugin._extract_batch_command_fragment(
                DummyEvent([None, "/批量2 aiimg 一只猫"])
            ),
            "",
        )
        self.assertEqual(
            plugin._extract_batch_command_fragment(
                DummyEvent(["聊天里提到 .批量2 aiimg 一只猫"])
            ),
            "",
        )

    async def test_preset_fallback_requires_complete_prefixed_chain_command(self):
        mod, _ = _load_module()
        plugin = mod.GiteeAIImagePlugin(
            context=types.SimpleNamespace(),
            config={},
        )
        plugin._wake_prefixes = (".",)
        plugin.edit = types.SimpleNamespace(get_preset_names=lambda: ["表情包"])

        calls = []

        async def fake_has_images(event):
            return True

        async def fake_do_edit_direct(event, prompt, preset=None):
            calls.append((prompt, preset))

        plugin._has_message_images_or_avatar_mentions = fake_has_images
        plugin._do_edit_direct = fake_do_edit_direct

        class DummyEvent:
            message_str = ""

            def __init__(self, command_text):
                self.stopped = False
                plain = mod.Plain()
                plain.text = command_text
                self._messages = [object(), plain]

            def get_messages(self):
                return self._messages

            def stop_event(self):
                self.stopped = True

        wrong_prefix = DummyEvent("/表情包 加字")
        await plugin.preset_regex_fallback(wrong_prefix)
        embedded = DummyEvent("聊天里提到 .表情包 加字")
        await plugin.preset_regex_fallback(embedded)
        partial = DummyEvent(".表情包风格 加字")
        await plugin.preset_regex_fallback(partial)

        self.assertEqual(calls, [])
        self.assertFalse(wrong_prefix.stopped)
        self.assertFalse(embedded.stopped)
        self.assertFalse(partial.stopped)

        valid = DummyEvent(".表情包 加字")
        await plugin.preset_regex_fallback(valid)

        self.assertEqual(calls, [("加字", "表情包")])
        self.assertTrue(valid.stopped)

    async def test_chain_command_extraction_rejects_bare_group_text(self):
        mod, _ = _load_module()
        plugin = mod.GiteeAIImagePlugin(
            context=types.SimpleNamespace(),
            config={},
        )
        plugin._wake_prefixes = (".",)

        class DummyEvent:
            def __init__(self, text):
                plain = mod.Plain()
                plain.text = text
                self._messages = [plain]

            def get_messages(self):
                return self._messages

        self.assertEqual(
            plugin._extract_command_arg_from_chain(
                DummyEvent("改图 加点光影"),
                "改图",
            ),
            (False, ""),
        )
        self.assertEqual(
            plugin._extract_command_arg_from_chain(
                DummyEvent(".改图 加点光影"),
                "改图",
            ),
            (True, "加点光影"),
        )


if __name__ == "__main__":
    unittest.main()
