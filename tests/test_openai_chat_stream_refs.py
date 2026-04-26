import importlib.util
import sys
import types
import unittest
from base64 import b64decode
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "openai_chat_stream_testpkg"
CORE_PACKAGE_NAME = f"{PACKAGE_NAME}.core"
OPENAI_COMPAT_MODULE_NAME = f"{CORE_PACKAGE_NAME}.openai_compat_backend"
MODULE_NAME = f"{CORE_PACKAGE_NAME}.openai_chat_image_backend"


class _Logger:
    def debug(self, *args, **kwargs):
        return None

    def info(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None

    def error(self, *args, **kwargs):
        return None


class _MessageImage:
    def __init__(self, path: str):
        self.path = path

    @staticmethod
    def fromFileSystem(path: str):
        return _MessageImage(path)

    async def register_to_file_service(self):
        return f"https://files.example/{Path(self.path).name}"


class _DummyImageManager:
    def __init__(self, *, fail_download_error: Exception | None = None):
        self.saved_inputs: list[bytes] = []
        self.downloaded_urls: list[str] = []
        self.fail_download_error = fail_download_error

    async def save_image(self, data: bytes):
        self.saved_inputs.append(data)
        return Path(f"/tmp/input_{len(self.saved_inputs)}.png")

    async def download_image(self, url: str):
        self.downloaded_urls.append(url)
        if self.fail_download_error is not None:
            raise self.fail_download_error
        return Path("/tmp/result.png")


class _DummyChatCompletions:
    def __init__(self, results: list[object]):
        self.results = list(results)
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


class _DummyClient:
    def __init__(self, results: list[object]):
        self.chat = types.SimpleNamespace(completions=_DummyChatCompletions(results))


def _clear_modules():
    for name in [
        MODULE_NAME,
        OPENAI_COMPAT_MODULE_NAME,
        CORE_PACKAGE_NAME,
        PACKAGE_NAME,
        "astrbot",
        "astrbot.api",
        "astrbot.api.message_components",
    ]:
        sys.modules.pop(name, None)


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

    message_components_mod = types.ModuleType("astrbot.api.message_components")
    message_components_mod.Image = _MessageImage
    sys.modules["astrbot.api.message_components"] = message_components_mod

    openai_compat_spec = importlib.util.spec_from_file_location(
        OPENAI_COMPAT_MODULE_NAME,
        ROOT / "core" / "openai_compat_backend.py",
    )
    openai_compat_module = importlib.util.module_from_spec(openai_compat_spec)
    sys.modules[OPENAI_COMPAT_MODULE_NAME] = openai_compat_module
    assert openai_compat_spec and openai_compat_spec.loader
    openai_compat_spec.loader.exec_module(openai_compat_module)

    spec = importlib.util.spec_from_file_location(
        MODULE_NAME,
        ROOT / "core" / "openai_chat_image_backend.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[MODULE_NAME] = module
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


class OpenAIChatStreamRefTests(unittest.TestCase):
    def test_request_mode_auto_uses_backend_defaults(self):
        mod = _load_module()
        backend = mod.OpenAIChatImageBackend(
            imgr=_DummyImageManager(),
            base_url="https://api.example.com/v1",
            api_keys=["test-key"],
            default_model="gemini-3.1-flash-image-preview-4k",
        )

        self.assertEqual(backend.generate_request_mode, "auto")
        self.assertEqual(backend.edit_request_mode, "auto")
        self.assertTrue(backend._should_try_stream("generate"))
        self.assertFalse(backend._should_try_stream("edit"))

    def test_request_mode_overrides_legacy_stream_flags(self):
        mod = _load_module()
        backend = mod.OpenAIChatImageBackend(
            imgr=_DummyImageManager(),
            base_url="https://api.example.com/v1",
            api_keys=["test-key"],
            default_model="gemini-3.1-flash-image-preview-4k",
            generate_request_mode="non_stream",
            edit_request_mode="stream",
            enable_stream_generate=True,
            enable_stream_edit=False,
        )

        self.assertEqual(backend.generate_request_mode, "non_stream")
        self.assertEqual(backend.edit_request_mode, "stream")
        self.assertFalse(backend._should_try_stream("generate"))
        self.assertTrue(backend._should_try_stream("edit"))

    def test_request_mode_auto_preserves_legacy_generate_flag(self):
        mod = _load_module()
        backend = mod.OpenAIChatImageBackend(
            imgr=_DummyImageManager(),
            base_url="https://api.example.com/v1",
            api_keys=["test-key"],
            default_model="gemini-3.1-flash-image-preview-4k",
            generate_request_mode="auto",
            enable_stream_generate=False,
        )

        self.assertEqual(backend.generate_request_mode, "non_stream")
        self.assertFalse(backend._should_try_stream("generate"))

    def test_request_mode_auto_preserves_legacy_edit_flag(self):
        mod = _load_module()
        backend = mod.OpenAIChatImageBackend(
            imgr=_DummyImageManager(),
            base_url="https://api.example.com/v1",
            api_keys=["test-key"],
            default_model="gemini-3.1-flash-image-preview-4k",
            edit_request_mode="auto",
            enable_stream_edit=True,
        )

        self.assertEqual(backend.edit_request_mode, "stream")
        self.assertTrue(backend._should_try_stream("edit"))

    def test_extracts_delta_images_from_sse(self):
        mod = _load_module()
        sse_text = (
            'data: {"choices":[{"delta":{"images":[{"type":"image_url","image_url":{"url":"'
            "data:image/png;base64,"
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
            '"}}]}}]}\n'
            "data: [DONE]\n"
        )

        image_refs, video_refs = mod._extract_media_refs_from_sse_text(sse_text)

        self.assertEqual(video_refs, [])
        self.assertEqual(len(image_refs), 1)
        self.assertTrue(image_refs[0].startswith("data:image/png;base64,"))

    def test_flags_tiny_placeholder_png(self):
        mod = _load_module()
        tiny_png = (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAE"
            "hQGAhKmMIQAAAABJRU5ErkJggg=="
        )
        raw = mod._decode_base64_bytes(tiny_png)

        self.assertTrue(mod._looks_like_placeholder_image_bytes(raw))
        self.assertFalse(mod._looks_like_placeholder_image_bytes(b"\xff\xd8\xff" + b"0" * 256))

    def test_apply_gemini_image_config_adds_common_size_aliases(self):
        mod = _load_module()

        payload = mod.OpenAIChatImageBackend._apply_gemini_image_config(
            {},
            model="gemini-3.1-flash-image-preview-4k",
            size=None,
            resolution="4K",
        )

        self.assertEqual(payload["image_config"]["image_size"], "1K")
        self.assertEqual(payload["image_config"]["imageSize"], "1K")
        self.assertEqual(payload["image_size"], "1K")
        self.assertEqual(payload["imageSize"], "1K")
        self.assertEqual(payload["size"], "1K")
        self.assertEqual(payload["generation_config"]["image_size"], "1K")
        self.assertEqual(payload["generation_config"]["imageSize"], "1K")
        self.assertEqual(payload["generationConfig"]["imageConfig"]["image_size"], "1K")
        self.assertEqual(payload["generationConfig"]["imageConfig"]["imageSize"], "1K")
        self.assertEqual(payload["modalities"], ["image", "text"])

    def test_apply_gemini_image_config_skips_non_gemini_4k_model(self):
        mod = _load_module()

        payload = mod.OpenAIChatImageBackend._apply_gemini_image_config(
            {"existing": True},
            model="custom-image-preview-4k",
            size=None,
            resolution="4K",
        )

        self.assertEqual(payload, {"existing": True})
        self.assertEqual(
            mod.OpenAIChatImageBackend._chat_request_attempts(
                model="custom-image-preview-4k"
            ),
            1,
        )

    def test_rewrite_local_media_url_preserves_result_port(self):
        mod = _load_module()

        for ref in (
            "http://host.docker.internal:38000/tmp/final_4k.jpg",
            "http://172.17.0.1:38000/tmp/final_4k.jpg",
            "http://172.22.0.1:38000/tmp/final_4k.jpg",
        ):
            rewritten = mod._rewrite_local_media_url(
                ref,
                base_url="https://newapi.165201.xyz/v1",
            )
            self.assertEqual(
                rewritten,
                "http://newapi.165201.xyz:38000/tmp/final_4k.jpg",
            )


class OpenAIChatEditFallbackTests(unittest.IsolatedAsyncioTestCase):
    async def test_edit_retries_with_file_service_url_when_data_uri_is_rejected(self):
        mod = _load_module()
        imgr = _DummyImageManager()
        final_response = types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    message=types.SimpleNamespace(
                        content="![image1](https://cdn.example.com/final.png)"
                    )
                )
            ]
        )
        client = _DummyClient(
            [
                RuntimeError(
                    'get file base64 from url "data:image/png;base64,..." failed: '
                    'unsupported protocol scheme "data"'
                ),
                RuntimeError(
                    'get file base64 from url "data:image/png;base64,..." failed: '
                    'unsupported protocol scheme "data"'
                ),
                final_response,
            ]
        )
        backend = mod.OpenAIChatImageBackend(
            imgr=imgr,
            base_url="https://api.example.com/v1",
            api_keys=["test-key"],
            default_model="gemini-3.1-flash-image-preview-4k",
        )
        backend._get_client = lambda key: client

        async def _stream_stub(**kwargs):
            return [], [], ""

        backend._stream_chat_completion = _stream_stub

        png_bytes = b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+X2ioAAAAASUVORK5CYII="
        )
        out_path = await backend.edit("改成赛博朋克", [png_bytes])

        self.assertEqual(out_path, Path("/tmp/result.png"))

    async def test_edit_skips_stream_when_disabled(self):
        mod = _load_module()
        imgr = _DummyImageManager()
        final_response = types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    message=types.SimpleNamespace(
                        content="![image1](https://cdn.example.com/final-edit.png)"
                    )
                )
            ]
        )
        client = _DummyClient([final_response])
        backend = mod.OpenAIChatImageBackend(
            imgr=imgr,
            base_url="https://api.example.com/v1",
            api_keys=["test-key"],
            default_model="gemini-3.1-flash-image-preview-4k",
            edit_request_mode="non_stream",
        )
        backend._get_client = lambda key: client

        stream_called = {"value": False}

        async def _stream_stub(**kwargs):
            stream_called["value"] = True
            return [], [], ""

        backend._stream_chat_completion = _stream_stub

        png_bytes = b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+X2ioAAAAASUVORK5CYII="
        )
        out_path = await backend.edit("改成赛博朋克", [png_bytes])

        self.assertEqual(out_path, Path("/tmp/result.png"))
        self.assertFalse(stream_called["value"])


class OpenAIChatGenerateFallbackTests(unittest.IsolatedAsyncioTestCase):
    async def test_generate_skips_stream_when_disabled(self):
        mod = _load_module()
        imgr = _DummyImageManager()
        final_response = types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    message=types.SimpleNamespace(
                        content="![image1](https://cdn.example.com/final-generate.png)"
                    )
                )
            ]
        )
        client = _DummyClient([final_response])
        backend = mod.OpenAIChatImageBackend(
            imgr=imgr,
            base_url="https://api.example.com/v1",
            api_keys=["test-key"],
            default_model="gemini-3.1-flash-image-preview-4k",
            generate_request_mode="non_stream",
        )
        backend._get_client = lambda key: client

        stream_called = {"value": False}

        async def _stream_stub(**kwargs):
            stream_called["value"] = True
            return [], [], ""

        backend._stream_chat_completion = _stream_stub

        out_path = await backend.generate("画一个赛博朋克少女")

        self.assertEqual(out_path, Path("/tmp/result.png"))
        self.assertFalse(stream_called["value"])
        self.assertEqual(
            imgr.downloaded_urls, ["https://cdn.example.com/final-generate.png"]
        )
        self.assertEqual(len(client.chat.completions.calls), 1)

    async def test_save_single_ref_rewrites_relative_ref_to_origin(self):
        mod = _load_module()
        imgr = _DummyImageManager()
        backend = mod.OpenAIChatImageBackend(
            imgr=imgr,
            base_url="https://api.example.com/v1",
            api_keys=["test-key"],
            default_model="gemini-3.1-flash-image-preview-4k",
        )

        out_path = await backend._save_single_ref("/tmp/final.png")

        self.assertEqual(out_path, Path("/tmp/result.png"))
        self.assertEqual(imgr.downloaded_urls, ["https://api.example.com/tmp/final.png"])

    async def test_save_single_ref_rewrites_local_result_host_to_origin_host(self):
        mod = _load_module()
        imgr = _DummyImageManager()
        backend = mod.OpenAIChatImageBackend(
            imgr=imgr,
            base_url="https://newapi.165201.xyz/v1",
            api_keys=["test-key"],
            default_model="gemini-3.1-flash-image-4k",
        )

        out_path = await backend._save_single_ref(
            "http://host.docker.internal:38000/tmp/final.png"
        )

        self.assertEqual(out_path, Path("/tmp/result.png"))
        self.assertEqual(
            imgr.downloaded_urls,
            ["http://newapi.165201.xyz:38000/tmp/final.png"],
        )

    async def test_save_from_ref_prefers_latest_candidate(self):
        mod = _load_module()
        imgr = _DummyImageManager()
        backend = mod.OpenAIChatImageBackend(
            imgr=imgr,
            base_url="https://api.example.com/v1",
            api_keys=["test-key"],
            default_model="gemini-3.1-flash-image-preview-4k",
        )

        out_path = await backend._save_from_ref(
            "https://cdn.example.com/preview.png",
            fallback_refs=[
                "https://cdn.example.com/final.png",
            ],
        )

        self.assertEqual(out_path, Path("/tmp/result.png"))
        self.assertEqual(imgr.downloaded_urls, ["https://cdn.example.com/final.png"])

    async def test_save_single_ref_uses_trusted_direct_download_when_guard_blocks(self):
        mod = _load_module()
        imgr = _DummyImageManager(
            fail_download_error=RuntimeError("Disallowed resolved IP address")
        )
        backend = mod.OpenAIChatImageBackend(
            imgr=imgr,
            base_url="https://api.bltcy.ai/v1",
            api_keys=["test-key"],
            default_model="gemini-3.1-flash-image-preview-4k",
        )

        async def _direct_download(url: str):
            self.assertEqual(url, "https://files.closeai.fans/output/final.jpg")
            return Path("/tmp/input_1.png")

        backend._download_trusted_result_url = _direct_download

        out_path = await backend._save_single_ref(
            "https://files.closeai.fans/output/final.jpg"
        )

        self.assertEqual(out_path, Path("/tmp/input_1.png"))
        self.assertEqual(
            imgr.downloaded_urls,
            ["https://files.closeai.fans/output/final.jpg"],
        )

    async def test_download_trusted_result_url_checks_redirect_targets(self):
        mod = _load_module()
        imgr = _DummyImageManager()
        backend = mod.OpenAIChatImageBackend(
            imgr=imgr,
            base_url="https://api.bltcy.ai/v1",
            api_keys=["test-key"],
            default_model="gemini-3.1-flash-image-preview-4k",
        )

        class _FakeResponse:
            def __init__(self, status_code: int, headers: dict[str, str], body: bytes = b""):
                self.status_code = status_code
                self.headers = headers
                self._body = body

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def aread(self):
                return self._body

            async def aiter_bytes(self):
                if self._body:
                    yield self._body

        class _FakeClient:
            def __init__(self):
                self.calls: list[str] = []

            def stream(self, method: str, url: str, **kwargs):
                self.calls.append(url)
                return _FakeResponse(
                    302,
                    {"location": "http://127.0.0.1/private.png"},
                )

        fake_client = _FakeClient()
        backend.proxy_url = "http://proxy.local"
        backend._http_client = fake_client

        with self.assertRaisesRegex(RuntimeError, "Disallowed IP address"):
            await backend._download_trusted_result_url(
                "https://files.closeai.fans/output/final.jpg"
            )

        self.assertEqual(
            fake_client.calls,
            ["https://files.closeai.fans/output/final.jpg"],
        )

    async def test_edit_uses_images_api_fallback_before_file_service(self):
        mod = _load_module()
        imgr = _DummyImageManager()
        client = _DummyClient(
            [
                RuntimeError("image_url is required for image edits"),
            ]
        )
        backend = mod.OpenAIChatImageBackend(
            imgr=imgr,
            base_url="https://api.example.com/v1",
            api_keys=["test-key"],
            default_model="gemini-3.1-flash-image-preview-4k",
        )
        backend._get_client = lambda key: client

        async def _stream_stub(**kwargs):
            return [], [], ""

        async def _images_stub(**kwargs):
            return Path("/tmp/from-images-api.png")

        async def _register_stub(images):
            raise AssertionError("should not use file service fallback when images api succeeds")

        backend._stream_chat_completion = _stream_stub
        backend._edit_via_images_api = _images_stub
        backend._register_input_image_urls = _register_stub

        png_bytes = b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+X2ioAAAAASUVORK5CYII="
        )
        out_path = await backend.edit("改成赛博朋克", [png_bytes])

        self.assertEqual(out_path, Path("/tmp/from-images-api.png"))
        self.assertEqual(len(client.chat.completions.calls), 1)


if __name__ == "__main__":
    unittest.main()
