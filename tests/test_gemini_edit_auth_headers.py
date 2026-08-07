import importlib
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "gemini_edit_auth_testpkg"
CORE_PACKAGE_NAME = f"{PACKAGE_NAME}.core"
IMAGE_FORMAT_MODULE_NAME = f"{CORE_PACKAGE_NAME}.image_format"
MODULE_NAME = f"{CORE_PACKAGE_NAME}.gemini_edit"


class _Logger:
    def debug(self, *args, **kwargs):
        return None

    def info(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None

    def error(self, *args, **kwargs):
        return None


def _clear_modules():
    for name in [
        MODULE_NAME,
        IMAGE_FORMAT_MODULE_NAME,
        CORE_PACKAGE_NAME,
        PACKAGE_NAME,
        "astrbot",
        "astrbot.api",
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

    image_format_spec = importlib.util.spec_from_file_location(
        IMAGE_FORMAT_MODULE_NAME,
        ROOT / "core" / "image_format.py",
    )
    image_format_module = importlib.util.module_from_spec(image_format_spec)
    sys.modules[IMAGE_FORMAT_MODULE_NAME] = image_format_module
    assert image_format_spec and image_format_spec.loader
    image_format_spec.loader.exec_module(image_format_module)

    spec = importlib.util.spec_from_file_location(
        MODULE_NAME,
        ROOT / "core" / "gemini_edit.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[MODULE_NAME] = module
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


class _FakeResponse:
    def __init__(self, *, status=200, payload=None, text=""):
        self.status = status
        self.payload = payload if payload is not None else {"candidates": []}
        self.response_text = text

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def json(self):
        return self.payload

    async def text(self):
        return self.response_text


class _FakeSession:
    def __init__(self, responses=None):
        self.last_headers = None
        self.last_json = None
        self.headers = []
        self.responses = list(responses or [])

    def post(self, *args, **kwargs):
        self.last_headers = kwargs.get("headers")
        self.last_json = kwargs.get("json")
        self.headers.append(self.last_headers)
        if self.responses:
            response = self.responses.pop(0)
            if isinstance(response, BaseException):
                raise response
            return response
        return _FakeResponse()


class GeminiEditAuthHeaderTests(unittest.IsolatedAsyncioTestCase):
    def test_gemini_native_uses_new_runtime_defaults(self):
        mod = _load_module()
        backend = mod.GeminiEditBackend(
            imgr=object(),
            settings={"api_keys": ["test-key"], "api_url": "https://example.com"},
        )

        self.assertEqual(backend.timeout, 600)
        self.assertEqual(backend.max_retries, 2)

    def test_gemini_native_allows_disabling_retries(self):
        mod = _load_module()
        backend = mod.GeminiEditBackend(
            imgr=object(),
            settings={
                "api_keys": ["test-key"],
                "api_url": "https://example.com",
                "max_retries": 0,
            },
        )

        self.assertEqual(backend.max_retries, 0)

    async def test_gemini_native_retries_server_errors_with_next_key(self):
        mod = _load_module()
        backend = mod.GeminiEditBackend(
            imgr=object(),
            settings={
                "api_keys": ["first-key", "second-key"],
                "api_url": "https://example.com",
                "max_retries": 1,
            },
        )
        session = _FakeSession(
            [
                _FakeResponse(status=503, text="temporarily unavailable"),
                _FakeResponse(payload={"candidates": [{"content": {}}]}),
            ]
        )

        async def fake_get_session():
            return session

        backend._get_session = fake_get_session

        with patch.object(mod.asyncio, "sleep", new=AsyncMock()) as sleep:
            data = await backend._request([{"text": "draw"}])

        self.assertIn("candidates", data)
        self.assertEqual(
            [headers["x-goog-api-key"] for headers in session.headers],
            ["first-key", "second-key"],
        )
        sleep.assert_awaited_once_with(1)

    async def test_gemini_native_retries_timeouts(self):
        mod = _load_module()
        backend = mod.GeminiEditBackend(
            imgr=object(),
            settings={
                "api_keys": ["first-key", "second-key"],
                "api_url": "https://example.com",
                "max_retries": 1,
            },
        )
        session = _FakeSession(
            [
                mod.asyncio.TimeoutError(),
                _FakeResponse(payload={"candidates": []}),
            ]
        )

        async def fake_get_session():
            return session

        backend._get_session = fake_get_session

        with patch.object(mod.asyncio, "sleep", new=AsyncMock()) as sleep:
            data = await backend._request([{"text": "draw"}])

        self.assertEqual(data, {"candidates": []})
        self.assertEqual(
            [headers["x-goog-api-key"] for headers in session.headers],
            ["first-key", "second-key"],
        )
        sleep.assert_awaited_once_with(1)

    async def test_gemini_native_does_not_retry_non_retryable_client_errors(self):
        mod = _load_module()
        backend = mod.GeminiEditBackend(
            imgr=object(),
            settings={
                "api_keys": ["test-key"],
                "api_url": "https://example.com",
                "max_retries": 2,
            },
        )
        session = _FakeSession([_FakeResponse(status=400, text="bad request")])

        async def fake_get_session():
            return session

        backend._get_session = fake_get_session

        with patch.object(mod.asyncio, "sleep", new=AsyncMock()) as sleep:
            with self.assertRaisesRegex(RuntimeError, "Gemini API 错误 \\(400\\)"):
                await backend._request([{"text": "draw"}])

        self.assertEqual(len(session.headers), 1)
        sleep.assert_not_awaited()

    async def test_gemini_native_uses_api_key_header_without_bearer_auth(self):
        mod = _load_module()
        backend = mod.GeminiEditBackend(
            imgr=object(),
            settings={"api_keys": ["test-key"], "api_url": "https://example.com"},
        )
        session = _FakeSession()

        async def fake_get_session():
            return session

        backend._get_session = fake_get_session

        await backend._request([{"text": "draw"}])

        self.assertEqual(session.last_headers["x-goog-api-key"], "test-key")
        self.assertNotIn("Authorization", session.last_headers)

    async def test_gemini_native_sends_adaptive_image_config(self):
        mod = _load_module()
        backend = mod.GeminiEditBackend(
            imgr=object(),
            settings={"api_keys": ["test-key"], "api_url": "https://example.com"},
        )
        session = _FakeSession()

        async def fake_get_session():
            return session

        backend._get_session = fake_get_session

        await backend._request(
            [{"text": "draw"}],
            resolution="4K",
            aspect_ratio="16:9",
        )

        image_config = session.last_json["generationConfig"]["imageConfig"]
        self.assertEqual(
            image_config,
            {"imageSize": "4K", "aspectRatio": "16:9"},
        )

    async def test_gemini_native_payload_uses_prompt_output_intent(self):
        mod = _load_module()
        output_spec = importlib.import_module(f"{CORE_PACKAGE_NAME}.output_spec")
        backend = mod.GeminiEditBackend(
            imgr=object(),
            settings={"api_keys": ["test-key"], "api_url": "https://example.com"},
        )
        session = _FakeSession()

        async def fake_get_session():
            return session

        backend._get_session = fake_get_session

        intent = output_spec.resolve_llm_output_intent(
            "电影感海边日落, 16:9, 4K",
            output="1024x1024",
        )
        await backend._request(
            [{"text": "draw"}],
            **backend.resolve_output_intent(intent),
        )

        image_config = session.last_json["generationConfig"]["imageConfig"]
        self.assertEqual(
            image_config,
            {"imageSize": "4K", "aspectRatio": "16:9"},
        )


if __name__ == "__main__":
    unittest.main()
