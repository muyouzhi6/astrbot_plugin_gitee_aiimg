import importlib.util
import sys
import types
import unittest
from pathlib import Path


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
    status = 200

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def json(self):
        return {"candidates": []}

    async def text(self):
        return ""


class _FakeSession:
    def __init__(self):
        self.last_headers = None

    def post(self, *args, **kwargs):
        self.last_headers = kwargs.get("headers")
        return _FakeResponse()


class GeminiEditAuthHeaderTests(unittest.IsolatedAsyncioTestCase):
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


if __name__ == "__main__":
    unittest.main()
