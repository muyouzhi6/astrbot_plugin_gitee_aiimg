import importlib.util
import sys
import types
import unittest
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


def _clear_modules():
    for name in [
        MODULE_NAME,
        OPENAI_COMPAT_MODULE_NAME,
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


if __name__ == "__main__":
    unittest.main()
