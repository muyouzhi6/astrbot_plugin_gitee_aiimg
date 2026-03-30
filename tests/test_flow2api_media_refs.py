import base64
import importlib.util
import sys
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "flow2api_media_testpkg"
CORE_PACKAGE_NAME = f"{PACKAGE_NAME}.core"
IMAGE_FORMAT_MODULE_NAME = f"{CORE_PACKAGE_NAME}.image_format"
MODULE_NAME = f"{CORE_PACKAGE_NAME}.gemini_flow2api"


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
        ROOT / "core" / "gemini_flow2api.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[MODULE_NAME] = module
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


class Flow2ApiMediaRefTests(unittest.TestCase):
    def test_prefers_new_upscaled_asset_url_over_origin(self):
        mod = _load_module()

        payload = {
            "url": "https://origin.example/original.jpg",
            "generated_assets": {
                "type": "image",
                "origin_image_url": "https://origin.example/original.jpg",
                "upscaled_image": {
                    "resolution": "4K",
                    "local_url": "http://0.0.0.0:8000/tmp/final_4k.jpg",
                    "url": "http://0.0.0.0:8000/tmp/final_4k.jpg",
                },
            },
        }

        ref = mod._extract_first_image_ref_from_obj(payload)

        self.assertEqual(ref, "http://0.0.0.0:8000/tmp/final_4k.jpg")

    def test_prefers_upscaled_local_url_over_base64_when_both_exist(self):
        mod = _load_module()

        payload = {
            "generated_assets": {
                "upscaled_image": {
                    "base64": (
                        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8"
                        "/x8AAwMCAO+X2ioAAAAASUVORK5CYII="
                    ),
                    "local_url": "http://154.36.187.54:38000/tmp/final_4k.jpg",
                    "url": "http://154.36.187.54:38000/tmp/final_4k.jpg",
                }
            }
        }

        ref = mod._extract_first_image_ref_from_obj(payload)

        self.assertEqual(ref, "http://154.36.187.54:38000/tmp/final_4k.jpg")

    def test_rewrites_local_flow2api_host_to_configured_origin(self):
        mod = _load_module()

        rewritten = mod._rewrite_flow2api_media_ref(
            "http://0.0.0.0:8000/tmp/final_4k.jpg",
            endpoint_url="http://154.36.187.54:38000/v1/chat/completions",
        )

        self.assertEqual(
            rewritten,
            "http://154.36.187.54:38000/tmp/final_4k.jpg",
        )

    def test_rewrites_relative_tmp_path_to_configured_origin(self):
        mod = _load_module()

        ref = mod._extract_first_image_ref("/tmp/final_4k.jpg")
        rewritten = mod._rewrite_flow2api_media_ref(
            ref,
            endpoint_url="http://154.36.187.54:38000/v1/chat/completions",
        )

        self.assertEqual(ref, "/tmp/final_4k.jpg")
        self.assertEqual(
            rewritten,
            "http://154.36.187.54:38000/tmp/final_4k.jpg",
        )

    def test_accepts_base64_asset_from_new_payload_shape(self):
        mod = _load_module()

        tiny_png = (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8"
            "/x8AAwMCAO+X2ioAAAAASUVORK5CYII="
        )
        payload = {
            "generated_assets": {
                "upscaled_image": {
                    "base64": tiny_png,
                }
            }
        }

        ref = mod._extract_first_image_ref_from_obj(payload)

        self.assertTrue(ref.startswith("data:image/png;base64,"))
        decoded = base64.b64decode(ref.split(",", 1)[1])
        self.assertGreater(len(decoded), 0)

    def test_rejects_non_image_base64_payload(self):
        mod = _load_module()

        text = base64.b64encode(b"hello world").decode()

        ref = mod._base64_to_data_image_ref(text, min_length=1)

        self.assertIsNone(ref)


if __name__ == "__main__":
    unittest.main()
