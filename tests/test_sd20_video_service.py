import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "sd20_video_service_testpkg"
CORE_PACKAGE_NAME = f"{PACKAGE_NAME}.core"
MODULE_NAME = f"{CORE_PACKAGE_NAME}.sd20_video_service"


class _Logger:
    def info(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None


def _load_module():
    for name in list(sys.modules):
        if name.startswith(PACKAGE_NAME) or name in {"astrbot", "astrbot.api"}:
            sys.modules.pop(name, None)

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

    image_mod = types.ModuleType(f"{CORE_PACKAGE_NAME}.image_format")
    image_mod.guess_image_mime_and_ext = lambda _value: ("image/png", "png")
    sys.modules[image_mod.__name__] = image_mod

    spec = importlib.util.spec_from_file_location(
        MODULE_NAME,
        ROOT / "core" / "sd20_video_service.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[MODULE_NAME] = module
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


class SD20VideoServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_text_to_video_payload_uses_documented_endpoint(self):
        mod = _load_module()
        calls = []

        class Service(mod.SD20VideoService):
            async def _request_with_retries(self, *args, **kwargs):
                calls.append((args, kwargs))
                return {
                    "task_id": "task-1",
                    "status": "SUCCESS",
                    "result_url": "/v1/videos/task-1/content",
                }

        service = Service(
            settings={
                "base_url": "https://api.3365api.cn/v1",
                "api_key": "test-key",
                "model": "video-v1-5s",
                "ratio": "9:16",
            }
        )

        result = await service.generate_video_url("a red paper airplane")

        self.assertIsInstance(result, mod.VideoResult)
        self.assertEqual(result.url, "https://api.3365api.cn/v1/videos/task-1/content")
        self.assertEqual(result.download_headers, {"Authorization": "Bearer test-key"})
        self.assertEqual(calls[0][0][1], "POST")
        self.assertEqual(calls[0][0][2], "https://api.3365api.cn/v1/video/generations")
        self.assertEqual(
            calls[0][1]["payload"],
            {
                "model": "video-v1-5s",
                "prompt": "a red paper airplane",
                "ratio": "9:16",
            },
        )

    async def test_image_to_video_uses_data_uri_and_polls_uppercase_success(self):
        mod = _load_module()
        calls = []
        image_bytes = b"\x89PNG\r\n\x1a\n" + b"0" * 32

        async def _no_sleep(_seconds):
            return None

        class Service(mod.SD20VideoService):
            async def _request_with_retries(self, *args, **kwargs):
                calls.append((args, kwargs))
                if args[0] is not None and args[1] == "POST":
                    return {"data": {"task_id": "task-2", "status": "queued"}}
                return {
                    "data": {
                        "task_id": "task-2",
                        "status": "SUCCESS",
                        "result_url": "/v1/videos/task-2/content",
                    }
                }

        service = Service(
            settings={
                "base_url": "https://api.3365api.cn",
                "api_keys": ["test-key"],
                "poll_interval_seconds": 1,
            }
        )

        with patch.object(mod.asyncio, "sleep", new=AsyncMock(side_effect=_no_sleep)):
            result = await service.generate_video_url(
                "animate it", image_bytes=image_bytes
            )

        self.assertEqual(result.url, "https://api.3365api.cn/v1/videos/task-2/content")
        payload = calls[0][1]["payload"]
        self.assertTrue(payload["image"].startswith("data:image/png;base64,"))
        self.assertEqual(calls[1][0][1], "GET")

    async def test_failed_task_surfaces_upstream_message(self):
        mod = _load_module()

        class Service(mod.SD20VideoService):
            async def _request_with_retries(self, *args, **kwargs):
                return {
                    "task_id": "task-3",
                    "status": "FAILURE",
                    "message": "no channel",
                }

        service = Service(settings={"api_keys": ["test-key"]})
        with self.assertRaisesRegex(RuntimeError, "no channel"):
            await service.generate_video_url("prompt")


if __name__ == "__main__":
    unittest.main()
