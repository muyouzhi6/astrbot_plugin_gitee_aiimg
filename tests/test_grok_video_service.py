import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch


ROOT = Path(__file__).resolve().parents[1]
MODULE_NAME = "grok_video_service_endpoint_test"


class _Logger:
    def info(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None


def _load_module():
    sys.modules.pop(MODULE_NAME, None)
    astrbot_mod = types.ModuleType("astrbot")
    sys.modules["astrbot"] = astrbot_mod
    api_mod = types.ModuleType("astrbot.api")
    api_mod.logger = _Logger()
    sys.modules["astrbot.api"] = api_mod

    spec = importlib.util.spec_from_file_location(
        MODULE_NAME,
        ROOT / "core" / "grok_video_service.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[MODULE_NAME] = module
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


class GrokVideoServiceEndpointTests(unittest.IsolatedAsyncioTestCase):
    def test_root_server_url_appends_v1_endpoint(self):
        mod = _load_module()

        service = mod.GrokVideoService(
            settings={"server_url": "https://gateway.example"}
        )

        self.assertEqual(
            service.api_url,
            "https://gateway.example/v1/videos/generations",
        )

    def test_v1_server_url_does_not_duplicate_v1(self):
        mod = _load_module()

        service = mod.GrokVideoService(
            settings={"server_url": "https://gateway.example/v1"}
        )

        self.assertEqual(
            service.api_url,
            "https://gateway.example/v1/videos/generations",
        )

    def test_full_endpoint_is_preserved(self):
        mod = _load_module()

        service = mod.GrokVideoService(
            settings={"server_url": "https://gateway.example/v1/videos/generations"}
        )

        self.assertEqual(
            service.api_url,
            "https://gateway.example/v1/videos/generations",
        )

    def test_numeric_resolution_is_normalized_to_xai_enum(self):
        mod = _load_module()

        service = mod.GrokVideoService(settings={"resolution": 1080})

        self.assertEqual(service.resolution, "1080p")

    async def test_request_payload_uses_xai_async_image_shape_and_parses_content(self):
        mod = _load_module()
        calls = []

        class Service(mod.GrokVideoService):
            async def _request_json(
                self, client, method, url, *, headers, payload=None
            ):
                calls.append((method, url, headers, payload))
                if method == "POST":
                    return {"request_id": "req-1"}
                return {
                    "status": "done",
                    "video": {"url": "/v1/videos/req-1/content"},
                }

        service = Service(
            settings={
                "server_url": "https://gateway.example/v1",
                "api_key": "test-key",
                "model": "grok-imagine-video-1.5",
                "duration": 1,
                "aspect_ratio": "9:16",
                "resolution": "480p",
                "poll_interval_seconds": 1,
            }
        )

        with patch.object(mod.asyncio, "sleep", new=AsyncMock()):
            result = await service.generate_video_url(
                "animate this", image_bytes=b"\x89PNG\r\n\x1a\n"
            )

        self.assertIsInstance(result, mod.VideoResult)
        self.assertEqual(result.url, "https://gateway.example/v1/videos/req-1/content")
        self.assertEqual(result.download_headers, {"Authorization": "Bearer test-key"})
        self.assertEqual(calls[0][0], "POST")
        self.assertEqual(calls[0][3]["model"], "grok-imagine-video-1.5")
        self.assertEqual(calls[0][3]["prompt"], "animate this")
        self.assertEqual(calls[0][3]["duration"], 1)
        self.assertEqual(calls[0][3]["aspect_ratio"], "9:16")
        self.assertEqual(calls[0][3]["resolution"], "480p")
        self.assertEqual(
            calls[0][3]["image"]["url"].split(",", 1)[0], "data:image/png;base64"
        )
        self.assertEqual(calls[1][0], "GET")

    async def test_text_to_video_omits_image_field(self):
        mod = _load_module()

        class Service(mod.GrokVideoService):
            async def _request_json(
                self, client, method, url, *, headers, payload=None
            ):
                if method == "POST":
                    self.payload = payload
                    return {"request_id": "req-1"}
                return {"status": "done", "video": {"url": "https://cdn.example/v.mp4"}}

        service = Service(settings={"api_key": "test-key", "duration": 5})
        with patch.object(mod.asyncio, "sleep", new=AsyncMock()):
            result = await service.generate_video_url("a cat")

        self.assertEqual(result, "https://cdn.example/v.mp4")
        self.assertNotIn("image", service.payload)

    async def test_request_timeout_applies_to_image_upload(self):
        mod = _load_module()
        captured = {}

        class _Client:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return None

        def client_factory(*, timeout, follow_redirects):
            captured["timeout"] = timeout
            captured["follow_redirects"] = follow_redirects
            return _Client()

        class Service(mod.GrokVideoService):
            async def _request_json(
                self, client, method, url, *, headers, payload=None
            ):
                return {"video_url": "https://cdn.example/v.mp4"}

        service = Service(
            settings={
                "api_key": "test-key",
                "request_timeout_seconds": 120,
            }
        )
        with patch.object(mod.httpx, "AsyncClient", side_effect=client_factory):
            result = await service.generate_video_url(
                "animate this", image_bytes=b"\x89PNG\r\n\x1a\n"
            )

        self.assertEqual(result, "https://cdn.example/v.mp4")
        self.assertTrue(captured["follow_redirects"])
        self.assertEqual(captured["timeout"].read, 120.0)
        self.assertEqual(captured["timeout"].write, 120.0)


if __name__ == "__main__":
    unittest.main()
