import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "sora2_video_service_testpkg"
CORE_PACKAGE_NAME = f"{PACKAGE_NAME}.core"
MODULE_NAME = f"{CORE_PACKAGE_NAME}.sora2_video_service"


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

    spec = importlib.util.spec_from_file_location(
        MODULE_NAME,
        ROOT / "core" / "sora2_video_service.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[MODULE_NAME] = module
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


class Sora2VideoServiceTests(unittest.IsolatedAsyncioTestCase):
    def test_extract_video_url_ignores_authenticated_content_url(self):
        mod = _load_module()

        result = mod._extract_video_url(
            {"status": "completed", "content_url": "/v1/videos/task/content"},
            base_origin="https://x666.me",
        )

        self.assertEqual(result, "")

    def test_extract_video_url_resolves_nested_relative_video_url(self):
        mod = _load_module()

        result = mod._extract_video_url(
            {"data": {"videos": [{"video_url": "/video/result.mp4"}]}},
            base_origin="https://x666.me",
        )

        self.assertEqual(result, "https://x666.me/video/result.mp4")

    def test_extract_video_url_reads_new_api_metadata_url(self):
        mod = _load_module()

        result = mod._extract_video_url(
            {"status": "completed", "metadata": {"url": "/video/result.mp4"}},
            base_origin="https://gateway.example",
        )

        self.assertEqual(result, "https://gateway.example/video/result.mp4")

    async def test_completed_task_without_public_url_returns_authenticated_content(
        self,
    ):
        mod = _load_module()
        calls = []

        async def _no_sleep(_seconds):
            return None

        mod.asyncio.sleep = _no_sleep

        class Service(mod.Sora2VideoService):
            async def _request_json_with_retries(self, *args, **kwargs):
                calls.append((args, kwargs))
                if args[1] == "POST":
                    return {"id": "task-public", "status": "queued"}
                return {"id": "task-public", "status": "completed"}

        service = Service(
            settings={
                "base_url": "https://gateway.example/v1",
                "api_keys": ["test-key"],
                "poll_interval_seconds": 1,
            }
        )

        result = await service.generate_video_url("prompt")

        self.assertIsInstance(result, mod.VideoResult)
        self.assertEqual(
            result.url,
            "https://gateway.example/v1/videos/task-public/content",
        )
        self.assertEqual(
            result.download_headers,
            {"Authorization": "Bearer test-key"},
        )
        self.assertEqual(calls[1][0][1], "GET")

    async def test_metadata_content_url_keeps_download_authentication(self):
        mod = _load_module()

        class Service(mod.Sora2VideoService):
            async def _request_json_with_retries(self, *args, **kwargs):
                return {
                    "id": "task-public",
                    "status": "completed",
                    "metadata": {"url": "/v1/videos/task-public/content"},
                }

        service = Service(
            settings={
                "base_url": "https://gateway.example/v1",
                "api_keys": ["test-key"],
            }
        )

        result = await service.generate_video_url("prompt")

        self.assertIsInstance(result, mod.VideoResult)
        self.assertEqual(
            result.url,
            "https://gateway.example/v1/videos/task-public/content",
        )
        self.assertEqual(
            result.download_headers,
            {"Authorization": "Bearer test-key"},
        )

    async def test_queued_metadata_url_waits_for_completed_status(self):
        mod = _load_module()
        calls = []

        async def _no_sleep(_seconds):
            return None

        mod.asyncio.sleep = _no_sleep

        class Service(mod.Sora2VideoService):
            async def _request_json_with_retries(self, *args, **kwargs):
                calls.append(args[1])
                if args[1] == "POST":
                    return {
                        "id": "task-pending",
                        "status": "queued",
                        "metadata": {"url": "/video/not-ready.mp4"},
                    }
                return {
                    "id": "task-pending",
                    "status": "completed",
                    "metadata": {"url": "/video/ready.mp4"},
                }

        service = Service(
            settings={
                "base_url": "https://gateway.example/v1",
                "api_keys": ["test-key"],
                "poll_interval_seconds": 1,
            }
        )

        result = await service.generate_video_url("prompt")

        self.assertEqual(result, "https://gateway.example/video/ready.mp4")
        self.assertEqual(calls, ["POST", "GET"])

    async def test_extra_body_cannot_override_core_payload_fields(self):
        mod = _load_module()
        calls = []

        class Service(mod.Sora2VideoService):
            async def _request_json_with_retries(self, *args, **kwargs):
                calls.append(kwargs)
                return {"video_url": "https://cdn.example/video.mp4"}

        service = Service(
            settings={
                "api_keys": ["key-a"],
                "model": "sora-2",
                "seconds": "5",
                "size": "720x1280",
                "extra_body": {
                    "model": "wrong",
                    "prompt": "wrong",
                    "seconds": 9,
                    "size": "1x1",
                    "n": 4,
                    "metadata": {"ok": True},
                },
            }
        )

        result = await service.generate_video_url("real prompt")

        self.assertEqual(result, "https://cdn.example/video.mp4")
        payload = calls[0]["json_body"]
        self.assertEqual(payload["model"], "sora-2")
        self.assertEqual(payload["prompt"], "real prompt")
        self.assertEqual(payload["seconds"], "5")
        self.assertEqual(payload["size"], "720x1280")
        self.assertEqual(payload["n"], 1)
        self.assertEqual(payload["metadata"], {"ok": True})

    def test_meinianda_replaces_unsupported_default_duration(self):
        mod = _load_module()

        service = mod.Sora2VideoService(
            settings={
                "base_url": "https://meinianda.top",
                "api_keys": ["test-key"],
                "model": "gemini-omni-flash",
            }
        )

        self.assertEqual(service.seconds, "4")

    async def test_create_request_uses_zero_retries_by_default(self):
        mod = _load_module()
        create_retry_values = []

        class Service(mod.Sora2VideoService):
            async def _request_json_with_retries(self, *args, **kwargs):
                create_retry_values.append(kwargs.get("max_retries"))
                return {"video_url": "https://cdn.example/video.mp4"}

        service = Service(settings={"api_keys": ["key-a"]})

        await service.generate_video_url("prompt")

        self.assertEqual(create_retry_values, [0])

    async def test_reference_image_uses_multipart_input_reference(self):
        mod = _load_module()
        calls = []
        png_bytes = b"\x89PNG\r\n\x1a\n" + b"0" * 32

        class Service(mod.Sora2VideoService):
            async def _request_json_with_retries(self, *args, **kwargs):
                calls.append(kwargs)
                return {"video_url": "https://cdn.example/video.mp4"}

        service = Service(
            settings={
                "api_keys": ["key-a"],
                "model": "sora-2",
                "seconds": "5",
                "size": "720x1280",
            }
        )

        result = await service.generate_video_url("prompt", image_bytes=png_bytes)

        self.assertEqual(result, "https://cdn.example/video.mp4")
        call = calls[0]
        self.assertIsNone(call["json_body"])
        self.assertNotIn("Content-Type", call["headers"])
        self.assertEqual(
            call["data_fields"],
            {
                "model": "sora-2",
                "prompt": "prompt",
                "seconds": "5",
                "size": "720x1280",
                "n": "1",
            },
        )
        filename, file_bytes, mime = call["files"]["input_reference"]
        self.assertEqual(filename, "input_reference.png")
        self.assertEqual(file_bytes, png_bytes)
        self.assertEqual(mime, "image/png")

    async def test_create_request_falls_back_to_next_key_on_auth_failure(self):
        mod = _load_module()
        attempted_keys = []

        class Service(mod.Sora2VideoService):
            async def _request_json_with_retries(self, *args, **kwargs):
                headers = kwargs["headers"]
                attempted_keys.append(headers["Authorization"].removeprefix("Bearer "))
                if len(attempted_keys) == 1:
                    raise mod.Sora2APIError("bad key", 401)
                return {"video_url": "https://cdn.example/video.mp4"}

        service = Service(settings={"api_keys": ["bad", "good"]})

        result = await service.generate_video_url("prompt")

        self.assertEqual(result, "https://cdn.example/video.mp4")
        self.assertEqual(attempted_keys, ["bad", "good"])

    async def test_auth_failure_does_not_retry_same_key_before_fallback(self):
        mod = _load_module()
        attempted_keys = []

        class Service(mod.Sora2VideoService):
            async def _request_json_with_retries(self, *args, **kwargs):
                headers = kwargs["headers"]
                attempted_keys.append(headers["Authorization"].removeprefix("Bearer "))
                if len(attempted_keys) == 1:
                    raise mod.Sora2APIError("bad key", 401)
                return {"video_url": "https://cdn.example/video.mp4"}

        service = Service(
            settings={
                "api_keys": ["bad", "good"],
                "create_max_retries": 3,
            }
        )

        await service.generate_video_url("prompt")

        self.assertEqual(attempted_keys, ["bad", "good"])

    async def test_auth_failed_key_is_skipped_on_next_request(self):
        mod = _load_module()
        attempted_keys = []

        class Service(mod.Sora2VideoService):
            async def _request_json_with_retries(self, *args, **kwargs):
                headers = kwargs["headers"]
                key = headers["Authorization"].removeprefix("Bearer ")
                attempted_keys.append(key)
                if key == "bad":
                    raise mod.Sora2APIError("bad key", 401)
                return {"video_url": "https://cdn.example/video.mp4"}

        service = Service(settings={"api_keys": ["bad", "good"]})

        await service.generate_video_url("first")
        await service.generate_video_url("second")

        self.assertEqual(attempted_keys, ["bad", "good", "good"])

    async def test_create_request_does_not_fall_back_after_ambiguous_timeout(self):
        mod = _load_module()
        attempted_keys = []

        class Service(mod.Sora2VideoService):
            async def _request_json_with_retries(self, *args, **kwargs):
                headers = kwargs["headers"]
                attempted_keys.append(headers["Authorization"].removeprefix("Bearer "))
                raise TimeoutError("response lost after create")

        service = Service(settings={"api_keys": ["key-a", "key-b"]})

        with self.assertRaises(TimeoutError):
            await service.generate_video_url("prompt")

        self.assertEqual(attempted_keys, ["key-a"])

    async def test_rate_limited_key_uses_retry_after_cooldown(self):
        mod = _load_module()

        class Service(mod.Sora2VideoService):
            async def _request_json_with_retries(self, *args, **kwargs):
                headers = kwargs["headers"]
                key = headers["Authorization"].removeprefix("Bearer ")
                if key == "limited":
                    raise mod.Sora2APIError("rate limited", 429, retry_after_seconds=12)
                return {"video_url": "https://cdn.example/video.mp4"}

        service = Service(settings={"api_keys": ["limited", "good"]})

        await service.generate_video_url("prompt")

        cooldown_remaining = service._key_disabled_until[0] - mod.time.monotonic()
        self.assertGreater(cooldown_remaining, 5)
        self.assertLessEqual(cooldown_remaining, 12.5)

    async def test_env_key_has_priority_over_configured_key_pool(self):
        mod = _load_module()
        attempted_keys = []
        old_value = os.environ.get("X666_TEST_KEY")
        os.environ["X666_TEST_KEY"] = "env-key"
        self.addCleanup(
            lambda: (
                os.environ.pop("X666_TEST_KEY", None)
                if old_value is None
                else os.environ.__setitem__("X666_TEST_KEY", old_value)
            )
        )

        class Service(mod.Sora2VideoService):
            async def _request_json_with_retries(self, *args, **kwargs):
                headers = kwargs["headers"]
                attempted_keys.append(headers["Authorization"].removeprefix("Bearer "))
                return {"video_url": "https://cdn.example/video.mp4"}

        service = Service(
            settings={
                "api_key_env": "X666_TEST_KEY",
                "api_keys": ["pool-key"],
            }
        )

        await service.generate_video_url("prompt")

        self.assertEqual(attempted_keys, ["env-key"])

    async def test_successful_key_advances_next_starting_key(self):
        mod = _load_module()
        attempted_keys = []

        class Service(mod.Sora2VideoService):
            async def _request_json_with_retries(self, *args, **kwargs):
                headers = kwargs["headers"]
                attempted_keys.append(headers["Authorization"].removeprefix("Bearer "))
                return {"video_url": "https://cdn.example/video.mp4"}

        service = Service(settings={"api_keys": ["key-a", "key-b"]})

        await service.generate_video_url("first")
        await service.generate_video_url("second")

        self.assertEqual(attempted_keys, ["key-a", "key-b"])


if __name__ == "__main__":
    unittest.main()
