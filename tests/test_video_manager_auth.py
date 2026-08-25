import importlib.util
import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, patch


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "video_manager_auth_testpkg"
CORE_PACKAGE_NAME = f"{PACKAGE_NAME}.core"
MODULE_NAME = f"{CORE_PACKAGE_NAME}.video_manager"


class _Logger:
    def info(self, *args, **kwargs):
        return None

    def debug(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None


def _load_module():
    for name in list(sys.modules):
        if name.startswith(PACKAGE_NAME) or name in {
            "aiofiles",
            "astrbot",
            "astrbot.api",
        }:
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

    class _AsyncFile:
        def __init__(self, path, mode):
            self._path = path
            self._mode = mode
            self._file = None

        async def __aenter__(self):
            self._file = open(self._path, self._mode)
            return self

        async def __aexit__(self, exc_type, exc, tb):
            self._file.close()
            return False

        async def write(self, data):
            return self._file.write(data)

    aiofiles_mod = types.ModuleType("aiofiles")
    aiofiles_mod.open = lambda path, mode: _AsyncFile(path, mode)
    sys.modules["aiofiles"] = aiofiles_mod

    spec = importlib.util.spec_from_file_location(
        MODULE_NAME,
        ROOT / "core" / "video_manager.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[MODULE_NAME] = module
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


class _Response:
    def __init__(self, *, status_code, headers=None, chunks=None):
        self.status_code = status_code
        self.headers = headers or {}
        self._chunks = chunks or []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    async def aiter_bytes(self, chunk_size):
        for chunk in self._chunks:
            yield chunk


class VideoManagerAuthTests(unittest.IsolatedAsyncioTestCase):
    async def test_cross_origin_redirect_strips_authorization_header(self):
        mod = _load_module()
        requests = []
        responses = [
            _Response(
                status_code=302,
                headers={"location": "https://cdn.example/video.mp4"},
            ),
            _Response(
                status_code=200,
                headers={"content-type": "video/mp4"},
                chunks=[b"video-bytes"],
            ),
        ]

        class _Client:
            def __init__(self, *args, **kwargs):
                return None

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            def stream(self, method, url, *, headers=None):
                requests.append((method, url, headers))
                return responses.pop(0)

        async def _allow_url(*args, **kwargs):
            return None

        mod.httpx.AsyncClient = _Client
        mod.ensure_url_allowed = _allow_url

        with TemporaryDirectory() as temp_dir:
            manager = mod.VideoManager(
                {"storage": {"max_cached_videos": 20}},
                Path(temp_dir),
            )
            result = await manager.download_video(
                "https://gateway.example/v1/videos/task/content",
                headers={"Authorization": "Bearer test-key"},
            )

            self.assertTrue(result.exists())
            self.assertEqual(result.read_bytes(), b"video-bytes")

        self.assertEqual(
            requests,
            [
                (
                    "GET",
                    "https://gateway.example/v1/videos/task/content",
                    {"Authorization": "Bearer test-key"},
                ),
                ("GET", "https://cdn.example/video.mp4", None),
            ],
        )

    async def test_interrupted_download_resumes_from_existing_part_file(self):
        mod = _load_module()
        requests = []
        responses = [
            _Response(
                status_code=200,
                headers={"content-type": "video/mp4", "content-length": "10"},
                chunks=[b"1234"],
            ),
            _Response(
                status_code=206,
                headers={
                    "content-type": "video/mp4",
                    "content-length": "6",
                    "content-range": "bytes 4-9/10",
                },
                chunks=[b"567890"],
            ),
        ]

        class _Client:
            def __init__(self, *args, **kwargs):
                return None

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            def stream(self, method, url, *, headers=None):
                requests.append((method, url, headers))
                return responses.pop(0)

        async def _allow_url(*args, **kwargs):
            return None

        mod.httpx.AsyncClient = _Client
        mod.ensure_url_allowed = _allow_url

        with TemporaryDirectory() as temp_dir:
            manager = mod.VideoManager(
                {"storage": {"max_cached_videos": 20}},
                Path(temp_dir),
            )
            with patch.object(mod.asyncio, "sleep", new=AsyncMock()):
                result = await manager.download_video(
                    "https://gateway.example/v1/videos/task/content",
                    headers={"Authorization": "Bearer test-key"},
                )

            self.assertEqual(result.read_bytes(), b"1234567890")

        self.assertEqual(requests[0][2], {"Authorization": "Bearer test-key"})
        self.assertEqual(
            requests[1][2],
            {"Authorization": "Bearer test-key", "Range": "bytes=4-"},
        )

    async def test_json_error_body_is_not_saved_as_video(self):
        mod = _load_module()

        class _Client:
            def __init__(self, *args, **kwargs):
                return None

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            def stream(self, method, url, *, headers=None):
                return _Response(
                    status_code=200,
                    headers={"content-type": "application/json"},
                    chunks=[b'{"error":"upstream failed"}'],
                )

        async def _allow_url(*args, **kwargs):
            return None

        mod.httpx.AsyncClient = _Client
        mod.ensure_url_allowed = _allow_url

        with TemporaryDirectory() as temp_dir:
            manager = mod.VideoManager(
                {"storage": {"max_cached_videos": 20}},
                Path(temp_dir),
            )
            with self.assertRaisesRegex(RuntimeError, "non-video content"):
                await manager.download_video(
                    "https://gateway.example/v1/videos/task/content",
                    headers={"Authorization": "Bearer test-key"},
                )

            self.assertEqual(list((Path(temp_dir) / "videos").iterdir()), [])

    async def test_authenticated_range_download_reassembles_video(self):
        mod = _load_module()
        video_bytes = b"\x00\x00\x00\x18ftypmp42demo"
        requests = []
        responses = [
            _Response(
                status_code=206,
                headers={
                    "content-type": "video/mp4",
                    "content-range": f"bytes 0-0/{len(video_bytes)}",
                },
                chunks=[video_bytes[:1]],
            ),
            _Response(
                status_code=206,
                headers={
                    "content-type": "video/mp4",
                    "content-range": f"bytes 0-{len(video_bytes) - 1}/{len(video_bytes)}",
                },
                chunks=[video_bytes],
            ),
        ]

        class _Client:
            def __init__(self, *args, **kwargs):
                return None

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            def stream(self, method, url, *, headers=None):
                requests.append((method, url, headers))
                return responses.pop(0)

        async def _allow_url(*args, **kwargs):
            return None

        mod.httpx.AsyncClient = _Client
        mod.ensure_url_allowed = _allow_url

        with TemporaryDirectory() as temp_dir:
            manager = mod.VideoManager(
                {
                    "network": {"video_range_download": True},
                    "storage": {"max_cached_videos": 20},
                },
                Path(temp_dir),
            )
            result = await manager.download_video(
                "https://gateway.example/v1/videos/task/content",
                headers={"Authorization": "Bearer test-key"},
            )
            self.assertEqual(result.read_bytes(), video_bytes)

        self.assertEqual(len(requests), 2)
        self.assertEqual(requests[0][2]["Range"], "bytes=0-0")
        self.assertEqual(requests[1][2]["Range"], f"bytes=0-{len(video_bytes) - 1}")

    async def test_range_probe_recovers_when_zero_byte_probe_returns_200(self):
        mod = _load_module()
        video_bytes = b"\x00\x00\x00\x18ftypmp42range-fallback"
        requests = []
        responses = [
            _Response(
                status_code=200,
                headers={"content-type": "video/mp4"},
                chunks=[video_bytes],
            ),
            _Response(
                status_code=206,
                headers={
                    "content-type": "video/mp4",
                    "content-range": f"bytes 0-{len(video_bytes) - 1}/{len(video_bytes)}",
                },
                chunks=[video_bytes],
            ),
            _Response(
                status_code=206,
                headers={
                    "content-type": "video/mp4",
                    "content-range": f"bytes 0-{len(video_bytes) - 1}/{len(video_bytes)}",
                },
                chunks=[video_bytes],
            ),
        ]

        class _Client:
            def __init__(self, *args, **kwargs):
                return None

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            def stream(self, method, url, *, headers=None):
                requests.append((method, url, headers))
                return responses.pop(0)

        async def _allow_url(*args, **kwargs):
            return None

        mod.httpx.AsyncClient = _Client
        mod.ensure_url_allowed = _allow_url

        with TemporaryDirectory() as temp_dir:
            manager = mod.VideoManager(
                {
                    "network": {"video_range_download": True},
                    "storage": {"max_cached_videos": 20},
                },
                Path(temp_dir),
            )
            result = await manager.download_video(
                "https://gateway.example/v1/videos/task/content",
                headers={"Authorization": "Bearer test-key"},
            )
            self.assertEqual(result.read_bytes(), video_bytes)

        self.assertEqual(len(requests), 3)
        self.assertEqual(requests[0][2]["Range"], "bytes=0-0")
        self.assertEqual(
            requests[1][2]["Range"],
            f"bytes=0-{manager._video_range_chunk_bytes - 1}",
        )
        self.assertEqual(requests[2][2]["Range"], f"bytes=0-{len(video_bytes) - 1}")


if __name__ == "__main__":
    unittest.main()
