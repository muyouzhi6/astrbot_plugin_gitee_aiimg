import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "modelscope_async_testpkg"
CORE_PACKAGE_NAME = f"{PACKAGE_NAME}.core"
MODULE_NAME = f"{CORE_PACKAGE_NAME}.modelscope_async_backend"


class _Logger:
    def info(self, *args, **kwargs):
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
        ROOT / "core" / "modelscope_async_backend.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[MODULE_NAME] = module
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


class _FakeResponse:
    def __init__(self, payload, status=200):
        self.payload = payload
        self.status = status

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def text(self):
        return str(self.payload)

    async def json(self, **kwargs):
        return self.payload


class _FakeSession:
    responses = []
    instances = []

    def __init__(self, **kwargs):
        self.requests = []
        self.__class__.instances.append(self)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def request(self, method, url, **kwargs):
        self.requests.append((method, url, kwargs))
        return _FakeResponse(self.__class__.responses.pop(0))


class _ImageManager:
    def __init__(self):
        self.urls = []

    async def download_image(self, url, **kwargs):
        self.urls.append(url)
        return Path("/tmp/modelscope.png")


class ModelScopeAsyncBackendTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        _FakeSession.responses = []
        _FakeSession.instances = []

    async def test_generate_submits_async_task_and_downloads_result(self):
        mod = _load_module()
        imgr = _ImageManager()
        backend = mod.ModelScopeAsyncImageBackend(
            imgr=imgr,
            base_url="https://api-inference.modelscope.cn/v1",
            api_keys=["secret"],
            default_model="Qwen/Qwen-Image",
            poll_interval=0.5,
        )
        _FakeSession.responses = [
            {"task_id": "task/1"},
            {"task_status": "RUNNING"},
            {"task_status": "SUCCEED", "output_images": ["https://cdn.example/out.png"]},
        ]

        with patch.object(mod.aiohttp, "ClientSession", _FakeSession), patch.object(
            mod.asyncio, "sleep", return_value=None
        ):
            output = await backend.generate("draw", size="1024x1024")

        self.assertEqual(output, Path("/tmp/modelscope.png"))
        self.assertEqual(imgr.urls, ["https://cdn.example/out.png"])
        requests = _FakeSession.instances[0].requests
        self.assertEqual(requests[0][0:2], ("POST", "https://api-inference.modelscope.cn/v1/images/generations"))
        self.assertEqual(requests[0][2]["headers"]["X-ModelScope-Async-Mode"], "true")
        self.assertEqual(requests[0][2]["json"]["model"], "Qwen/Qwen-Image")
        self.assertEqual(requests[1][0:2], ("GET", "https://api-inference.modelscope.cn/v1/tasks/task%2F1"))
        self.assertEqual(requests[1][2]["headers"]["X-ModelScope-Task-Type"], "image_generation")

    async def test_poll_reports_upstream_failure(self):
        mod = _load_module()
        backend = mod.ModelScopeAsyncImageBackend(
            imgr=_ImageManager(),
            base_url="https://api-inference.modelscope.cn/v1",
            api_keys=["secret"],
            default_model="Qwen/Qwen-Image",
        )
        session = _FakeSession()
        _FakeSession.responses = [
            {"task_status": "FAILED", "errors": {"message": "quota exceeded"}}
        ]

        with self.assertRaisesRegex(RuntimeError, "quota exceeded"):
            await backend._poll_task(session, task_id="task-2", api_key="secret")

    async def test_poll_times_out(self):
        mod = _load_module()
        backend = mod.ModelScopeAsyncImageBackend(
            imgr=_ImageManager(),
            base_url="https://api-inference.modelscope.cn/v1",
            api_keys=["secret"],
            default_model="Qwen/Qwen-Image",
            poll_timeout=10,
        )
        session = _FakeSession()

        with patch.object(mod.time, "monotonic", side_effect=[0, 11]):
            with self.assertRaisesRegex(TimeoutError, "task-3"):
                await backend._poll_task(session, task_id="task-3", api_key="secret")


if __name__ == "__main__":
    unittest.main()
