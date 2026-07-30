import importlib.util
import sys
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "jimeng_backend_testpkg"
CORE_PACKAGE_NAME = f"{PACKAGE_NAME}.core"
MODULE_NAME = f"{CORE_PACKAGE_NAME}.jimeng_api_backend"


class _Logger:
    def info(self, *args, **kwargs):
        return None


def _load_module():
    for name in list(sys.modules):
        if name.startswith(PACKAGE_NAME) or name.startswith("astrbot"):
            sys.modules.pop(name, None)

    package = types.ModuleType(PACKAGE_NAME)
    package.__path__ = [str(ROOT)]
    sys.modules[PACKAGE_NAME] = package
    core_package = types.ModuleType(CORE_PACKAGE_NAME)
    core_package.__path__ = [str(ROOT / "core")]
    sys.modules[CORE_PACKAGE_NAME] = core_package

    astrbot_module = types.ModuleType("astrbot")
    astrbot_api = types.ModuleType("astrbot.api")
    astrbot_api.logger = _Logger()
    message_components = types.ModuleType("astrbot.api.message_components")
    message_components.Image = type("Image", (), {})
    sys.modules["astrbot"] = astrbot_module
    sys.modules["astrbot.api"] = astrbot_api
    sys.modules["astrbot.api.message_components"] = message_components

    spec = importlib.util.spec_from_file_location(
        MODULE_NAME,
        ROOT / "core" / "jimeng_api_backend.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[MODULE_NAME] = module
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


class JimengApiBackendTests(unittest.IsolatedAsyncioTestCase):
    async def test_constructor_accepts_and_clamps_timeout(self):
        mod = _load_module()

        backend = mod.JimengApiBackend(
            imgr=object(),
            data_dir=Path("/tmp"),
            api_url="https://example.invalid/jimeng",
            apikey="test-key",
            cookie_list=["conversation:cookie"],
            timeout=7200,
            output_format="webp_lossless",
        )

        self.assertEqual(backend.timeout, 3600)
        self.assertEqual(backend.output_format, "webp_lossless")
        await backend.close()


if __name__ == "__main__":
    unittest.main()
