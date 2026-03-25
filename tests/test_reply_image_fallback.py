import importlib.util
import sys
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "gitee_aiimg_testpkg"
CORE_PACKAGE_NAME = f"{PACKAGE_NAME}.core"
UTILS_MODULE_NAME = f"{CORE_PACKAGE_NAME}.utils"
NET_SAFETY_MODULE_NAME = f"{CORE_PACKAGE_NAME}.net_safety"


class _Logger:
    def debug(self, *args, **kwargs):
        return None

    def info(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None

    def error(self, *args, **kwargs):
        return None


class _BaseImage:
    def __init__(self, file=None, **kwargs):
        self.file = file
        self.url = kwargs.get("url", "")
        self.path = kwargs.get("path", "")

    @staticmethod
    def fromURL(url: str, **kwargs):
        return _BaseImage(file=url, url=url, **kwargs)

    @staticmethod
    def fromBase64(data: str, **kwargs):
        return _BaseImage(file=f"base64://{data}", **kwargs)

    @staticmethod
    def fromFileSystem(path: str, **kwargs):
        return _BaseImage(file=f"file:///{path}", path=path, **kwargs)


class _At:
    def __init__(self, qq):
        self.qq = qq


class _Reply:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class _BrokenReply(_Reply):
    @property
    def chain(self):
        raise TypeError("CQHttp.call_action() takes 2 positional arguments but 3 were given")


class _DummyAPI:
    def __init__(self, handler):
        self._handler = handler
        self.calls = []

    async def call_action(self, action: str, **params):
        self.calls.append((action, dict(params)))
        result = self._handler(action, dict(params))
        if hasattr(result, "__await__"):
            return await result
        return result


class _DummyEvent:
    def __init__(self, chain, api=None, *, sender_id="42", self_id="99", group_id=""):
        self._chain = list(chain)
        self._sender_id = sender_id
        self._self_id = self_id
        self._group_id = group_id
        self.message_obj = types.SimpleNamespace(message=list(chain))
        if api is None:
            self.bot = types.SimpleNamespace(api=None)
        else:
            self.bot = types.SimpleNamespace(api=api, call_action=api.call_action)

    def get_messages(self):
        return list(self._chain)

    def get_sender_id(self):
        return self._sender_id

    def get_self_id(self):
        return self._self_id

    def get_group_id(self):
        return self._group_id


def _clear_modules():
    for name in [
        UTILS_MODULE_NAME,
        NET_SAFETY_MODULE_NAME,
        CORE_PACKAGE_NAME,
        PACKAGE_NAME,
        "astrbot",
        "astrbot.api",
        "astrbot.core",
        "astrbot.core.message",
        "astrbot.core.message.components",
        "astrbot.core.platform",
        "astrbot.core.platform.astr_message_event",
        "astrbot.core.utils",
        "astrbot.core.utils.quoted_message_parser",
    ]:
        sys.modules.pop(name, None)


def _load_utils_module(*, parser_impl=None):
    _clear_modules()

    pkg = types.ModuleType(PACKAGE_NAME)
    pkg.__path__ = [str(ROOT)]
    sys.modules[PACKAGE_NAME] = pkg

    core_pkg = types.ModuleType(CORE_PACKAGE_NAME)
    core_pkg.__path__ = [str(ROOT / "core")]
    sys.modules[CORE_PACKAGE_NAME] = core_pkg

    astrbot_mod = types.ModuleType("astrbot")
    astrbot_mod.logger = _Logger()
    sys.modules["astrbot"] = astrbot_mod

    api_mod = types.ModuleType("astrbot.api")
    api_mod.logger = _Logger()
    sys.modules["astrbot.api"] = api_mod

    components_mod = types.ModuleType("astrbot.core.message.components")
    components_mod.At = _At
    components_mod.Image = _BaseImage
    components_mod.Reply = _Reply
    sys.modules["astrbot.core.message.components"] = components_mod
    sys.modules["astrbot.core.message"] = types.ModuleType("astrbot.core.message")

    platform_mod = types.ModuleType("astrbot.core.platform.astr_message_event")
    platform_mod.AstrMessageEvent = object
    sys.modules["astrbot.core.platform.astr_message_event"] = platform_mod
    sys.modules["astrbot.core.platform"] = types.ModuleType("astrbot.core.platform")
    sys.modules["astrbot.core"] = types.ModuleType("astrbot.core")
    sys.modules["astrbot.core.utils"] = types.ModuleType("astrbot.core.utils")

    if parser_impl is not None:
        parser_mod = types.ModuleType("astrbot.core.utils.quoted_message_parser")
        parser_mod.extract_quoted_message_images = parser_impl
        sys.modules["astrbot.core.utils.quoted_message_parser"] = parser_mod

    net_safety_mod = types.ModuleType(NET_SAFETY_MODULE_NAME)
    net_safety_mod.URLFetchPolicy = type("URLFetchPolicy", (), {})

    async def _allow_url(*args, **kwargs):
        return None

    net_safety_mod.ensure_url_allowed = _allow_url
    sys.modules[NET_SAFETY_MODULE_NAME] = net_safety_mod

    spec = importlib.util.spec_from_file_location(
        UTILS_MODULE_NAME,
        ROOT / "core" / "utils.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[UTILS_MODULE_NAME] = module
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


class ReplyImageFallbackTests(unittest.IsolatedAsyncioTestCase):
    async def test_embedded_reply_chain_image_is_used_directly(self):
        utils = _load_utils_module()
        reply = _Reply(id="1", chain=[_BaseImage.fromURL("https://example.com/a.png")])
        event = _DummyEvent([reply])

        images = await utils.get_images_from_event(event, include_avatar=False)

        self.assertEqual(len(images), 1)
        self.assertEqual(images[0].url, "https://example.com/a.png")

    async def test_reply_chain_failure_falls_back_to_get_msg(self):
        utils = _load_utils_module()

        def handler(action, params):
            if action == "get_msg":
                self.assertIn("message_id", params)
                return {
                    "data": {
                        "message": [
                            {
                                "type": "image",
                                "data": {"url": "https://example.com/replied.png"},
                            }
                        ]
                    }
                }
            raise AssertionError(f"unexpected action={action} params={params}")

        api = _DummyAPI(handler)
        reply = _BrokenReply(id="123")
        event = _DummyEvent([reply], api=api)

        images = await utils.get_images_from_event(event, include_avatar=False)

        self.assertEqual(len(images), 1)
        self.assertEqual(images[0].url, "https://example.com/replied.png")
        self.assertEqual(api.calls[0][0], "get_msg")

    async def test_reply_lookup_tries_id_when_message_id_fails(self):
        utils = _load_utils_module()

        def handler(action, params):
            if action != "get_msg":
                raise AssertionError(f"unexpected action={action} params={params}")
            if "message_id" in params:
                raise RuntimeError("message_id unsupported")
            return {
                "data": {
                    "message": [
                        {
                            "type": "image",
                            "data": {"url": "https://example.com/by-id.png"},
                        }
                    ]
                }
            }

        api = _DummyAPI(handler)
        reply = _BrokenReply(id="456")
        event = _DummyEvent([reply], api=api)

        images = await utils.get_images_from_event(event, include_avatar=False)

        self.assertEqual(len(images), 1)
        self.assertEqual(images[0].url, "https://example.com/by-id.png")
        self.assertGreaterEqual(len(api.calls), 2)
        self.assertIn("message_id", api.calls[0][1])
        self.assertIn("id", api.calls[1][1])

    async def test_reply_lookup_continues_when_message_id_returns_none(self):
        utils = _load_utils_module()

        def handler(action, params):
            if action != "get_msg":
                raise AssertionError(f"unexpected action={action} params={params}")
            if "message_id" in params:
                return None
            return {
                "data": {
                    "message": [
                        {
                            "type": "image",
                            "data": {"url": "https://example.com/from-none-fallback.png"},
                        }
                    ]
                }
            }

        api = _DummyAPI(handler)
        reply = _BrokenReply(id="654")
        event = _DummyEvent([reply], api=api)

        images = await utils.get_images_from_event(event, include_avatar=False)

        self.assertEqual(len(images), 1)
        self.assertEqual(images[0].url, "https://example.com/from-none-fallback.png")
        self.assertGreaterEqual(len(api.calls), 2)
        self.assertIn("message_id", api.calls[0][1])
        self.assertIn("id", api.calls[1][1])

    async def test_reply_image_id_is_resolved_via_get_image(self):
        utils = _load_utils_module()

        def handler(action, params):
            if action == "get_msg":
                return {"data": {"message": [{"type": "image", "data": {"file": "img_abc"}}]}}
            if action == "get_image" and params.get("file") == "img_abc":
                return {"data": {"url": "https://example.com/resolved.png"}}
            raise RuntimeError(f"no mock response for action={action} params={params}")

        api = _DummyAPI(handler)
        reply = _BrokenReply(id="789")
        event = _DummyEvent([reply], api=api)

        images = await utils.get_images_from_event(event, include_avatar=False)

        self.assertEqual(len(images), 1)
        self.assertEqual(images[0].url, "https://example.com/resolved.png")
        self.assertTrue(any(call[0] == "get_image" for call in api.calls))

    async def test_prefers_astrbot_quoted_message_parser_when_available(self):
        async def parser_impl(event, reply_component=None):
            return ["https://example.com/from-parser.png"]

        utils = _load_utils_module(parser_impl=parser_impl)
        def handler(action, params):
            raise AssertionError(f"should not call api: action={action} params={params}")

        api = _DummyAPI(handler)
        reply = _BrokenReply(id="999")
        event = _DummyEvent([reply], api=api)

        images = await utils.get_images_from_event(event, include_avatar=False)

        self.assertEqual(len(images), 1)
        self.assertEqual(images[0].url, "https://example.com/from-parser.png")
        self.assertEqual(api.calls, [])


if __name__ == "__main__":
    unittest.main()
