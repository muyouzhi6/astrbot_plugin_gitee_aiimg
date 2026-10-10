import json
import logging
import sys
import types
from pathlib import Path

if "astrbot.api" not in sys.modules:
    _astrbot = sys.modules.setdefault("astrbot", types.ModuleType("astrbot"))
    _api = types.ModuleType("astrbot.api")
    _api.logger = logging.getLogger("test")
    _astrbot.api = _api
    sys.modules["astrbot.api"] = _api

import httpx
import pytest

from core.openai_full_url_backend import OpenAIFullURLBackend, _rewrite_loopback_ref

PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02"
    b"\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\xff\xff?\x00\x05\xfe\x02"
    b"\xfe\xa7\x35\x81\x84\x00\x00\x00\x00IEND\xaeB`\x82"
)


class FakeImgr:
    def __init__(self):
        self.downloaded = []

    async def save_image(self, data, output_format="jpeg"):
        return Path("/tmp/saved.jpg")

    async def download_image(self, url, output_format="jpeg"):
        self.downloaded.append(url)
        return Path("/tmp/dl.jpg")


def make_backend(handler, **kw):
    imgr = FakeImgr()
    b = OpenAIFullURLBackend(
        imgr=imgr,
        full_generate_url="http://gw.example:3000/v1/images/generations",
        full_edit_url="http://gw.example:3000/v1/images/edits",
        api_keys=["k"],
        default_model="gen-model",
        max_retries=0,
        **kw,
    )
    b._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return b, imgr


def ok_url(url="http://127.0.0.1:8000/v1/media/images/a.jpg"):
    return httpx.Response(200, json={"data": [{"url": url}]})


def body(req):
    ct = req.headers.get("content-type", "")
    return json.loads(req.content) if ct.startswith("application/json") else None


@pytest.mark.asyncio
async def test_edit_falls_back_to_object_image_format_and_uses_edit_model():
    seen = []

    def handler(req):
        p = body(req)
        seen.append(p)
        if p is None:
            return httpx.Response(415, json={"error": "json only"})
        if isinstance(p.get("image"), dict):
            assert p["image"]["type"] == "image_url"
            assert p["image"]["url"].startswith("data:image/")
            return ok_url()
        return httpx.Response(400, json={"error": "bad image"})

    b, imgr = make_backend(handler, default_edit_model="edit-model")
    await b.edit("x", [PNG])
    assert all(p["model"] == "edit-model" for p in seen)
    assert len(seen) == 4
    assert imgr.downloaded == ["http://gw.example:8000/v1/media/images/a.jpg"]


@pytest.mark.asyncio
async def test_edit_without_edit_model_keeps_legacy_order_and_model():
    seen = []

    def handler(req):
        p = body(req)
        seen.append(p)
        if p is not None:
            return httpx.Response(400, json={"error": "x"})
        return httpx.Response(200, json={"data": [{"url": "https://cdn.x/a.png"}]})

    b, imgr = make_backend(handler)
    await b.edit("x", [PNG])
    keys = [
        next(k for k in ("image", "images", "image_url") if k in p) for p in seen[:-1]
    ]
    assert keys == ["image", "images", "image_url", "image", "images"]
    assert isinstance(seen[0]["image"], str)
    assert seen[0]["model"] == "gen-model"
    assert seen[-1] is None  # multipart fallback preserved
    assert imgr.downloaded == ["https://cdn.x/a.png"]


@pytest.mark.asyncio
async def test_generate_retries_with_size_auto_once():
    sizes = []

    def handler(req):
        p = body(req)
        sizes.append(p["size"])
        if p["size"] != "auto":
            return httpx.Response(400, json={"error": "invalid size 4096x4096"})
        return ok_url("https://cdn.x/b.png")

    b, _ = make_backend(handler)
    await b.generate("cat")
    assert sizes == ["4096x4096", "auto"]


@pytest.mark.asyncio
async def test_non_size_error_does_not_retry_auto():
    sizes = []

    def handler(req):
        sizes.append(body(req)["size"])
        return httpx.Response(400, json={"error": "bad prompt"})

    b, _ = make_backend(handler)
    with pytest.raises(RuntimeError):
        await b.generate("cat")
    assert sizes == ["4096x4096"]


@pytest.mark.parametrize(
    "ref,endpoint,expected",
    [
        (
            "http://127.0.0.1:8000/m/a.jpg",
            "http://gw:3000/v1",
            "http://gw:8000/m/a.jpg",
        ),
        ("http://localhost/m/a.jpg", "https://gw.io/v1", "http://gw.io/m/a.jpg"),
        ("https://cdn.x/a.jpg", "http://gw:3000/v1", "https://cdn.x/a.jpg"),
        (
            "http://127.0.0.1:8000/a",
            "http://127.0.0.1:3000/v1",
            "http://127.0.0.1:8000/a",
        ),
    ],
)
def test_rewrite_loopback_ref(ref, endpoint, expected):
    assert _rewrite_loopback_ref(ref, endpoint) == expected


def test_registry_passes_edit_model():
    src = (Path(__file__).parents[1] / "core" / "provider_registry.py").read_text()
    assert 'default_edit_model=str(conf.get("edit_model") or "").strip()' in src
