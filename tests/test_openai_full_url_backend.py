"""Images URL compatibility through the real router and HTTP serialization."""

import asyncio
import base64
import importlib.util
import io
import json
import logging
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from PIL import Image

from test_main_initialize_request_mode import _load_module as _load_main
from test_output_routing import _load_module, _Registry
from test_output_routing import modules as modules

if "astrbot.api" not in sys.modules:
    _astrbot = sys.modules.setdefault("astrbot", types.ModuleType("astrbot"))
    _api = types.ModuleType("astrbot.api")
    _api.logger = logging.getLogger("test")
    _astrbot.api = _api
    sys.modules["astrbot.api"] = _api

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


@pytest.fixture
def runtime(monkeypatch, tmp_path):
    main, _ = _load_main()
    spec = importlib.util.spec_from_file_location(
        "issue_debouncer", Path(__file__).resolve().parents[1] / "core/debouncer.py"
    )
    debouncer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(debouncer)
    plugin = main.GiteeAIImagePlugin(
        SimpleNamespace(),
        {"features": {"draw": {"enabled": True, "llm_tool_enabled": True}}},
    )
    plugin.data_dir = tmp_path
    plugin.debouncer = debouncer.Debouncer(plugin.config)
    plugin.registry = SimpleNamespace(provider_ids=lambda: ["images"])
    plugin._concurrency_lock = asyncio.Lock()
    plugin._image_inflight = {}
    plugin._video_inflight = {}
    plugin.draw = SimpleNamespace(
        generate=AsyncMock(side_effect=RuntimeError("HTTP 503"))
    )
    plugin._signal_llm_tool_failure = AsyncMock()
    plugin._has_message_images = AsyncMock(return_value=False)
    plugin._save_last_image_task_meta = AsyncMock()
    for name in ("mark_processing", "mark_success", "mark_failed"):
        monkeypatch.setattr(main, name, AsyncMock())
    return main, plugin, debouncer


def png(color):
    output = io.BytesIO()
    Image.new("RGB", (4, 3), color).save(output, format="PNG")
    return output.getvalue()


@pytest.fixture
def backend(modules):
    mod = _load_module("openai_full_url_backend")
    imgr = SimpleNamespace(
        save_image=AsyncMock(return_value=Path("saved.png")),
        download_image=AsyncMock(return_value=Path("saved.png")),
    )
    return mod.OpenAIFullURLBackend(
        imgr=imgr,
        full_generate_url="https://images.example/custom/generate",
        full_edit_url="https://images.example/custom/edit",
        api_keys=["offline-test"],
        default_model="image-model",
        max_retries=0,
    )


def router(modules, backend):
    return modules[2].EditRouter(
        {"features": {"edit": {"chain": ["images"]}}},
        imgr=backend.imgr,
        data_dir=Path("."),
        registry=_Registry({"images": backend}),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("multipart", [False, True])
async def test_39_router_preserves_separate_ordered_references(
    modules, backend, multipart
):
    inputs = [png("red"), png("blue"), png("green")]
    requests = []

    def respond(request):
        requests.append(request)
        assert request.url.path == "/custom/edit"
        if request.headers["content-type"].startswith("application/json"):
            payload = json.loads(request.content)
            assert payload["model"] == "image-model"
            assert payload["size"] == "1024x1024"
            assert [
                base64.b64decode(
                    (ref["url"] if isinstance(ref, dict) else ref).split(",", 1)[1]
                )
                for ref in payload["images"]
            ] == inputs
            assert "image" not in payload and "image_url" not in payload
            if multipart:
                return httpx.Response(415, json={"error": "multipart required"})
        else:
            assert request.content.count(b'name="image[]";') == 3
            offsets = [request.content.index(raw) for raw in inputs]
            assert offsets == sorted(offsets)
        return httpx.Response(200, json={"data": [{"url": "/output/result.png"}]})

    backend._client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    try:
        result = await router(modules, backend).edit(
            "图一人物穿图二服装，图三背景",
            inputs,
            size="1024x1024",
            require_ordered_references=True,
        )
        assert result == Path("saved.png")
        assert len(requests) == (3 if multipart else 1)
        backend.imgr.download_image.assert_awaited_once_with(
            "https://images.example/output/result.png", output_format="jpeg"
        )
    finally:
        await backend.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("count", "ordered", "field", "expected_requests"),
    [(1, False, "image", 4), (1, False, "images", 5), (3, True, "images", 2)],
)
async def test_39_json_only_gateway_accepts_object_images(
    backend, count, ordered, field, expected_requests
):
    inputs = [png((index * 40, 20, 30)) for index in range(count)]
    requests = []

    def respond(request):
        requests.append(request)
        if not request.headers["content-type"].startswith("application/json"):
            return httpx.Response(415, json={"error": "JSON only"})
        payload = json.loads(request.content)
        refs = payload.get(field)
        if field == "image" and isinstance(refs, dict):
            refs = [refs]
        if not isinstance(refs, list) or not all(isinstance(ref, dict) for ref in refs):
            return httpx.Response(400, json={"error": "image must be an object"})
        assert [ref["type"] for ref in refs] == ["image_url"] * count
        assert [base64.b64decode(ref["url"].split(",", 1)[1]) for ref in refs] == inputs
        assert "image_url" not in payload
        return httpx.Response(200, json={"data": [{"url": "/output/result.png"}]})

    backend._client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    try:
        assert await backend.edit(
            "ordered edit", inputs, require_ordered_references=ordered
        ) == Path("saved.png")
        assert len(requests) == expected_requests
        assert all(
            "application/json" in req.headers["content-type"] for req in requests
        )
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_39_edit_model_is_separate_from_generate_model(backend):
    backend.default_edit_model = "edit-model"
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(
            200, content=png("white"), headers={"content-type": "image/png"}
        )

    backend._client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    try:
        assert await backend.generate("generate") == Path("saved.png")
        assert await backend.edit("edit", [png("red")]) == Path("saved.png")
        assert await backend.edit("edit", [png("red")], model="explicit-model") == Path(
            "saved.png"
        )
        assert [json.loads(req.content)["model"] for req in requests] == [
            "image-model",
            "edit-model",
            "explicit-model",
        ]
    finally:
        await backend.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["generate", "edit"])
@pytest.mark.parametrize("status", [400, 422])
@pytest.mark.parametrize(
    "size_error",
    [
        {"error": {"message": "unsupported size"}},
        {"error": {"code": "invalid_resolution"}},
        {"error": "4096x4096 is unsupported"},
    ],
)
async def test_39_size_rejection_retries_same_json_payload_with_auto(
    backend, mode, status, size_error
):
    requests = []

    def respond(request):
        payload = json.loads(request.content)
        requests.append(payload)
        if payload["size"] != "auto":
            return httpx.Response(status, json=size_error)
        return httpx.Response(
            200, content=png("white"), headers={"content-type": "image/png"}
        )

    backend._client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    try:
        result = (
            await backend.generate("generate")
            if mode == "generate"
            else await backend.edit("edit", [png("red")])
        )
        assert result == Path("saved.png")
        assert [req["size"] for req in requests] == ["4096x4096", "auto"]
        assert {k: v for k, v in requests[0].items() if k != "size"} == {
            k: v for k, v in requests[1].items() if k != "size"
        }
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_39_auto_size_persists_through_json_formats_and_multipart(backend):
    requests = []

    def respond(request):
        requests.append(request)
        if "application/json" in request.headers["content-type"]:
            payload = json.loads(request.content)
            if payload["size"] != "auto":
                return httpx.Response(
                    400, json={"error": {"message": "size unsupported"}}
                )
            return httpx.Response(415, json={"error": "multipart required"})
        if b'name="size"\r\n\r\nauto' not in request.content:
            return httpx.Response(422, json={"error": {"message": "invalid size"}})
        return httpx.Response(
            200, content=png("white"), headers={"content-type": "image/png"}
        )

    backend._client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    try:
        assert await backend.edit("edit", [png("red")]) == Path("saved.png")
        json_sizes = [
            json.loads(req.content)["size"]
            for req in requests
            if "application/json" in req.headers["content-type"]
        ]
        assert json_sizes == ["4096x4096"] + ["auto"] * 5
        assert len(requests) == 7
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_39_multipart_size_rejection_retries_once_with_auto(backend):
    requests = []

    def respond(request):
        requests.append(request)
        if "application/json" in request.headers["content-type"]:
            return httpx.Response(415, json={"error": "multipart required"})
        if b'name="size"\r\n\r\nauto' not in request.content:
            return httpx.Response(422, json={"error": {"param": "size"}})
        return httpx.Response(
            200, content=png("white"), headers={"content-type": "image/png"}
        )

    backend._client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    try:
        assert await backend.edit("edit", [png("red")]) == Path("saved.png")
        assert len(requests) == 7
        assert all(
            "application/json" in req.headers["content-type"] for req in requests[:5]
        )
        assert b'name="size"\r\n\r\n4096x4096' in requests[5].content
        assert b'name="size"\r\n\r\nauto' in requests[6].content
    finally:
        await backend.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [401, 429, 500, 503])
async def test_39_non_protocol_errors_do_not_resubmit_as_multipart(backend, status):
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(status, json={"error": "upstream failure"})

    backend._client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    try:
        with pytest.raises(RuntimeError, match=f"HTTP {status}"):
            await backend.edit("edit", [png("red")])
        assert len(requests) == 1
        assert "application/json" in requests[0].headers["content-type"]
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_39_retryable_server_error_stays_on_json_transport(backend):
    backend.max_retries = 1
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(500, json={"error": "upstream failure"})

    backend._client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    try:
        with pytest.raises(RuntimeError, match="HTTP 500"):
            await backend.edit("edit", [png("red")])
        assert len(requests) == 2
        assert all(
            "application/json" in req.headers["content-type"] for req in requests
        )
    finally:
        await backend.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("host", ["127.0.0.1", "localhost"])
async def test_39_loopback_media_url_uses_endpoint_host_and_media_port(backend, host):
    backend.full_edit_url = "https://images.example:3000/v1/images/edits"
    ref = f"http://{host}:8000/v1/media/images/result.png?v=1"
    response = httpx.Response(200, json={"data": [{"url": ref}]})
    assert await backend._save_response(
        response, endpoint_url=backend.full_edit_url
    ) == Path("saved.png")
    backend.imgr.download_image.assert_awaited_once_with(
        "http://images.example:8000/v1/media/images/result.png?v=1",
        output_format="jpeg",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("count", [1, 3])
async def test_legacy_edit_keeps_existing_single_image_payload(backend, count):
    inputs = [png("red")] * count
    requests = []

    def respond(request):
        requests.append(request)
        payload = json.loads(request.content)
        assert "image" in payload and "images" not in payload
        data = base64.b64decode(payload["image"].split(",", 1)[1])
        with Image.open(io.BytesIO(data)) as decoded:
            assert decoded.width > 0
        if count == 1:
            assert data == inputs[0]
        return httpx.Response(
            200, content=png("white"), headers={"content-type": "image/png"}
        )

    backend._client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    try:
        assert await backend.edit("edit", inputs) == Path("saved.png")
        assert len(requests) == 1
    finally:
        await backend.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [400, 415, 422, 401, 429, 500])
async def test_39_failed_ordered_upload_never_collages_or_drops_references(
    modules, backend, status
):
    inputs = [png("red"), png("blue")]
    requests = []

    def respond(request):
        requests.append(request)
        if "json" in request.headers["content-type"]:
            assert len(json.loads(request.content)["images"]) == 2
        else:
            assert request.content.count(b'name="image[]";') == 2
            assert all(raw in request.content for raw in inputs)
        return httpx.Response(status, json={"error": "offline rejection"})

    backend._client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    try:
        with pytest.raises(RuntimeError, match=f"HTTP {status}"):
            await router(modules, backend).edit(
                "edit", inputs, require_ordered_references=True
            )
        assert len(requests) == (3 if status in {400, 415, 422} else 1)
        backend.imgr.save_image.assert_not_awaited()
        backend.imgr.download_image.assert_not_awaited()
    finally:
        await backend.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case", ["empty", "disabled", "missing_endpoint", "missing_model", "missing_key"]
)
async def test_39_invalid_edit_inputs_stop_before_http(backend, case):
    inputs = [png("red"), png("blue")]
    if case == "empty":
        inputs = []
    elif case == "disabled":
        backend.supports_edit = False
    elif case == "missing_endpoint":
        backend.full_edit_url = backend.full_generate_url = ""
    elif case == "missing_model":
        backend.default_model = ""
    else:
        backend.api_keys = []
    with pytest.raises((ValueError, RuntimeError)):
        await backend.edit("edit", inputs, require_ordered_references=True)
    assert backend._client is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "ref",
    [
        "https://cdn.example/result.png",
        "http://cdn.example/result.png",
        "/output/result.png",
        "data:image/png;base64," + base64.b64encode(png("red")).decode(),
    ],
)
async def test_21_full_url_response_schemes_are_saved_locally(backend, ref):
    response = httpx.Response(200, json={"data": [{"url": ref}]})
    result = await backend._save_response(
        response, endpoint_url=backend.full_generate_url
    )
    assert result == Path("saved.png")
    if ref.startswith("data:"):
        backend.imgr.save_image.assert_awaited_once_with(
            png("red"), output_format="jpeg"
        )
        backend.imgr.download_image.assert_not_awaited()
    else:
        expected = "https://images.example" + ref if ref.startswith("/") else ref
        backend.imgr.download_image.assert_awaited_once_with(
            expected, output_format="jpeg"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "ref",
    [
        "",
        "ftp://cdn.example/result.png",
        "file:///missing.png",
        "data:image/png;base64,???",
    ],
)
async def test_21_invalid_response_references_fail_without_download(backend, ref):
    with pytest.raises(RuntimeError):
        await backend._save_response(
            httpx.Response(200, json={"data": [{"url": ref}]}),
            endpoint_url=backend.full_generate_url,
        )
    backend.imgr.download_image.assert_not_awaited()
    backend.imgr.save_image.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("count", [2, 8])
@pytest.mark.parametrize(
    "outcome", ["json", "multipart", "rejected", "timeout", "malformed"]
)
async def test_39_llm_entry_routes_explicit_references_in_order(
    runtime, modules, backend, monkeypatch, count, outcome
):
    from test_background_pipeline_event import _Event

    main, plugin, _ = runtime
    inputs = [png((index * 20, 30, 40)) for index in range(count)]
    selection = main.ReferenceSelection(
        images=tuple(inputs),
        sources=tuple(
            {"id": f"ca_{index}", "role": "subject"} for index in range(count)
        ),
    )
    plugin._prepare_context_reference_request = AsyncMock(return_value=selection)
    plugin._image_segs_to_bytes = AsyncMock(return_value=[])
    monkeypatch.setattr(main, "get_images_from_event", AsyncMock(return_value=[]))
    plugin.edit = router(modules, backend)
    plugin._finalize_llm_tool_image = AsyncMock(return_value="delivered")
    backend.extra_body = {
        "image": "stale",
        "images": ["stale"],
        "image_url": "stale",
        "image[]": "stale",
        "quality": "high",
    }
    requests = []

    def respond(request):
        requests.append(request)
        assert b"stale" not in request.content
        if "json" in request.headers["content-type"]:
            payload = json.loads(request.content)
            assert [
                base64.b64decode(
                    (ref["url"] if isinstance(ref, dict) else ref).split(",", 1)[1]
                )
                for ref in payload["images"]
            ] == inputs
            assert not {"image", "image_url", "image[]"}.intersection(payload)
            assert payload["quality"] == "high"
            if outcome == "multipart":
                return httpx.Response(415, json={"error": "multipart required"})
        else:
            assert request.content.count(b'name="image[]";') == count
            offsets = [request.content.index(raw) for raw in inputs]
            assert offsets == sorted(offsets)
            assert b'name="quality"\r\n\r\nhigh' in request.content
        if outcome == "rejected":
            return httpx.Response(500, json={"error": "offline rejection"})
        if outcome == "timeout":
            raise httpx.ReadTimeout("offline timeout", request=request)
        if outcome == "malformed":
            return httpx.Response(200, json={"data": []})
        return httpx.Response(
            200, content=png("white"), headers={"content-type": "image/png"}
        )

    backend._client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    try:
        event = _Event()
        result = await plugin.aiimg_generate(
            event,
            "按参考图次序合成",
            mode="edit",
            reference_image_ids=[f"ca_{index}" for index in range(count)],
            reference_roles=["subject"] * count,
        )
        assert len(requests) == (3 if outcome == "multipart" else 1)
        if outcome in {"json", "multipart"}:
            assert result == "delivered"
            plugin._finalize_llm_tool_image.assert_awaited_once()
            assert plugin._finalize_llm_tool_image.call_args.args[1] == Path(
                "saved.png"
            )
            plugin._signal_llm_tool_failure.assert_not_awaited()
        else:
            assert result.content
            plugin._finalize_llm_tool_image.assert_not_awaited()
            plugin._signal_llm_tool_failure.assert_awaited_once()
            backend.imgr.save_image.assert_not_awaited()
            backend.imgr.download_image.assert_not_awaited()
        plugin.draw.generate.assert_not_awaited()
        assert plugin._image_inflight == {}
        assert not event.get_extra("_gitee_image_reference_call_active")
    finally:
        await backend.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("background", [False, True])
@pytest.mark.parametrize("reference_count,extra_count", [(2, 0), (8, 0), (1, 1)])
@pytest.mark.parametrize("protocol", ["json", "object", "multipart", "rejected"])
async def test_39_fixed_selfie_references_without_history_keep_order(
    runtime,
    modules,
    backend,
    monkeypatch,
    tmp_path,
    background,
    reference_count,
    extra_count,
    protocol,
):
    from test_background_pipeline_event import _Event, _target

    main, plugin, _ = runtime
    inputs = [
        png((index * 20, 30, 40)) for index in range(reference_count + extra_count)
    ]
    references = []
    for index, data in enumerate(inputs[:reference_count]):
        path = tmp_path / f"reference-{index}.png"
        path.write_bytes(data)
        references.append(path)
    plugin.config["features"]["selfie"] = {"enabled": True, "llm_tool_enabled": True}
    plugin._get_selfie_reference_paths = AsyncMock(return_value=(references, "webui"))
    plugin._get_life_context_without_llm = AsyncMock(return_value={})
    plugin._image_segs_to_bytes = AsyncMock(return_value=inputs[reference_count:])
    monkeypatch.setattr(main, "get_images_from_event", AsyncMock(return_value=[]))
    plugin.edit = router(modules, backend)
    requests = []
    validated_requests = []

    def respond(request):
        requests.append(request)
        if "json" in request.headers["content-type"]:
            payload = json.loads(request.content)
            assert "images" in payload, "selfie references were collapsed to one image"
            refs = payload["images"]
            assert [
                base64.b64decode(
                    (ref["url"] if isinstance(ref, dict) else ref).split(",", 1)[1]
                )
                for ref in refs
            ] == inputs
            validated_requests.append(request)
            if protocol == "multipart" or (
                protocol == "object" and not isinstance(refs[0], dict)
            ):
                return httpx.Response(415, json={"error": "unsupported image format"})
        else:
            assert protocol == "multipart"
            assert request.content.count(b'name="image[]";') == len(inputs)
            offsets = [request.content.index(raw) for raw in inputs]
            assert offsets == sorted(offsets)
            validated_requests.append(request)
        if protocol == "rejected":
            return httpx.Response(500, json={"error": "offline rejection"})
        return httpx.Response(
            200, content=png("white"), headers={"content-type": "image/png"}
        )

    backend._client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    manager = None
    try:
        event = _Event()
        assert plugin._context_selection(event) is None
        if background:
            manager = main.BackgroundImageTaskManager(
                tmp_path / "queue", heartbeat_seconds=60
            )
            await manager.start()
            plugin._background_manager_for_event = lambda event: manager
            plugin._build_background_delivery_target = AsyncMock(
                return_value=_target(main)
            )
            factories, jobs = [], []
            manager.start_worker = lambda task_id, factory: factories.append(factory)

            async def capture_job(manager, task_id, job, target, **kwargs):
                jobs.append(job)

            plugin._run_background_single = capture_job
            receipt = await plugin._accept_background_single(
                event,
                prompt="自然自拍",
                mode="selfie_ref",
                backend="auto",
                output="",
                aspect_ratio="",
                resolution="",
            )
            assert json.loads(receipt.content[0].text)["status"] == "accepted"
            await factories[0]()
            # The queued task must use the frozen references without an old event.
            for path in references:
                path.unlink()
            event._extras.clear()
            assert not jobs[0].task_meta.get("reference_sources")
            generate = plugin._execute_prepared_image_job(manager, jobs[0])
        else:
            generate = plugin._generate_selfie_image_with_meta(event, "自然自拍", None)
        if protocol == "rejected":
            with pytest.raises(RuntimeError):
                await generate
            backend.imgr.save_image.assert_not_awaited()
            backend.imgr.download_image.assert_not_awaited()
        else:
            result, meta = await generate
            assert result == Path("saved.png")
            assert meta["reference_count"] == reference_count
            assert meta["extra_reference_count"] == extra_count
        assert (
            len(requests)
            == {"json": 1, "object": 2, "multipart": 3, "rejected": 1}[protocol]
        )
        assert validated_requests == requests
        plugin.draw.generate.assert_not_awaited()
    finally:
        await backend.close()
        if manager:
            await manager.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("ordered", [False, True])
async def test_legacy_single_edit_json_and_multipart_fallbacks(backend, ordered):
    inputs = [png("red")]
    requests = []

    def respond(request):
        requests.append(request)
        index = len(requests) - 1
        if index < 5:
            payload = json.loads(request.content)
            key = ("image", "images", "image_url", "image", "images")[index]
            value = payload[key][0] if key == "images" else payload[key]
            if isinstance(value, dict):
                assert value["type"] == "image_url"
                value = value["url"]
            assert base64.b64decode(value.split(",", 1)[1]) == inputs[0]
            return httpx.Response(422, json={"error": "multipart required"})
        assert b'name="image";' in request.content
        assert inputs[0] in request.content
        return httpx.Response(
            200, json={"data": [{"b64_json": base64.b64encode(png("white")).decode()}]}
        )

    backend._client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    try:
        assert await backend.edit(
            "edit", inputs, require_ordered_references=ordered
        ) == Path("saved.png")
        assert len(requests) == 6
        backend.imgr.save_image.assert_awaited_once_with(
            png("white"), output_format="jpeg"
        )
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_generate_keeps_custom_endpoint_size_and_extra_body(backend):
    requests = []

    def respond(request):
        requests.append(request)
        assert request.url.path == "/custom/generate"
        assert json.loads(request.content) == {
            "model": "image-model",
            "prompt": "a cat",
            "size": "2048x2048",
            "quality": "high",
        }
        return httpx.Response(
            200, json={"data": [{"url": "https://cdn.example/result.png"}]}
        )

    backend._client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    try:
        assert await backend.generate(
            "a cat", resolution="2K", extra_body={"quality": "high"}
        ) == Path("saved.png")
        assert len(requests) == 1
    finally:
        await backend.close()
