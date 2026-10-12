import importlib.util
import json
import sys
from unittest.mock import AsyncMock, Mock

import httpx
import openai
import pytest

from test_provider_registry_request_modes import CORE_PACKAGE_NAME, ROOT, _load_module


BACKENDS = [
    ("openai_images", "openai_compat_backend", "OpenAICompatBackend"),
    ("openai_chat", "openai_chat_image_backend", "OpenAIChatImageBackend"),
    ("grok_images", "grok_images_backend", "GrokImagesBackend"),
    ("openai_full_url_images", "openai_full_url_backend", "OpenAIFullURLBackend"),
    (
        "modelscope_openai_images",
        "modelscope_async_backend",
        "ModelScopeAsyncImageBackend",
    ),
    ("gemini_native", "gemini_edit", "GeminiEditBackend"),
    ("vertex_ai_anonymous", "vertex_ai_anonymous_backend", "VertexAIAnonymousBackend"),
]


def registry_for(template, module_name, class_name, value):
    registry_module = _load_module()
    for stub in ("gitee_sizes", "openai_compat_backend", "openai_chat_image_backend"):
        sys.modules.pop(f"{CORE_PACKAGE_NAME}.{stub}", None)
    name = f"{CORE_PACKAGE_NAME}.{module_name}"
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "core" / f"{module_name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    setattr(registry_module, class_name, getattr(module, class_name))
    if template == "vertex_ai_anonymous":
        registry_module.VertexAIAnonymousSettings = module.VertexAIAnonymousSettings
    conf = {
        "id": "test",
        "__template_key": template,
        "base_url": "https://offline.example/v1",
        "full_generate_url": "https://offline.example/v1/images/generations",
        "api_keys": ["offline-test"],
        "graphql_api_key": "offline-test",
        "model": "gemini-image",
    }
    if value != "missing":
        conf["max_retries"] = value
    return registry_module.ProviderRegistry(
        {"providers": [conf]}, imgr=object(), data_dir=ROOT
    )


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("value", [0, "0", 1, None, "missing"])
def test_registry_preserves_retry_values(backend, value):
    instance = registry_for(*backend, value).get_backend("test")
    actual = (
        instance.settings.max_retries
        if backend[0] == "vertex_ai_anonymous"
        else instance.max_retries
    )
    default = 10 if backend[0] == "vertex_ai_anonymous" else 2
    assert actual == (default if value in (None, "missing") else int(value))


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("value", [-1, "-1", "", "oops", 1.5, True, [], {}])
def test_registry_rejects_invalid_retry_values(backend, value):
    with pytest.raises(ValueError, match="max_retries"):
        registry_for(*backend, value).get_backend("test")


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", BACKENDS[:2])
async def test_zero_reaches_sdk_and_sends_one_failed_request(backend):
    instance = registry_for(*backend, 0).get_backend("test")
    client = instance._get_client("offline-test")
    assert client.max_retries == 0
    calls = []

    def fail(request):
        calls.append(request)
        return httpx.Response(500, json={"error": {"message": "offline failure"}})

    await client._client.aclose()
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(fail))
    try:
        with pytest.raises(openai.InternalServerError):
            if backend[0] == "openai_images":
                await instance.generate("test")
            else:
                await instance._run_chat_request_with_retries(
                    lambda: client.chat.completions.create(
                        model="gemini-image", messages=[]
                    ),
                    model="gemini-image",
                    log_tag="test",
                    input_mode="text",
                )
        assert len(calls) == 1
    finally:
        await instance.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("count", [0, 1, 2, "missing"])
@pytest.mark.parametrize("operation", ["generate", "edit"])
async def test_grok_timeout_attempt_count(count, operation, monkeypatch):
    instance = registry_for(*BACKENDS[2], count).get_backend("test")
    session = Mock()
    session.post.side_effect = TimeoutError("offline timeout")
    instance._ensure_session = AsyncMock(return_value=session)
    monkeypatch.setattr("asyncio.sleep", AsyncMock())
    with pytest.raises(TimeoutError if count == 0 else RuntimeError):
        await getattr(instance, operation)(
            "test", **({"images": [b"offline-image"]} if operation == "edit" else {})
        )
    # Positive values retain the legacy format/size fallback traversal.
    expected = (
        1
        if count == 0
        else (2 if count == "missing" else count)
        * (3 if operation == "generate" else 6)
    )
    assert session.post.call_count == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("count", [0, 1, 2, "missing"])
@pytest.mark.parametrize("operation", ["generate", "edit"])
@pytest.mark.parametrize(
    "status, body, error, message",
    [
        (
            503,
            b'{"error": {"message": "offline failure"}}',
            RuntimeError,
            "offline failure",
        ),
        (400, b'{"error": {"message": "bad request"}}', RuntimeError, "bad request"),
        (200, b"not-json", json.JSONDecodeError, "Expecting value"),
    ],
)
async def test_grok_response_failure_attempt_count(
    count, operation, status, body, error, message, monkeypatch
):
    instance = registry_for(*BACKENDS[2], count).get_backend("test")
    response = Mock(status=status)
    response.read = AsyncMock(return_value=body)
    context = AsyncMock()
    context.__aenter__.return_value = response
    session = Mock()
    session.post.return_value = context
    instance._ensure_session = AsyncMock(return_value=session)
    monkeypatch.setattr("asyncio.sleep", AsyncMock())

    if status == 200 and count != 0:
        error, message = RuntimeError, "API 响应格式异常"
    with pytest.raises(error, match=message):
        await getattr(instance, operation)(
            "test", **({"images": [b"offline-image"]} if operation == "edit" else {})
        )

    expected = max(1, 2 if count == "missing" else count)
    if status == 200 and count != 0:
        expected *= 3 if operation == "generate" else 6
    assert session.post.call_count == expected
    assert response.read.await_count == expected
    endpoint = "edits" if operation == "edit" else "generations"
    assert all(
        call.args[0].endswith(f"/v1/images/{endpoint}")
        for call in session.post.call_args_list
    )


def test_retry_schema_describes_compatibility_requests():
    schema = json.loads((ROOT / "_conf_schema.json").read_text(encoding="utf-8"))
    fields = [
        schema["providers"]["templates"][name]["items"]["max_retries"]
        for name in [
            *(backend[0] for backend in BACKENDS),
            "grok_chat",
            "gemini_openai_images",
            "gemini_openai_chat",
            "gitee_images",
        ]
    ]
    assert len(fields) == 11
    for field in fields:
        assert "0 禁用失败重试" in field["description"]
        assert "协议兼容请求和多服务商回退仍会执行" in field["description"]
        assert "只尝试一次" not in field["description"]


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [0, "0"])
@pytest.mark.parametrize("operation", ["generate", "edit"])
async def test_grok_zero_preserves_response_format_fallback(
    value, operation, monkeypatch
):
    instance = registry_for(*BACKENDS[2], value).get_backend("test")
    response = Mock(status=400)
    response.read = AsyncMock(
        return_value=b'{"error": {"message": "unsupported response_format"}}'
    )
    context = AsyncMock()
    context.__aenter__.return_value = response
    session = Mock()
    session.post.return_value = context
    instance._ensure_session = AsyncMock(return_value=session)
    sleep = AsyncMock()
    monkeypatch.setattr("asyncio.sleep", sleep)

    with pytest.raises(RuntimeError, match="unsupported response_format"):
        await getattr(instance, operation)(
            "test", **({"images": [b"offline-image"]} if operation == "edit" else {})
        )

    assert session.post.call_count == 3
    assert response.read.await_count == 3
    payloads = []
    for call in session.post.call_args_list:
        if operation == "generate":
            assert call.args[0].endswith("/v1/images/generations")
            payloads.append(call.kwargs["json"])
        else:
            assert call.args[0].endswith("/v1/images/edits")
            payloads.append(
                {field[0]["name"]: field[2] for field in call.kwargs["data"]._fields}
            )
    assert [payload.get("response_format") for payload in payloads] == [
        "b64_json",
        "url",
        None,
    ]
    sleep.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("count", [0, 1, 2, 10, "missing"])
async def test_vertex_failure_attempt_count(count):
    instance = registry_for(*BACKENDS[6], count).get_backend("test")
    instance._get_recaptcha_token = AsyncMock(return_value="offline")
    instance._call_api = AsyncMock(return_value=(None, 500, "offline failure"))
    with pytest.raises(RuntimeError, match="offline failure"):
        await instance._generate_images("test", image_bytes_list=None)
    assert instance._call_api.await_count == max(1, 10 if count == "missing" else count)


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [0, "0"])
@pytest.mark.parametrize("failure", ["http", "timeout"])
@pytest.mark.parametrize("backend", [BACKENDS[4], BACKENDS[5]])
async def test_aiohttp_zero_sends_one_failed_request(
    backend, value, failure, monkeypatch
):
    instance = registry_for(*backend, value).get_backend("test")
    response = Mock(status=503)
    response.text = AsyncMock(return_value="offline failure")
    request_context = AsyncMock()
    request_context.__aenter__.return_value = response
    request = Mock(return_value=request_context)
    if failure == "timeout":
        request.side_effect = TimeoutError("offline timeout")
    session = Mock(post=request, request=request)
    sleep = AsyncMock()
    monkeypatch.setattr("asyncio.sleep", sleep)
    if backend[0] == "gemini_native":
        instance._get_session = AsyncMock(return_value=session)
    else:
        session_context = AsyncMock()
        session_context.__aenter__.return_value = session
        module = sys.modules[type(instance).__module__]
        monkeypatch.setattr(
            module.aiohttp, "ClientSession", Mock(return_value=session_context)
        )
    with pytest.raises(RuntimeError, match="503|请求超时|offline timeout"):
        await instance.generate("test")
    assert request.call_count == 1
    sleep.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [0, "0"])
@pytest.mark.parametrize("failure", ["http", "timeout"])
@pytest.mark.parametrize("operation", ["generate", "multipart"])
async def test_full_url_zero_sends_one_failed_request(
    value, failure, operation, monkeypatch
):
    instance = registry_for(*BACKENDS[3], value).get_backend("test")
    calls = []

    def fail(request):
        calls.append(request)
        if failure == "timeout":
            raise httpx.ReadTimeout("offline timeout", request=request)
        return httpx.Response(503, json={"error": {"message": "offline failure"}})

    sleep = AsyncMock()
    monkeypatch.setattr("asyncio.sleep", sleep)
    async with httpx.AsyncClient(transport=httpx.MockTransport(fail)) as client:
        monkeypatch.setattr(instance, "_get_client", lambda: client)

        async def run():
            if operation == "generate":
                return await instance.generate("test")
            response = await instance._post_multipart(
                instance.full_generate_url,
                "offline-test",
                {"prompt": "test"},
                {"image": ("image.png", b"offline-image", "image/png")},
            )
            return await instance._save_response(
                response, endpoint_url=instance.full_generate_url
            )

        with pytest.raises(RuntimeError, match="offline failure|offline timeout"):
            await run()
    assert len(calls) == 1
    sleep.assert_not_awaited()


@pytest.mark.parametrize("value", [-1, "oops", False, 0.5])
def test_direct_constructor_rejects_invalid_value(value):
    instance = registry_for(*BACKENDS[0], 0).get_backend("test")
    with pytest.raises(ValueError, match="max_retries"):
        type(instance)(
            imgr=object(),
            base_url="https://offline.example",
            api_keys=[],
            max_retries=value,
        )
