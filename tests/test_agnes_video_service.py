import asyncio
import importlib
import json
import time
from unittest.mock import AsyncMock

import httpx
import pytest
from test_sora2_video_service import _load_module


@pytest.fixture
def module():
    sora = _load_module()
    return importlib.import_module(sora.__package__ + ".agnes_video_service")


@pytest.fixture
def service(module, tmp_path):
    return module.AgnesVideoService(
        settings={"api_key": "private-test-key"}, data_dir=tmp_path
    )


def mock_transport(module, service, monkeypatch, responses):
    calls = []

    def handle(request):
        calls.append(request)
        item = responses.pop(0)
        if isinstance(item, Exception):
            raise item
        status, data, *headers = item
        return httpx.Response(status, json=data, headers=headers[0] if headers else {})

    original = httpx.AsyncClient
    monkeypatch.setattr(
        module.httpx,
        "AsyncClient",
        lambda **kw: original(transport=httpx.MockTransport(handle), **kw),
    )
    service._wait_request = AsyncMock()
    service.poll_interval = 0
    return calls


@pytest.mark.asyncio
async def test_uses_video_id_and_model_name_and_only_completed_url(
    module, service, monkeypatch
):
    calls = mock_transport(
        module,
        service,
        monkeypatch,
        [
            (200, {"id": "task-wrong", "video_id": "video-right", "status": "queued"}),
            (
                200,
                {
                    "status": "in_progress",
                    "metadata": {"url": "https://cdn.example/not-ready.mp4"},
                },
            ),
            (
                200,
                {
                    "status": "completed",
                    "metadata": {"url": "https://cdn.example/ready.mp4"},
                },
            ),
        ],
    )
    assert await service.generate_video_url("a cup") == "https://cdn.example/ready.mp4"
    assert calls[0].url.path == "/v1/videos"
    payload = json.loads(calls[0].content)
    assert payload == {
        "model": "agnes-video-2.5-flash",
        "mode": "text",
        "seconds": "5",
        "size": "720P",
        "aspect_ratio": "16:9",
        "n": 1,
        "prompt": "a cup",
    }
    for request in calls[1:]:
        assert request.url.path == "/agnesapi"
        assert dict(request.url.params) == {
            "video_id": "video-right",
            "model_name": "agnes-video-2.5-flash",
        }
        assert request.headers["authorization"] == "Bearer private-test-key"
    assert service._wait_request.await_count == 3


@pytest.mark.parametrize(
    "extra,match",
    [
        ({"size": "1080P"}, "720P"),
        ({"n": 2}, "n=1"),
        ({"seconds": 4}, "字符串"),
        ({"seconds": "13"}, "4 至 12"),
        ({"aspect_ratio": "2:3"}, "画幅"),
        ({"mode": "text", "images": ["https://cdn.example/a.png"]}, "模式"),
        ({"mode": "keyframe"}, "模式"),
        ({"mode": "reference", "images": ["https://cdn.example/a.png"] * 6}, "最多 5"),
        ({"mode": "reference", "audios": ["https://cdn.example/a.mp3"] * 4}, "最多 3"),
        (
            {"mode": "reference", "videos": [{"url": "https://cdn.example/a.mp4"}]},
            "模式",
        ),
        ({"mode": "reference", "images": "https://cdn.example/a.png"}, "列表"),
        ({"mode": "reference", "images": ["file:///secret"]}, "URL"),
        ({"seed": True}, "整数"),
    ],
)
def test_flash_rejects_invalid_parameters_before_submission(service, extra, match):
    service.settings["extra_body"] = extra
    with pytest.raises(ValueError, match=match):
        service.build_payload("a cup")


def test_first_frame_and_reference_payloads_do_not_mutate_settings(service):
    image = b"\x89PNG\r\n\x1a\n"
    first = service.build_payload("animate", image)
    assert first["mode"] == "keyframe"
    assert first["first_frame"].startswith("data:image/png;base64,")
    service.settings.update(
        mode="reference",
        extra_body={"images": ["https://cdn.example/other.png"], "seed": 7},
    )
    reference = service.build_payload("animate", image)
    assert reference["mode"] == "reference" and len(reference["images"]) == 2
    assert reference["seed"] == 7
    assert service.settings["extra_body"]["images"] == ["https://cdn.example/other.png"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "result",
    [
        httpx.ReadTimeout("timeout"),
        asyncio.TimeoutError(),
        (408, {"message": "request timeout"}),
        (503, {"message": "busy"}),
        (200, {"id": "task-only"}),
    ],
)
async def test_uncertain_create_never_replays_or_falls_through(
    module, service, monkeypatch, result
):
    calls = mock_transport(module, service, monkeypatch, [result])
    with pytest.raises(module.AgnesVideoError) as exc:
        await service.generate_video_url("a cup")
    assert exc.value.stop_provider_chain
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_poll_retry_keeps_original_task_and_key(module, service, monkeypatch):
    calls = mock_transport(
        module,
        service,
        monkeypatch,
        [
            (200, {"video_id": "video-one"}),
            (429, {"detail": "limited"}, {"retry-after": "0"}),
            (
                200,
                {
                    "status": "completed",
                    "metadata": {"url": "https://cdn.example/a.mp4"},
                },
            ),
        ],
    )
    assert await service.generate_video_url("a cup") == "https://cdn.example/a.mp4"
    assert [r.method for r in calls] == ["POST", "GET", "GET"]
    assert calls[1].url == calls[2].url


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout_type", [TimeoutError, asyncio.TimeoutError])
async def test_timeout_keeps_task_identity_and_never_recreates(
    module, service, monkeypatch, timeout_type
):
    calls = mock_transport(
        module, service, monkeypatch, [(200, {"video_id": "retained-video"})]
    )
    service.poll = AsyncMock(side_effect=timeout_type())
    with pytest.raises(module.AgnesVideoError, match="retained-video") as exc:
        await service.generate_video_url("a cup")
    assert exc.value.stop_provider_chain and len(calls) == 1


@pytest.mark.asyncio
async def test_oversized_poll_response_does_not_start_another_task(
    module, service, monkeypatch
):
    calls = mock_transport(
        module,
        service,
        monkeypatch,
        [(200, {"video_id": "retained-video"}), (200, {"data": "x" * 2100000})],
    )
    with pytest.raises(module.AgnesVideoError, match="响应过大") as exc:
        await service.generate_video_url("a cup")
    assert exc.value.stop_provider_chain
    assert [r.method for r in calls] == ["POST", "GET"]


@pytest.mark.asyncio
async def test_failed_and_malformed_completion_are_not_success(
    module, service, monkeypatch
):
    mock_transport(
        module,
        service,
        monkeypatch,
        [(200, {"video_id": "one"}), (200, {"status": "completed", "metadata": {}})],
    )
    with pytest.raises(module.AgnesVideoError, match="未返回视频地址"):
        await service.generate_video_url("a cup")


@pytest.mark.asyncio
async def test_live_gateway_top_level_url_is_supported(module, service, monkeypatch):
    mock_transport(
        module,
        service,
        monkeypatch,
        [
            (200, {"video_id": "live-id", "id": "live-id"}),
            (
                200,
                {
                    "id": "live-id",
                    "status": "completed",
                    "url": "https://cdn.example/real.mp4",
                },
            ),
        ],
    )
    assert await service.generate_video_url("a cup") == "https://cdn.example/real.mp4"


@pytest.mark.asyncio
async def test_rate_budget_is_shared_and_survives_new_backend(
    module, service, tmp_path
):
    second = module.AgnesVideoService(
        settings={"api_key": "private-test-key"}, data_dir=tmp_path
    )
    waits = await asyncio.gather(
        asyncio.to_thread(service._reserve), asyncio.to_thread(second._reserve)
    )
    assert len([w for w in waits if w == 0]) == 1
    assert max(waits) > 59
    third = module.AgnesVideoService(
        settings={"api_key": "private-test-key"}, data_dir=tmp_path
    )
    assert third._reserve() > 59
    assert b"private-test-key" not in service.rate_path.read_bytes()
    with pytest.raises(TimeoutError):
        await third._wait_request(time.monotonic() + 1)


def test_schema_contains_agnes_and_removes_3365():
    from pathlib import Path

    schema = json.loads(
        (Path(__file__).parents[1] / "_conf_schema.json").read_text(
            encoding="utf-8-sig"
        )
    )
    templates = schema["providers"]["templates"]
    assert "agnes_video" in templates
    assert "3365_video" not in templates and "sd20_video" not in templates
