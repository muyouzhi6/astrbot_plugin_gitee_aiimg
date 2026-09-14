import asyncio
import importlib
import io
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from PIL import Image

from core.studio_store import StudioConflict, StudioStore
from test_main_initialize_request_mode import _load_module
from test_background_pipeline_event import _Event, _plugin


def picture(color):
    buffer = io.BytesIO()
    Image.new("RGB", (30, 40), color).save(buffer, "PNG")
    return buffer.getvalue()


def character(store, name, color, kind="person"):
    asset = store.add_image(picture(color))
    return store.save_document(
        "character",
        {
            "name": name,
            "kind": kind,
            "bot_id": "bot" if kind == "bot" else "",
            "owner_sender": "user" if kind == "person" else "",
            "allowed_senders": ["user"],
            "scopes": [],
            "active_look": "daily",
            "looks": [{"name": "daily", "assets": [asset["id"]]}],
        },
    )


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    main, _ = _load_module()
    module = importlib.import_module(main.__package__ + ".core.studio")
    characters = importlib.import_module(main.__package__ + ".core.studio_characters")
    workflow_module = importlib.import_module(
        main.__package__ + ".core.studio_workflows"
    )
    real_planner = importlib.import_module("core.llm_batch_planner")
    for name in ("parse_planned_prompt_items", "validate_planned_prompt_items"):
        monkeypatch.setattr(workflow_module, name, getattr(real_planner, name))

    class Config(dict):
        def save_config(self):
            pass

    conf = Config(
        providers=[
            {
                "id": "test",
                "__template_key": "openai_images",
                "base_url": "https://example.com/v1",
                "api_keys": ["secret-value"],
                "model": "gpt-image-2",
            }
        ],
        features={
            "draw": {"chain": [{"provider_id": "test"}]},
            "edit": {"chain": [{"provider_id": "test"}]},
        },
    )
    plugin = _plugin(main, None)
    plugin.data_dir = tmp_path
    plugin.config = conf
    plugin.imgr = SimpleNamespace()
    plugin.draw = SimpleNamespace(generate=AsyncMock())
    plugin.edit = SimpleNamespace(edit=AsyncMock())
    plugin.context.get_all_providers = lambda: []
    plugin.registry = module.ProviderRegistry(conf, imgr=plugin.imgr, data_dir=tmp_path)
    studio = module.Studio(plugin)
    plugin.studio = studio
    return main, module, characters, plugin, studio


def test_original_assets_survive_caches_and_thumbnail_is_private(tmp_path):
    store = StudioStore(tmp_path)
    data = picture("red")
    a = store.add_image(data, kind="history", metadata={"prompt": "test prompt"})
    assert "file" not in a
    assert store.asset_path(a["id"]).read_bytes() == data
    assert store.asset_path(a["id"], thumbnail=True).suffix == ".jpg"
    with pytest.raises(ValueError):
        store.asset_path("../../secret")
    assert StudioStore(tmp_path).list_assets()["items"][0]["prompt"] == "test prompt"


def test_stale_document_cannot_overwrite_newer_canvas(tmp_path):
    store = StudioStore(tmp_path)
    first = store.save_document("workspace", {"name": "one", "layers": []})
    second = store.save_document("workspace", {**first, "name": "two"})
    with pytest.raises(StudioConflict):
        store.save_document("workspace", {**first, "name": "stale"})
    assert store.document("workspace", first["id"])["name"] == "two"
    assert second["revision"] == 2


def test_removing_look_clears_scoped_selection_and_keeps_reference_assets(tmp_path):
    store = StudioStore(tmp_path)
    c = character(store, "User", "blue")
    original_asset = c["looks"][0]["assets"][0]
    store.save_document("appearance", {"id": "chat:" + c["id"], "look": "daily"})
    c["looks"][0]["name"] = "new"
    c["active_look"] = "new"
    saved = store.save_document("character", c)
    assert store.document("appearance", "chat:" + c["id"]) is None
    with pytest.raises(StudioConflict):
        store.delete_character(c["id"], c["revision"])
    store.save_document("appearance", {"id": "chat:" + c["id"], "look": "new"})
    assert store.delete_character(saved["id"], saved["revision"])["deleted"]
    assert store.document("character", c["id"]) is None
    assert store.document("appearance", "chat:" + c["id"]) is None
    assert store.asset_path(original_asset).is_file()


def test_portrait_keeps_faces_and_wardrobes_individually_bound(runtime):
    _, _, chars, _, studio = runtime
    character(studio.store, "Bot", "red", "bot")
    character(studio.store, "User", "blue")
    selection, note, cast = chars.resolve_characters(
        studio.store,
        ["self", "me"],
        ["", "black jacket"],
        sender="user",
        scope="chat",
        bot_id="bot",
        life_context={"outfit": "white dress", "schedule": "park"},
    )
    assert len(selection.images) == 2
    assert selection.images[0] != selection.images[1]
    assert cast[0]["outfit"] == "white dress"
    assert cast[1]["outfit"] == "black jacket"
    assert cast[0]["reference_indices"] == [1]
    assert cast[1]["reference_indices"] == [2]
    assert "不互换脸" in note


def test_photograph_requester_does_not_add_bot_or_schedule(runtime):
    _, _, chars, _, studio = runtime
    character(studio.store, "Bot", "red", "bot")
    character(studio.store, "User", "blue")
    selection, note, cast = chars.resolve_characters(
        studio.store,
        ["me"],
        ["suit"],
        sender="user",
        scope="chat",
        bot_id="bot",
        life_context={"outfit": "white dress", "schedule": "private bot plan"},
    )
    assert len(selection.images) == 1
    assert cast[0]["kind"] == "person"
    assert "white dress" not in note
    assert "private bot plan" not in note


def test_character_acl_and_identity_alias_deduplication(runtime):
    _, _, chars, _, studio = runtime
    c = character(studio.store, "User", "blue")
    with pytest.raises(ValueError, match="无权"):
        chars.resolve_characters(
            studio.store,
            [c["id"]],
            [""],
            sender="stranger",
            scope="chat",
            bot_id="bot",
            life_context={},
        )
    with pytest.raises(ValueError, match="重复"):
        chars.resolve_characters(
            studio.store,
            ["me", c["id"]],
            ["", ""],
            sender="user",
            scope="chat",
            bot_id="bot",
            life_context={},
        )


def test_switch_is_scoped_and_pending_inputs_are_immutable(runtime):
    _, _, chars, _, studio = runtime
    c = character(studio.store, "Bot", "red", "bot")
    other = studio.store.add_image(picture("green"))
    c["looks"].append({"name": "other", "assets": [other["id"]]})
    studio.store.save_document("character", c)
    args = dict(sender="user", bot_id="bot", life_context={})
    first, _, _ = chars.resolve_characters(
        studio.store, ["self"], [""], scope="one", **args
    )
    studio.store.save_document("appearance", {"id": "one:" + c["id"], "look": "other"})
    after, _, _ = chars.resolve_characters(
        studio.store, ["self"], [""], scope="one", **args
    )
    separate, _, _ = chars.resolve_characters(
        studio.store, ["self"], [""], scope="two", **args
    )
    assert first.images == separate.images
    assert first.images != after.images


def test_same_photo_for_two_people_is_rejected(runtime):
    _, _, chars, _, studio = runtime
    character(studio.store, "Bot", "red", "bot")
    character(studio.store, "User", "red")
    with pytest.raises(ValueError, match="相同身份图"):
        chars.resolve_characters(
            studio.store,
            ["self", "me"],
            ["", ""],
            sender="user",
            scope="chat",
            bot_id="bot",
            life_context={},
        )


def test_masked_keys_never_follow_changed_endpoint(runtime):
    _, module, _, _, studio = runtime
    p = studio.config_view()["providers"][0]
    assert p["api_keys"] == module.MASK
    assert studio.provider_input(p)["api_keys"] == ["secret-value"]
    p["base_url"] = "https://other.example/v1"
    with pytest.raises(ValueError, match="重新填写"):
        studio.provider_input(p)


@pytest.mark.asyncio
async def test_live_model_listing_uses_selected_credentials(runtime, monkeypatch):
    _, module, _, _, studio = runtime
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(
            200, json={"data": [{"id": "model-a"}, {"id": "model-b"}]}
        )

    original = httpx.AsyncClient
    monkeypatch.setattr(
        module.httpx,
        "AsyncClient",
        lambda **kw: original(transport=httpx.MockTransport(handler), **kw),
    )
    r = await studio.models({"provider": studio.config_view()["providers"][0]})
    assert r == {"models": ["model-a", "model-b"], "source": "live"}
    assert str(calls[0].url) == "https://example.com/v1/models"
    assert calls[0].headers["authorization"] == "Bearer secret-value"


@pytest.mark.asyncio
async def test_config_preserves_keys_and_rejects_stale_writes(runtime):
    _, module, _, plugin, studio = runtime
    view = studio.config_view()
    view["providers"][0]["label"] = "Renamed"
    view["chains"] = {"draw": [{"provider_id": "test", "output": "4K"}]}
    result = await studio.save_config(view)
    assert plugin.config["providers"][0]["api_keys"] == ["secret-value"]
    assert plugin.config["features"]["draw"]["chain"][0]["output"] == "4K"
    assert result["revision"] != view["revision"]
    with pytest.raises(module.StudioConflict):
        await studio.save_config(view)


@pytest.mark.asyncio
async def test_tool_character_preparation_keeps_reference_order(runtime):
    main, _, _, plugin, studio = runtime
    character(studio.store, "Bot", "red", "bot")
    character(studio.store, "User", "blue")
    plugin._get_life_context_without_llm = AsyncMock(
        return_value={"outfit": "white dress"}
    )
    prompt, selection = await plugin._prepare_character_portrait(
        _Event(), "together", ["self", "me"], ["", "suit"], None
    )
    assert isinstance(selection, main.ReferenceSelection)
    assert len(selection.images) == 2
    assert "white dress" in prompt and "suit" in prompt


@pytest.mark.asyncio
async def test_studio_job_is_deduplicated_and_survives_reopen(runtime):
    _, module, _, plugin, studio = runtime
    manager = module.BackgroundImageTaskManager(
        studio.store.root / "queue", max_running=1, max_queued=2
    )
    await manager.start()
    studio.own_manager = manager
    calls = []

    async def generate(prompt, **kw):
        calls.append(prompt)
        cap = importlib.import_module(module.__package__ + ".studio_capture")
        a = studio.store.add_image(
            picture("green"), kind="history", metadata={"prompt": prompt}
        )
        cap.capture_context.get()["asset_id"] = a["id"]
        return studio.store.asset_path(a["id"])

    plugin.draw = SimpleNamespace(generate=generate)
    plugin.edit = SimpleNamespace(edit=AsyncMock())
    body = {"request_id": "test-request-01", "prompt": "a green image"}
    try:
        result = await studio.submit(body)
        assert await studio.submit(body) == result
        await asyncio.gather(*studio.tasks)
        task = await manager.get_task(result["task_id"])
        assert task["state"] == "completed"
        assert task["gallery_asset_id"]
        assert calls == ["a green image"]
        reopened = module.Studio(plugin)
        reopened.own_manager = manager
        assert await reopened.submit(body) == result
        assert calls == ["a green image"]
    finally:
        await studio.close()


@pytest.mark.asyncio
async def test_studio_batch_partial_retry_preserves_inputs_and_completed_images(
    runtime,
):
    _, module, _, plugin, studio = runtime
    manager = module.BackgroundImageTaskManager(
        studio.store.root / "batch", max_running=2, max_queued=4
    )
    await manager.start()
    studio.own_manager = manager
    ref = studio.store.add_image(picture("blue"))
    calls, running, peak = [], 0, 0
    release = asyncio.Event()
    cap = importlib.import_module(module.__package__ + ".studio_capture")

    async def edit(prompt, images, **kwargs):
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        calls.append((prompt, images))
        attempt = len(calls)
        if attempt == 2:
            release.set()
        await release.wait()
        await asyncio.sleep(0)
        running -= 1
        if attempt == 2:
            raise ValueError("fixture failure")
        a = studio.store.add_image(
            picture("green"), kind="history", metadata={"prompt": prompt}
        )
        cap.capture_context.get()["asset_id"] = a["id"]
        return studio.store.asset_path(a["id"])

    plugin.edit = SimpleNamespace(edit=edit)
    body = {
        "request_id": "batch-request-01",
        "prompt": "keep cup",
        "assets": [ref["id"]],
        "count": 3,
    }
    try:
        result = await studio.submit(body)
        assert await studio.submit(body) == result
        await asyncio.gather(*studio.tasks)
        row = await manager.get_task(result["task_id"])
        assert peak == 2
        assert row["state"] == "partial"
        assert len(row["gallery_asset_ids"]) == 2
        originals = list(row["gallery_asset_ids"])
        assert len(calls) == 3
        retry_body = {"request_id": "retry-request-01", "task_id": result["task_id"]}
        retry = await studio.retry(retry_body)
        assert await studio.retry(retry_body) == retry
        await asyncio.gather(*studio.tasks)
        assert len(calls) == 4
        assert calls[-1] == calls[1]
        assert (await manager.get_task(retry["task_id"]))["state"] == "completed"
        assert (await manager.get_task(result["task_id"]))[
            "gallery_asset_ids"
        ] == originals
        assert (await manager.health_snapshot())["reservation_remaining"] == 0
    finally:
        await studio.close()


@pytest.mark.asyncio
async def test_visual_plan_review_excludes_target_face_and_freezes_wardrobe(runtime):
    import json

    _, module, _, plugin, studio = runtime
    manager = module.BackgroundImageTaskManager(
        studio.store.root / "planned", max_running=1, max_queued=4
    )
    await manager.start()
    studio.own_manager = manager
    person = character(studio.store, "User", "blue")
    target = studio.store.add_image(picture("red"))
    planner = SimpleNamespace(
        meta=lambda: SimpleNamespace(id="vision"),
        get_model=lambda: "vision-model",
        text_chat=AsyncMock(
            return_value=SimpleNamespace(
                completion_text=json.dumps(
                    [
                        {
                            "title": "portrait",
                            "prompt": "black suit, studio, frontal",
                            "variation_focus": ["pose"],
                            "aspect_ratio": "3:4",
                        },
                        {
                            "title": "side",
                            "prompt": "black suit, studio, side view",
                            "variation_focus": ["angle"],
                            "aspect_ratio": "3:4",
                        },
                    ]
                )
            )
        ),
    )
    plugin.context.get_all_providers = lambda: [planner]
    plugin.context.get_provider_by_id = lambda key: planner if key == "vision" else None
    plugin._get_life_context_without_llm = AsyncMock(
        return_value={"outfit": "white dress"}
    )
    captured = []
    cap = importlib.import_module(module.__package__ + ".studio_capture")

    async def edit(prompt, images, **kwargs):
        captured.append((prompt, images))
        a = studio.store.add_image(picture("green"), kind="history")
        cap.capture_context.get()["asset_id"] = a["id"]
        return studio.store.asset_path(a["id"])

    plugin.edit = SimpleNamespace(edit=edit)
    body = {
        "request_id": "plan-request-01",
        "workflow": "recreate",
        "source_asset": target["id"],
        "characters": [person["id"]],
        "target_character": "",
        "planner": "vision",
        "count": 2,
        "prompt": "",
        "output": "3:4 4K",
    }
    try:
        initial = await studio.workflows.start(body)
        assert (await studio.workflows.start(body))["id"] == initial["id"]
        await asyncio.gather(*studio.workflows.tasks.values())
        plan = studio.store.document("plan", initial["id"])
        assert plan["state"] == "ready"
        assert planner.text_chat.await_count == 1
        assert not captured
        assert planner.text_chat.call_args.kwargs["image_urls"] == [
            str(studio.store.asset_path(target["id"], thumbnail="preview"))
        ]
        request = {
            **body,
            "request_id": "planned-shoot-01",
            "plan_id": plan["id"],
            "shots": plan["shots"],
            "outfits": [""],
        }
        with pytest.raises(ValueError, match="变化"):
            await studio.submit(
                {**request, "source_asset": person["looks"][0]["assets"][0]}
            )
        await studio.submit(request)
        await asyncio.gather(*studio.tasks)
        assert len(captured) == 2
        assert all(images == [picture("blue")] for _, images in captured)
        assert all("目标人物: " + person["id"] in prompt for prompt, _ in captured)
        assert all("white dress" not in prompt for prompt, _ in captured)
    finally:
        await studio.close()


@pytest.mark.asyncio
async def test_planning_failure_does_not_generate_and_is_recoverable(runtime):
    _, module, _, plugin, studio = runtime
    manager = module.BackgroundImageTaskManager(studio.store.root / "plan-failure")
    await manager.start()
    studio.own_manager = manager
    planner = SimpleNamespace(
        meta=lambda: SimpleNamespace(id="vision"),
        get_model=lambda: "vision",
        text_chat=AsyncMock(return_value=SimpleNamespace(completion_text="bad json")),
    )
    plugin.context.get_all_providers = lambda: [planner]
    plugin.context.get_provider_by_id = lambda key: planner
    source = studio.store.add_image(picture("red"))
    try:
        body = {
            "request_id": "failed-plan-01",
            "workflow": "variants",
            "source_asset": source["id"],
            "planner": "vision",
            "count": 4,
        }
        plan = await studio.workflows.start(body)
        await asyncio.gather(*studio.workflows.tasks.values())
        assert studio.store.document("plan", plan["id"])["state"] == "failed"
        assert studio.store.documents("job") == []
        studio.store.save_document(
            "plan", {"id": "interrupted-plan", "state": "planning"}
        )
        studio.workflows.recover()
        assert (
            studio.store.document("plan", "interrupted-plan")["state"] == "interrupted"
        )
    finally:
        await studio.close()


@pytest.mark.asyncio
async def test_batch_capacity_is_reserved_atomically(runtime):
    _, module, _, plugin, studio = runtime
    manager = module.BackgroundImageTaskManager(
        studio.store.root / "capacity", max_queued=2
    )
    await manager.start()
    studio.own_manager = manager
    try:
        with pytest.raises(RuntimeError, match="queue"):
            await studio.submit(
                {"request_id": "capacity-test-01", "prompt": "cup", "count": 3}
            )
        assert studio.store.documents("job") == []
        assert (await manager.health_snapshot())["reservation_remaining"] == 0
        plugin.draw.generate.assert_not_awaited()
    finally:
        await studio.close()


@pytest.mark.asyncio
async def test_studio_batch_cancel_and_restart_keep_completed_items(runtime):
    _, module, _, _, studio = runtime
    folder = studio.store.root / "batch-recovery"
    first = module.BackgroundImageTaskManager(folder)
    await first.start()
    for task_id in ("cancel-group", "restart-group"):
        await first.create_task_record(
            {
                "task_id": task_id,
                "task_kind": "studio",
                "scope_hash": "studio",
                "request_fingerprint": task_id,
                "items": [
                    {
                        "item_id": "one",
                        "state": "completed",
                        "asset_id": "saved-image",
                        "image_generated": True,
                    },
                    {"item_id": "two", "state": "running"},
                ],
            },
            reservation=1,
        )
    await first.cancel_task("cancel-group", "cancel")
    await first.close()
    second = module.BackgroundImageTaskManager(folder)
    try:
        await second.start()
        cancelled = await second.get_task("cancel-group")
        restarted = await second.get_task("restart-group")
        assert cancelled["items"][0]["asset_id"] == "saved-image"
        assert cancelled["items"][1]["state"] == "cancelled"
        assert restarted["items"][0]["asset_id"] == "saved-image"
        assert restarted["items"][1]["state"] == "interrupted"
        assert (await second.health_snapshot())["reservation_remaining"] == 0
    finally:
        await second.close()


@pytest.mark.asyncio
async def test_queued_work_does_not_inherit_previous_request_context(runtime):
    from contextvars import ContextVar

    _, module, _, _, studio = runtime
    manager = module.BackgroundImageTaskManager(
        studio.store.root / "context-queue", max_running=1
    )
    await manager.start()
    tag = ContextVar("test-request-tag")
    started, release = asyncio.Event(), asyncio.Event()

    async def first():
        started.set()
        await release.wait()
        return tag.get()

    async def second():
        return tag.get()

    try:
        tag.set("first-person")
        a = asyncio.create_task(manager.run_provider("a", first))
        await started.wait()
        tag.set("second-person")
        b = asyncio.create_task(manager.run_provider("b", second))
        await asyncio.sleep(0)
        release.set()
        assert await asyncio.gather(a, b) == ["first-person", "second-person"]
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_studio_cancel_and_restart_never_enqueue_chat_notification(runtime):
    import sqlite3

    _, module, _, _, studio = runtime
    folder = studio.store.root / "recovery"
    first = module.BackgroundImageTaskManager(folder)
    await first.start()
    for task_id in ("studio-cancel", "studio-restart"):
        await first.create_task_record(
            {
                "task_id": task_id,
                "task_kind": "studio",
                "scope_hash": "studio",
                "request_fingerprint": task_id,
            },
            reservation=1,
        )
    await first.cancel_task("studio-cancel", "cancel")
    await first.close()
    second = module.BackgroundImageTaskManager(folder)
    try:
        recovered = await second.start()
        assert recovered == []
        assert (await second.get_task("studio-restart"))["state"] == "interrupted"
        with sqlite3.connect(second.db_path) as db:
            assert (
                db.execute("select count(*) from notification_outbox").fetchone()[0]
                == 0
            )
            assert (
                db.execute("select sum(remaining) from reservations").fetchone()[0] == 0
            )
    finally:
        await second.close()


@pytest.mark.asyncio
async def test_api_bridge_envelope_and_crop_geometry(runtime, monkeypatch):
    import sys
    import types

    _, _, _, _, studio = runtime
    request = SimpleNamespace(method="GET", query={})
    web = types.ModuleType("astrbot.api.web")
    web.request = request
    web.json_response = lambda value, status_code=200: (value, status_code)
    web.file_response = lambda *a, **kw: None
    monkeypatch.setitem(sys.modules, "astrbot.api.web", web)
    result, status = await studio.handle("state")
    assert status == 200
    assert result["status"] == "ok"
    assert result["data"]["ok"] is True
    a = studio.store.add_image(picture("red"))
    request.method = "POST"
    request.json = AsyncMock(return_value={"id": a["id"], "rect": [0, 0, 0.5, 0.5]})
    result, status = await studio.handle("crop")
    assert status == 200
    cropped = result["data"]["data"]
    assert (cropped["width"], cropped["height"]) == (15, 20)
    assert studio.store.asset_path(a["id"]).is_file()
    request.json = AsyncMock(return_value={"id": a["id"], "rect": [0, 0, 2, 2]})
    assert (await studio.handle("crop"))[1] == 400


@pytest.mark.asyncio
async def test_group_portrait_without_identity_ids_stops_before_debounce(runtime):
    _, _, _, plugin, studio = runtime
    character(studio.store, "Bot", "red", "bot")
    character(studio.store, "User", "blue")
    plugin.debouncer = None
    event = _Event()
    event.message_str = "我们合照一张"
    result = await plugin.aiimg_generate(event, prompt="我们合照", mode="selfie_ref")
    assert "character_ids" in result.content[0].text
    assert "尚未开始" in result.content[0].text


def test_clothing_reference_requires_explicit_person_and_overrides_schedule(runtime):
    _, _, chars, _, studio = runtime
    character(studio.store, "Bot", "red", "bot")
    character(studio.store, "User", "blue")
    _, _, cast = chars.resolve_characters(
        studio.store,
        ["self", "me"],
        ["", ""],
        sender="user",
        scope="chat",
        bot_id="bot",
        life_context={"outfit": "daily clothes"},
    )
    with pytest.raises(ValueError, match="指定"):
        chars.targeted_reference_note(
            [{"role": "clothing"}], [""], cast, ["self", "me"], 2
        )
    note = chars.targeted_reference_note(
        [{"role": "clothing"}], ["me"], cast, ["self", "me"], 2
    )
    assert "人物 2" in note
    assert "参考图 3" in note
    assert "优先于" in note
    assert "不影响其他人物" in note
