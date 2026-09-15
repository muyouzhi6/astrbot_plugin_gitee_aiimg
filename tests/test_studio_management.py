import asyncio
import importlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from core.studio_store import StudioStore
from core.studio_graph import validate_graph
from test_studio import runtime as studio_runtime, picture, character

runtime = studio_runtime


def graph_spec():
    return {
        "name": "test",
        "nodes": [
            {"id": "text", "type": "text", "text": "green cup", "x": 0, "y": 0},
            {"id": "generate", "type": "generate", "count": 2, "x": 250, "y": 0},
            {"id": "output", "type": "output", "x": 500, "y": 0},
        ],
        "edges": [
            {"from": "text", "to": "generate", "port": "text"},
            {"from": "generate", "to": "output", "port": "image"},
        ],
    }


def age(store, asset, timestamp):
    a = store.asset(asset["id"])
    a["created"] = timestamp
    with store.connect() as db:
        db.execute(
            "UPDATE assets SET created=?,data=? WHERE id=?",
            (timestamp, json.dumps(a), a["id"]),
        )


def test_delete_shared_file_favorite_and_reference_protection(tmp_path):
    store = StudioStore(tmp_path)
    first = store.add_image(picture("red"))
    same = store.add_image(picture("red"))
    path = store.asset_path(first["id"])
    store.favorite([same["id"]], True)
    result = store.delete_assets([first["id"], same["id"]])
    assert result["deleted"] == [first["id"]]
    assert result["skipped"][0]["id"] == same["id"]
    assert path.exists()
    assert store.list_assets(kind="favorite")["items"][0]["id"] == same["id"]
    store.favorite([same["id"]], False)
    store.delete_assets([same["id"]])
    assert not path.exists()
    assert not store.available(same["id"])
    person = character(store, "Person", "blue")
    protected = person["looks"][0]["assets"][0]
    assert store.delete_assets([protected])["deleted"] == []
    assert store.asset_path(protected).exists()


def test_delete_updates_canvas_revision_and_stale_write_cannot_restore(tmp_path):
    store = StudioStore(tmp_path)
    a = store.add_image(picture("red"))
    workspace = store.save_document(
        "workspace",
        {
            "name": "one",
            "layers": [{"asset_id": a["id"]}],
            "shoot": {"picked": [a["id"]], "source": a["id"], "plan_id": "plan"},
        },
    )
    store.delete_assets([a["id"]])
    current = store.document("workspace", workspace["id"])
    assert current["layers"] == [] and current["shoot"]["picked"] == []
    assert current["shoot"]["source"] == ""
    assert current["revision"] == workspace["revision"] + 1
    with pytest.raises(ValueError):
        store.save_document("workspace", workspace)


def test_limit_cleans_oldest_skips_favorites_jobs_and_graph_inputs(tmp_path):
    store = StudioStore(tmp_path)
    assets = [
        store.add_image(picture(color))
        for color in ("red", "blue", "green", "yellow", "white")
    ]
    for i, a in enumerate(assets):
        age(store, a, 100 + i)
    store.favorite([assets[0]["id"]], True)
    store.save_document(
        "graph", {"nodes": [{"type": "image", "asset_id": assets[1]["id"]}]}
    )
    store.save_document("settings", {"id": "library", "max_count": 2})
    result = store.enforce_limit([assets[2]["id"]])
    assert result["deleted"] == [assets[3]["id"], assets[4]["id"]]
    assert result["remaining_over_limit"] == 1
    assert store.library_status()["count"] == 3
    assert store.library_status()["favorites"] == 1


def test_new_assets_not_immediately_deleted_by_limit(tmp_path):
    store = StudioStore(tmp_path)
    store.save_document("settings", {"id": "library", "max_count": 1})
    a = store.add_image(picture("red"))
    b = store.add_image(picture("blue"))
    assert store.enforce_limit()["deleted"] == []
    assert store.available(a["id"]) and store.available(b["id"])


def test_graph_rejects_cycles_wrong_ports_and_orphan_nodes():
    graph = graph_spec()
    assert validate_graph(graph, executable=True)[1] == ["text", "generate", "output"]
    graph["edges"][0]["port"] = "image"
    with pytest.raises(ValueError, match="类型"):
        validate_graph(graph)
    graph = graph_spec()
    graph["edges"].append({"from": "generate", "to": "generate", "port": "image"})
    with pytest.raises(ValueError):
        validate_graph(graph)
    graph = graph_spec()
    graph["nodes"].append({"id": "unused", "type": "image"})
    with pytest.raises(ValueError, match="未连接"):
        validate_graph(graph, executable=True)
    graph = graph_spec()
    graph["nodes"].append({"id": "g2", "type": "generate"})
    graph["edges"] += [
        {"from": "generate", "to": "g2", "port": "image"},
        {"from": "g2", "to": "generate", "port": "image"},
    ]
    with pytest.raises(ValueError, match="循环"):
        validate_graph(graph)


@pytest.mark.asyncio
async def test_graph_executes_selected_provider_and_deduplicates(runtime):
    _, module, _, plugin, studio = runtime
    manager = module.BackgroundImageTaskManager(
        studio.store.root / "graph", max_running=2, max_queued=4
    )
    await manager.start()
    studio.own_manager = manager
    cap = importlib.import_module(module.__package__ + ".studio_capture")
    calls = []

    async def generate(prompt, **kwargs):
        calls.append((prompt, kwargs))
        a = studio.store.add_image(picture("green"), kind="history")
        cap.capture_context.get()["asset_id"] = a["id"]
        return studio.store.asset_path(a["id"])

    plugin.draw.generate = generate
    graph = graph_spec()
    graph["nodes"][1]["provider"] = "test"
    saved = studio.graphs.save(graph)
    try:
        body = {
            "graph_id": saved["id"],
            "revision": saved["revision"],
            "request_id": "graph-run-test-01",
        }
        run = await studio.graphs.start(body)
        assert (await studio.graphs.start(body))["id"] == run["id"]
        await asyncio.gather(*studio.graphs.tasks.values())
        result = studio.store.document("graph_run", run["id"])
        assert result["state"] == "completed" and len(result["assets"]) == 2
        assert len(calls) == 2 and all(k["provider_id"] == "test" for _, k in calls)
        assert all(n["state"] == "completed" for n in result["nodes"].values())
        assert (await manager.health_snapshot())["reservation_remaining"] == 0
    finally:
        await studio.close()


@pytest.mark.asyncio
async def test_graph_partial_failure_stops_downstream_and_preserves_outputs(runtime):
    _, module, _, plugin, studio = runtime
    manager = module.BackgroundImageTaskManager(
        studio.store.root / "graph-fail", max_running=1, max_queued=4
    )
    await manager.start()
    studio.own_manager = manager
    cap = importlib.import_module(module.__package__ + ".studio_capture")
    calls = []

    async def generate(prompt, **kwargs):
        calls.append(prompt)
        if len(calls) == 2:
            raise ValueError("test failure")
        a = studio.store.add_image(picture("green"), kind="history")
        cap.capture_context.get()["asset_id"] = a["id"]
        return studio.store.asset_path(a["id"])

    plugin.draw.generate = generate
    saved = studio.graphs.save(graph_spec())
    try:
        run = await studio.graphs.start(
            {
                "graph_id": saved["id"],
                "revision": saved["revision"],
                "request_id": "graph-failure-01",
            }
        )
        await asyncio.gather(*studio.graphs.tasks.values())
        result = studio.store.document("graph_run", run["id"])
        assert result["state"] == "failed"
        assert len(result["nodes"]["generate"]["assets"]) == 1
        assert result["nodes"]["output"]["state"] == "skipped"
    finally:
        await studio.close()


@pytest.mark.asyncio
async def test_graph_cancel_releases_child_and_preserves_saved_definition(runtime):
    _, module, _, plugin, studio = runtime
    manager = module.BackgroundImageTaskManager(
        studio.store.root / "graph-cancel", max_running=1, max_queued=4
    )
    await manager.start()
    studio.own_manager = manager
    started = asyncio.Event()

    async def generate(*args, **kwargs):
        started.set()
        await asyncio.Event().wait()

    plugin.draw.generate = generate
    saved = studio.graphs.save(graph_spec())
    try:
        run = await studio.graphs.start(
            {
                "graph_id": saved["id"],
                "revision": saved["revision"],
                "request_id": "graph-cancel-01",
            }
        )
        await asyncio.wait_for(started.wait(), 2)
        result = await studio.graphs.cancel(run["id"])
        assert result["state"] == "cancelled"
        assert (await manager.health_snapshot())["reservation_remaining"] == 0
        assert studio.store.document("graph", saved["id"])["nodes"] == saved["nodes"]
    finally:
        await studio.close()


@pytest.mark.asyncio
async def test_graph_visual_plan_freezes_identity_before_async_planning(runtime):
    _, module, _, plugin, studio = runtime
    manager = module.BackgroundImageTaskManager(
        studio.store.root / "graph-plan", max_running=1, max_queued=4
    )
    await manager.start()
    studio.own_manager = manager
    person = character(studio.store, "User", "blue")
    reference = studio.store.add_image(picture("red"))
    replacement = studio.store.add_image(picture("yellow"))
    planner = SimpleNamespace(
        meta=lambda: SimpleNamespace(id="vision"),
        get_model=lambda: "vision",
        text_chat=AsyncMock(
            return_value=SimpleNamespace(
                completion_text=json.dumps(
                    [
                        {
                            "title": "shot",
                            "prompt": "green jacket portrait",
                            "variation_focus": ["pose"],
                            "aspect_ratio": "3:4",
                        }
                    ]
                )
            )
        ),
    )
    plugin.context.get_all_providers = lambda: [planner]
    plugin.context.get_provider_by_id = lambda key: planner
    plugin._get_life_context_without_llm = AsyncMock(
        return_value={"outfit": "white dress"}
    )
    cap = importlib.import_module(module.__package__ + ".studio_capture")
    received = []

    async def edit(prompt, images, **kwargs):
        received.extend(images)
        a = studio.store.add_image(picture("green"), kind="history")
        cap.capture_context.get()["asset_id"] = a["id"]
        return studio.store.asset_path(a["id"])

    plugin.edit.edit = edit
    graph = {
        "name": "portrait",
        "nodes": [
            {"id": "ref", "type": "image", "asset_id": reference["id"]},
            {"id": "person", "type": "person", "characters": [person["id"]]},
            {
                "id": "plan",
                "type": "plan",
                "workflow": "recreate",
                "planner": "vision",
                "count": 1,
            },
            {"id": "gen", "type": "generate"},
            {"id": "out", "type": "output"},
        ],
        "edges": [
            {"from": "ref", "to": "plan", "port": "image"},
            {"from": "person", "to": "plan", "port": "people"},
            {"from": "plan", "to": "gen", "port": "plan"},
            {"from": "gen", "to": "out", "port": "image"},
        ],
    }
    saved = studio.graphs.save(graph)
    try:
        run = await studio.graphs.start(
            {
                "graph_id": saved["id"],
                "revision": saved["revision"],
                "request_id": "graph-portrait-01",
            }
        )
        person["looks"][0]["assets"] = [replacement["id"]]
        studio.store.save_document("character", person)
        await asyncio.gather(*studio.graphs.tasks.values())
        assert studio.store.document("graph_run", run["id"])["state"] == "completed"
        assert received == [picture("blue")]
        assert planner.text_chat.await_count == 1
    finally:
        await studio.close()
