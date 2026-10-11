"""Reference gallery regressions exercised through a real HTTP listener."""

import base64
import hashlib
import sys
from contextvars import ContextVar
from types import ModuleType

import aiohttp
from aiohttp import web
import pytest
import pytest_asyncio

from core.ref_store import ReferenceStore
from test_studio import picture, runtime as studio_runtime

runtime = studio_runtime


@pytest_asyncio.fixture
async def http_studio(runtime, monkeypatch):
    _, _, _, plugin, studio = runtime
    plugin.refs = ReferenceStore(plugin.data_dir)
    plugin.config["features"].setdefault("selfie", {})
    current = ContextVar("studio_http_request")

    class Request:
        @property
        def method(self):
            return current.get().method

        @property
        def query(self):
            return current.get().query

        async def json(self):
            return await current.get().json()

        async def files(self):
            form = await current.get().post()

            class Upload:
                def __init__(self, part):
                    self.filename = part.filename
                    self.file = part.file

                async def read(self, size):
                    return self.file.read(size)

            return {
                key: Upload(part)
                for key, part in form.items()
                if isinstance(part, web.FileField)
            }

    bridge = ModuleType("astrbot.api.web")
    bridge.request = Request()
    bridge.json_response = lambda value, status_code=200, **kw: web.json_response(
        value, status=status_code, **kw
    )
    bridge.file_response = lambda path, **kw: web.FileResponse(path)
    monkeypatch.setitem(sys.modules, "astrbot.api.web", bridge)
    app = web.Application(client_max_size=22 * 1024 * 1024)

    def register(path, handler, methods, description):
        async def endpoint(request):
            token = current.set(request)
            try:
                return await handler(**request.match_info)
            finally:
                current.reset(token)

        app.router.add_route("*", path.replace("<asset_id>", "{asset_id}"), endpoint)

    plugin.context.register_web_api = register
    await studio.start()
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    try:
        async with aiohttp.ClientSession() as client:
            yield (
                plugin,
                studio,
                client,
                f"http://127.0.0.1:{port}/astrbot_plugin_gitee_aiimg/studio/",
            )
    finally:
        await runner.cleanup()
        await studio.close()


async def listing(client, base):
    async with client.get(base + "library", params={"kind": "reference"}) as response:
        assert response.status == 200
        return (await response.json())["data"]["data"]


@pytest.mark.asyncio
async def test_config_change_replaces_legacy_snapshot(http_studio):
    plugin, studio, client, base = http_studio
    old = plugin.data_dir / "old.png"
    new = plugin.data_dir / "new.png"
    old.write_bytes(picture("red"))
    new.write_bytes(picture("blue"))
    # Reproduce the sqlite state created by v5.7.3's one-time import.
    studio.store.add_image(
        old.read_bytes(), kind="reference", metadata={"source": "legacy", "name": "old"}
    )
    plugin.config["features"]["selfie"]["reference_images"] = ["old.png"]
    first = (await listing(client, base))["items"]
    assert len(first) == 1
    plugin.config["features"]["selfie"]["reference_images"] = ["new.png"]
    second = (await listing(client, base))["items"]
    assert len(second) == 1
    assert second[0]["sha256"] == hashlib.sha256(new.read_bytes()).hexdigest()
    assert second[0]["version"] == second[0]["sha256"]
    assert second[0]["id"] != first[0]["id"]
    plugin.config["features"]["selfie"]["reference_images"] = []
    assert (await listing(client, base))["items"] == []


async def upload(client, base, blob, asset_id="", filename="photo.png"):
    form = aiohttp.FormData()
    form.add_field("file", blob, filename=filename, content_type="image/png")
    # Invalid IDs still exercise the legacy query route's validation.
    endpoint = "reference-upload" + (
        "/" + asset_id if asset_id and "/" not in asset_id else ""
    )
    params = {"id": asset_id} if "/" in asset_id else {}
    async with client.post(base + endpoint, params=params, data=form) as response:
        return response.status, await response.json()


async def remove(client, base, ids, action="reference-delete"):
    async with client.post(base + action, json={"ids": ids}) as response:
        return response.status, await response.json()


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["reference-delete", "delete-assets"])
@pytest.mark.parametrize("source", ["config", "store"])
async def test_batch_delete_skips_favorite_references(http_studio, action, source):
    plugin, studio, client, base = http_studio
    if source == "config":
        for color in ["red", "blue"]:
            assert (await upload(client, base, picture(color)))[0] == 200
        paths = plugin._get_config_selfie_reference_paths()
    else:
        await plugin.refs.set("bot_selfie", [picture("red"), picture("blue")])
        paths = await plugin.refs.get_paths("bot_selfie")
    items = (await listing(client, base))["items"]
    favorite = next(
        item
        for item in items
        if item["sha256"] == hashlib.sha256(picture("red")).hexdigest()
    )
    removable = next(item for item in items if item["id"] != favorite["id"])
    studio.store.favorite([favorite["id"]], True)
    ids = [favorite["id"], removable["id"]]
    deleted = [removable["id"]]
    skipped = [{"id": favorite["id"], "reason": "已收藏, 请先取消收藏"}]
    ordinary = studio.store.add_image(picture("green"))
    ordinary_path = studio.store.asset_path(ordinary["id"])
    if action == "delete-assets":
        protected = studio.store.add_image(picture("yellow"))
        studio.store.favorite([protected["id"]], True)
        ids.extend([ordinary["id"], protected["id"]])
        deleted.append(ordinary["id"])
        skipped.append({"id": protected["id"], "reason": "已收藏, 请先取消收藏"})
    status, response = await remove(client, base, ids, action)
    assert status == 200
    assert response["data"]["data"] == {"deleted": deleted, "skipped": skipped}
    assert [item["id"] for item in (await listing(client, base))["items"]] == [
        favorite["id"]
    ]
    remaining_paths = (
        plugin._get_config_selfie_reference_paths()
        if source == "config"
        else await plugin.refs.get_paths("bot_selfie")
    )
    assert [path.read_bytes() for path in remaining_paths] == [picture("red")]
    assert [path.read_bytes() for path in paths] == [picture("red"), picture("blue")]
    assert studio.store.asset_path(removable["id"]).read_bytes() == picture("blue")
    assert ordinary_path.exists() == (action == "reference-delete")
    if action == "delete-assets":
        assert studio.store.asset_path(protected["id"]).read_bytes() == picture(
            "yellow"
        )

    # An all-favorite batch succeeds without persisting new bindings or backups.
    before_config = list(plugin._get_selfie_conf().get("reference_images", []))
    before_index = await plugin.refs._read_index()
    before_backups = list((studio.store.root / "config-backups").glob("*"))
    status, response = await remove(client, base, [favorite["id"]], action)
    assert status == 200
    assert response["data"]["data"] == {"deleted": [], "skipped": skipped[:1]}
    assert (await upload(client, base, picture("black"), favorite["id"]))[0] == 400
    assert plugin._get_selfie_conf().get("reference_images", []) == before_config
    assert await plugin.refs._read_index() == before_index
    assert list((studio.store.root / "config-backups").glob("*")) == before_backups
    assert [item["id"] for item in (await listing(client, base))["items"]] == [
        favorite["id"]
    ]


@pytest.mark.asyncio
async def test_http_upload_replace_delete_and_external_config_reload(
    http_studio, monkeypatch
):
    plugin, studio, client, base = http_studio
    persisted = []
    monkeypatch.setattr(
        plugin.config,
        "save_config",
        lambda: persisted.append(list(plugin._get_selfie_conf()["reference_images"])),
    )
    assert (await listing(client, base))["items"] == []
    status, result = await upload(client, base, picture("red"))
    assert status == 200
    first = result["data"]["data"]
    first_paths = plugin._get_config_selfie_reference_paths()
    assert len(first_paths) == 1 and first_paths[0].read_bytes() == picture("red")
    assert len(persisted) == 1
    async with client.get(
        base + "asset", params={"id": first["id"], "thumbnail": "0"}
    ) as response:
        assert response.status == 200
        assert response.headers["Cache-Control"] == "no-store"
        first_url = (await response.json())["data"]["data"]["data_url"]
        assert base64.b64decode(first_url.split(",", 1)[1]) == picture("red")
    status, result = await upload(client, base, picture("blue"), first["id"])
    assert status == 200
    second = result["data"]["data"]
    assert second["id"] != first["id"] and second["sha256"] != first["sha256"]
    assert len(plugin._get_config_selfie_reference_paths()) == 1
    assert plugin._get_config_selfie_reference_paths()[0].read_bytes() == picture(
        "blue"
    )
    assert len(persisted) == 2
    assert [a["id"] for a in (await listing(client, base))["items"]] == [second["id"]]
    async with client.get(
        base + "asset", params={"id": second["id"], "thumbnail": "0"}
    ) as response:
        assert (await response.json())["data"]["data"]["data_url"] != first_url
    assert (await remove(client, base, [second["id"]]))[0] == 200
    assert plugin._get_config_selfie_reference_paths() == [] and persisted[-1] == []
    assert (await listing(client, base))["items"] == []
    # Archived originals and generated images remain usable after detaching references.
    assert studio.store.asset_path(first["id"]).read_bytes() == picture("red")
    assert studio.store.asset_path(second["id"]).read_bytes() == picture("blue")
    assert first_paths[0].is_file()
    configured = plugin.data_dir / "configured.png"
    configured.write_bytes(picture("green"))
    plugin.config["features"]["selfie"]["reference_images"] = [configured.name]
    current = (await listing(client, base))["items"][0]
    assert current["sha256"] == hashlib.sha256(picture("green")).hexdigest()
    configured.write_bytes(picture("yellow"))
    changed = (await listing(client, base))["items"][0]
    assert changed["id"] != current["id"]
    assert changed["sha256"] == hashlib.sha256(picture("yellow")).hexdigest()
    async with client.get(base + "library", params={"kind": "reference"}) as response:
        assert response.headers["Cache-Control"] == "no-store"


@pytest.mark.asyncio
async def test_scoped_command_references_and_generation_use_same_files(http_studio):
    from types import SimpleNamespace

    plugin, studio, client, base = http_studio
    await plugin.refs.set("bot_selfie_bot", [picture("red"), picture("blue")])
    await plugin.refs.set("bot_selfie_other", [picture("green")])
    items = (await listing(client, base))["items"]
    assert len(items) == 3
    red = next(
        a for a in items if a["sha256"] == hashlib.sha256(picture("red")).hexdigest()
    )
    status, response = await upload(client, base, picture("yellow"), red["id"])
    assert status == 200
    updated = response["data"]["data"]
    event = SimpleNamespace(get_self_id=lambda: "bot", unified_msg_origin="test")
    paths, source = await plugin._get_selfie_reference_paths(event)
    assert source == "store" and [p.read_bytes() for p in paths] == [
        picture("yellow"),
        picture("blue"),
    ]
    assert [
        p.read_bytes() for p in await plugin.refs.get_paths("bot_selfie_other")
    ] == [picture("green")]
    assert (await remove(client, base, [updated["id"]], "delete-assets"))[0] == 200
    paths, _ = await plugin._get_selfie_reference_paths(event)
    assert [p.read_bytes() for p in paths] == [picture("blue")]
    await plugin.refs.delete("bot_selfie_bot")
    assert len((await listing(client, base))["items"]) == 1
    assert studio.store.asset_path(red["id"]).is_file()


@pytest.mark.asyncio
async def test_invalid_uploads_and_paths_do_not_change_sources(http_studio):
    plugin, _, client, base = http_studio
    outside = plugin.data_dir.parent / "outside.png"
    outside.write_bytes(picture("green"))
    plugin.config["features"]["selfie"]["reference_images"] = ["../outside.png"]
    await plugin.refs._write_index({"bot_selfie": ["../../outside.png"]})
    assert (await listing(client, base))["items"] == []
    assert await plugin.refs.get_paths("bot_selfie") == []
    for blob in [b"not an image", picture("red")[:40], b"x" * (20 * 1024 * 1024 + 1)]:
        assert (await upload(client, base, blob))[0] == 400
    for key in ["missing", "../outside.png", "../../studio.sqlite3"]:
        assert (await upload(client, base, picture("red"), key))[0] == 409
        assert (await remove(client, base, [key]))[0] == 409
    assert outside.read_bytes() == picture("green")
    assert (await listing(client, base))["items"] == []
    async with client.get(base + "reference-delete") as response:
        assert response.status == 405
    # A filename is only a label, never a server path.
    assert (await upload(client, base, picture("red"), filename="../../outside.png"))[
        0
    ] == 200
    assert outside.read_bytes() == picture("green")
    assert plugin._get_config_selfie_reference_paths()[0].is_relative_to(
        plugin.data_dir
    )


@pytest.mark.asyncio
async def test_concurrent_add_replace_delete_does_not_lose_updates(http_studio):
    import asyncio

    plugin, _, client, base = http_studio
    results = await asyncio.gather(
        *(upload(client, base, picture(color)) for color in ["red", "blue", "green"])
    )
    assert [status for status, _ in results] == [200, 200, 200]
    assert len(plugin._get_config_selfie_reference_paths()) == 3
    target = (await listing(client, base))["items"][0]
    results = await asyncio.gather(
        upload(client, base, picture("yellow"), target["id"]),
        upload(client, base, picture("white"), target["id"]),
    )
    assert sorted(status for status, _ in results) == [200, 409]
    assert len(plugin._get_config_selfie_reference_paths()) == 3
    ids = [a["id"] for a in (await listing(client, base))["items"]]
    results = await asyncio.gather(*(remove(client, base, [key]) for key in ids))
    assert [status for status, _ in results] == [200, 200, 200]
    assert plugin._get_config_selfie_reference_paths() == []
    assert (await listing(client, base))["items"] == []


@pytest.mark.asyncio
async def test_failed_config_save_rolls_back_files_and_index(http_studio, monkeypatch):
    plugin, _, client, base = http_studio
    status, result = await upload(client, base, picture("red"))
    assert status == 200
    original = list(plugin._get_selfie_conf()["reference_images"])
    directory = plugin.data_dir / "files/features/selfie/reference_images"
    files = set(directory.iterdir())
    await plugin.refs.set("bot_selfie", [picture("blue")])
    previous_index = await plugin.refs._read_index()

    def fail():
        raise OSError("test save failure")

    monkeypatch.setattr(plugin.config, "save_config", fail)
    assert (await upload(client, base, picture("green"), result["data"]["data"]["id"]))[
        0
    ] == 500
    assert set(directory.iterdir()) == files
    assert plugin._get_selfie_conf()["reference_images"] == original
    ids = [a["id"] for a in (await listing(client, base))["items"]]
    assert (await remove(client, base, ids))[0] == 500
    assert await plugin.refs._read_index() == previous_index
    assert plugin._get_selfie_conf()["reference_images"] == original
    assert len((await listing(client, base))["items"]) == 2


@pytest.mark.asyncio
async def test_migration_retains_ids_snapshots_generated_media_and_restart(http_studio):
    plugin, studio, client, base = http_studio
    generated = studio.store.add_image(picture("green"), kind="history")
    legacy = studio.store.add_image(
        picture("red"), kind="reference", metadata={"source": "legacy"}
    )
    stale = studio.store.add_image(
        picture("blue"), kind="reference", metadata={"source": "legacy"}
    )
    configured = plugin.data_dir / "old.png"
    configured.write_bytes(picture("red"))
    plugin.config["features"]["selfie"]["reference_images"] = [configured.name]
    assert [a["id"] for a in (await listing(client, base))["items"]] == [legacy["id"]]
    assert studio.store.asset(stale["id"])["kind"] == "upload"
    assert studio.store.asset_path(stale["id"]).read_bytes() == picture("blue")
    from core.studio_store import StudioStore

    restarted = StudioStore(plugin.data_dir)
    assert restarted.list_assets(kind="reference")["items"][0]["id"] == legacy["id"]
    assert restarted.asset_path(generated["id"]).read_bytes() == picture("green")
    assert (await remove(client, base, [legacy["id"]]))[0] == 200
    assert restarted.asset_path(legacy["id"]).read_bytes() == picture("red")
    assert restarted.asset_path(generated["id"]).read_bytes() == picture("green")
    assert (await listing(client, base))["items"] == []


@pytest.mark.asyncio
async def test_config_order_generation_and_restart_share_reference_bindings(
    http_studio,
):
    from types import SimpleNamespace

    plugin, studio, client, base = http_studio
    paths = []
    for index, color in enumerate(["red", "blue", "green"]):
        path = plugin.data_dir / f"configured-{index}.png"
        path.write_bytes(picture(color))
        paths.append(path.name)
    plugin.config["features"]["selfie"]["reference_images"] = paths.copy()
    items = (await listing(client, base))["items"]
    middle = next(
        a for a in items if a["sha256"] == hashlib.sha256(picture("blue")).hexdigest()
    )
    assert (await upload(client, base, picture("yellow"), middle["id"]))[0] == 200
    event = SimpleNamespace(get_self_id=lambda: "bot", unified_msg_origin="test")
    effective, source = await plugin._get_selfie_reference_paths(event)
    assert source == "webui"
    assert [p.read_bytes() for p in effective] == [
        picture("red"),
        picture("yellow"),
        picture("green"),
    ]
    assert plugin._get_selfie_conf()["reference_images"][::2] == paths[::2]
    # A fresh Studio instance reuses the mapped IDs, even with legacy-v1 set.
    restarted = type(studio)(plugin)
    await restarted.references.sync()
    assert {
        a["id"] for a in restarted.store.list_assets(kind="reference")["items"]
    } == {a["id"] for a in (await listing(client, base))["items"]}
    plugin.config["features"]["selfie"]["reference_images"] = []
    await restarted.references.sync()
    assert restarted.store.list_assets(kind="reference")["items"] == []
    assert (await plugin._get_selfie_reference_paths(event))[0] == []


@pytest.mark.asyncio
async def test_external_config_change_during_batch_delete_is_not_overwritten(
    http_studio, monkeypatch
):
    plugin, _, client, base = http_studio
    assert (await upload(client, base, picture("red")))[0] == 200
    await plugin.refs.set("bot_selfie", [picture("blue")])
    original_index = await plugin.refs._read_index()
    ids = [a["id"] for a in (await listing(client, base))["items"]]
    external = plugin.data_dir / "external.png"
    external.write_bytes(picture("green"))
    original_write = plugin.refs._write_index

    async def interleaved_write(index):
        await original_write(index)
        plugin.config["features"]["selfie"]["reference_images"] = [external.name]

    monkeypatch.setattr(plugin.refs, "_write_index", interleaved_write)
    assert (await remove(client, base, ids))[0] == 409
    assert plugin._get_selfie_conf()["reference_images"] == [external.name]
    assert await plugin.refs._read_index() == original_index


@pytest.mark.asyncio
async def test_command_same_filename_update_symlink_and_archived_retention(http_studio):
    from test_studio_management import age

    plugin, studio, client, base = http_studio
    await plugin.refs.set("bot_selfie", [picture("red")])
    old = (await listing(client, base))["items"][0]
    await plugin.refs.set("bot_selfie", [picture("blue")])
    changed = (await listing(client, base))["items"][0]
    assert old["id"] != changed["id"] and old["sha256"] != changed["sha256"]
    external = plugin.data_dir.parent / "unrelated.png"
    external.write_bytes(picture("green"))
    link = plugin.data_dir / "linked.png"
    link.symlink_to(external)
    plugin.config["features"]["selfie"]["reference_images"] = [link.name]
    assert len((await listing(client, base))["items"]) == 1
    assert external.read_bytes() == picture("green")
    age(studio.store, old, 1)
    studio.store.save_document("settings", {"id": "library", "max_count": 1})
    assert studio.store.enforce_limit()["deleted"] == []
    assert studio.store.asset_path(old["id"]).read_bytes() == picture("red")
