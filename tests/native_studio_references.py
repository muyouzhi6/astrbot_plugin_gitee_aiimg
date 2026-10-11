"""Independent tests: real AstrBot 4.26.8 APIs, TCP HTTP, config and old DB."""

# AstrBot reads its runtime paths during import; isolate those paths first.
# ruff: noqa: E402

import asyncio
import copy
import hashlib
import importlib
import io
import json
import logging
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
os.umask(0o077)
os.environ.setdefault("ASTRBOT_ROOT", str(ROOT / "data" / "native-reference-tests"))
Path(os.environ["ASTRBOT_ROOT"]).mkdir(parents=True, exist_ok=True)
os.chdir(os.environ["ASTRBOT_ROOT"])
sys.path.insert(0, str(ROOT.parent))

import astrbot
from astrbot.api.star import Context
from astrbot.core.config.astrbot_config import AstrBotConfig
from astrbot.dashboard.api.plugins import legacy_router, router
from astrbot.dashboard.responses import ApiError
from astrbot.dashboard.services.config_service import ConfigFileService, validate_config
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import httpx
import jwt
from PIL import Image
import uvicorn

main = importlib.import_module(ROOT.name + ".main")
store_module = importlib.import_module(ROOT.name + ".core.studio_store")
reference_module = importlib.import_module(ROOT.name + ".core.studio_references")
ref_module = importlib.import_module(ROOT.name + ".core.ref_store")
SCHEMA = json.loads((ROOT / "_conf_schema.json").read_text())
NAME = "astrbot_plugin_gitee_aiimg"
logging.disable(logging.INFO)


def picture(color):
    buffer = io.BytesIO()
    Image.new("RGB", (30, 40), color).save(buffer, "PNG")
    return buffer.getvalue()


class NativeRuntime(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory(
            prefix="case-", dir=os.environ["ASTRBOT_ROOT"]
        )
        self.root = Path(self.tmp.name)
        self.config = AstrBotConfig(
            str(self.root / "config.json"),
            default_config={
                "providers": [],
                "features": {"selfie": {"reference_images": []}},
            },
        )
        object.__setattr__(self.config, "schema", SCHEMA)
        context = object.__new__(Context)
        context.registered_web_apis = []
        context.provider_manager = SimpleNamespace(provider_insts=[])
        context.get_config = lambda: {"wake_prefix": ["/"]}
        self.plugin = main.GiteeAIImagePlugin(context, self.config)
        self.plugin.data_dir = self.root / "plugin-data"
        self.plugin.data_dir.mkdir()
        await self.plugin.initialize()
        self.assertIsNotNone(self.plugin.studio)
        self.app = FastAPI()
        self.app.state.jwt_secret = secrets.token_hex(32)
        self.app.state.core_lifecycle = SimpleNamespace(star_context=context)
        self.app.include_router(legacy_router)
        self.app.include_router(router, prefix="/api/v1")

        @self.app.exception_handler(ApiError)
        async def api_error(request, exc):
            return JSONResponse({"error": exc.message}, status_code=exc.status_code)

        @self.app.exception_handler(ValueError)
        async def value_error(request, exc):
            return JSONResponse({"error": str(exc)}, status_code=400)

        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(64)
        self.server = uvicorn.Server(
            uvicorn.Config(
                self.app, log_level="critical", access_log=False, lifespan="off"
            )
        )
        self.server_task = asyncio.create_task(self.server.serve(sockets=[self.sock]))
        for _ in range(100):
            if self.server.started:
                break
            await asyncio.sleep(0.01)
        self.assertTrue(self.server.started)
        self.origin = f"http://127.0.0.1:{self.sock.getsockname()[1]}"
        self.base = self.origin + f"/api/plug/{NAME}/studio/"
        token = jwt.encode(
            {"username": "reference-test-fixture"},
            self.app.state.jwt_secret,
            algorithm="HS256",
        )
        self.client = httpx.AsyncClient(
            base_url=self.base, headers={"Authorization": "Bearer " + token}
        )
        metadata = SimpleNamespace(config=self.config)
        self.reloads = []

        async def reload(name):
            self.reloads.append(name)

        self.service = ConfigFileService(
            SimpleNamespace(plugin_manager=SimpleNamespace(reload=reload))
        )
        self.service.get_plugin_metadata_by_name = lambda name: metadata
        self.service._plugin_root_path = lambda name: self.plugin.data_dir
        self.app.state.services = SimpleNamespace(config_files=self.service)

    async def asyncTearDown(self):
        try:
            await self.client.aclose()
        finally:
            self.server.should_exit = True
            await asyncio.wait_for(self.server_task, 5)
            self.sock.close()
            await self.plugin.terminate()
            self.tmp.cleanup()

    async def listing(self):
        response = await self.client.get("library", params={"kind": "reference"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers.get("cache-control"), "no-store")
        return response.json()["data"]["data"]["items"]

    async def upload(self, color, asset_id=""):
        endpoint = "reference-upload" + ("/" + asset_id if asset_id else "")
        response = await self.client.post(
            endpoint, files={"file": ("new.png", picture(color), "image/png")}
        )
        self.assertEqual(response.status_code, 200)
        return response.json()["data"]["data"]

    async def test_real_http_auth_lifecycle_and_persistence(self):
        async with httpx.AsyncClient(base_url=self.base) as anonymous:
            for method, endpoint, kwargs in [
                ("GET", "library", {}),
                (
                    "POST",
                    "reference-upload",
                    {"files": {"file": ("x.png", picture("red"))}},
                ),
                ("POST", "reference-delete", {"json": {"ids": ["missing"]}}),
            ]:
                self.assertEqual(
                    (await anonymous.request(method, endpoint, **kwargs)).status_code,
                    401,
                )
            self.assertEqual(
                (
                    await anonymous.get(
                        "library", headers={"Authorization": "Bearer invalid-fixture"}
                    )
                ).status_code,
                401,
            )
        self.assertEqual(await self.listing(), [])
        first = await self.upload("red")
        second = await self.upload("blue", first["id"])
        self.assertNotEqual(first["id"], second["id"])
        self.assertEqual([a["id"] for a in await self.listing()], [second["id"]])
        saved = json.loads(
            Path(self.config.config_path).read_text(encoding="utf-8-sig")
        )
        self.assertEqual(
            saved["features"]["selfie"]["reference_images"],
            self.config["features"]["selfie"]["reference_images"],
        )
        self.assertEqual(
            self.plugin._get_config_selfie_reference_paths()[0].read_bytes(),
            picture("blue"),
        )
        response = await self.client.get(
            "asset", params={"id": second["id"], "thumbnail": "0"}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers.get("cache-control"), "no-store")
        self.assertEqual(
            (
                await self.client.post("reference-delete", json={"ids": [second["id"]]})
            ).status_code,
            200,
        )
        self.assertEqual(await self.listing(), [])
        self.assertEqual(
            json.loads(Path(self.config.config_path).read_text(encoding="utf-8-sig"))[
                "features"
            ]["selfie"]["reference_images"],
            [],
        )
        self.assertEqual(
            self.plugin.studio.store.asset_path(first["id"]).read_bytes(),
            picture("red"),
        )
        self.assertEqual(
            self.plugin.studio.store.asset_path(second["id"]).read_bytes(),
            picture("blue"),
        )

    async def test_mixed_batch_skips_favorites_and_persists_other_deletions(self):
        favorite = await self.upload("red")
        removable = await self.upload("blue")
        store = self.plugin.studio.store
        store.favorite([favorite["id"]], True)
        ordinary = store.add_image(picture("green"))
        ordinary_path = store.asset_path(ordinary["id"])
        response = await self.client.post(
            "delete-assets",
            json={"ids": [favorite["id"], removable["id"], ordinary["id"]]},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["data"]["data"],
            {
                "deleted": [removable["id"], ordinary["id"]],
                "skipped": [{"id": favorite["id"], "reason": "已收藏, 请先取消收藏"}],
            },
        )
        self.assertEqual(
            [item["id"] for item in await self.listing()], [favorite["id"]]
        )
        self.assertEqual(
            [
                path.read_bytes()
                for path in self.plugin._get_config_selfie_reference_paths()
            ],
            [picture("red")],
        )
        saved = json.loads(
            Path(self.config.config_path).read_text(encoding="utf-8-sig")
        )
        self.assertEqual(
            saved["features"]["selfie"]["reference_images"],
            self.config["features"]["selfie"]["reference_images"],
        )
        self.assertFalse(ordinary_path.exists())
        self.assertEqual(
            store.asset_path(removable["id"]).read_bytes(), picture("blue")
        )

    async def test_gallery_upload_must_allow_native_config_save(self):
        before, _ = validate_config(
            copy.deepcopy(dict(self.config)), SCHEMA, is_core=False
        )
        self.assertEqual(before, [])
        await self.upload("red")
        response = await self.client.put(
            self.origin + f"/api/v1/plugins/{NAME}/config",
            json={"config": dict(self.config)},
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self.reloads, [NAME])
        first = (await self.listing())[0]
        await self.upload("blue", first["id"])
        response = await self.client.put(
            self.origin + f"/api/v1/plugins/{NAME}/config",
            json={"config": dict(self.config)},
        )
        self.assertEqual(response.status_code, 200, response.text)

    async def test_gallery_upload_must_appear_in_native_file_picker(self):
        await self.upload("red")
        selected = self.config["features"]["selfie"]["reference_images"]
        response = await self.client.get(
            self.origin
            + f"/api/v1/plugins/{NAME}/config-files/features.selfie.reference_images"
        )
        self.assertEqual(response.status_code, 200)
        native = response.json()["data"]["files"]
        self.assertIn(selected[0], native)
        first = (await self.listing())[0]
        await self.upload("blue", first["id"])
        response = await self.client.get(
            self.origin
            + f"/api/v1/plugins/{NAME}/config-files/features.selfie.reference_images"
        )
        self.assertIn(
            self.config["features"]["selfie"]["reference_images"][0],
            response.json()["data"]["files"],
        )

    async def test_v1_bridge_target_route_auth_and_upload(self):
        path = self.origin + f"/api/v1/plugins/extensions/{NAME}/studio/"
        async with httpx.AsyncClient(base_url=self.base) as anonymous:
            self.assertEqual((await anonymous.get(path + "library")).status_code, 401)
            expired = jwt.encode(
                {"username": "reference-test-fixture", "exp": 1},
                self.app.state.jwt_secret,
                algorithm="HS256",
            )
            self.assertEqual(
                (
                    await anonymous.get(
                        path + "library", headers={"Authorization": "Bearer " + expired}
                    )
                ).status_code,
                401,
            )
        response = await self.client.post(
            path + "reference-upload",
            files={"file": ("photo.png", picture("red"), "image/png")},
        )
        self.assertEqual(response.status_code, 200)
        response = await self.client.get(path + "library", params={"kind": "reference"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json()["data"]["data"]["items"]), 1)
        first = response.json()["data"]["data"]["items"][0]
        response = await self.client.post(
            path + "reference-upload/" + first["id"],
            files={"file": ("photo.png", picture("blue"), "image/png")},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["data"]["data"]["sha256"],
            hashlib.sha256(picture("blue")).hexdigest(),
        )

    async def seed_wrong_directory(self):
        relative = "files/selfie/old.png"
        file = self.plugin.data_dir / relative
        file.parent.mkdir(parents=True)
        file.write_bytes(picture("red"))
        self.config["features"]["selfie"]["reference_images"] = [relative, relative]
        self.config.save_config()
        refs = self.plugin.studio.references
        old = self.plugin.studio.store.sync_references(refs.sources({}))[0]
        self.plugin.studio.store.favorite([old["id"]], True)
        return relative, file, old

    async def test_wrong_directory_migrates_on_restart_preserving_ids_and_media(self):
        relative, file, old = await self.seed_wrong_directory()
        store = self.plugin.studio.store
        generated = store.add_image(picture("green"), kind="history")
        store.save_document(
            "workspace", {"id": "canvas", "layers": [{"asset_id": old["id"]}]}
        )
        native_dir = self.plugin.data_dir / "files/features/selfie/reference_images"
        native_dir.mkdir(parents=True)
        collision = native_dir / "old.png"
        collision.write_bytes(picture("blue"))
        before = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in store.media.iterdir()
        }
        await self.plugin.terminate()
        await self.plugin.initialize()
        selected = self.config["features"]["selfie"]["reference_images"]
        self.assertNotEqual(selected, [relative, relative])
        self.assertEqual(selected[0], selected[1])
        self.assertTrue(
            selected[0].startswith("files/features/selfie/reference_images/")
        )
        self.assertEqual(
            (self.plugin.data_dir / selected[0]).read_bytes(), picture("red")
        )
        current = (await self.listing())[0]
        self.assertEqual(current["id"], old["id"])
        self.assertTrue(current["favorite"])
        self.assertEqual(file.read_bytes(), picture("red"))
        self.assertEqual(collision.read_bytes(), picture("blue"))
        self.assertEqual(
            store.document("workspace", "canvas")["layers"][0]["asset_id"], old["id"]
        )
        self.assertEqual(
            store.asset_path(generated["id"]).read_bytes(), picture("green")
        )
        self.assertEqual(
            before,
            {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in store.media.iterdir()
            },
        )
        backups = list((store.root / "config-backups").glob("*.json"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].stat().st_mode & 0o777, 0o600)
        self.assertEqual(
            json.loads(backups[0].read_text())["features"]["selfie"][
                "reference_images"
            ],
            [relative, relative],
        )
        response = await self.client.put(
            self.origin + f"/api/v1/plugins/{NAME}/config",
            json={"config": dict(self.config)},
        )
        self.assertEqual(response.status_code, 200, response.text)
        response = await self.client.get(
            self.origin
            + f"/api/v1/plugins/{NAME}/config-files/features.selfie.reference_images"
        )
        self.assertIn(selected[0], response.json()["data"]["files"])
        await self.listing()
        self.assertEqual(len(list((store.root / "config-backups").glob("*.json"))), 1)
        with store.connect() as db:
            self.assertEqual(db.execute("PRAGMA integrity_check").fetchone()[0], "ok")

    async def test_wrong_directory_migration_save_failure_rolls_back(self):
        from unittest.mock import patch

        relative, file, old = await self.seed_wrong_directory()
        refs = self.plugin.studio.references
        with patch.object(
            AstrBotConfig, "save_config", side_effect=OSError("fixture save failure")
        ):
            with self.assertRaises(OSError):
                await refs.sync()
        self.assertEqual(
            self.config["features"]["selfie"]["reference_images"], [relative, relative]
        )
        self.assertEqual(file.read_bytes(), picture("red"))
        self.assertEqual(
            self.plugin.studio.store.asset(old["id"])["reference_origin"]["path"],
            relative,
        )
        native_dir = self.plugin.data_dir / "files/features/selfie/reference_images"
        self.assertEqual(list(native_dir.glob("*")), [])
        self.assertEqual((await self.listing())[0]["id"], old["id"])

    async def test_wrong_directory_external_symlink_is_not_migrated(self):
        outside = self.root / "outside.png"
        outside.write_bytes(picture("red"))
        link = self.plugin.data_dir / "files/selfie/link.png"
        link.parent.mkdir(parents=True)
        link.symlink_to(outside)
        relative = link.relative_to(self.plugin.data_dir).as_posix()
        self.config["features"]["selfie"]["reference_images"] = [relative]
        self.assertEqual(await self.listing(), [])
        self.assertEqual(outside.read_bytes(), picture("red"))
        self.assertEqual(
            self.config["features"]["selfie"]["reference_images"], [relative]
        )

    async def test_native_setting_update_and_path_security(self):
        relative = "files/features/selfie/reference_images/native.png"
        file = self.plugin.data_dir / relative
        file.parent.mkdir(parents=True)
        file.write_bytes(picture("red"))
        draft = copy.deepcopy(dict(self.config))
        draft["features"]["selfie"]["reference_images"] = [relative]
        response = await self.client.put(
            self.origin + f"/api/v1/plugins/{NAME}/config", json={"config": draft}
        )
        self.assertEqual(response.status_code, 200)
        first = (await self.listing())[0]
        file.write_bytes(picture("green"))
        second = (await self.listing())[0]
        self.assertNotEqual(first["id"], second["id"])
        self.assertEqual(second["sha256"], hashlib.sha256(picture("green")).hexdigest())
        response = await self.client.request(
            "DELETE",
            self.origin + f"/api/v1/plugins/{NAME}/config-files",
            json={"path": relative},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(await self.listing(), [])
        outside = self.root / "outside.png"
        outside.write_bytes(picture("yellow"))
        (self.plugin.data_dir / "symlink.png").symlink_to(outside)
        self.config["features"]["selfie"]["reference_images"] = [
            "../outside.png",
            "symlink.png",
        ]
        self.assertEqual(await self.listing(), [])
        self.assertEqual(
            (
                await self.client.post(
                    "reference-delete", json={"ids": ["../../studio.sqlite3"]}
                )
            ).status_code,
            409,
        )
        self.assertEqual(
            (
                await self.client.post(
                    "reference-upload", files={"file": ("bad.png", b"not-image")}
                )
            ).status_code,
            400,
        )
        self.assertEqual(outside.read_bytes(), picture("yellow"))

    async def test_native_http_invalid_uploads_and_concurrent_replacement(self):
        for payload in [b"not-image", b"x" * (20 * 1024 * 1024 + 1)]:
            response = await self.client.post(
                "reference-upload", files={"file": ("bad.png", payload)}
            )
            self.assertEqual(response.status_code, 400)
        for asset_id in ["missing", "../outside.png"]:
            response = await self.client.post(
                "reference-delete", json={"ids": [asset_id]}
            )
            self.assertEqual(response.status_code, 409)
            response = await self.client.post(
                "reference-upload",
                params={"id": asset_id},
                files={"file": ("photo.png", picture("red"))},
            )
            self.assertEqual(response.status_code, 409)
        self.assertEqual(await self.listing(), [])
        first, _ = await asyncio.gather(self.upload("red"), self.upload("blue"))
        results = await asyncio.gather(
            *(
                self.client.post(
                    "reference-upload/" + first["id"],
                    files={"file": ("photo.png", picture(color))},
                )
                for color in ["green", "yellow"]
            )
        )
        self.assertEqual(
            sorted(response.status_code for response in results), [200, 409]
        )
        items = await self.listing()
        self.assertEqual(len(items), 2)
        self.assertEqual(len(self.config["features"]["selfie"]["reference_images"]), 2)
        results = await asyncio.gather(
            *(
                self.client.post("reference-delete", json={"ids": [item["id"]]})
                for item in items
            )
        )
        self.assertEqual([response.status_code for response in results], [200, 200])
        self.assertEqual(await self.listing(), [])

    async def test_migrate_actual_baseline_database_without_losing_media(self):
        root = self.root / "migration"
        root.mkdir()
        self.plugin.data_dir = root
        self.plugin.refs = ref_module.ReferenceStore(root)
        await self.plugin.refs.set("bot_selfie_bot", [picture("green")])
        for name, color in [("active.png", "red"), ("stale.png", "blue")]:
            (root / name).write_bytes(picture(color))
        old_source = subprocess.check_output(
            [
                "git",
                "show",
                "0aab814f2533ac798a5cd0a8727155da21ff8280:core/studio_store.py",
            ],
            cwd=ROOT,
            text=True,
        )
        namespace = {"__name__": "review1_baseline_store"}
        exec(compile(old_source, "baseline/core/studio_store.py", "exec"), namespace)
        old = namespace["StudioStore"](root)
        old.import_existing(root, ["active.png", "stale.png"])
        previous = old.list_assets(kind="reference")["items"]
        stale = next(
            a
            for a in previous
            if a["sha256"] == hashlib.sha256(picture("blue")).hexdigest()
        )
        history = old.add_image(picture("yellow"), kind="history")
        old.save_document(
            "workspace", {"id": "canvas", "layers": [{"asset_id": stale["id"]}]}
        )
        files_before = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in old.media.iterdir()
        }
        self.config["features"]["selfie"]["reference_images"] = ["active.png"]
        current = store_module.StudioStore(root)
        refs = reference_module.StudioReferences(
            SimpleNamespace(plugin=self.plugin, store=current)
        )
        active = await refs.sync()
        self.assertEqual(
            {a["id"] for a in active}, {a["id"] for a in previous} - {stale["id"]}
        )
        self.assertTrue(current.asset(stale["id"])["reference_archive"])
        self.assertEqual(
            current.asset_path(history["id"]).read_bytes(), picture("yellow")
        )
        self.assertEqual(
            current.document("workspace", "canvas")["layers"][0]["asset_id"],
            stale["id"],
        )
        self.assertEqual(
            files_before,
            {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in current.media.iterdir()
            },
        )
        with current.connect() as db:
            self.assertEqual(db.execute("PRAGMA integrity_check").fetchone()[0], "ok")
        second_sync = await refs.sync()
        self.assertEqual([a["id"] for a in active], [a["id"] for a in second_sync])


if __name__ == "__main__":
    print("RUNTIME:", astrbot.__version__, flush=True)
    unittest.main(verbosity=2)
