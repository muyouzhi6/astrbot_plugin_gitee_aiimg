"""Authenticated AstrBot studio API and durable image job integration."""

from __future__ import annotations

import asyncio
import base64
import copy
import hashlib
import json
import re
import time
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from .background_tasks import BackgroundImageTaskManager
from .draw_service import ImageDrawService
from .edit_router import EditRouter
from .output_spec import parse_output_intent
from .provider_registry import ProviderRegistry
from .studio_capture import capture_context
from .studio_characters import (
    resolve_characters,
    validate_character,
    targeted_reference_note,
)
from .studio_store import StudioConflict, StudioStore
from .studio_workflows import StudioWorkflows, request_key

MASK = "******** (已保存)"
SECRET = re.compile(r"api.?key|token|password|secret|cookie|authorization", re.I)
ENDPOINTS = ("base_url", "api_url", "server_url", "full_generate_url", "full_edit_url")


def redact(value):
    if isinstance(value, dict):
        return {
            k: (MASK if SECRET.search(k) and v else redact(v)) for k, v in value.items()
        }
    if isinstance(value, list):
        return [redact(v) for v in value]
    return value


def restore_secrets(value, original):
    if value == MASK:
        if original is None:
            raise ValueError("请重新填写密钥")
        return copy.deepcopy(original)
    if isinstance(value, dict):
        original = original if isinstance(original, dict) else {}
        return {k: restore_secrets(v, original.get(k)) for k, v in value.items()}
    if isinstance(value, list):
        old = original if isinstance(original, list) else []
        return [
            restore_secrets(v, old[i] if i < len(old) else None)
            for i, v in enumerate(value)
        ]
    return value


class Studio:
    def __init__(self, plugin):
        self.plugin = plugin
        self.store = StudioStore(plugin.data_dir)
        self.schema = json.loads(
            (Path(__file__).parents[1] / "_conf_schema.json").read_text(
                encoding="utf-8-sig"
            )
        )
        self.lock = asyncio.Lock()
        self.own_manager = None
        self.retired = []
        self.tasks = set()
        self.workflows = StudioWorkflows(self)

    async def start(self):
        self.workflows.recover()
        self.plugin.imgr.studio_store = self.store
        await asyncio.to_thread(
            self.store.import_existing,
            self.plugin.data_dir,
            self.plugin._get_selfie_conf().get("reference_images", []),
        )
        if self.plugin.background_tasks is None and not self.plugin._get_feature(
            "background_llm_image"
        ).get("enabled", False):
            self.own_manager = BackgroundImageTaskManager(
                self.store.root / "queue", max_running=2, max_queued=16
            )
            await self.own_manager.start()
        for action in (
            "state",
            "config",
            "models",
            "library",
            "asset",
            "upload",
            "character",
            "appearance",
            "workspace",
            "generate",
            "jobs",
            "cancel",
            "download",
            "crop",
            "plan",
            "retry",
        ):

            async def handler(action=action):
                return await self.handle(action)

            self.plugin.context.register_web_api(
                f"/astrbot_plugin_gitee_aiimg/studio/{action}",
                handler,
                ["GET", "POST"],
                "Image studio",
            )

    @property
    def manager(self):
        return self.plugin.background_tasks or self.own_manager

    async def close(self):
        await self.workflows.close()
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        if self.own_manager:
            await self.own_manager.close()
        for registry in self.retired:
            await registry.close()

    def revision(self):
        return hashlib.sha256(
            json.dumps(
                dict(self.plugin.config), sort_keys=True, ensure_ascii=False
            ).encode("utf-8")
        ).hexdigest()

    def config_view(self):
        return {
            "revision": self.revision(),
            "providers": redact(self.plugin.config.get("providers", [])),
            "features": redact(self.plugin.config.get("features", {})),
            "templates": self.schema["providers"]["templates"],
        }

    def provider_input(self, draft):
        original = next(
            (
                p
                for p in self.plugin.config.get("providers", [])
                if p.get("id") == draft.get("id")
            ),
            {},
        )
        if MASK in json.dumps(draft, ensure_ascii=False) and any(
            draft.get(k, "") != original.get(k, "") for k in ENDPOINTS
        ):
            raise ValueError("接口地址已更改, 请重新填写密钥后获取模型或保存")
        return restore_secrets(draft, original)

    async def save_config(self, body):
        async with self.lock:
            if self.manager and self.manager.started:
                health = await self.manager.health_snapshot()
                if health.get("active_tasks", 0):
                    raise ValueError(
                        "仍有图片任务执行或排队, 请等待结束后保存服务商配置"
                    )
            if (
                any(getattr(self.plugin, "_image_inflight", {}).values())
                or any(getattr(self.plugin, "_video_inflight", {}).values())
                or getattr(self.plugin, "_video_tasks", set())
            ):
                raise ValueError("仍有图片或视频任务执行, 请等待结束后保存")
            if body.get("revision") != self.revision():
                raise StudioConflict("配置已在另一处修改, 请刷新后重试")
            providers = body.get("providers")
            if not isinstance(providers, list) or len(providers) > 100:
                raise ValueError("服务商列表格式错误")
            merged = copy.deepcopy(dict(self.plugin.config))
            merged["providers"] = [self.provider_input(p) for p in providers]
            for p in merged["providers"]:
                if not isinstance(p.get("extra_body", {}), dict):
                    raise ValueError("额外请求体必须是 JSON 对象")
            for feature, chain in body.get("chains", {}).items():
                if feature not in {"draw", "edit", "selfie", "video"} or not isinstance(
                    chain, list
                ):
                    raise ValueError("功能链路格式错误")
                if any(
                    not isinstance(x, dict)
                    or x.get("provider_id") not in {p.get("id") for p in providers}
                    for x in chain
                ):
                    raise ValueError("链路引用了不存在的服务商")
                merged.setdefault("features", {}).setdefault(feature, {})["chain"] = (
                    chain
                )
            for feature in ("draw", "edit", "selfie", "video"):
                for entry in (
                    merged.get("features", {}).get(feature, {}).get("chain", [])
                ):
                    if entry.get("provider_id") not in {p.get("id") for p in providers}:
                        raise ValueError("服务商仍被功能链路使用, 请先移除引用")
            registry = ProviderRegistry(
                merged, imgr=self.plugin.imgr, data_dir=self.plugin.data_dir
            )
            errors = registry.validate()
            if errors:
                raise ValueError("配置未通过校验: " + "; ".join(errors[:4]))
            backups = self.store.root / "config-backups"
            backups.mkdir(exist_ok=True)
            backup = backups / f"{time.time_ns()}.json"
            backup.touch(mode=0o600)
            backup.write_text(
                json.dumps(dict(self.plugin.config), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            original = copy.deepcopy(dict(self.plugin.config))
            self.plugin.config.clear()
            self.plugin.config.update(merged)
            try:
                self.plugin.config.save_config()
            except Exception:
                self.plugin.config.clear()
                self.plugin.config.update(original)
                raise
            self.retired.append(self.plugin.registry)
            self.plugin.registry = registry
            self.plugin.draw = ImageDrawService(
                self.plugin.config,
                self.plugin.imgr,
                self.plugin.data_dir,
                registry=registry,
            )
            self.plugin.edit = EditRouter(
                self.plugin.config,
                self.plugin.imgr,
                self.plugin.data_dir,
                registry=registry,
            )
            for old in self.retired:
                await old.close()
            self.retired.clear()
            return self.config_view()

    async def models(self, body):
        provider = self.provider_input(body["provider"])
        kind = ProviderRegistry._resolve_template_key(provider)
        url = str(
            provider.get("base_url")
            or provider.get("api_url")
            or provider.get("server_url")
            or ""
        ).rstrip("/")
        parsed = urlsplit(url)
        if (
            parsed.scheme not in {"https", "http"}
            or not parsed.netloc
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("请填写有效的接口地址")
        keys = provider.get("api_keys") or [
            provider.get("api_key") or provider.get("apikey")
        ]
        if not isinstance(keys, list) or not keys or not str(keys[0] or "").strip():
            raise ValueError("请先填写 API Key")
        key = str(keys[0]).strip()
        if kind == "gemini_native":
            url = re.sub(r"/v1(?:beta)?(?:/models.*)?$", "", url) + "/v1beta/models"
            headers = {"x-goog-api-key": key}
        else:
            url = re.sub(
                r"/(?:chat/completions|images/(?:generations|edits))$", "", url
            )
            url += "/models" if re.search(r"/v\d+$", url) else "/v1/models"
            headers = {"Authorization": "Bearer " + key}
        async with httpx.AsyncClient(timeout=20, follow_redirects=False) as client:
            response = await client.get(url, headers=headers)
            if response.status_code != 200:
                raise ValueError(
                    f"获取模型失败 (HTTP {response.status_code}), 请检查地址和密钥"
                )
            if len(response.content) > 4 * 1024 * 1024:
                raise ValueError("模型列表响应过大")
            data = response.json()
        rows = data.get("models", data.get("data", []))
        result = sorted(
            {
                str(r.get("id") or r.get("name") or "").removeprefix("models/")
                for r in rows
                if isinstance(r, dict)
            }
            - {""}
        )
        if not result:
            raise ValueError("上游未返回模型列表, 可手动填写模型 ID")
        return {"models": result, "source": "live"}

    async def submit(self, body):
        request_id = request_key(body.get("request_id"))
        manager = self.manager
        if not manager or not manager.accepting:
            raise ValueError("任务系统尚未就绪, 请稍后重试")
        prior = self.store.document("job", request_id)
        if prior:
            return {"task_id": prior["task_id"]}
        body, shots = self.workflows.generation(body)
        images, portrait_note, cast = [], "", []
        ids = body.get("characters", [])
        if ids:
            life = await self.plugin._get_life_context_without_llm()
            selection, portrait_note, cast = await asyncio.to_thread(
                resolve_characters,
                self.store,
                ids,
                body.get("outfits", [""] * len(ids)),
                sender="",
                scope="studio",
                bot_id="",
                life_context=life,
                administrator=True,
            )
            images.extend(selection.images)
        asset_ids = body.get("assets", [])
        if not isinstance(asset_ids, list) or len(asset_ids) + len(images) > 8:
            raise ValueError("每次最多使用 8 张参考图")
        for asset_id in asset_ids:
            images.append(
                await asyncio.to_thread(self.store.asset_path(asset_id).read_bytes)
            )
        if cast and asset_ids:
            roles = body.get("asset_roles") or ["background"] * len(asset_ids)
            if (
                not isinstance(roles, list)
                or len(roles) != len(asset_ids)
                or any(
                    r
                    not in {
                        "background",
                        "style",
                        "clothing",
                        "pose",
                        "subject",
                        "object",
                    }
                    for r in roles
                )
            ):
                raise ValueError("参考图用途与图片数量不匹配")
            portrait_note += "\n" + targeted_reference_note(
                [{"role": r} for r in roles],
                body.get("asset_targets"),
                cast,
                ids,
                len(images) - len(asset_ids),
            )
            portrait_note += "\n额外参考图按指定用途使用, 不新增人物、不覆盖人物身份. 未指定人物用途的图片只参考背景、构图或物体."
        from .image_reference_bridge import validate_inputs

        validate_inputs(images)
        mode = "edit" if images else "text"
        feature = "edit" if images else "draw"
        if not self.plugin._get_feature(feature).get("enabled", True):
            raise ValueError("此生成功能已关闭")
        backend = str(body.get("provider") or "") or None
        if backend and backend not in self.plugin.registry.provider_ids():
            raise ValueError("所选服务商已不存在")
        specs = [
            {
                "item_id": f"image-{i + 1}",
                "title": s["title"],
                "prompt": s["prompt"],
                "effective_prompt": s["prompt"]
                + ("\n\n" + portrait_note if portrait_note else "")
                + ("\n" + body["workflow_note"] if body.get("workflow_note") else ""),
                "state": "queued",
            }
            for i, s in enumerate(shots)
        ]
        output = parse_output_intent(str(body.get("output") or ""))
        return await self.enqueue(
            body, request_id, images, cast, asset_ids, specs, backend, mode, output
        )

    async def retry(self, body):
        key = request_key(body.get("request_id"))
        existing = self.store.document("job", key)
        if existing:
            return {"task_id": existing["task_id"]}
        job = next(
            (
                j
                for j in self.store.documents("job")
                if j["task_id"] == body.get("task_id")
            ),
            None,
        )
        if not job:
            raise ValueError("任务不存在")
        record = await self.manager.get_task(job["task_id"])
        if not record or record["state"] not in {"failed", "partial"}:
            raise ValueError("只能重试已结束任务中的失败镜头")
        snapshot = record.get("studio_snapshot")
        failed = [s for s in record.get("items", []) if s["state"] == "failed"]
        if not snapshot or not failed:
            raise ValueError("原始输入已不可用, 请重新选择图片")
        images = await self.manager.read_spooled_inputs(
            tuple(snapshot["paths"]), snapshot["manifest"]
        )
        specs = [
            {k: s[k] for k in ("item_id", "title", "prompt", "effective_prompt")}
            | {"state": "queued"}
            for s in failed
        ]
        original = snapshot["body"]
        return await self.enqueue(
            original,
            key,
            images,
            snapshot["cast"],
            snapshot["asset_ids"],
            specs,
            snapshot["backend"],
            record["mode"],
            parse_output_intent(original.get("output") or ""),
        )

    async def enqueue(
        self, body, request_id, images, cast, asset_ids, specs, backend, mode, output
    ):
        manager = self.manager
        prompt = body["prompt"]
        task_id = manager.new_task_id("studio")
        # The manager owns reservations, cancellation, restart recovery and scheduling.
        record, created = await manager.create_task_record(
            {
                "task_id": task_id,
                "task_kind": "studio",
                "state": "queued",
                "platform_name": "studio",
                "umo": "studio",
                "scope_hash": "studio",
                "request_fingerprint": "studio:" + request_id,
                "user_prompt": prompt,
                "effective_prompt": specs[0]["effective_prompt"],
                "requested_count": len(specs),
                "items": specs,
                "mode": mode,
                "notification_state": "not_required",
                "ack_state": "confirmed",
            },
            reservation=len(specs),
        )
        task_id = record["task_id"]
        try:
            self.store.save_document(
                "job",
                {
                    "id": request_id,
                    "task_id": task_id,
                    "prompt": prompt,
                    "created": time.time(),
                    "workspace_id": body.get("workspace_id", ""),
                    "workflow": body.get("workflow", "generate"),
                    "count": len(specs),
                    "source_asset": body.get("source_asset", ""),
                    "request": {
                        k: body.get(k)
                        for k in (
                            "workflow",
                            "prompt",
                            "assets",
                            "characters",
                            "outfits",
                            "asset_roles",
                            "asset_targets",
                            "provider",
                            "output",
                            "workspace_id",
                            "source_asset",
                            "plan_id",
                        )
                    },
                },
            )
            paths, manifest = (
                await manager.spool_inputs(task_id, images) if created else ((), [])
            )
            if created:
                await manager.transition(
                    task_id,
                    "queued",
                    {
                        "studio_snapshot": {
                            "paths": list(paths),
                            "manifest": manifest,
                            "cast": cast,
                            "asset_ids": asset_ids,
                            "backend": backend,
                            "body": {
                                k: body.get(k)
                                for k in (
                                    "prompt",
                                    "output",
                                    "workflow",
                                    "source_asset",
                                    "workspace_id",
                                )
                            },
                        }
                    },
                )
        except BaseException:
            if created:
                await manager.transition(
                    task_id, "failed", {"error": "任务输入保存失败, 尚未请求模型"}
                )
            raise
        if created:
            draw, edit = self.plugin.draw, self.plugin.edit

            async def run():
                try:

                    async def one(spec):
                        token = capture_context.set(
                            {
                                "job_id": task_id,
                                "item_id": spec["item_id"],
                                "title": spec["title"],
                                "user_prompt": prompt,
                                "characters": cast,
                                "parent_assets": asset_ids,
                                "workflow": body.get("workflow", "generate"),
                                "source_asset": body.get("source_asset", ""),
                                "workspace_id": body.get("workspace_id", ""),
                            }
                        )
                        try:

                            async def call():
                                await manager.transition(task_id, "running")
                                await manager.update_item(
                                    task_id, spec["item_id"], {"state": "running"}
                                )
                                inputs = await manager.read_spooled_inputs(
                                    paths, manifest
                                )
                                if mode == "edit":
                                    return await edit.edit(
                                        spec["effective_prompt"],
                                        inputs,
                                        backend=backend,
                                        output_intent=output,
                                        require_ordered_references=len(inputs) > 1,
                                    )
                                return await draw.generate(
                                    spec["effective_prompt"],
                                    provider_id=backend,
                                    output_intent=output,
                                )

                            path = await asyncio.wait_for(
                                manager.run_provider(task_id, call), 7200
                            )
                            if manager.is_cancelled(task_id):
                                raise asyncio.CancelledError
                            asset_id = capture_context.get().get("asset_id")
                            if not asset_id:
                                raise ValueError(
                                    "图片已生成但归档失败, 请检查存储空间; 不要自动重新生成"
                                )
                            await manager.update_item(
                                task_id,
                                spec["item_id"],
                                {
                                    "state": "completed",
                                    "image_generated": True,
                                    "asset_id": asset_id,
                                    "result_name": Path(path).name,
                                },
                                release_if_terminal=True,
                            )
                        except asyncio.CancelledError:
                            raise
                        except Exception as exc:
                            if not manager.is_cancelled(task_id):
                                await manager.update_item(
                                    task_id,
                                    spec["item_id"],
                                    {
                                        "state": "failed",
                                        "error": BackgroundImageTaskManager.sanitize_error(
                                            exc
                                        ),
                                    },
                                    release_if_terminal=True,
                                )
                        finally:
                            capture_context.reset(token)

                    await asyncio.gather(*(one(spec) for spec in specs))
                    result = await manager.get_task(task_id)
                    assets = [
                        s["asset_id"] for s in result["items"] if s.get("asset_id")
                    ]
                    await manager.transition(
                        task_id,
                        "completed"
                        if len(assets) == len(specs)
                        else "partial"
                        if assets
                        else "failed",
                        {
                            "image_generated": bool(assets),
                            "gallery_asset_id": assets[0] if assets else "",
                            "gallery_asset_ids": assets,
                        },
                    )
                except asyncio.CancelledError:
                    await manager.transition(
                        task_id,
                        "cancelled",
                        {"error": "已停止等待; 已发送的请求可能仍计费"},
                    )
                except Exception as exc:
                    await manager.transition(
                        task_id,
                        "failed",
                        {"error": BackgroundImageTaskManager.sanitize_error(exc)},
                    )

            task = manager.start_worker(task_id, run)
            if task:
                self.tasks.add(task)
                task.add_done_callback(self.tasks.discard)
        return {"task_id": task_id}

    async def handle(self, action):
        from astrbot.api.web import request, json_response, file_response

        try:
            body = (
                await request.json()
                if request.method == "POST"
                else dict(request.query.items())
            )
            body = body or {}
            writes = {
                "upload",
                "character",
                "appearance",
                "workspace",
                "generate",
                "cancel",
                "models",
                "crop",
                "retry",
            }
            if action in writes and request.method != "POST":
                return json_response({"error": "请使用 POST 请求"}, status_code=405)
            if action == "state":
                result = {
                    "version": 1,
                    "characters": self.store.documents("character"),
                    "workspaces": self.store.documents("workspace"),
                    "config": self.config_view(),
                    "planners": self.workflows.providers(),
                    "plans": self.store.documents("plan")[-30:],
                }
            elif action == "config":
                result = (
                    await self.save_config(body)
                    if request.method == "POST"
                    else self.config_view()
                )
            elif action == "models":
                result = await self.models(body)
            elif action == "library":
                result = self.store.list_assets(
                    kind=body.get("kind", ""), offset=body.get("offset", 0)
                )
            elif action == "asset":
                path = self.store.asset_path(
                    body["id"],
                    thumbnail="preview"
                    if body.get("thumbnail") == "preview"
                    else body.get("thumbnail") in {"1", True},
                )
                result = {
                    "asset": self.store.public_asset(self.store.asset(body["id"])),
                    "data_url": "data:image/"
                    + ("jpeg" if path.suffix == ".jpg" else path.suffix[1:])
                    + ";base64,"
                    + base64.b64encode(await asyncio.to_thread(path.read_bytes)).decode(
                        "ascii"
                    ),
                }
            elif action == "download":
                path = self.store.asset_path(body["id"])
                return file_response(path, filename=path.name)
            elif action == "upload":
                files = await request.files()
                upload = files.get("file")
                if not upload:
                    raise ValueError("请选择图片")
                blob = await upload.read(20 * 1024 * 1024 + 1)
                result = await asyncio.to_thread(
                    self.store.add_image,
                    blob,
                    metadata={"name": str(upload.filename or "图片")[:160]},
                )
            elif action == "crop":
                from PIL import Image, ImageOps
                import io

                source = self.store.asset_path(body["id"])
                rect = body.get("rect", [])
                if len(rect) != 4 or any(
                    not isinstance(v, (int, float)) or not 0 <= v <= 1 for v in rect
                ):
                    raise ValueError("裁剪区域无效")
                x, y, width, height = rect
                if width <= 0 or height <= 0 or x + width > 1.001 or y + height > 1.001:
                    raise ValueError("裁剪区域超出图片")

                def crop():
                    with Image.open(source) as raw:
                        image = ImageOps.exif_transpose(raw)
                        box = tuple(
                            round(v)
                            for v in (
                                x * image.width,
                                y * image.height,
                                (x + width) * image.width,
                                (y + height) * image.height,
                            )
                        )
                        output = io.BytesIO()
                        image.crop(box).save(output, "PNG")
                    return self.store.add_image(
                        output.getvalue(),
                        metadata={"name": "裁剪图片", "parent_assets": [body["id"]]},
                    )

                result = await asyncio.to_thread(crop)
            elif action == "character":
                if body.get("delete"):
                    result = self.store.delete_character(body["id"], body["revision"])
                else:
                    result = self.store.save_document(
                        "character", validate_character(self.store, body)
                    )
            elif action == "appearance":
                c = self.store.document("character", body["character_id"])
                if not c or body["look"] not in {x["name"] for x in c["looks"]}:
                    raise ValueError("形象不存在")
                c["active_look"] = body["look"]
                result = self.store.save_document("character", c)
            elif action == "workspace":
                if (
                    not str(body.get("name", "")).strip()
                    or len(body.get("layers", [])) > 100
                ):
                    raise ValueError("请输入工作区名称, 图层最多 100 个")
                for layer in body.get("layers", []):
                    self.store.asset_path(layer["asset_id"])
                    for key in ("x", "y", "width", "height", "rotation"):
                        value = layer.get(key, 0)
                        if not isinstance(value, (int, float)) or abs(value) > 100000:
                            raise ValueError("画布图层参数错误")
                        if key in {"width", "height"} and value <= 0:
                            raise ValueError("图层宽高必须大于零")
                result = self.store.save_document("workspace", body)
            elif action == "generate":
                async with self.lock:
                    result = await self.submit(body)
            elif action == "plan":
                if request.method == "POST":
                    async with self.lock:
                        result = await self.workflows.start(body)
                else:
                    result = self.store.document("plan", body["id"])
            elif action == "retry":
                async with self.lock:
                    result = await self.retry(body)
            elif action == "jobs":
                result = []
                for j in sorted(
                    self.store.documents("job"),
                    key=lambda x: x["created"],
                    reverse=True,
                )[:50]:
                    row = await self.manager.get_task(j["task_id"])
                    result.append(
                        {
                            **j,
                            "state": row.get("state", "interrupted")
                            if row
                            else "expired",
                            "asset_id": row.get("gallery_asset_id", "") if row else "",
                            "error": row.get("error", "") if row else "",
                            "items": [
                                {
                                    k: s[k]
                                    for k in (
                                        "item_id",
                                        "title",
                                        "prompt",
                                        "state",
                                        "asset_id",
                                        "error",
                                    )
                                    if k in s
                                }
                                for s in (row.get("items", []) if row else [])
                            ],
                        }
                    )
            elif action == "cancel":
                known = {j["task_id"] for j in self.store.documents("job")}
                if body.get("task_id") not in known:
                    raise ValueError("工作台任务不存在")
                result = await self.manager.cancel_task(
                    body["task_id"], "工作台取消任务", suppress_future_injection=True
                )
            return json_response({"status": "ok", "data": {"ok": True, "data": result}})
        except (ValueError, KeyError) as exc:
            return json_response(
                {
                    "status": "error",
                    "message": str(exc),
                    "ok": False,
                    "error": str(exc),
                },
                status_code=409 if isinstance(exc, StudioConflict) else 400,
            )
        except Exception as exc:
            from astrbot.api import logger

            logger.warning("[studio] %s failed: %s", action, type(exc).__name__)
            return json_response(
                {
                    "status": "error",
                    "message": "操作未完成, 请检查配置或稍后重试",
                    "ok": False,
                    "error": "操作未完成, 请检查配置或稍后重试",
                },
                status_code=500,
            )
