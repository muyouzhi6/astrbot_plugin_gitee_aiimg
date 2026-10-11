"""Live gallery projection of the same sources used by selfie generation."""

from __future__ import annotations

import copy
import io
import json
import time
import uuid
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from .studio_store import StudioConflict

MAX_REFERENCE_BYTES = 20 * 1024 * 1024
CONFIG_REFERENCE_DIR = Path("files/features/selfie/reference_images")


class StudioReferences:
    def __init__(self, studio):
        self.studio = studio
        self.plugin = studio.plugin
        self.store = studio.store

    def sources(self, index):
        sources = []
        values = self.plugin._get_selfie_conf().get("reference_images", [])
        if isinstance(values, list):
            for value in values:
                path = self.plugin._resolve_data_rel_path(value)
                if path and path.is_file():
                    sources.append(
                        ({"kind": "config", "path": value}, path, "插件设置")
                    )
        for key, files in index.items():
            for filename in files:
                path = self.plugin.refs.resolve_path(filename)
                if path and path.is_file():
                    sources.append(
                        (
                            {"kind": "store", "key": key, "path": filename},
                            path,
                            "命令参考图 · " + key,
                        )
                    )
        # Multiple occurrences of one path represent one editable source.
        return list(
            {
                json.dumps(origin, sort_keys=True): (origin, path, label)
                for origin, path, label in sources
            }.values()
        )

    async def sync(self):
        async with self.plugin.refs._lock:
            self._migrate_config_paths()
            index = await self.plugin.refs._read_index()
            return self.store.sync_references(self.sources(index))

    def _config_image_path(self, ext):
        relative = CONFIG_REFERENCE_DIR / f"{uuid.uuid4().hex}.{ext}"
        path = self.plugin._resolve_data_rel_path(relative.as_posix())
        if path is None:
            raise ValueError("参考图保存目录无效")
        return path, relative.as_posix()

    def _migrate_config_paths(self):
        """Copy the earlier gallery's misplaced files into the native file field."""
        original = self.plugin._get_selfie_conf().get("reference_images", [])
        if not isinstance(original, list):
            return
        replacements, created = {}, []
        saved = False
        try:
            for value in original:
                if not isinstance(value, str) or value in replacements:
                    continue
                parts = value.replace("\\", "/").split("/")
                if len(parts) != 3 or parts[:2] != ["files", "selfie"]:
                    continue
                source = self.plugin._resolve_data_rel_path(value)
                if not source or not source.is_file():
                    continue
                if source.stat().st_size > MAX_REFERENCE_BYTES:
                    continue
                blob = source.read_bytes()
                try:
                    ext = self.validate_image(blob)
                except ValueError:
                    continue
                destination, relative = self._config_image_path(ext)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.touch(mode=0o600, exist_ok=False)
                created.append(destination)
                destination.write_bytes(blob)
                replacements[value] = relative
            if not replacements:
                return
            values = [replacements.get(value, value) for value in original]
            # Keep each existing asset ID, favorite and canvas binding. A failed
            # config save rolls back both its in-memory values and this transaction.
            with self.store.connect() as db:
                for before, after in replacements.items():
                    db.execute(
                        "UPDATE assets SET data=json_set(data,'$.reference_origin.path',?) "
                        "WHERE kind='reference' "
                        "AND json_extract(data,'$.reference_origin.kind')='config' "
                        "AND json_extract(data,'$.reference_origin.path')=?",
                        (after, before),
                    )
                self._save_config(values)
                saved = True
        except Exception:
            if saved:
                # A database commit failure must not leave the config referencing
                # files we are about to remove. Retain copies if restoration fails.
                self._save_config(original)
            for path in created:
                path.unlink(missing_ok=True)
            raise

    def _save_config(self, values):
        original = copy.deepcopy(dict(self.plugin.config))
        directory = self.store.root / "config-backups"
        directory.mkdir(exist_ok=True)
        backup = directory / f"{time.time_ns()}.json"
        backup.touch(mode=0o600)
        backup.write_text(json.dumps(original, ensure_ascii=False), encoding="utf-8")
        self.plugin.config.setdefault("features", {}).setdefault("selfie", {})[
            "reference_images"
        ] = values
        try:
            self.plugin.config.save_config()
        except Exception:
            self.plugin.config.clear()
            self.plugin.config.update(original)
            raise

    @staticmethod
    def validate_image(blob):
        if not blob or len(blob) > MAX_REFERENCE_BYTES:
            raise ValueError("图片需小于 20 MB")
        try:
            with Image.open(io.BytesIO(blob)) as image:
                if image.width * image.height > 40_000_000:
                    raise ValueError("图片像素过大")
                ext = {"PNG": "png", "JPEG": "jpg", "WEBP": "webp", "GIF": "gif"}.get(
                    image.format
                )
                if not ext:
                    raise ValueError("请使用 PNG, JPG, WebP 或 GIF 图片")
                image.load()
                return ext
        except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
            raise ValueError("图片文件无效") from exc

    async def change(self, ids, *, blob=None):
        """Remove selected bindings or replace one, serialized with chat commands."""
        if ids:
            self.store.validate_asset_ids(ids)
        if blob is not None and len(ids) > 1:
            raise ValueError("每次替换一张参考图")
        ext = self.validate_image(blob) if blob is not None else None
        async with self.plugin.refs._lock:
            self._migrate_config_paths()
            index = await self.plugin.refs._read_index()
            previous_index = copy.deepcopy(index)
            active = self.store.sync_references(self.sources(index))
            by_id = {a["id"]: a for a in active}
            if any(key not in by_id for key in ids):
                raise StudioConflict("参考图已更改或删除, 请刷新后重试")
            skipped = [
                {"id": key, "reason": "已收藏, 请先取消收藏"}
                for key in ids
                if by_id[key].get("favorite")
            ]
            if blob is not None and skipped:
                raise ValueError("请先取消收藏再更改参考图")
            selected = [by_id[key] for key in ids if not by_id[key].get("favorite")]
            values = copy.deepcopy(
                self.plugin._get_selfie_conf().get("reference_images", [])
            )
            previous_values = copy.deepcopy(values)
            if not isinstance(values, list):
                values = []
            config_changed = not ids and blob is not None
            index_changed = False
            new_path = None
            origin = selected[0]["reference_origin"] if selected else {"kind": "config"}
            if blob is not None:
                if origin["kind"] == "store":
                    new_path = self.plugin.refs.refs_dir / f"{uuid.uuid4().hex}.{ext}"
                    new_value = new_path.name
                else:
                    new_path, new_value = self._config_image_path(ext)
                new_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    new_path.touch(mode=0o600)
                    new_path.write_bytes(blob)
                except Exception:
                    new_path.unlink(missing_ok=True)
                    raise
                if not ids:
                    values.append(new_value)
            for asset in selected:
                source = asset["reference_origin"]
                entries = values if source["kind"] == "config" else index[source["key"]]
                entries[:] = [
                    new_value if value == source["path"] and blob is not None else value
                    for value in entries
                    if value != source["path"] or blob is not None
                ]
                config_changed |= source["kind"] == "config"
                index_changed |= source["kind"] == "store"
            try:
                # A batch deletion may affect both sources. Persist index first and
                # roll it back if saving the AstrBot config fails.
                if index_changed:
                    await self.plugin.refs._write_index(index)
                try:
                    if config_changed:
                        if (
                            self.plugin._get_selfie_conf().get("reference_images", [])
                            != previous_values
                        ):
                            raise StudioConflict("参考图配置已更改, 请刷新后重试")
                        self._save_config(values)
                except Exception:
                    if index_changed:
                        await self.plugin.refs._write_index(previous_index)
                    raise
            except Exception:
                if new_path:
                    new_path.unlink(missing_ok=True)
                raise
            current = self.store.sync_references(self.sources(index))
            if blob is not None:
                result = next(
                    a
                    for a in current
                    if a["reference_origin"]["path"] == new_value
                    and a["reference_origin"]["kind"] == origin["kind"]
                )
                return self.store.public_asset(result)
            return {"deleted": [asset["id"] for asset in selected], "skipped": skipped}
