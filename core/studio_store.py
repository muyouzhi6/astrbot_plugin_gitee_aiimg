"""Private studio assets and revisioned documents, outside the plugin checkout."""

from __future__ import annotations

import hashlib
import io
import json
import re
import sqlite3
import time
import threading
import uuid
from contextlib import contextmanager
from pathlib import Path

from PIL import Image, ImageOps


class StudioConflict(ValueError):
    pass


class StudioStore:
    def __init__(self, data_dir: Path):
        self.root = Path(data_dir) / "studio"
        self.media = self.root / "media"
        self.media.mkdir(parents=True, exist_ok=True)
        self.root.chmod(0o700)
        self.media.chmod(0o700)
        self.db_path = self.root / "studio.sqlite3"
        self.media_lock = threading.RLock()
        with self.connect() as db:
            db.executescript(
                "CREATE TABLE IF NOT EXISTS assets (id TEXT PRIMARY KEY, "
                "sha TEXT NOT NULL, kind TEXT NOT NULL, created REAL NOT NULL, data TEXT NOT NULL);"
                "CREATE INDEX IF NOT EXISTS assets_created ON assets(kind, created DESC);"
                "CREATE TABLE IF NOT EXISTS documents (kind TEXT NOT NULL, id TEXT NOT NULL, "
                "revision INTEGER NOT NULL, data TEXT NOT NULL, PRIMARY KEY(kind,id));"
            )

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.db_path, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA journal_mode=WAL")
        try:
            with db:
                yield db
        finally:
            db.close()

    def add_image(self, blob: bytes, *, kind="upload", metadata=None):
        with self.media_lock:
            return self._add_image(blob, kind=kind, metadata=metadata)

    def _add_image(self, blob: bytes, *, kind="upload", metadata=None):
        limit = 64 if kind == "history" else 20
        if not blob or len(blob) > limit * 1024 * 1024:
            raise ValueError(f"图片需小于 {limit} MB")
        with Image.open(io.BytesIO(blob)) as source:
            if source.width * source.height > 40_000_000:
                raise ValueError("图片像素过大")
            if source.format not in {"PNG", "JPEG", "WEBP", "GIF"}:
                raise ValueError("请使用 PNG, JPG, WebP 或 GIF 图片")
            source.load()
            fmt = source.format
            oriented = ImageOps.exif_transpose(source)
            width, height = oriented.size
            thumb = oriented.convert("RGB")
            preview = thumb.copy()
            preview.thumbnail((2048, 2048))
            preview_buffer = io.BytesIO()
            preview.save(preview_buffer, "JPEG", quality=92)
            thumb.thumbnail((640, 640))
            buf = io.BytesIO()
            thumb.save(buf, "JPEG", quality=85)
        digest = hashlib.sha256(blob).hexdigest()
        ext = {"PNG": "png", "JPEG": "jpg", "WEBP": "webp", "GIF": "gif"}[fmt]
        path = self.media / f"{digest}.{ext}"
        if not path.exists():
            temp = self.media / f".{uuid.uuid4().hex}.tmp"
            temp.write_bytes(blob)
            temp.replace(path)
        thumbnail = self.media / f"{digest}.thumb.jpg"
        if not thumbnail.exists():
            temp = self.media / f".{uuid.uuid4().hex}.tmp"
            temp.write_bytes(buf.getvalue())
            temp.replace(thumbnail)
        preview_path = self.media / f"{digest}.preview.jpg"
        if not preview_path.exists():
            temp = self.media / f".{uuid.uuid4().hex}.tmp"
            temp.write_bytes(preview_buffer.getvalue())
            temp.replace(preview_path)
        data = {
            **(metadata or {}),
            "id": "asset_" + uuid.uuid4().hex,
            "sha256": digest,
            "kind": kind,
            "width": width,
            "height": height,
            "bytes": len(blob),
            "file": path.name,
            "created": time.time(),
        }
        with self.connect() as db:
            db.execute(
                "INSERT INTO assets VALUES (?,?,?,?,?)",
                (
                    data["id"],
                    digest,
                    kind,
                    data["created"],
                    json.dumps(data, ensure_ascii=False),
                ),
            )
        return self.public_asset(data)

    @staticmethod
    def public_asset(data):
        return {k: v for k, v in data.items() if k != "file"}

    def asset(self, asset_id, *, include_deleted=False):
        with self.connect() as db:
            row = db.execute(
                "SELECT data FROM assets WHERE id=?", (asset_id,)
            ).fetchone()
        if not row:
            raise ValueError("图片不存在或已移除")
        data = json.loads(row[0])
        if data.get("deleted_at") and not include_deleted:
            raise ValueError("图片已删除")
        return data

    def available(self, asset_id):
        try:
            return bool(asset_id and self.asset(asset_id))
        except ValueError:
            return False

    def asset_path(self, asset_id, *, thumbnail=False):
        a = self.asset(asset_id)
        suffix = "preview" if thumbnail == "preview" else "thumb"
        p = self.media / (f"{a['sha256']}.{suffix}.jpg" if thumbnail else a["file"])
        if thumbnail == "preview" and not p.exists():
            original = self.media / a["file"]
            if original.parent.resolve() != self.media.resolve():
                raise ValueError("图片路径无效")
            with Image.open(original) as source:
                preview = ImageOps.exif_transpose(source).convert("RGB")
                preview.thumbnail((2048, 2048))
                temp = self.media / f".{uuid.uuid4().hex}.tmp"
                preview.save(temp, "JPEG", quality=92)
                temp.replace(p)
        if p.parent.resolve() != self.media.resolve() or not p.is_file():
            raise ValueError("原图文件不可用")
        return p

    def list_assets(self, *, kind="", offset=0, limit=48):
        offset, limit = max(0, int(offset)), max(1, min(100, int(limit)))
        query, args = (
            "SELECT data FROM assets WHERE json_extract(data,'$.deleted_at') IS NULL",
            [],
        )
        if kind == "favorite":
            query += " AND json_extract(data,'$.favorite')=1"
        elif kind:
            query += " AND kind=?"
            args.append(kind)
        with self.connect() as db:
            rows = db.execute(
                query + " ORDER BY created DESC LIMIT ? OFFSET ?",
                [*args, limit + 1, offset],
            ).fetchall()
        return {
            "items": [self.public_asset(json.loads(r[0])) for r in rows[:limit]],
            "more": len(rows) > limit,
        }

    def protected_assets(self, extra=()):
        protected = {key: "任务正在使用" for key in extra}
        for c in self.documents("character"):
            for look in c.get("looks", []):
                for key in look["assets"]:
                    protected[key] = "人物身份参考"
        for graph in self.documents("graph"):
            for node in graph.get("nodes", []):
                if node.get("type") == "image" and node.get("asset_id"):
                    protected[node["asset_id"]] = "工作流参考图"
        return protected

    def library_settings(self):
        return self.document("settings", "library") or {
            "id": "library",
            "revision": 0,
            "max_count": 0,
        }

    def library_status(self, extra=()):
        protected = self.protected_assets(extra)
        with self.connect() as db:
            assets = [
                json.loads(r[0])
                for r in db.execute(
                    "SELECT data FROM assets WHERE json_extract(data,'$.deleted_at') IS NULL"
                )
            ]
        return {
            **self.library_settings(),
            "count": len(assets),
            "favorites": sum(bool(a.get("favorite")) for a in assets),
            "protected": sum(
                bool(
                    a.get("favorite")
                    or a["kind"] == "reference"
                    or a["id"] in protected
                )
                for a in assets
            ),
            "bytes": sum(a["bytes"] for a in {a["sha256"]: a for a in assets}.values()),
        }

    def favorite(self, ids, value):
        if not isinstance(value, bool):
            raise ValueError("收藏状态无效")
        self.validate_asset_ids(ids)
        with self.media_lock, self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            rows = []
            for key in ids:
                a = self.asset(key)
                a["favorite"] = value
                rows.append(a)
            db.executemany(
                "UPDATE assets SET data=? WHERE id=?",
                [(json.dumps(a, ensure_ascii=False), a["id"]) for a in rows],
            )
        return {"updated": ids, "favorite": value}

    @staticmethod
    def validate_asset_ids(ids):
        if (
            not isinstance(ids, list)
            or not 1 <= len(ids) <= 500
            or any(not isinstance(x, str) for x in ids)
            or len(set(ids)) != len(ids)
        ):
            raise ValueError("请选择 1 至 500 张不同图片")

    def delete_assets(self, ids, *, extra=(), automatic=False):
        self.validate_asset_ids(ids)
        deleted, skipped = [], []
        with self.media_lock, self.connect() as db:
            protected = self.protected_assets(extra)
            db.execute("BEGIN IMMEDIATE")
            removed = []
            for key in ids:
                row = db.execute(
                    "SELECT data FROM assets WHERE id=?", (key,)
                ).fetchone()
                if not row:
                    skipped.append({"id": key, "reason": "图片不存在"})
                    continue
                a = json.loads(row[0])
                if a.get("deleted_at"):
                    continue
                reason = (
                    "已收藏, 请先取消收藏" if a.get("favorite") else protected.get(key)
                )
                if a["kind"] == "reference":
                    reason = reason or "原有参考图库正在使用"
                if automatic and a["created"] > time.time() - 300:
                    reason = reason or "新图片保留期"
                if reason:
                    skipped.append({"id": key, "reason": reason})
                    continue
                a["deleted_at"] = time.time()
                db.execute(
                    "UPDATE assets SET data=? WHERE id=?",
                    (json.dumps(a, ensure_ascii=False), key),
                )
                removed.append(a)
                deleted.append(key)
            gone = set(deleted)
            for row in db.execute(
                "SELECT id,data,revision FROM documents WHERE kind='workspace'"
            ).fetchall():
                doc = json.loads(row["data"])
                before = json.dumps(doc, sort_keys=True)
                doc["layers"] = [
                    layer
                    for layer in doc.get("layers", [])
                    if layer.get("asset_id") not in gone
                ]
                shoot = doc.get("shoot", {})
                shoot["picked"] = [
                    key for key in shoot.get("picked", []) if key not in gone
                ]
                if shoot.get("source") in gone:
                    shoot.update(source="", plan_id="", plan_shots=[])
                if json.dumps(doc, sort_keys=True) != before:
                    db.execute(
                        "UPDATE documents SET data=?,revision=revision+1 WHERE kind='workspace' AND id=?",
                        (json.dumps(doc, ensure_ascii=False), row["id"]),
                    )
            db.commit()
            for a in removed:
                if not db.execute(
                    "SELECT 1 FROM assets WHERE sha=? AND json_extract(data,'$.deleted_at') IS NULL LIMIT 1",
                    (a["sha256"],),
                ).fetchone():
                    for name in (
                        a["file"],
                        a["sha256"] + ".thumb.jpg",
                        a["sha256"] + ".preview.jpg",
                    ):
                        path = self.media / name
                        if path.parent.resolve() != self.media.resolve():
                            raise ValueError("图片路径无效")
                        try:
                            path.unlink(missing_ok=True)
                        except OSError:
                            pass
        return {"deleted": deleted, "skipped": skipped}

    def enforce_limit(self, extra=()):
        self.collect_deleted_files()
        limit = self.library_settings()["max_count"]
        if not limit:
            return {"deleted": [], "remaining_over_limit": 0}
        with self.connect() as db:
            assets = [
                json.loads(r[0])
                for r in db.execute(
                    "SELECT data FROM assets WHERE json_extract(data,'$.deleted_at') IS NULL ORDER BY created ASC"
                )
            ]
        excess = max(0, len(assets) - limit)
        protected = self.protected_assets(extra)
        candidates = [
            a["id"]
            for a in assets
            if not a.get("favorite")
            and a["kind"] != "reference"
            and a["id"] not in protected
            and a["created"] <= time.time() - 300
        ][:excess]
        deleted = []
        for i in range(0, len(candidates), 500):
            deleted.extend(
                self.delete_assets(
                    candidates[i : i + 500], extra=extra, automatic=True
                )["deleted"]
            )
        return {
            "deleted": deleted,
            "remaining_over_limit": max(0, excess - len(deleted)),
        }

    def collect_deleted_files(self):
        with self.media_lock, self.connect() as db:
            rows = db.execute(
                "SELECT data FROM assets a WHERE json_extract(data,'$.deleted_at') IS NOT NULL AND NOT EXISTS (SELECT 1 FROM assets b WHERE b.sha=a.sha AND json_extract(b.data,'$.deleted_at') IS NULL)"
            ).fetchall()
            for row in rows:
                a = json.loads(row[0])
                for name in (
                    a["file"],
                    a["sha256"] + ".thumb.jpg",
                    a["sha256"] + ".preview.jpg",
                ):
                    path = self.media / name
                    if path.parent.resolve() != self.media.resolve():
                        continue
                    try:
                        path.unlink(missing_ok=True)
                    except OSError:
                        pass

    def documents(self, kind):
        with self.connect() as db:
            return [
                dict(json.loads(r["data"]), id=r["id"], revision=r["revision"])
                for r in db.execute(
                    "SELECT * FROM documents WHERE kind=? ORDER BY rowid", (kind,)
                )
            ]

    def document(self, kind, doc_id):
        with self.connect() as db:
            row = db.execute(
                "SELECT revision,data FROM documents WHERE kind=? AND id=?",
                (kind, doc_id),
            ).fetchone()
        return (
            dict(json.loads(row["data"]), id=doc_id, revision=row["revision"])
            if row
            else None
        )

    def save_document(self, kind, value):
        with self.media_lock:
            if kind == "character":
                for look in value.get("looks", []):
                    for key in look.get("assets", []):
                        self.asset(key)
            if kind == "graph":
                for node in value.get("nodes", []):
                    if node.get("asset_id"):
                        self.asset(node["asset_id"])
            if kind == "workspace":
                for layer in value.get("layers", []):
                    self.asset(layer["asset_id"])
            return self._save_document(kind, value)

    def _save_document(self, kind, value):
        data = dict(value)
        doc_id = str(data.get("id") or uuid.uuid4().hex)
        if kind in {"character", "workspace"} and not re.fullmatch(
            r"[A-Za-z0-9_-]{1,100}", doc_id
        ):
            raise ValueError("记录标识无效")
        revision = int(data.pop("revision", 0))
        payload = json.dumps(data, ensure_ascii=False, allow_nan=False)
        if len(payload.encode("utf-8")) > 1024 * 1024:
            raise ValueError("内容过大")
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT revision FROM documents WHERE kind=? AND id=?", (kind, doc_id)
            ).fetchone()
            if (row[0] if row else 0) != revision:
                raise StudioConflict("内容已在另一处修改, 请刷新后重试")
            db.execute(
                "INSERT OR REPLACE INTO documents VALUES (?,?,?,?)",
                (kind, doc_id, revision + 1, payload),
            )
            if kind == "character":
                names = {look["name"] for look in data["looks"]}
                for row in db.execute(
                    "SELECT id,data FROM documents WHERE kind='appearance'"
                ).fetchall():
                    if (
                        row["id"].endswith(":" + doc_id)
                        and json.loads(row["data"])["look"] not in names
                    ):
                        db.execute(
                            "DELETE FROM documents WHERE kind='appearance' AND id=?",
                            (row["id"],),
                        )
        return dict(data, id=doc_id, revision=revision + 1)

    def delete_graph(self, key, revision):
        with self.media_lock, self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT revision FROM documents WHERE kind='graph' AND id=?", (key,)
            ).fetchone()
            if not row or row[0] != revision:
                raise StudioConflict("工作流已变化, 请刷新")
            db.execute("DELETE FROM documents WHERE kind='graph' AND id=?", (key,))
        return {"deleted": key}

    def delete_character(self, doc_id, revision):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT revision FROM documents WHERE kind='character' AND id=?",
                (doc_id,),
            ).fetchone()
            if not row or row[0] != revision:
                raise StudioConflict("人物已在另一处修改, 请刷新后重试")
            db.execute(
                "DELETE FROM documents WHERE kind='character' AND id=?", (doc_id,)
            )
            for row in db.execute(
                "SELECT id FROM documents WHERE kind='appearance'"
            ).fetchall():
                if row["id"].endswith(":" + doc_id):
                    db.execute(
                        "DELETE FROM documents WHERE kind='appearance' AND id=?",
                        (row["id"],),
                    )
        return {"id": doc_id, "deleted": True}

    def import_existing(self, data_dir, configured_refs):
        if self.document("migration", "legacy-v1"):
            return
        root = Path(data_dir).resolve()
        ref_paths = set()
        for value in configured_refs:
            path = (root / str(value)).resolve()
            if path.is_relative_to(root) and path.is_file():
                ref_paths.add(path)
        index = root / "refs/index.json"
        if index.is_file():
            for files in json.loads(index.read_text(encoding="utf-8-sig")).values():
                for filename in files:
                    path = (root / "refs" / filename).resolve()
                    if path.is_relative_to(root / "refs") and path.is_file():
                        ref_paths.add(path)
        for path in sorted(ref_paths):
            self.add_image(
                path.read_bytes(),
                kind="reference",
                metadata={"name": path.stem, "source": "legacy"},
            )
        # Old caches have no trustworthy prompt-to-file index. Label them honestly.
        for path in sorted(
            (root / "images").glob("*"), key=lambda p: p.stat().st_mtime, reverse=True
        ):
            if path.is_file() and path.suffix.lower() in {
                ".png",
                ".jpg",
                ".jpeg",
                ".webp",
            }:
                try:
                    self.add_image(
                        path.read_bytes(),
                        kind="history",
                        metadata={
                            "name": path.stem,
                            "prompt": "",
                            "source": "legacy",
                            "original_created": path.stat().st_mtime,
                        },
                    )
                except (ValueError, OSError):
                    continue
        self.save_document("migration", {"id": "legacy-v1"})
