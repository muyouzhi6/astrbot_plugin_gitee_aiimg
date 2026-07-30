"""Durable background task management for LLM image generation."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import shutil
import sqlite3
import time
import uuid
from collections import deque
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, TypeVar


SCHEMA_VERSION = 1
ACTIVE_STATES = {
    "preparing",
    "planning",
    "queued",
    "running",
    "generated",
    "sending",
}
TERMINAL_STATES = {
    "completed",
    "partial",
    "failed",
    "cancelled",
    "interrupted",
}
DELIVERY_STATES = {"not_started", "attempting", "confirmed", "unknown"}
NOTIFICATION_TERMINAL_STATES = {"sent", "unknown", "failed", "expired"}
PROMPT_LIMIT = 32768
RECORD_LIMIT_BYTES = 512 * 1024
INPUT_FILE_LIMIT_BYTES = 20 * 1024 * 1024
INPUT_TASK_LIMIT_BYTES = 64 * 1024 * 1024

_T = TypeVar("_T")


class BackgroundTaskError(RuntimeError):
    """Base error for durable background task operations."""


class BackgroundTaskCapacityError(BackgroundTaskError):
    """Raised when accepting a task would exceed configured capacity."""


class BackgroundTaskOwnerError(BackgroundTaskError):
    """Raised when another live plugin instance owns the task database."""


class BackgroundTaskStateError(BackgroundTaskError):
    """Raised for an invalid task state transition."""


@dataclass(frozen=True, slots=True)
class TaskDeliveryTarget:
    """Persistable platform routing data for a background image task.

    Args:
        platform_id: Unique AstrBot platform instance ID.
        platform_name: Adapter name, such as ``aiocqhttp``.
        message_type: AstrBot message type value.
        umo: Unified message origin.
        session_id: Platform session ID.
        group_id: Group ID, or an empty string for private messages.
        self_id: Bot account ID.
        sender_id: Original requester ID.
        sender_name: Original requester display name.
        source_message_id: Message that created the task.
        conversation_id: Conversation ID observed when the task was accepted.
    """

    platform_id: str
    platform_name: str
    message_type: str
    umo: str
    session_id: str
    group_id: str
    self_id: str
    sender_id: str
    sender_name: str
    source_message_id: str
    conversation_id: str = ""


@dataclass(frozen=True, slots=True)
class PreparedImageJob:
    """Event-independent input for one image provider call.

    Args:
        mode: Resolved execution mode.
        user_prompt: User-facing request text.
        effective_prompt: Exact prompt sent to the image provider.
        backend: Requested provider ID, or ``None`` for configured routing.
        output: Serialized output intent.
        input_paths: Spool files containing immutable image inputs.
        task_meta: Metadata used by existing follow-up behavior.
        options: Mode-specific provider options.
    """

    mode: str
    user_prompt: str
    effective_prompt: str
    backend: str | None
    output: dict[str, Any]
    input_paths: tuple[str, ...] = ()
    task_meta: dict[str, Any] = field(default_factory=dict)
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PreparedBatchJob:
    """Event-independent input for a background batch planner.

    Args:
        mode: Resolved execution mode shared by all children.
        user_prompt: Original batch request.
        requested_count: Number of child images reserved atomically.
        backend: Requested provider ID, or ``None`` for configured routing.
        output: Serialized common output intent.
        input_paths: Shared immutable image inputs.
        options: Planner and mode-specific options.
    """

    mode: str
    user_prompt: str
    requested_count: int
    backend: str | None
    output: dict[str, Any]
    input_paths: tuple[str, ...] = ()
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class _ScheduledWork:
    parent_id: str
    factory: Callable[[], Awaitable[Any]]
    future: asyncio.Future[Any]


class BackgroundImageTaskManager:
    """Own durable task records and bounded provider execution.

    The manager deliberately does not import AstrBot event classes. Callers
    provide immutable prepared jobs and delivery callbacks, so workers never
    retain an event after its pipeline turn has ended.

    Args:
        data_dir: Plugin data directory.
        max_running: Maximum simultaneous image provider calls.
        max_queued: Maximum reserved image slots across active tasks.
        log: Optional logger compatible with the standard logging API.
        lease_seconds: Owner heartbeat expiry.
        heartbeat_seconds: Owner heartbeat interval.
        terminal_ttl_seconds: Retention for terminal task records.
    """

    def __init__(
        self,
        data_dir: Path,
        *,
        max_running: int = 2,
        max_queued: int = 16,
        log: Any | None = None,
        lease_seconds: int = 45,
        heartbeat_seconds: int = 10,
        terminal_ttl_seconds: int = 24 * 60 * 60,
    ) -> None:
        self.base_dir = Path(data_dir) / "background_tasks"
        self.db_path = self.base_dir / "background_tasks.sqlite3"
        self.max_running = max(1, min(8, int(max_running)))
        self.max_queued = max(self.max_running, min(128, int(max_queued)))
        self.max_notification_backlog = max(32, self.max_queued * 4)
        self.log = log
        self.lease_seconds = max(15, int(lease_seconds))
        self.heartbeat_seconds = max(5, int(heartbeat_seconds))
        self.terminal_ttl_seconds = max(3600, int(terminal_ttl_seconds))

        self.owner_instance_id = f"plugin_{uuid.uuid4().hex}"
        self.owner_epoch = 0
        self.started = False
        self.accepting = False
        self._closing = False
        self._db_lock = asyncio.Lock()
        self._managed_tasks: set[asyncio.Task[Any]] = set()
        self._root_tasks: dict[str, asyncio.Task[Any]] = {}
        self._cancel_events: dict[str, asyncio.Event] = {}

        self._scheduler_lock = asyncio.Lock()
        self._ready_by_parent: dict[str, deque[_ScheduledWork]] = {}
        self._parent_ring: deque[str] = deque()
        self._parent_in_ring: set[str] = set()
        self._provider_running = 0
        self._planner_semaphore = asyncio.Semaphore(1)
        self._notification_map_lock = asyncio.Lock()
        self._notification_locks: dict[str, asyncio.Lock] = {}
        self._notification_lock_users: dict[str, int] = {}
        self._notification_events: dict[str, asyncio.Event] = {}
        self._health_failure_count = 0

    @property
    def is_closing(self) -> bool:
        """Return whether shutdown has stopped task intake."""

        return self._closing

    @staticmethod
    def now_ms() -> int:
        """Return the current UTC Unix time in milliseconds."""

        return int(time.time() * 1000)

    @staticmethod
    def new_task_id(kind: str = "img") -> str:
        """Create an opaque sortable-enough task identifier.

        Args:
            kind: Human-readable identifier prefix.

        Returns:
            A unique task ID.
        """

        return f"{kind}_{int(time.time() * 1000):013x}_{uuid.uuid4().hex[:12]}"

    @staticmethod
    def scope_hash(
        umo: str,
        self_id: str,
        sender_id: str,
        conversation_id: str = "",
    ) -> str:
        """Build the stable privacy-preserving task scope key.

        Args:
            umo: Unified message origin.
            self_id: Bot account ID.
            sender_id: Requester ID.
            conversation_id: Optional AstrBot conversation ID.

        Returns:
            SHA-256 scope digest.
        """

        raw = "\x1f".join((umo, self_id, sender_id, conversation_id))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    @staticmethod
    def request_fingerprint(
        scope_hash: str,
        source_message_id: str,
        normalized_args: dict[str, Any],
    ) -> str:
        """Build a deterministic duplicate-request fingerprint.

        Args:
            scope_hash: Stable task scope digest.
            source_message_id: Source platform message ID.
            normalized_args: Normalized tool arguments.

        Returns:
            SHA-256 fingerprint.
        """

        payload = json.dumps(normalized_args, ensure_ascii=False, sort_keys=True)
        raw = "\x1f".join((scope_hash, source_message_id, payload))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    @staticmethod
    def sanitize_error(error: BaseException | str, limit: int = 1000) -> str:
        """Remove credentials and oversized upstream content from an error.

        Args:
            error: Exception or raw error text.
            limit: Maximum returned character count.

        Returns:
            Sanitized error summary.
        """

        text = str(error or "unknown error")
        text = re.sub(
            r"(?i)(authorization\s*[:=]\s*(?:bearer\s+)?)[^\s,;]+",
            r"\1[REDACTED]",
            text,
        )
        text = re.sub(
            r"(?i)((?:api[_-]?key|access[_-]?token|secret)\s*[:=]\s*)[^\s,;]+",
            r"\1[REDACTED]",
            text,
        )
        text = re.sub(r"(https?://[^\s?]+)\?[^\s]+", r"\1?[REDACTED]", text)
        return text[: max(64, int(limit))]

    async def start(self) -> list[dict[str, Any]]:
        """Initialize storage, acquire the owner lease, and recover tasks.

        Returns:
            Task records recovered into an interrupted terminal state.

        Raises:
            BackgroundTaskOwnerError: Another live owner holds the database.
            BackgroundTaskError: Storage integrity or schema setup failed.
        """

        if self.started:
            return []
        await asyncio.to_thread(self._prepare_storage)
        async with self._db_lock:
            recovered = await asyncio.to_thread(self._acquire_owner_and_recover)
        self.started = True
        self.accepting = True
        self._track(self._heartbeat_loop(), name="background-image-heartbeat")
        self._track(self._gc_loop(), name="background-image-gc")
        self._track(self._health_loop(), name="background-image-health")
        return recovered

    def _prepare_storage(self) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(self.base_dir, 0o700)
        except OSError:
            pass

        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "CREATE TABLE IF NOT EXISTS schema_meta ("
                "singleton INTEGER PRIMARY KEY CHECK(singleton=1), "
                "schema_version INTEGER NOT NULL)"
            )
            row = conn.execute(
                "SELECT schema_version FROM schema_meta WHERE singleton=1"
            ).fetchone()
            if row is not None and int(row[0]) > SCHEMA_VERSION:
                conn.rollback()
                raise BackgroundTaskError(
                    f"Unsupported task database schema version: {row[0]}"
                )
            conn.execute(
                "INSERT INTO schema_meta(singleton, schema_version) VALUES(1, ?) "
                "ON CONFLICT(singleton) DO UPDATE SET schema_version=excluded.schema_version",
                (SCHEMA_VERSION,),
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS runtime_owner ("
                "singleton INTEGER PRIMARY KEY CHECK(singleton=1), "
                "owner_instance_id TEXT NOT NULL, owner_epoch INTEGER NOT NULL, "
                "heartbeat_at_ms INTEGER NOT NULL, state TEXT NOT NULL)"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS tasks ("
                "task_id TEXT PRIMARY KEY, scope_hash TEXT NOT NULL, "
                "state TEXT NOT NULL, task_kind TEXT NOT NULL, "
                "owner_epoch INTEGER NOT NULL, request_fingerprint TEXT NOT NULL, "
                "record_json TEXT NOT NULL, created_at_ms INTEGER NOT NULL, "
                "updated_at_ms INTEGER NOT NULL, expires_at_ms INTEGER NOT NULL)"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS reservations ("
                "task_id TEXT PRIMARY KEY REFERENCES tasks(task_id) ON DELETE CASCADE, "
                "total INTEGER NOT NULL, remaining INTEGER NOT NULL, released INTEGER NOT NULL DEFAULT 0)"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS receipts ("
                "send_attempt_id TEXT PRIMARY KEY, "
                "task_id TEXT NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE, "
                "item_id TEXT NOT NULL DEFAULT '', kind TEXT NOT NULL, "
                "delivery_state TEXT NOT NULL, transport TEXT NOT NULL, "
                "response_digest TEXT NOT NULL DEFAULT '', created_at_ms INTEGER NOT NULL)"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS notification_outbox ("
                "token TEXT PRIMARY KEY, "
                "task_id TEXT NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE, "
                "kind TEXT NOT NULL, state TEXT NOT NULL, attempt_id TEXT NOT NULL DEFAULT '', "
                "payload_json TEXT NOT NULL, queued_at_ms INTEGER NOT NULL, updated_at_ms INTEGER NOT NULL)"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS request_dedupe ("
                "request_fingerprint TEXT PRIMARY KEY, task_id TEXT NOT NULL, expires_at_ms INTEGER NOT NULL)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_tasks_scope_updated "
                "ON tasks(scope_hash, updated_at_ms DESC)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_tasks_state_updated "
                "ON tasks(state, updated_at_ms DESC)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_tasks_expires ON tasks(expires_at_ms)"
            )
            conn.commit()

            check = conn.execute("PRAGMA quick_check").fetchone()
            if check is None or str(check[0]).lower() != "ok":
                raise BackgroundTaskError(f"Task database quick_check failed: {check}")
        try:
            os.chmod(self.db_path, 0o600)
        except OSError:
            pass

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=5, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=FULL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def _acquire_owner_and_recover(self) -> list[dict[str, Any]]:
        now = self.now_ms()
        lease_cutoff = now - self.lease_seconds * 1000
        recovered: list[dict[str, Any]] = []
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            owner = conn.execute(
                "SELECT owner_instance_id, owner_epoch, heartbeat_at_ms, state "
                "FROM runtime_owner WHERE singleton=1"
            ).fetchone()
            if (
                owner is not None
                and owner["owner_instance_id"] != self.owner_instance_id
                and owner["state"] in {"active", "draining"}
                and int(owner["heartbeat_at_ms"]) >= lease_cutoff
            ):
                conn.rollback()
                raise BackgroundTaskOwnerError(
                    "Another live Gitee background task owner holds the database"
                )

            self.owner_epoch = int(owner["owner_epoch"] if owner else 0) + 1
            conn.execute(
                "INSERT INTO runtime_owner(singleton, owner_instance_id, owner_epoch, heartbeat_at_ms, state) "
                "VALUES(1, ?, ?, ?, 'active') "
                "ON CONFLICT(singleton) DO UPDATE SET "
                "owner_instance_id=excluded.owner_instance_id, "
                "owner_epoch=excluded.owner_epoch, heartbeat_at_ms=excluded.heartbeat_at_ms, state='active'",
                (self.owner_instance_id, self.owner_epoch, now),
            )

            rows = conn.execute(
                "SELECT task_id, state, record_json FROM tasks WHERE state IN ("
                + ",".join("?" for _ in ACTIVE_STATES)
                + ")",
                tuple(sorted(ACTIVE_STATES)),
            ).fetchall()
            for row in rows:
                record = json.loads(row["record_json"])
                old_state = str(row["state"])
                record["state"] = "interrupted"
                record["terminal_reason"] = "process_restarted"
                record["error_code"] = "process_restarted"
                record["error_message"] = (
                    "The image task was interrupted by a plugin or AstrBot restart."
                )
                items = record.get("items")
                if record.get("task_kind") == "batch" and isinstance(items, list):
                    for item in items:
                        item_state = str(item.get("state") or "queued")
                        if item_state in {
                            "completed",
                            "failed",
                            "cancelled",
                            "unknown",
                        }:
                            continue
                        if (
                            item_state == "sending"
                            or item.get("delivery_state") == "attempting"
                        ):
                            item.update(
                                {
                                    "state": "unknown",
                                    "delivery_state": "unknown",
                                    "image_sent": False,
                                    "error_code": "delivery_unknown",
                                    "error_message": (
                                        "The process restarted while image delivery was in progress."
                                    ),
                                }
                            )
                        else:
                            item.update(
                                {
                                    "state": "cancelled",
                                    "image_sent": False,
                                    "error_code": "process_restarted",
                                    "error_message": (
                                        "The process restarted before this batch item was delivered."
                                    ),
                                }
                            )
                    self._refresh_batch_counts(record)
                    record["image_generated"] = bool(record.get("generated_count"))
                    record["image_sent"] = bool(record.get("sent_count"))
                    if int(record.get("unknown_count") or 0) > 0:
                        record["delivery_state"] = "unknown"
                    elif int(record.get("sent_count") or 0) > 0:
                        record["delivery_state"] = "confirmed"
                    else:
                        record["delivery_state"] = "not_started"
                elif (
                    old_state == "sending"
                    or record.get("delivery_state") == "attempting"
                ):
                    record["delivery_state"] = "unknown"
                record["owner_instance_id"] = self.owner_instance_id
                record["owner_epoch"] = self.owner_epoch
                record["updated_at"] = now
                record["finished_at"] = now
                record["notification_state"] = "pending"
                token = str(record.get("notification_token") or uuid.uuid4().hex)
                record["notification_token"] = token
                payload = self._encode_record(record)
                conn.execute(
                    "UPDATE tasks SET state='interrupted', owner_epoch=?, record_json=?, "
                    "updated_at_ms=?, expires_at_ms=? WHERE task_id=?",
                    (
                        self.owner_epoch,
                        payload,
                        now,
                        now + self.terminal_ttl_seconds * 1000,
                        row["task_id"],
                    ),
                )
                conn.execute(
                    "UPDATE receipts SET delivery_state='unknown' "
                    "WHERE task_id=? AND delivery_state='attempting'",
                    (row["task_id"],),
                )
                conn.execute(
                    "UPDATE reservations SET remaining=0, released=1 WHERE task_id=?",
                    (row["task_id"],),
                )
                conn.execute(
                    "INSERT OR IGNORE INTO notification_outbox("
                    "token, task_id, kind, state, payload_json, queued_at_ms, updated_at_ms"
                    ") VALUES(?, ?, 'terminal', 'pending', ?, ?, ?)",
                    (token, row["task_id"], payload, now, now),
                )
                recovered.append(record)

            pending_rows = conn.execute(
                "SELECT o.token, o.state, t.record_json FROM notification_outbox o "
                "JOIN tasks t ON t.task_id=o.task_id "
                "WHERE o.state='pending'"
            ).fetchall()
            recovered_ids = {str(record.get("task_id") or "") for record in recovered}
            for row in pending_rows:
                record = json.loads(row["record_json"])
                task_id = str(record.get("task_id") or "")
                if not task_id or task_id in recovered_ids:
                    continue
                record["notification_state"] = "pending"
                record["notification_token"] = str(row["token"])
                record["owner_instance_id"] = self.owner_instance_id
                record["owner_epoch"] = self.owner_epoch
                payload = self._encode_record(record)
                conn.execute(
                    "UPDATE tasks SET owner_epoch=?, record_json=?, updated_at_ms=? WHERE task_id=?",
                    (self.owner_epoch, payload, now, task_id),
                )
                conn.execute(
                    "UPDATE notification_outbox SET payload_json=?, updated_at_ms=? "
                    "WHERE token=? AND state='pending'",
                    (payload, now, row["token"]),
                )
                recovered.append(record)
                recovered_ids.add(task_id)
            ambiguous_rows = conn.execute(
                "SELECT o.token, o.task_id, t.record_json FROM notification_outbox o "
                "JOIN tasks t ON t.task_id=o.task_id "
                "WHERE o.state IN ('queued', 'claimed')"
            ).fetchall()
            for row in ambiguous_rows:
                record = json.loads(row["record_json"])
                record["notification_state"] = "unknown"
                record["notification_token"] = str(row["token"])
                record["owner_instance_id"] = self.owner_instance_id
                record["owner_epoch"] = self.owner_epoch
                conn.execute(
                    "UPDATE notification_outbox SET state='unknown', updated_at_ms=? WHERE token=?",
                    (now, row["token"]),
                )
                conn.execute(
                    "UPDATE tasks SET owner_epoch=?, record_json=?, updated_at_ms=? WHERE task_id=?",
                    (
                        self.owner_epoch,
                        self._encode_record(record),
                        now,
                        row["task_id"],
                    ),
                )
            conn.commit()
        return recovered

    async def create_task_record(
        self,
        record: dict[str, Any],
        *,
        reservation: int,
        dedupe_seconds: int = 600,
    ) -> tuple[dict[str, Any], bool]:
        """Atomically reserve capacity and insert a new task record.

        Args:
            record: Versioned JSON-compatible task record.
            reservation: Provider work slots reserved by the task.
            dedupe_seconds: Fingerprint retention period.

        Returns:
            A ``(record, created)`` tuple. A duplicate returns the original
            record with ``created=False``.

        Raises:
            BackgroundTaskCapacityError: The bounded queue is full.
            BackgroundTaskOwnerError: The owner lease is no longer valid.
            BackgroundTaskError: The record is invalid or too large.
        """

        if not self.started or not self.accepting or self._closing:
            raise BackgroundTaskError("Background task manager is not accepting work")
        reservation = max(1, int(reservation))
        now = self.now_ms()
        record = dict(record)
        task_id = str(record.get("task_id") or self.new_task_id())
        task_kind = str(record.get("task_kind") or "single")
        state = str(record.get("state") or "queued")
        if state not in ACTIVE_STATES:
            raise BackgroundTaskStateError(f"Invalid initial state: {state}")
        for key in ("user_prompt", "effective_prompt"):
            if len(str(record.get(key) or "")) > PROMPT_LIMIT:
                raise BackgroundTaskError(f"{key} exceeds {PROMPT_LIMIT} characters")

        record.update(
            {
                "schema_version": SCHEMA_VERSION,
                "revision": int(record.get("revision") or 0) + 1,
                "task_id": task_id,
                "task_kind": task_kind,
                "state": state,
                "owner_instance_id": self.owner_instance_id,
                "owner_epoch": self.owner_epoch,
                "created_at": int(record.get("created_at") or now),
                "updated_at": now,
                "started_at": int(record.get("started_at") or 0),
                "finished_at": 0,
                "notification_token": str(
                    record.get("notification_token") or f"notify_{uuid.uuid4().hex}"
                ),
                "notification_state": str(
                    record.get("notification_state") or "pending"
                ),
                "ack_token": str(record.get("ack_token") or f"ack_{uuid.uuid4().hex}"),
                "ack_state": str(record.get("ack_state") or "pending"),
            }
        )
        scope = str(record.get("scope_hash") or "")
        fingerprint = str(record.get("request_fingerprint") or "")
        if not scope or not fingerprint:
            raise BackgroundTaskError(
                "Task scope_hash and request_fingerprint are required"
            )
        payload = self._encode_record(record)

        async with self._db_lock:
            result = await asyncio.to_thread(
                self._insert_task_transaction,
                record,
                payload,
                reservation,
                now,
                max(60, int(dedupe_seconds)),
            )
        self._cancel_events.setdefault(task_id, asyncio.Event())
        return result

    def _insert_task_transaction(
        self,
        record: dict[str, Any],
        payload: str,
        reservation: int,
        now: int,
        dedupe_seconds: int,
    ) -> tuple[dict[str, Any], bool]:
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            self._assert_owner(conn, now)
            conn.execute("DELETE FROM request_dedupe WHERE expires_at_ms < ?", (now,))
            duplicate = conn.execute(
                "SELECT task_id FROM request_dedupe WHERE request_fingerprint=?",
                (record["request_fingerprint"],),
            ).fetchone()
            if duplicate is not None:
                row = conn.execute(
                    "SELECT record_json FROM tasks WHERE task_id=?",
                    (duplicate["task_id"],),
                ).fetchone()
                conn.commit()
                if row is not None:
                    return json.loads(row["record_json"]), False

            reserved = conn.execute(
                "SELECT COALESCE(SUM(remaining), 0) AS total FROM reservations WHERE released=0"
            ).fetchone()
            current = int(reserved["total"] if reserved else 0)
            if current + reservation > self.max_queued:
                conn.rollback()
                raise BackgroundTaskCapacityError(
                    f"Background image queue is full ({current}/{self.max_queued})"
                )
            notification_backlog = int(
                conn.execute(
                    "SELECT COUNT(*) FROM notification_outbox "
                    "WHERE state IN ('pending', 'queued', 'claimed')"
                ).fetchone()[0]
            )
            if notification_backlog >= self.max_notification_backlog:
                conn.rollback()
                raise BackgroundTaskCapacityError(
                    "Background notification queue is full "
                    f"({notification_backlog}/{self.max_notification_backlog})"
                )
            task_id = record["task_id"]
            expires = now + 2 * 60 * 60 * 1000
            conn.execute(
                "INSERT INTO tasks(task_id, scope_hash, state, task_kind, owner_epoch, "
                "request_fingerprint, record_json, created_at_ms, updated_at_ms, expires_at_ms) "
                "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    task_id,
                    record["scope_hash"],
                    record["state"],
                    record["task_kind"],
                    self.owner_epoch,
                    record["request_fingerprint"],
                    payload,
                    record["created_at"],
                    now,
                    expires,
                ),
            )
            conn.execute(
                "INSERT INTO reservations(task_id, total, remaining, released) VALUES(?, ?, ?, 0)",
                (task_id, reservation, reservation),
            )
            conn.execute(
                "INSERT INTO request_dedupe(request_fingerprint, task_id, expires_at_ms) VALUES(?, ?, ?)",
                (
                    record["request_fingerprint"],
                    task_id,
                    now + dedupe_seconds * 1000,
                ),
            )
            conn.commit()
        return record, True

    async def transition(
        self,
        task_id: str,
        state: str,
        updates: dict[str, Any] | None = None,
        *,
        release: int | None = None,
        queue_notification: bool = False,
    ) -> dict[str, Any]:
        """Apply one fenced state update and optional capacity release.

        Args:
            task_id: Task to update.
            state: New state.
            updates: Additional record fields.
            release: Reserved child slots to release. Terminal parent updates
                release all remaining capacity when omitted.
            queue_notification: Whether to atomically insert the terminal outbox.

        Returns:
            Updated task record.

        Raises:
            BackgroundTaskStateError: The transition is invalid.
            BackgroundTaskOwnerError: The owner epoch no longer matches.
        """

        updates = dict(updates or {})
        now = self.now_ms()
        async with self._db_lock:
            record = await asyncio.to_thread(
                self._transition_transaction,
                task_id,
                state,
                updates,
                release,
                queue_notification,
                now,
            )
        if state in TERMINAL_STATES:
            event = self._cancel_events.get(task_id)
            if state in {"cancelled", "interrupted"} and event is not None:
                event.set()
        return record

    def _transition_transaction(
        self,
        task_id: str,
        state: str,
        updates: dict[str, Any],
        release: int | None,
        queue_notification: bool,
        now: int,
    ) -> dict[str, Any]:
        if state not in ACTIVE_STATES | TERMINAL_STATES:
            raise BackgroundTaskStateError(f"Unknown task state: {state}")
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            self._assert_owner(conn, now)
            row = conn.execute(
                "SELECT state, record_json, owner_epoch FROM tasks WHERE task_id=?",
                (task_id,),
            ).fetchone()
            if row is None:
                conn.rollback()
                raise BackgroundTaskError(f"Unknown task: {task_id}")
            if int(row["owner_epoch"]) != self.owner_epoch:
                conn.rollback()
                raise BackgroundTaskOwnerError("Task owner epoch no longer matches")
            old_state = str(row["state"])
            if old_state in TERMINAL_STATES:
                if old_state == state:
                    record = json.loads(row["record_json"])
                    if not updates and not queue_notification:
                        conn.commit()
                        return record
                    record.update(updates)
                    insert_notification = False
                    if queue_notification:
                        token = str(
                            record.get("notification_token")
                            or f"notify_{uuid.uuid4().hex}"
                        )
                        record["notification_token"] = token
                        existing = conn.execute(
                            "SELECT state FROM notification_outbox WHERE token=?",
                            (token,),
                        ).fetchone()
                        if existing is None:
                            record["notification_state"] = "pending"
                            insert_notification = True
                        else:
                            record["notification_state"] = str(existing["state"])
                    self._validate_terminal_record(record, state)
                    record["revision"] = int(record.get("revision") or 0) + 1
                    record["updated_at"] = now
                    payload = self._encode_record(record)
                    changed = conn.execute(
                        "UPDATE tasks SET record_json=?, updated_at_ms=? "
                        "WHERE task_id=? AND owner_epoch=?",
                        (payload, now, task_id, self.owner_epoch),
                    ).rowcount
                    if changed != 1:
                        conn.rollback()
                        raise BackgroundTaskOwnerError(
                            "Fenced terminal task update was rejected"
                        )
                    if insert_notification:
                        conn.execute(
                            "INSERT INTO notification_outbox("
                            "token, task_id, kind, state, payload_json, queued_at_ms, updated_at_ms"
                            ") VALUES(?, ?, 'terminal', 'pending', ?, ?, ?)",
                            (record["notification_token"], task_id, payload, now, now),
                        )
                    elif queue_notification and str(existing["state"]) in {
                        "pending",
                        "queued",
                    }:
                        conn.execute(
                            "UPDATE notification_outbox SET payload_json=?, updated_at_ms=? "
                            "WHERE token=? AND state IN ('pending', 'queued')",
                            (payload, now, record["notification_token"]),
                        )
                    conn.commit()
                    return record
                conn.rollback()
                raise BackgroundTaskStateError(
                    f"Terminal task cannot move from {old_state} to {state}"
                )

            record = json.loads(row["record_json"])
            record.update(updates)
            record["state"] = state
            record["revision"] = int(record.get("revision") or 0) + 1
            record["updated_at"] = now
            if state == "running" and not record.get("started_at"):
                record["started_at"] = now
            if state in TERMINAL_STATES:
                record["finished_at"] = now
                self._validate_terminal_record(record, state)
            payload = self._encode_record(record)
            expires = now + (
                self.terminal_ttl_seconds * 1000
                if state in TERMINAL_STATES
                else 2 * 60 * 60 * 1000
            )
            changed = conn.execute(
                "UPDATE tasks SET state=?, record_json=?, updated_at_ms=?, expires_at_ms=? "
                "WHERE task_id=? AND owner_epoch=?",
                (state, payload, now, expires, task_id, self.owner_epoch),
            ).rowcount
            if changed != 1:
                conn.rollback()
                raise BackgroundTaskOwnerError("Fenced task update was rejected")

            reservation_row = conn.execute(
                "SELECT remaining FROM reservations WHERE task_id=?", (task_id,)
            ).fetchone()
            remaining = int(reservation_row["remaining"] if reservation_row else 0)
            release_count = 0
            if release is not None:
                release_count = min(remaining, max(0, int(release)))
            elif state in TERMINAL_STATES:
                release_count = remaining
            if release_count:
                remaining -= release_count
                conn.execute(
                    "UPDATE reservations SET remaining=?, released=? WHERE task_id=?",
                    (remaining, 1 if remaining == 0 else 0, task_id),
                )

            if queue_notification and state in TERMINAL_STATES:
                token = str(
                    record.get("notification_token") or f"notify_{uuid.uuid4().hex}"
                )
                record["notification_token"] = token
                record["notification_state"] = "pending"
                payload = self._encode_record(record)
                conn.execute(
                    "UPDATE tasks SET record_json=? WHERE task_id=?",
                    (payload, task_id),
                )
                conn.execute(
                    "INSERT OR IGNORE INTO notification_outbox("
                    "token, task_id, kind, state, payload_json, queued_at_ms, updated_at_ms"
                    ") VALUES(?, ?, 'terminal', 'pending', ?, ?, ?)",
                    (token, task_id, payload, now, now),
                )
            conn.commit()
        return record

    @staticmethod
    def _validate_terminal_record(record: dict[str, Any], state: str) -> None:
        """Reject terminal summaries that overstate confirmed image delivery.

        Args:
            record: Candidate durable task record.
            state: Candidate terminal parent state.

        Raises:
            BackgroundTaskStateError: The terminal facts contradict the state.
        """

        if state == "completed":
            if record.get("task_kind") == "batch":
                requested = int(record.get("requested_count") or 0)
                sent = int(record.get("sent_count") or 0)
                unknown = int(record.get("unknown_count") or 0)
                valid = requested > 0 and sent == requested and unknown == 0
            else:
                valid = (
                    bool(record.get("image_generated"))
                    and bool(record.get("image_sent"))
                    and record.get("delivery_state") == "confirmed"
                )
            if not valid:
                raise BackgroundTaskStateError(
                    "Completed tasks require confirmed delivery for every image"
                )
        if state == "partial" and record.get("task_kind") == "batch":
            requested = int(record.get("requested_count") or 0)
            sent = int(record.get("sent_count") or 0)
            unknown = int(record.get("unknown_count") or 0)
            if not (0 < sent < requested and unknown == 0):
                raise BackgroundTaskStateError(
                    "Partial batches require known confirmed and unsent child results"
                )

    async def update_item(
        self,
        task_id: str,
        item_id: str,
        updates: dict[str, Any],
        *,
        release_if_terminal: bool = False,
    ) -> dict[str, Any]:
        """Update one batch item and release its reservation once.

        Args:
            task_id: Batch parent task ID.
            item_id: Stable child item ID.
            updates: Child fields to merge.
            release_if_terminal: Release one capacity slot if this call moves
                the child from active to terminal.

        Returns:
            Updated parent record.
        """

        now = self.now_ms()
        async with self._db_lock:
            return await asyncio.to_thread(
                self._update_item_transaction,
                task_id,
                item_id,
                dict(updates),
                release_if_terminal,
                now,
            )

    def _update_item_transaction(
        self,
        task_id: str,
        item_id: str,
        updates: dict[str, Any],
        release_if_terminal: bool,
        now: int,
    ) -> dict[str, Any]:
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            self._assert_owner(conn, now)
            row = conn.execute(
                "SELECT state, record_json, owner_epoch FROM tasks WHERE task_id=?",
                (task_id,),
            ).fetchone()
            if row is None:
                conn.rollback()
                raise BackgroundTaskError(f"Unknown task: {task_id}")
            if int(row["owner_epoch"]) != self.owner_epoch:
                conn.rollback()
                raise BackgroundTaskOwnerError("Task owner epoch no longer matches")
            if str(row["state"]) in TERMINAL_STATES:
                conn.rollback()
                raise BackgroundTaskStateError(
                    "Terminal parent items cannot be updated"
                )
            record = json.loads(row["record_json"])
            items = record.get("items")
            if not isinstance(items, list):
                items = []
            item = next(
                (
                    candidate
                    for candidate in items
                    if candidate.get("item_id") == item_id
                ),
                None,
            )
            if item is None:
                conn.rollback()
                raise BackgroundTaskError(f"Unknown batch item: {item_id}")
            old_state = str(item.get("state") or "planned")
            new_state = str(updates.get("state") or old_state)
            child_terminal = {"completed", "failed", "cancelled", "unknown"}
            if old_state in child_terminal and new_state != old_state:
                conn.rollback()
                raise BackgroundTaskStateError(
                    f"Terminal item cannot move from {old_state} to {new_state}"
                )
            item.update(updates)
            record["revision"] = int(record.get("revision") or 0) + 1
            record["updated_at"] = now
            self._refresh_batch_counts(record)
            payload = self._encode_record(record)
            changed = conn.execute(
                "UPDATE tasks SET record_json=?, updated_at_ms=? WHERE task_id=? AND owner_epoch=?",
                (payload, now, task_id, self.owner_epoch),
            ).rowcount
            if changed != 1:
                conn.rollback()
                raise BackgroundTaskOwnerError(
                    "Batch parent changed before the item update was persisted"
                )
            if (
                release_if_terminal
                and old_state not in child_terminal
                and new_state in child_terminal
            ):
                reservation = conn.execute(
                    "SELECT remaining FROM reservations WHERE task_id=?", (task_id,)
                ).fetchone()
                remaining = max(
                    0, int(reservation["remaining"] if reservation else 0) - 1
                )
                conn.execute(
                    "UPDATE reservations SET remaining=?, released=? WHERE task_id=?",
                    (remaining, 1 if remaining == 0 else 0, task_id),
                )
            conn.commit()
        return record

    @staticmethod
    def _refresh_batch_counts(record: dict[str, Any]) -> None:
        items = record.get("items") if isinstance(record.get("items"), list) else []
        record["planned_count"] = len(items)
        record["generated_count"] = sum(
            1 for item in items if bool(item.get("image_generated"))
        )
        record["sent_count"] = sum(1 for item in items if bool(item.get("image_sent")))
        record["failed_count"] = sum(
            1 for item in items if item.get("state") == "failed"
        )
        record["cancelled_count"] = sum(
            1 for item in items if item.get("state") == "cancelled"
        )
        record["unknown_count"] = sum(
            1 for item in items if item.get("state") == "unknown"
        )
        record["image_generated"] = record["generated_count"] > 0
        record["image_sent"] = record["sent_count"] > 0

    async def get_task(self, task_id: str) -> dict[str, Any] | None:
        """Read one task record by ID."""

        async with self._db_lock:
            return await asyncio.to_thread(self._get_task_sync, task_id)

    def _get_task_sync(self, task_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT record_json FROM tasks WHERE task_id=?", (task_id,)
            ).fetchone()
        return json.loads(row["record_json"]) if row else None

    async def list_scope_tasks(
        self,
        scope_hash: str,
        *,
        limit: int = 16,
        include_suppressed: bool = False,
    ) -> list[dict[str, Any]]:
        """List recent tasks for one requester scope.

        Args:
            scope_hash: Stable task scope digest.
            limit: Maximum records returned.
            include_suppressed: Include reset/new terminal records.

        Returns:
            Newest task records first.
        """

        async with self._db_lock:
            records = await asyncio.to_thread(
                self._list_scope_tasks_sync, scope_hash, max(1, min(32, int(limit)))
            )
        if include_suppressed:
            return records
        return [r for r in records if not r.get("suppress_future_injection")]

    def _list_scope_tasks_sync(
        self, scope_hash: str, limit: int
    ) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT record_json FROM tasks WHERE scope_hash=? "
                "ORDER BY updated_at_ms DESC LIMIT ?",
                (scope_hash, limit),
            ).fetchall()
        return [json.loads(row["record_json"]) for row in rows]

    async def list_active_for_umo(self, umo: str) -> list[dict[str, Any]]:
        """List active task records for an AstrBot conversation."""

        async with self._db_lock:
            return await asyncio.to_thread(self._list_active_for_umo_sync, umo)

    def _list_active_for_umo_sync(self, umo: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT record_json FROM tasks WHERE state IN ("
                + ",".join("?" for _ in ACTIVE_STATES)
                + ") ORDER BY updated_at_ms DESC LIMIT 32",
                tuple(sorted(ACTIVE_STATES)),
            ).fetchall()
        records = [json.loads(row["record_json"]) for row in rows]
        return [record for record in records if record.get("umo") == umo]

    async def cancel_task(
        self,
        task_id: str,
        reason: str,
        *,
        suppress_future_injection: bool = False,
    ) -> bool:
        """Persist cancellation and cancel the matching root worker.

        Args:
            task_id: Task to cancel.
            reason: User-visible cancellation reason.
            suppress_future_injection: Hide this terminal record from a new
                conversation created by reset/new.

        Returns:
            Whether an active task was cancelled.
        """

        now = self.now_ms()
        async with self._db_lock:
            record = await asyncio.shield(
                asyncio.to_thread(
                    self._cancel_task_transaction,
                    task_id,
                    self.sanitize_error(reason, 500),
                    suppress_future_injection,
                    now,
                )
            )
        if record is None:
            return False
        event = self._cancel_events.setdefault(task_id, asyncio.Event())
        event.set()
        worker = self._root_tasks.get(task_id)
        if worker is not None and not worker.done():
            worker.cancel()
        await self._drop_parent_work(task_id)
        return True

    def _cancel_task_transaction(
        self,
        task_id: str,
        reason: str,
        suppress_future_injection: bool,
        now: int,
    ) -> dict[str, Any] | None:
        """Atomically cancel a task using the latest item and receipt state.

        Args:
            task_id: Task to cancel.
            reason: Sanitized cancellation reason.
            suppress_future_injection: Hide the task from a reset/new conversation.
            now: Cancellation timestamp in milliseconds.

        Returns:
            The terminal task record, or ``None`` when it was already terminal.
        """

        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            self._assert_owner(conn, now)
            row = conn.execute(
                "SELECT state, record_json, owner_epoch FROM tasks WHERE task_id=?",
                (task_id,),
            ).fetchone()
            if row is None or str(row["state"]) in TERMINAL_STATES:
                conn.commit()
                return None
            if int(row["owner_epoch"]) != self.owner_epoch:
                conn.rollback()
                raise BackgroundTaskOwnerError("Task owner epoch no longer matches")

            record = json.loads(row["record_json"])
            receipt_states = {
                str(receipt["send_attempt_id"]): str(receipt["delivery_state"])
                for receipt in conn.execute(
                    "SELECT send_attempt_id, delivery_state FROM receipts WHERE task_id=?",
                    (task_id,),
                ).fetchall()
            }
            items = record.get("items")
            if record.get("task_kind") == "batch" and isinstance(items, list):
                for item in items:
                    item_state = str(item.get("state") or "queued")
                    if item_state in {
                        "completed",
                        "failed",
                        "cancelled",
                        "unknown",
                    }:
                        continue
                    attempt_id = str(item.get("send_attempt_id") or "")
                    receipt_state = receipt_states.get(attempt_id, "")
                    if receipt_state == "confirmed":
                        item.update(
                            {
                                "state": "completed",
                                "delivery_state": "confirmed",
                                "image_generated": True,
                                "image_sent": True,
                                "error_code": "",
                                "error_message": "",
                            }
                        )
                    elif (
                        item_state == "sending"
                        or item.get("delivery_state") == "attempting"
                    ):
                        item.update(
                            {
                                "state": "unknown",
                                "delivery_state": "unknown",
                                "image_sent": False,
                                "error_code": "delivery_unknown",
                                "error_message": (
                                    "Delivery was cancelled after transport started; receipt is unknown."
                                ),
                            }
                        )
                    else:
                        item.update(
                            {
                                "state": "cancelled",
                                "image_sent": False,
                                "error_code": "user_cancelled",
                                "error_message": reason,
                            }
                        )
                self._refresh_batch_counts(record)
                record["image_generated"] = bool(record.get("generated_count"))
                record["image_sent"] = bool(record.get("sent_count"))
                if int(record.get("unknown_count") or 0) > 0:
                    record["delivery_state"] = "unknown"
                elif int(record.get("sent_count") or 0) > 0:
                    record["delivery_state"] = "confirmed"
                else:
                    record["delivery_state"] = "not_started"
            elif (
                str(record.get("state") or "") == "sending"
                or record.get("delivery_state") == "attempting"
            ):
                attempt_id = str(record.get("send_attempt_id") or "")
                if receipt_states.get(attempt_id) == "confirmed":
                    record.update(
                        {
                            "delivery_state": "confirmed",
                            "image_generated": True,
                            "image_sent": True,
                            "error_code": "",
                            "error_message": "",
                        }
                    )
                else:
                    record.update(
                        {
                            "delivery_state": "unknown",
                            "image_sent": False,
                            "error_code": "delivery_unknown",
                            "error_message": (
                                "Delivery was cancelled after transport started; receipt is unknown."
                            ),
                        }
                    )

            record.update(
                {
                    "state": "cancelled",
                    "cancel_reason": reason,
                    "terminal_reason": "user_cancelled",
                    "suppress_future_injection": suppress_future_injection,
                    "revision": int(record.get("revision") or 0) + 1,
                    "updated_at": now,
                    "finished_at": now,
                }
            )
            token = str(
                record.get("notification_token") or f"notify_{uuid.uuid4().hex}"
            )
            record["notification_token"] = token
            existing = conn.execute(
                "SELECT state FROM notification_outbox WHERE token=?",
                (token,),
            ).fetchone()
            record["notification_state"] = (
                str(existing["state"]) if existing is not None else "pending"
            )
            payload = self._encode_record(record)
            expires = now + self.terminal_ttl_seconds * 1000
            changed = conn.execute(
                "UPDATE tasks SET state='cancelled', record_json=?, updated_at_ms=?, "
                "expires_at_ms=? WHERE task_id=? AND owner_epoch=?",
                (payload, now, expires, task_id, self.owner_epoch),
            ).rowcount
            if changed != 1:
                conn.rollback()
                raise BackgroundTaskOwnerError("Fenced task cancellation was rejected")
            conn.execute(
                "UPDATE reservations SET remaining=0, released=1 WHERE task_id=?",
                (task_id,),
            )
            conn.execute(
                "UPDATE receipts SET delivery_state='unknown' "
                "WHERE task_id=? AND delivery_state='attempting'",
                (task_id,),
            )
            if existing is None:
                conn.execute(
                    "INSERT INTO notification_outbox("
                    "token, task_id, kind, state, payload_json, queued_at_ms, updated_at_ms"
                    ") VALUES(?, ?, 'terminal', 'pending', ?, ?, ?)",
                    (token, task_id, payload, now, now),
                )
            elif str(existing["state"]) in {"pending", "queued"}:
                conn.execute(
                    "UPDATE notification_outbox SET payload_json=?, updated_at_ms=? "
                    "WHERE token=? AND state IN ('pending', 'queued')",
                    (payload, now, token),
                )
            conn.commit()
        return record

    async def cancel_scope(
        self,
        umo: str,
        sender_id: str,
        reason: str,
        *,
        suppress_future_injection: bool = False,
    ) -> int:
        """Cancel active tasks belonging to one sender in one UMO."""

        records = await self.list_active_for_umo(umo)
        count = 0
        for record in records:
            if str(record.get("sender_id") or "") != str(sender_id or ""):
                continue
            if await self.cancel_task(
                str(record["task_id"]),
                reason,
                suppress_future_injection=suppress_future_injection,
            ):
                count += 1
        return count

    def is_cancelled(self, task_id: str) -> bool:
        """Return whether a cancellation tombstone exists in this process."""

        event = self._cancel_events.get(task_id)
        return bool(event and event.is_set())

    async def run_provider(
        self,
        parent_id: str,
        work_factory: Callable[[], Awaitable[_T]],
    ) -> _T:
        """Run provider work through the fair, globally bounded scheduler.

        Args:
            parent_id: Single task or batch parent ID.
            work_factory: Zero-argument coroutine factory.

        Returns:
            Provider result.
        """

        if self._closing or self.is_cancelled(parent_id):
            raise asyncio.CancelledError
        loop = asyncio.get_running_loop()
        future: asyncio.Future[_T] = loop.create_future()
        work = _ScheduledWork(parent_id=parent_id, factory=work_factory, future=future)
        async with self._scheduler_lock:
            queue = self._ready_by_parent.setdefault(parent_id, deque())
            queue.append(work)
            if parent_id not in self._parent_in_ring:
                self._parent_ring.append(parent_id)
                self._parent_in_ring.add(parent_id)
            self._dispatch_locked()
        try:
            return await future
        except asyncio.CancelledError:
            if not future.done():
                future.cancel()
            raise

    async def run_planner(self, work_factory: Callable[[], Awaitable[_T]]) -> _T:
        """Run one batch planner through the global planner semaphore.

        Args:
            work_factory: Zero-argument planner coroutine factory.

        Returns:
            Planner result.
        """

        if self._closing:
            raise asyncio.CancelledError
        async with self._planner_semaphore:
            if self._closing:
                raise asyncio.CancelledError
            return await work_factory()

    def _dispatch_locked(self) -> None:
        while self._provider_running < self.max_running and self._parent_ring:
            parent_id = self._parent_ring.popleft()
            self._parent_in_ring.discard(parent_id)
            queue = self._ready_by_parent.get(parent_id)
            if not queue:
                self._ready_by_parent.pop(parent_id, None)
                continue
            work = queue.popleft()
            if queue:
                self._parent_ring.append(parent_id)
                self._parent_in_ring.add(parent_id)
            else:
                self._ready_by_parent.pop(parent_id, None)
            if work.future.cancelled() or self.is_cancelled(parent_id):
                if not work.future.done():
                    work.future.cancel()
                continue
            self._provider_running += 1
            self._track(
                self._execute_scheduled(work),
                name=f"background-provider-{parent_id}",
            )

    async def _execute_scheduled(self, work: _ScheduledWork) -> None:
        try:
            if self.is_cancelled(work.parent_id):
                raise asyncio.CancelledError
            result = await work.factory()
            if not work.future.done():
                work.future.set_result(result)
        except asyncio.CancelledError:
            if not work.future.done():
                work.future.cancel()
            raise
        except BaseException as exc:
            if not work.future.done():
                work.future.set_exception(exc)
        finally:
            async with self._scheduler_lock:
                self._provider_running = max(0, self._provider_running - 1)
                self._dispatch_locked()

    async def _drop_parent_work(self, parent_id: str) -> None:
        async with self._scheduler_lock:
            queue = self._ready_by_parent.pop(parent_id, deque())
            self._parent_in_ring.discard(parent_id)
            self._parent_ring = deque(
                candidate for candidate in self._parent_ring if candidate != parent_id
            )
            for work in queue:
                if not work.future.done():
                    work.future.cancel()

    def start_worker(
        self,
        task_id: str,
        runner: Callable[[], Awaitable[Any]],
    ) -> asyncio.Task[Any]:
        """Start and retain a root worker for an accepted task.

        Args:
            task_id: Durable task identifier.
            runner: Worker coroutine factory.

        Returns:
            Created asyncio task.
        """

        if task_id in self._root_tasks and not self._root_tasks[task_id].done():
            raise BackgroundTaskError(f"Task worker already exists: {task_id}")
        task = self._track(runner(), name=f"background-image-{task_id}")
        self._root_tasks[task_id] = task

        def cleanup(done: asyncio.Task[Any]) -> None:
            if self._root_tasks.get(task_id) is done:
                self._root_tasks.pop(task_id, None)

        task.add_done_callback(cleanup)
        return task

    def start_managed(
        self,
        coroutine: Awaitable[_T],
        *,
        name: str,
    ) -> asyncio.Task[_T]:
        """Start a retained non-root coroutine.

        Args:
            coroutine: Coroutine to execute.
            name: Diagnostic asyncio task name.

        Returns:
            Created asyncio task.
        """

        if self._closing:
            if hasattr(coroutine, "close"):
                coroutine.close()
            raise BackgroundTaskError("Background task manager is closing")
        return self._track(coroutine, name=name)

    def _track(
        self,
        coroutine: Awaitable[_T],
        *,
        name: str,
    ) -> asyncio.Task[_T]:
        task = asyncio.create_task(coroutine, name=name)
        self._managed_tasks.add(task)

        def consume(done: asyncio.Task[Any]) -> None:
            self._managed_tasks.discard(done)
            try:
                exc = done.exception()
            except asyncio.CancelledError:
                return
            except BaseException as exc:
                if self.log is not None:
                    self.log.error(
                        "[background-image] managed task callback failed: %s",
                        self.sanitize_error(exc),
                    )
                return
            if exc is not None and self.log is not None:
                self.log.error(
                    "[background-image] managed task failed: name=%s err=%s",
                    done.get_name(),
                    self.sanitize_error(exc),
                )

        task.add_done_callback(consume)
        return task

    @asynccontextmanager
    async def notification_turn(self, scope_key: str):
        """Serialize terminal notification delivery for one conversation scope.

        Args:
            scope_key: Stable UMO or equivalent conversation routing key.

        Yields:
            Control after all earlier notification turns for the scope finish.
        """

        key = str(scope_key or "").strip()
        if not key:
            raise BackgroundTaskError("A notification scope key is required")
        async with self._notification_map_lock:
            lock = self._notification_locks.setdefault(key, asyncio.Lock())
            self._notification_lock_users[key] = (
                self._notification_lock_users.get(key, 0) + 1
            )
        try:
            async with lock:
                yield
        finally:
            async with self._notification_map_lock:
                users = self._notification_lock_users.get(key, 1) - 1
                if users <= 0:
                    self._notification_lock_users.pop(key, None)
                    if not lock.locked():
                        self._notification_locks.pop(key, None)
                else:
                    self._notification_lock_users[key] = users

    async def wait_notification_terminal(
        self,
        token: str,
        *,
        timeout_seconds: float,
    ) -> bool:
        """Wait until an outbox token reaches an immutable terminal state.

        Args:
            token: Stable notification token.
            timeout_seconds: Maximum wait before returning ``False``.

        Returns:
            ``True`` when the durable outbox state is terminal.
        """

        event = self._notification_events.setdefault(token, asyncio.Event())
        async with self._db_lock:
            state = await asyncio.to_thread(self._notification_state_sync, token)
        if state in NOTIFICATION_TERMINAL_STATES:
            event.set()
        try:
            await asyncio.wait_for(
                event.wait(),
                timeout=max(0.01, float(timeout_seconds)),
            )
        except TimeoutError:
            return False
        finally:
            if self._notification_events.get(token) is event:
                self._notification_events.pop(token, None)
        async with self._db_lock:
            state = await asyncio.to_thread(self._notification_state_sync, token)
        return state in NOTIFICATION_TERMINAL_STATES

    def _notification_state_sync(self, token: str) -> str:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT state FROM notification_outbox WHERE token=?",
                (token,),
            ).fetchone()
        return str(row["state"] if row is not None else "")

    async def claim_notification(
        self,
        token: str,
        claimant: str,
        *,
        from_states: tuple[str, ...] = ("pending", "queued"),
    ) -> dict[str, Any] | None:
        """Claim one notification outbox row with SQLite CAS.

        Args:
            token: Stable notification token.
            claimant: Unique local delivery attempt ID.
            from_states: States eligible for claiming.

        Returns:
            Claimed payload, or ``None`` when another sender won the claim.
        """

        now = self.now_ms()
        async with self._db_lock:
            return await asyncio.to_thread(
                self._claim_notification_sync,
                token,
                claimant,
                from_states,
                now,
            )

    def _claim_notification_sync(
        self,
        token: str,
        claimant: str,
        from_states: tuple[str, ...],
        now: int,
    ) -> dict[str, Any] | None:
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            self._assert_owner(conn, now)
            row = conn.execute(
                "SELECT payload_json, state FROM notification_outbox WHERE token=?",
                (token,),
            ).fetchone()
            if row is None or row["state"] not in from_states:
                conn.commit()
                return None
            placeholders = ",".join("?" for _ in from_states)
            changed = conn.execute(
                "UPDATE notification_outbox SET state='claimed', attempt_id=?, updated_at_ms=? "
                f"WHERE token=? AND state IN ({placeholders})",
                (claimant, now, token, *from_states),
            ).rowcount
            conn.commit()
        return json.loads(row["payload_json"]) if changed == 1 else None

    async def mark_notification(
        self,
        token: str,
        state: str,
        *,
        attempt_id: str,
    ) -> bool:
        """Persist the final state of a claimed notification."""

        if state not in {"queued", "claimed", "sent", "unknown", "failed", "expired"}:
            raise BackgroundTaskStateError(f"Unknown notification state: {state}")
        if not str(attempt_id or "").strip():
            raise BackgroundTaskStateError(
                "A notification claim attempt_id is required"
            )
        now = self.now_ms()
        async with self._db_lock:
            changed = await asyncio.to_thread(
                self._mark_notification_sync, token, state, attempt_id, now
            )
        if changed and state in NOTIFICATION_TERMINAL_STATES:
            event = self._notification_events.get(token)
            if event is not None:
                event.set()
        return changed

    def _mark_notification_sync(
        self, token: str, state: str, attempt_id: str, now: int
    ) -> bool:
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            self._assert_owner(conn, now)
            row = conn.execute(
                "SELECT o.task_id, o.attempt_id, o.state, t.owner_epoch "
                "FROM notification_outbox o JOIN tasks t ON t.task_id=o.task_id "
                "WHERE o.token=?",
                (token,),
            ).fetchone()
            if row is None:
                conn.commit()
                return False
            if str(row["attempt_id"] or "") != attempt_id:
                conn.commit()
                return False
            if int(row["owner_epoch"]) != self.owner_epoch:
                conn.commit()
                return False
            current_state = str(row["state"] or "")
            if current_state in NOTIFICATION_TERMINAL_STATES:
                conn.commit()
                return current_state == state
            allowed = {
                "claimed": {
                    "claimed",
                    "queued",
                    *NOTIFICATION_TERMINAL_STATES,
                },
                "queued": {
                    "claimed",
                    "queued",
                    *NOTIFICATION_TERMINAL_STATES,
                },
            }
            if state not in allowed.get(current_state, set()):
                conn.commit()
                return False
            changed = conn.execute(
                "UPDATE notification_outbox SET state=?, updated_at_ms=? "
                "WHERE token=? AND attempt_id=? AND state=?",
                (state, now, token, attempt_id, current_state),
            ).rowcount
            if changed != 1:
                conn.rollback()
                return False
            task = conn.execute(
                "SELECT record_json FROM tasks WHERE task_id=?", (row["task_id"],)
            ).fetchone()
            if task is not None:
                record = json.loads(task["record_json"])
                record["notification_state"] = state
                record["notification_sent_at"] = now if state == "sent" else 0
                record["updated_at"] = now
                task_changed = conn.execute(
                    "UPDATE tasks SET record_json=?, updated_at_ms=? "
                    "WHERE task_id=? AND owner_epoch=?",
                    (
                        self._encode_record(record),
                        now,
                        row["task_id"],
                        self.owner_epoch,
                    ),
                ).rowcount
                if task_changed != 1:
                    conn.rollback()
                    return False
            conn.commit()
        return True

    async def mark_ack(self, task_id: str, state: str) -> dict[str, Any]:
        """Update the acceptance acknowledgement state."""

        if state not in {"pending", "decorated", "sent", "unknown"}:
            raise BackgroundTaskStateError(f"Unknown acknowledgement state: {state}")
        return await self.transition(
            task_id,
            str((await self.get_task(task_id) or {}).get("state") or "queued"),
            {
                "ack_state": state,
                "ack_sent_at": self.now_ms() if state == "sent" else 0,
            },
        )

    async def record_receipt(
        self,
        task_id: str,
        *,
        send_attempt_id: str,
        kind: str,
        delivery_state: str,
        transport: str,
        item_id: str = "",
        response_digest: str = "",
    ) -> bool:
        """Insert an immutable image or notification transport receipt."""

        if delivery_state not in DELIVERY_STATES:
            raise BackgroundTaskStateError(f"Unknown delivery state: {delivery_state}")
        now = self.now_ms()
        async with self._db_lock:
            return await asyncio.to_thread(
                self._record_receipt_sync,
                task_id,
                send_attempt_id,
                kind,
                delivery_state,
                transport,
                item_id,
                response_digest,
                now,
            )

    def _record_receipt_sync(
        self,
        task_id: str,
        send_attempt_id: str,
        kind: str,
        delivery_state: str,
        transport: str,
        item_id: str,
        response_digest: str,
        now: int,
    ) -> bool:
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            self._assert_owner(conn, now)
            existing = conn.execute(
                "SELECT task_id, item_id, kind, transport, delivery_state "
                "FROM receipts WHERE send_attempt_id=?",
                (send_attempt_id,),
            ).fetchone()
            if existing is not None:
                same_attempt = (
                    str(existing["task_id"] or "") == task_id
                    and str(existing["item_id"] or "") == item_id
                    and str(existing["kind"] or "") == kind
                    and str(existing["transport"] or "") == transport
                )
                if not same_attempt or existing["delivery_state"] != "attempting":
                    conn.commit()
                    return False
            changed = conn.execute(
                "INSERT INTO receipts(send_attempt_id, task_id, item_id, kind, "
                "delivery_state, transport, response_digest, created_at_ms) "
                "VALUES(?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(send_attempt_id) DO UPDATE SET "
                "delivery_state=excluded.delivery_state, "
                "response_digest=excluded.response_digest",
                (
                    send_attempt_id,
                    task_id,
                    item_id,
                    kind,
                    delivery_state,
                    transport,
                    response_digest,
                    now,
                ),
            ).rowcount
            conn.commit()
        return changed == 1

    async def spool_inputs(
        self,
        task_id: str,
        blobs: list[bytes],
    ) -> tuple[tuple[str, ...], list[dict[str, Any]]]:
        """Atomically spool event-dependent image inputs to private files.

        Args:
            task_id: Task that owns the files.
            blobs: Decoded image byte payloads.

        Returns:
            Absolute paths for the in-process job and a relative manifest for
            the durable task record.

        Raises:
            BackgroundTaskError: A size limit is exceeded.
        """

        total = sum(len(blob) for blob in blobs)
        if not blobs:
            return (), []
        if total > INPUT_TASK_LIMIT_BYTES:
            raise BackgroundTaskError(
                f"Task inputs exceed {INPUT_TASK_LIMIT_BYTES} bytes"
            )
        if any(len(blob) > INPUT_FILE_LIMIT_BYTES for blob in blobs):
            raise BackgroundTaskError(
                f"An input image exceeds {INPUT_FILE_LIMIT_BYTES} bytes"
            )
        return await asyncio.to_thread(self._spool_inputs_sync, task_id, blobs)

    def _spool_inputs_sync(
        self, task_id: str, blobs: list[bytes]
    ) -> tuple[tuple[str, ...], list[dict[str, Any]]]:
        input_dir = self.base_dir / task_id / "inputs"
        input_dir.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(input_dir.parent, 0o700)
            os.chmod(input_dir, 0o700)
        except OSError:
            pass
        paths: list[str] = []
        manifest: list[dict[str, Any]] = []
        for index, blob in enumerate(blobs, start=1):
            path = input_dir / f"{index:04d}.bin"
            part = path.with_suffix(".part")
            part.write_bytes(blob)
            try:
                os.chmod(part, 0o600)
            except OSError:
                pass
            part.replace(path)
            digest = hashlib.sha256(blob).hexdigest()
            paths.append(str(path))
            manifest.append(
                {
                    "relative_path": str(path.relative_to(self.base_dir)),
                    "size": len(blob),
                    "sha256": digest,
                }
            )
        return tuple(paths), manifest

    async def read_spooled_inputs(
        self,
        paths: tuple[str, ...],
        manifest: list[dict[str, Any]] | None = None,
    ) -> list[bytes]:
        """Read and validate private input files for a prepared job.

        Args:
            paths: Absolute private spool paths.
            manifest: Optional expected size and SHA-256 entries.

        Returns:
            Validated byte payloads.
        """

        return await asyncio.to_thread(
            self._read_spooled_inputs_sync,
            paths,
            manifest or [],
        )

    def _read_spooled_inputs_sync(
        self,
        paths: tuple[str, ...],
        manifest: list[dict[str, Any]],
    ) -> list[bytes]:
        base = self.base_dir.resolve(strict=False)
        blobs: list[bytes] = []
        total = 0
        for index, raw in enumerate(paths):
            path = Path(raw).resolve(strict=False)
            try:
                path.relative_to(base)
            except ValueError as exc:
                raise BackgroundTaskError(
                    "Spool path escaped the plugin data directory"
                ) from exc
            data = path.read_bytes()
            if len(data) > INPUT_FILE_LIMIT_BYTES:
                raise BackgroundTaskError("Spool input exceeded the per-file limit")
            total += len(data)
            if total > INPUT_TASK_LIMIT_BYTES:
                raise BackgroundTaskError("Spool inputs exceeded the per-task limit")
            if index < len(manifest):
                expected_size = int(manifest[index].get("size") or 0)
                expected_hash = str(manifest[index].get("sha256") or "")
                if expected_size and len(data) != expected_size:
                    raise BackgroundTaskError(
                        "Spool input size no longer matches manifest"
                    )
                if expected_hash and hashlib.sha256(data).hexdigest() != expected_hash:
                    raise BackgroundTaskError(
                        "Spool input hash no longer matches manifest"
                    )
            blobs.append(data)
        return blobs

    async def cleanup_task_files(self, task_id: str) -> None:
        """Best-effort removal of private spooled inputs."""

        await asyncio.to_thread(
            shutil.rmtree,
            self.base_dir / task_id,
            True,
        )

    async def close(self, *, grace_seconds: float = 10.0) -> None:
        """Stop intake, cancel workers, and release the owner lease.

        Args:
            grace_seconds: Maximum wait for managed tasks after cancellation.
        """

        if self._closing:
            return
        self._closing = True
        self.accepting = False
        for event in self._cancel_events.values():
            event.set()
        for event in self._notification_events.values():
            event.set()
        for task in list(self._managed_tasks):
            task.cancel()
        pending = list(self._managed_tasks)
        if pending:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*pending, return_exceptions=True),
                    timeout=max(1.0, float(grace_seconds)),
                )
            except TimeoutError:
                for task in pending:
                    task.cancel()

        if self.started:
            async with self._db_lock:
                await asyncio.to_thread(self._set_owner_state, "draining")
                await asyncio.to_thread(self._release_owner)
        self.started = False
        self._root_tasks.clear()
        self._cancel_events.clear()
        self._ready_by_parent.clear()
        self._parent_ring.clear()
        self._parent_in_ring.clear()
        self._notification_events.clear()
        self._notification_locks.clear()
        self._notification_lock_users.clear()

    async def _heartbeat_loop(self) -> None:
        while not self._closing:
            await asyncio.sleep(self.heartbeat_seconds)
            try:
                async with self._db_lock:
                    await asyncio.to_thread(self._heartbeat)
            except BackgroundTaskOwnerError as exc:
                self.accepting = False
                if self.log is not None:
                    self.log.error(
                        "[background-image] owner heartbeat stopped: %s",
                        self.sanitize_error(exc),
                    )
                return

    def _heartbeat(self) -> None:
        now = self.now_ms()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            changed = conn.execute(
                "UPDATE runtime_owner SET heartbeat_at_ms=? "
                "WHERE singleton=1 AND owner_instance_id=? AND owner_epoch=? AND state='active'",
                (now, self.owner_instance_id, self.owner_epoch),
            ).rowcount
            if changed != 1:
                conn.rollback()
                raise BackgroundTaskOwnerError("Background owner lease was lost")
            conn.commit()

    async def _gc_loop(self) -> None:
        while not self._closing:
            await asyncio.sleep(300)
            try:
                await self.gc()
            except asyncio.CancelledError:
                raise
            except BaseException as exc:
                if self.log is not None:
                    self.log.warning(
                        "[background-image] GC failed: %s",
                        self.sanitize_error(exc),
                    )

    async def _health_loop(self) -> None:
        while not self._closing:
            await asyncio.sleep(300)
            try:
                snapshot = await self.health_snapshot()
            except asyncio.CancelledError:
                raise
            except BaseException as exc:
                self._health_failure_count += 1
                if self.log is not None:
                    self.log.warning(
                        "[background-image] health check failed: consecutive=%s err=%s",
                        self._health_failure_count,
                        self.sanitize_error(exc),
                    )
                if self._health_failure_count >= 3:
                    self.accepting = False
                    if self.log is not None:
                        self.log.error(
                            "[background-image] intake disabled after repeated health check failures"
                        )
                continue

            self._health_failure_count = 0
            if self.log is not None:
                self.log.info(
                    "[background-image] health: epoch=%s managed=%s active=%s "
                    "reservation_remaining=%s provider_running=%s ready=%s "
                    "outbox_pending=%s outbox_sending=%s outbox_unknown=%s "
                    "oldest_active_seconds=%s db_bytes=%s wal_bytes=%s "
                    "wal_busy=%s wal_checkpointed=%s heartbeat_age_ms=%s",
                    snapshot["owner_epoch"],
                    snapshot["managed_tasks"],
                    snapshot["active_tasks"],
                    snapshot["reservation_remaining"],
                    snapshot["provider_running"],
                    snapshot["ready_work"],
                    snapshot["outbox_pending"],
                    snapshot["outbox_sending"],
                    snapshot["outbox_unknown"],
                    snapshot["oldest_active_seconds"],
                    snapshot["db_bytes"],
                    snapshot["wal_bytes"],
                    snapshot["wal_busy"],
                    snapshot["wal_checkpointed"],
                    snapshot["heartbeat_age_ms"],
                )

    async def health_snapshot(self) -> dict[str, int | str]:
        """Return a durable health snapshot and run a passive WAL checkpoint.

        Returns:
            Counts and storage metrics used by runtime monitoring.

        Raises:
            BackgroundTaskError: The database or reservation ledger is inconsistent.
            BackgroundTaskOwnerError: This manager no longer owns the ledger.
        """

        now = self.now_ms()
        async with self._db_lock:
            snapshot = await asyncio.to_thread(self._health_snapshot_sync, now)
        snapshot.update(
            {
                "managed_tasks": len(self._managed_tasks),
                "provider_running": self._provider_running,
                "ready_work": sum(
                    len(queue) for queue in self._ready_by_parent.values()
                ),
            }
        )
        theoretical_limit = self.max_queued * 2 + self.max_notification_backlog + 16
        if int(snapshot["managed_tasks"]) > theoretical_limit:
            raise BackgroundTaskError(
                "Managed background task count exceeded the bounded runtime limit"
            )
        if int(snapshot["outbox_active"]) * 5 >= self.max_notification_backlog * 4:
            raise BackgroundTaskError(
                "Background notification outbox reached 80% of its hard limit"
            )
        return snapshot

    def _health_snapshot_sync(self, now: int) -> dict[str, int | str]:
        with self._connect() as conn:
            self._assert_owner(conn, now)
            check = conn.execute("PRAGMA quick_check").fetchone()
            if check is None or str(check[0]).lower() != "ok":
                raise BackgroundTaskError(f"Task database quick_check failed: {check}")
            inconsistent = int(
                conn.execute(
                    "SELECT COUNT(*) FROM tasks t "
                    "JOIN reservations r ON r.task_id=t.task_id "
                    "WHERE (t.state IN ("
                    + ",".join("?" for _ in TERMINAL_STATES)
                    + ") AND (r.remaining<>0 OR r.released<>1)) "
                    "OR (t.state IN ("
                    + ",".join("?" for _ in ACTIVE_STATES)
                    + ") AND r.released<>0)",
                    (*tuple(sorted(TERMINAL_STATES)), *tuple(sorted(ACTIVE_STATES))),
                ).fetchone()[0]
            )
            if inconsistent:
                raise BackgroundTaskError(
                    f"Reservation ledger contains {inconsistent} inconsistent task rows"
                )
            active_states = tuple(sorted(ACTIVE_STATES))
            active_tasks = int(
                conn.execute(
                    "SELECT COUNT(*) FROM tasks WHERE state IN ("
                    + ",".join("?" for _ in active_states)
                    + ")",
                    active_states,
                ).fetchone()[0]
            )
            oldest = conn.execute(
                "SELECT MIN(created_at_ms) FROM tasks WHERE state IN ("
                + ",".join("?" for _ in active_states)
                + ")",
                active_states,
            ).fetchone()[0]
            reservation_remaining = int(
                conn.execute(
                    "SELECT COALESCE(SUM(remaining), 0) FROM reservations"
                ).fetchone()[0]
            )
            outbox = {
                str(row[0]): int(row[1])
                for row in conn.execute(
                    "SELECT state, COUNT(*) FROM notification_outbox GROUP BY state"
                ).fetchall()
            }
            owner = conn.execute(
                "SELECT heartbeat_at_ms FROM runtime_owner WHERE singleton=1"
            ).fetchone()
            checkpoint = conn.execute("PRAGMA wal_checkpoint(PASSIVE)").fetchone()
        wal_path = Path(f"{self.db_path}-wal")
        return {
            "quick_check": "ok",
            "owner_epoch": self.owner_epoch,
            "heartbeat_age_ms": max(
                0,
                now - int(owner[0] if owner is not None else 0),
            ),
            "active_tasks": active_tasks,
            "reservation_remaining": reservation_remaining,
            "outbox_pending": outbox.get("pending", 0),
            "outbox_sending": outbox.get("queued", 0) + outbox.get("claimed", 0),
            "outbox_active": outbox.get("pending", 0)
            + outbox.get("queued", 0)
            + outbox.get("claimed", 0),
            "outbox_unknown": outbox.get("unknown", 0),
            "oldest_active_seconds": max(
                0,
                (now - int(oldest)) // 1000 if oldest is not None else 0,
            ),
            "db_bytes": self.db_path.stat().st_size if self.db_path.exists() else 0,
            "wal_bytes": wal_path.stat().st_size if wal_path.exists() else 0,
            "wal_busy": int(checkpoint[0] if checkpoint is not None else -1),
            "wal_log_frames": int(checkpoint[1] if checkpoint is not None else -1),
            "wal_checkpointed": int(checkpoint[2] if checkpoint is not None else -1),
        }

    async def gc(self) -> list[str]:
        """Delete expired durable rows and return removed task IDs."""

        now = self.now_ms()
        async with self._db_lock:
            task_ids = await asyncio.to_thread(self._gc_sync, now)
        for task_id in task_ids:
            self._cancel_events.pop(task_id, None)
        return task_ids

    def _gc_sync(self, now: int) -> list[str]:
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            self._assert_owner(conn, now)
            rows = conn.execute(
                "SELECT task_id FROM tasks WHERE expires_at_ms < ? AND state IN ("
                + ",".join("?" for _ in TERMINAL_STATES)
                + ")",
                (now, *tuple(sorted(TERMINAL_STATES))),
            ).fetchall()
            task_ids = [str(row["task_id"]) for row in rows]
            removable: list[str] = []
            for task_id in task_ids:
                try:
                    shutil.rmtree(self.base_dir / task_id)
                except FileNotFoundError:
                    pass
                except OSError as exc:
                    if self.log is not None:
                        self.log.warning(
                            "[background-image] GC kept task row after spool cleanup failed: task=%s err=%s",
                            task_id,
                            self.sanitize_error(exc),
                        )
                    continue
                removable.append(task_id)
            if removable:
                conn.executemany(
                    "DELETE FROM tasks WHERE task_id=?", [(x,) for x in removable]
                )
            conn.execute("DELETE FROM request_dedupe WHERE expires_at_ms < ?", (now,))
            conn.commit()
        return removable

    def _set_owner_state(self, state: str) -> None:
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            changed = conn.execute(
                "UPDATE runtime_owner SET state=?, heartbeat_at_ms=? "
                "WHERE singleton=1 AND owner_instance_id=? AND owner_epoch=?",
                (state, self.now_ms(), self.owner_instance_id, self.owner_epoch),
            ).rowcount
            if changed != 1:
                conn.rollback()
                raise BackgroundTaskOwnerError("Background owner lease was lost")
            conn.commit()

    def _release_owner(self) -> None:
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "UPDATE runtime_owner SET state='released', heartbeat_at_ms=? "
                "WHERE singleton=1 AND owner_instance_id=? AND owner_epoch=?",
                (self.now_ms(), self.owner_instance_id, self.owner_epoch),
            )
            conn.commit()

    def _assert_owner(self, conn: sqlite3.Connection, now: int) -> None:
        row = conn.execute(
            "SELECT owner_instance_id, owner_epoch, heartbeat_at_ms, state "
            "FROM runtime_owner WHERE singleton=1"
        ).fetchone()
        if (
            row is None
            or row["owner_instance_id"] != self.owner_instance_id
            or int(row["owner_epoch"]) != self.owner_epoch
            or row["state"] != "active"
            or int(row["heartbeat_at_ms"]) < now - self.lease_seconds * 1000
        ):
            raise BackgroundTaskOwnerError("Background owner lease is not valid")

    def _encode_record(self, record: dict[str, Any]) -> str:
        payload = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        if len(payload.encode("utf-8")) > RECORD_LIMIT_BYTES:
            raise BackgroundTaskError(f"Task record exceeds {RECORD_LIMIT_BYTES} bytes")
        return payload

    @staticmethod
    def dataclass_dict(value: Any) -> dict[str, Any]:
        """Convert a prepared dataclass to a JSON-compatible dictionary."""

        return asdict(value)
