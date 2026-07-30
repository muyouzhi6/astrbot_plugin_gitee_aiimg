import asyncio
import importlib.util
import json
import sqlite3
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    name = "gitee_background_tasks_test"
    sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location(
        name,
        ROOT / "core" / "background_tasks.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


bg = _load_module()


def _record(
    manager,
    task_id: str,
    *,
    fingerprint: str | None = None,
    state: str = "queued",
    task_kind: str = "single",
):
    scope = manager.scope_hash("qq:GroupMessage:1", "bot", "user")
    return {
        "task_id": task_id,
        "task_kind": task_kind,
        "state": state,
        "scope_hash": scope,
        "request_fingerprint": fingerprint or f"fingerprint-{task_id}",
        "umo": "qq:GroupMessage:1",
        "sender_id": "user",
        "self_id": "bot",
        "user_prompt": "take a photo",
        "effective_prompt": "a detailed photo prompt",
        "delivery_state": "not_started",
        "image_generated": False,
        "image_sent": False,
        "items": [],
    }


@pytest.mark.asyncio
async def test_schema_capacity_dedupe_and_terminal_guard(tmp_path):
    manager = bg.BackgroundImageTaskManager(
        tmp_path,
        max_running=1,
        max_queued=2,
        heartbeat_seconds=60,
    )
    await manager.start()
    first, created = await manager.create_task_record(
        _record(manager, "img_1"),
        reservation=1,
    )
    assert created is True
    duplicate, created = await manager.create_task_record(
        _record(manager, "img_2", fingerprint=first["request_fingerprint"]),
        reservation=1,
    )
    assert created is False
    assert duplicate["task_id"] == "img_1"

    await manager.create_task_record(_record(manager, "img_3"), reservation=1)
    with pytest.raises(bg.BackgroundTaskCapacityError):
        await manager.create_task_record(_record(manager, "img_4"), reservation=1)

    completed = await manager.transition(
        "img_1",
        "completed",
        {
            "image_generated": True,
            "image_sent": True,
            "delivery_state": "confirmed",
        },
        queue_notification=True,
    )
    assert completed["state"] == "completed"
    with pytest.raises(bg.BackgroundTaskStateError):
        await manager.transition("img_1", "running")

    with sqlite3.connect(manager.db_path) as conn:
        assert conn.execute("PRAGMA quick_check").fetchone()[0] == "ok"
        remaining = conn.execute(
            "SELECT remaining FROM reservations WHERE task_id='img_1'"
        ).fetchone()[0]
    assert remaining == 0
    await manager.close()


@pytest.mark.asyncio
async def test_owner_lease_fails_closed_and_expired_owner_recovers(tmp_path):
    first = bg.BackgroundImageTaskManager(
        tmp_path,
        heartbeat_seconds=60,
    )
    await first.start()
    await first.create_task_record(_record(first, "img_recover"), reservation=1)

    second = bg.BackgroundImageTaskManager(tmp_path, heartbeat_seconds=60)
    with pytest.raises(bg.BackgroundTaskOwnerError):
        await second.start()

    for task in list(first._managed_tasks):
        task.cancel()
    await asyncio.gather(*list(first._managed_tasks), return_exceptions=True)
    with sqlite3.connect(first.db_path) as conn:
        payload = json.loads(
            conn.execute(
                "SELECT record_json FROM tasks WHERE task_id='img_recover'"
            ).fetchone()[0]
        )
        payload.pop("notification_token", None)
        conn.execute(
            "UPDATE tasks SET record_json=? WHERE task_id='img_recover'",
            (json.dumps(payload),),
        )
        conn.execute("UPDATE runtime_owner SET heartbeat_at_ms=0 WHERE singleton=1")
        conn.commit()

    recovered = await second.start()
    assert len(recovered) == 1
    assert recovered[0]["task_id"] == "img_recover"
    assert recovered[0]["state"] == "interrupted"
    token = recovered[0]["notification_token"]
    with sqlite3.connect(second.db_path) as conn:
        task_payload = json.loads(
            conn.execute(
                "SELECT record_json FROM tasks WHERE task_id='img_recover'"
            ).fetchone()[0]
        )
        outbox_token, outbox_payload = conn.execute(
            "SELECT token, payload_json FROM notification_outbox "
            "WHERE task_id='img_recover'"
        ).fetchone()
    assert task_payload["notification_token"] == token
    assert outbox_token == token
    assert json.loads(outbox_payload)["notification_token"] == token
    await second.close()


@pytest.mark.asyncio
async def test_pending_outbox_recovery_rewrites_payload_to_canonical_token(tmp_path):
    first = bg.BackgroundImageTaskManager(tmp_path, heartbeat_seconds=60)
    await first.start()
    record, _ = await first.create_task_record(
        _record(first, "img_pending_recovery"),
        reservation=1,
    )
    terminal = await first.transition(
        record["task_id"],
        "failed",
        queue_notification=True,
    )
    token = terminal["notification_token"]
    await first.close()

    with sqlite3.connect(first.db_path) as conn:
        task_payload = json.loads(
            conn.execute(
                "SELECT record_json FROM tasks WHERE task_id='img_pending_recovery'"
            ).fetchone()[0]
        )
        task_payload["notification_token"] = "stale-task-token"
        outbox_payload = dict(task_payload)
        outbox_payload["notification_token"] = "stale-payload-token"
        conn.execute(
            "UPDATE tasks SET record_json=? WHERE task_id='img_pending_recovery'",
            (json.dumps(task_payload),),
        )
        conn.execute(
            "UPDATE notification_outbox SET payload_json=? WHERE token=?",
            (json.dumps(outbox_payload), token),
        )
        conn.commit()

    second = bg.BackgroundImageTaskManager(tmp_path, heartbeat_seconds=60)
    recovered = await second.start()
    current = next(
        item for item in recovered if item["task_id"] == "img_pending_recovery"
    )
    assert current["notification_token"] == token
    claimed = await second.claim_notification(token, "recovery-sender")
    assert claimed["notification_token"] == token
    await second.mark_notification(
        token,
        "sent",
        attempt_id="recovery-sender",
    )
    await second.close()


@pytest.mark.asyncio
async def test_parent_round_robin_is_work_conserving(tmp_path):
    manager = bg.BackgroundImageTaskManager(
        tmp_path,
        max_running=1,
        max_queued=8,
        heartbeat_seconds=60,
    )
    await manager.start()
    order = []
    release_first = asyncio.Event()

    async def work(name, wait=False):
        order.append(name)
        if wait:
            await release_first.wait()
        await asyncio.sleep(0)
        return name

    first = asyncio.create_task(
        manager.run_provider("batch", lambda: work("batch-1", True))
    )
    await asyncio.sleep(0)
    batch_two = asyncio.create_task(
        manager.run_provider("batch", lambda: work("batch-2"))
    )
    single = asyncio.create_task(
        manager.run_provider("single", lambda: work("single-1"))
    )
    await asyncio.sleep(0)
    release_first.set()
    assert await first == "batch-1"
    assert await batch_two == "batch-2"
    assert await single == "single-1"
    assert order == ["batch-1", "batch-2", "single-1"]

    # A later parent enters after the batch has already been requeued. The
    # scheduler preserves FIFO parent-ring order without reserving idle slots.
    assert manager._provider_running == 0
    await manager.close()


@pytest.mark.asyncio
async def test_cancel_worker_releases_capacity_once(tmp_path):
    manager = bg.BackgroundImageTaskManager(tmp_path, heartbeat_seconds=60)
    await manager.start()
    await manager.create_task_record(_record(manager, "img_cancel"), reservation=1)
    entered = asyncio.Event()

    async def runner():
        await manager.transition("img_cancel", "running")
        entered.set()
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            raise

    manager.start_worker("img_cancel", runner)
    await entered.wait()
    assert await manager.cancel_task("img_cancel", "user requested stop") is True
    await asyncio.sleep(0)
    assert await manager.cancel_task("img_cancel", "duplicate stop") is False
    record = await manager.get_task("img_cancel")
    assert record["state"] == "cancelled"
    with sqlite3.connect(manager.db_path) as conn:
        remaining = conn.execute(
            "SELECT remaining FROM reservations WHERE task_id='img_cancel'"
        ).fetchone()[0]
    assert remaining == 0
    await manager.close()


@pytest.mark.asyncio
async def test_notification_outbox_compare_and_swap(tmp_path):
    manager = bg.BackgroundImageTaskManager(tmp_path, heartbeat_seconds=60)
    await manager.start()
    record, _ = await manager.create_task_record(
        _record(manager, "img_notify"),
        reservation=1,
    )
    terminal = await manager.transition(
        record["task_id"],
        "failed",
        {"error_code": "provider_error"},
        queue_notification=True,
    )
    token = terminal["notification_token"]
    claims = await asyncio.gather(
        manager.claim_notification(token, "agent"),
        manager.claim_notification(token, "watchdog"),
    )
    assert sum(claim is not None for claim in claims) == 1
    winner = "agent" if claims[0] is not None else "watchdog"
    loser = "watchdog" if winner == "agent" else "agent"
    with pytest.raises(bg.BackgroundTaskStateError, match="attempt_id"):
        await manager.mark_notification(token, "sent", attempt_id="")
    assert await manager.mark_notification(
        token,
        "sent",
        attempt_id=winner,
    )
    assert not await manager.mark_notification(
        token,
        "failed",
        attempt_id=loser,
    )
    updated = await manager.get_task(record["task_id"])
    assert updated["notification_state"] == "sent"
    await manager.close()


@pytest.mark.asyncio
async def test_terminal_same_state_retry_can_repair_missing_outbox(tmp_path):
    manager = bg.BackgroundImageTaskManager(tmp_path, heartbeat_seconds=60)
    await manager.start()
    record, _ = await manager.create_task_record(
        _record(manager, "img_notify_repair"),
        reservation=1,
    )
    await manager.transition(record["task_id"], "failed")

    repaired = await manager.transition(
        record["task_id"],
        "failed",
        queue_notification=True,
    )

    claimed = await manager.claim_notification(
        repaired["notification_token"],
        "repair-sender",
    )
    assert claimed is not None
    assert claimed["task_id"] == record["task_id"]
    await manager.mark_notification(
        repaired["notification_token"],
        "sent",
        attempt_id="repair-sender",
    )
    await manager.close()


@pytest.mark.asyncio
async def test_terminal_same_state_retry_refreshes_unsent_outbox_payload(tmp_path):
    manager = bg.BackgroundImageTaskManager(tmp_path, heartbeat_seconds=60)
    await manager.start()
    record, _ = await manager.create_task_record(
        _record(manager, "img_notify_refresh"),
        reservation=1,
    )
    terminal = await manager.transition(
        record["task_id"],
        "failed",
        {"error_message": "old failure"},
        queue_notification=True,
    )

    refreshed = await manager.transition(
        record["task_id"],
        "failed",
        {"error_message": "authoritative failure"},
        queue_notification=True,
    )
    claimed = await manager.claim_notification(
        terminal["notification_token"],
        "refresh-sender",
    )

    assert refreshed["error_message"] == "authoritative failure"
    assert claimed["error_message"] == "authoritative failure"
    await manager.mark_notification(
        terminal["notification_token"],
        "sent",
        attempt_id="refresh-sender",
    )
    await manager.close()


@pytest.mark.asyncio
async def test_spooled_inputs_are_private_and_bounded(tmp_path):
    manager = bg.BackgroundImageTaskManager(tmp_path, heartbeat_seconds=60)
    await manager.start()
    paths, manifest = await manager.spool_inputs("img_inputs", [b"one", b"two"])
    assert await manager.read_spooled_inputs(paths, manifest) == [b"one", b"two"]
    assert [item["size"] for item in manifest] == [3, 3]
    assert all(Path(path).is_file() for path in paths)
    Path(paths[0]).write_bytes(b"tampered")
    with pytest.raises(bg.BackgroundTaskError, match="manifest"):
        await manager.read_spooled_inputs(paths, manifest)
    with pytest.raises(bg.BackgroundTaskError):
        await manager.spool_inputs(
            "img_too_large",
            [b"x" * (bg.INPUT_FILE_LIMIT_BYTES + 1)],
        )
    await manager.cleanup_task_files("img_inputs")
    assert not (manager.base_dir / "img_inputs").exists()
    await manager.close()


@pytest.mark.asyncio
async def test_gc_keeps_durable_row_until_spool_cleanup_succeeds(tmp_path, monkeypatch):
    manager = bg.BackgroundImageTaskManager(tmp_path, heartbeat_seconds=60)
    await manager.start()
    await manager.create_task_record(_record(manager, "img_gc"), reservation=1)
    await manager.spool_inputs("img_gc", [b"input"])
    await manager.transition("img_gc", "failed")
    with sqlite3.connect(manager.db_path) as conn:
        conn.execute("UPDATE tasks SET expires_at_ms=0 WHERE task_id='img_gc'")
        conn.commit()

    original_rmtree = bg.shutil.rmtree

    def fail_task_cleanup(path, *args, **kwargs):
        if Path(path).name == "img_gc":
            raise PermissionError("simulated cleanup failure")
        return original_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(bg.shutil, "rmtree", fail_task_cleanup)
    assert await manager.gc() == []
    assert await manager.get_task("img_gc") is not None
    assert (manager.base_dir / "img_gc").exists()

    monkeypatch.setattr(bg.shutil, "rmtree", original_rmtree)
    assert await manager.gc() == ["img_gc"]
    assert await manager.get_task("img_gc") is None
    assert not (manager.base_dir / "img_gc").exists()
    await manager.close()
