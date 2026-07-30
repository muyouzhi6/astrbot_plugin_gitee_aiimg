import asyncio
import json
import sqlite3
import types
from pathlib import Path

import pytest

from test_main_initialize_request_mode import _load_module


class _Result:
    def __init__(self, chain=None):
        self.chain = list(chain or [])


class _Event:
    def __init__(self, *, group=True):
        self.unified_msg_origin = (
            "platform:GroupMessage:group" if group else "platform:FriendMessage:user"
        )
        self.message_str = ""
        self.message_obj = types.SimpleNamespace(message_id="source-message")
        self._extras = {}
        self._result = _Result()
        self._has_send_oper = False
        self._group = group
        self._stopped = False

    def get_platform_name(self):
        return "aiocqhttp"

    def get_platform_id(self):
        return "platform"

    def get_message_type(self):
        return types.SimpleNamespace(
            value="GroupMessage" if self._group else "FriendMessage"
        )

    def get_session_id(self):
        return "group" if self._group else "user"

    def get_group_id(self):
        return "group" if self._group else ""

    def get_self_id(self):
        return "bot"

    def get_sender_id(self):
        return "user"

    def get_sender_name(self):
        return "Alice"

    def get_extra(self, key, default=None):
        return self._extras.get(key, default)

    def set_extra(self, key, value):
        self._extras[key] = value

    def get_result(self):
        return self._result

    def set_result(self, result):
        self._result = result

    def plain_result(self, text):
        return _Result([types.SimpleNamespace(text=text)])

    def chain_result(self, chain):
        return _Result(chain)

    async def send(self, result):
        self.sent_result = result
        self._has_send_oper = True

    def stop_event(self):
        self._stopped = True

    def is_stopped(self):
        return self._stopped


class _Adapter:
    def __init__(self):
        self.committed = []

    def create_event(self, message):
        return message

    def commit_event(self, event):
        self.committed.append(event)


class _ConversationManager:
    def __init__(self, conversation_id="conversation"):
        self.conversation_id = conversation_id
        self.conversation = types.SimpleNamespace(cid=conversation_id)

    async def get_curr_conversation_id(self, umo):
        return self.conversation_id

    async def get_conversation(self, umo, conversation_id):
        if conversation_id != self.conversation_id:
            return None
        return self.conversation


class _Context:
    def __init__(self):
        self.adapter = _Adapter()
        self.conversation_manager = _ConversationManager()
        self.context_aware = None

    def get_config(self, umo=None):
        return {"provider_settings": {"streaming_response": False}}

    def get_platform_inst(self, platform_id):
        return self.adapter if platform_id == "platform" else None

    def get_registered_star(self, name):
        if name != "astrbot_plugin_context_aware" or self.context_aware is None:
            return None
        return types.SimpleNamespace(star_cls=self.context_aware)


def _target(mod):
    return mod.TaskDeliveryTarget(
        platform_id="platform",
        platform_name="aiocqhttp",
        message_type="GroupMessage",
        umo="platform:GroupMessage:group",
        session_id="group",
        group_id="group",
        self_id="bot",
        sender_id="user",
        sender_name="Alice",
        source_message_id="source-message",
        conversation_id="conversation",
    )


def _base_record(manager, task_id, target, *, kind="single", state="queued"):
    scope = manager.scope_hash(
        target.umo,
        target.self_id,
        target.sender_id,
        target.conversation_id,
    )
    return {
        "task_id": task_id,
        "task_kind": kind,
        "state": state,
        "scope_hash": scope,
        "request_fingerprint": f"fingerprint-{task_id}",
        **manager.dataclass_dict(target),
        "mode": "draw",
        "user_prompt": "take a portrait",
        "effective_prompt": "cinematic portrait with window light",
        "image_generated": False,
        "image_sent": False,
        "delivery_state": "not_started",
        "items": [],
    }


def _plugin(mod, manager):
    plugin = object.__new__(mod.GiteeAIImagePlugin)
    plugin.config = {
        "features": {
            "draw": {"enabled": True, "llm_tool_enabled": True},
            "batch": {"max_count": 8},
        }
    }
    plugin.background_tasks = manager
    plugin.context = _Context()
    plugin.registry = types.SimpleNamespace(provider_ids=lambda: [])
    plugin._last_image_by_user = {}
    plugin._last_image_task_meta_cache = {}
    plugin._background_send_gates = {}
    return plugin


@pytest.mark.asyncio
async def test_single_tool_returns_before_provider_finishes(tmp_path):
    mod, _ = _load_module()
    manager = mod.BackgroundImageTaskManager(tmp_path, heartbeat_seconds=60)
    await manager.start()
    plugin = _plugin(mod, manager)
    event = _Event()
    target = _target(mod)
    provider_started = asyncio.Event()
    provider_release = asyncio.Event()

    async def build_target(self, current_event):
        return target

    async def blocked_provider(self, current_manager, job):
        provider_started.set()
        await provider_release.wait()
        return Path(tmp_path / "never-sent.png"), dict(job.task_meta)

    async def no_images(self, current_event):
        return False

    plugin._build_background_delivery_target = types.MethodType(build_target, plugin)
    plugin._execute_prepared_image_job = types.MethodType(blocked_provider, plugin)
    plugin._has_message_images = types.MethodType(no_images, plugin)
    result = await plugin._accept_background_single(
        event,
        prompt="take a portrait",
        mode="text",
        backend="auto",
        output="",
        aspect_ratio="3:4",
        resolution="2K",
    )
    payload = json.loads(result.content[0].text)
    assert payload["status"] == "accepted"
    assert payload["effective_prompt"] == "take a portrait"
    await asyncio.wait_for(provider_started.wait(), timeout=1)
    record = await manager.get_task(payload["task_id"])
    assert record["state"] == "running"
    assert not provider_release.is_set()
    await manager.cancel_task(payload["task_id"], "test cleanup")
    await manager.close()


@pytest.mark.asyncio
async def test_batch_tool_returns_while_planner_is_still_running(tmp_path):
    mod, _ = _load_module()
    manager = mod.BackgroundImageTaskManager(tmp_path, heartbeat_seconds=60)
    await manager.start()
    plugin = _plugin(mod, manager)
    event = _Event()
    target = _target(mod)
    planner_started = asyncio.Event()
    planner_release = asyncio.Event()

    async def build_target(self, current_event):
        return target

    async def blocked_planner(self, **kwargs):
        planner_started.set()
        await planner_release.wait()
        return []

    plugin._build_background_delivery_target = types.MethodType(build_target, plugin)
    plugin._plan_batch_prompt_items = types.MethodType(blocked_planner, plugin)
    result = await plugin._accept_background_batch(
        event,
        prompt="four different portraits",
        count=4,
        mode="draw",
        backend="auto",
        output="",
        aspect_ratio="auto",
        resolution="2K",
    )
    payload = json.loads(result.content[0].text)
    assert payload["status"] == "accepted"
    assert payload["state"] == "planning"
    assert payload["requested_count"] == 4
    await asyncio.wait_for(planner_started.wait(), timeout=1)
    record = await manager.get_task(payload["task_id"])
    assert record["state"] == "planning"
    await manager.cancel_task(payload["task_id"], "test cleanup")
    await manager.close()


@pytest.mark.asyncio
async def test_llm_injection_contains_prompt_and_removes_recursive_tools(tmp_path):
    mod, _ = _load_module()
    manager = mod.BackgroundImageTaskManager(tmp_path, heartbeat_seconds=60)
    await manager.start()
    plugin = _plugin(mod, manager)
    target = _target(mod)
    record, _ = await manager.create_task_record(
        _base_record(manager, "img_inject", target),
        reservation=1,
    )
    event = _Event()
    event.set_extra("_gitee_bg_task_id", record["task_id"])
    req = types.SimpleNamespace(
        conversation=types.SimpleNamespace(cid="conversation"),
        extra_user_content_parts=[],
        system_prompt="",
        func_tool=types.SimpleNamespace(
            tools=[
                types.SimpleNamespace(name="aiimg_generate"),
                types.SimpleNamespace(name="safe_tool"),
            ]
        ),
    )
    await plugin.inject_background_image_tasks(event, req)
    assert len(req.extra_user_content_parts) == 1
    injected = req.extra_user_content_parts[0]
    assert not isinstance(injected, dict)
    serialized = injected.model_dump_for_context()
    assert serialized["_no_save"] is True
    assert "cinematic portrait with window light" in serialized["text"]
    assert [tool.name for tool in req.func_tool.tools] == ["safe_tool"]
    await manager.cancel_task(record["task_id"], "test cleanup")
    await manager.close()


@pytest.mark.asyncio
async def test_llm_injection_does_not_cross_conversation_when_request_is_unbound(
    tmp_path,
):
    mod, _ = _load_module()
    manager = mod.BackgroundImageTaskManager(tmp_path, heartbeat_seconds=60)
    await manager.start()
    plugin = _plugin(mod, manager)
    target = _target(mod)
    record = _base_record(manager, "img_old_conversation", target)
    record["conversation_id"] = "old-conversation"
    record["scope_hash"] = manager.scope_hash(
        target.umo,
        target.self_id,
        target.sender_id,
        "old-conversation",
    )
    stored, _ = await manager.create_task_record(record, reservation=1)
    event = _Event()
    req = types.SimpleNamespace(
        conversation=None,
        extra_user_content_parts=[],
        system_prompt="",
        func_tool=None,
    )

    await plugin.inject_background_image_tasks(event, req)

    assert req.conversation.cid == "conversation"
    assert req.extra_user_content_parts == []
    await manager.cancel_task(stored["task_id"], "test cleanup")
    await manager.close()


def test_background_manager_for_event_fails_closed_and_supports_weixin():
    mod, _ = _load_module()
    manager = types.SimpleNamespace(accepting=True)
    plugin = _plugin(mod, manager)
    event = _Event()

    assert plugin._background_manager_for_event(event) is manager
    event.get_platform_name = lambda: "weixin_oc"
    assert plugin._background_manager_for_event(event) is manager
    event.get_platform_name = lambda: "telegram"
    assert plugin._background_manager_for_event(event) is None

    event.get_platform_name = lambda: "aiocqhttp"
    plugin.context.get_config = lambda umo=None: {
        "provider_settings": {"streaming_response": True}
    }
    assert plugin._background_manager_for_event(event) is None
    plugin.background_tasks = None
    assert plugin._background_manager_for_event(event) is None


@pytest.mark.asyncio
async def test_ack_is_confirmed_only_with_plain_digest_and_transport_send(tmp_path):
    mod, _ = _load_module()
    manager = mod.BackgroundImageTaskManager(tmp_path, heartbeat_seconds=60)
    await manager.start()
    plugin = _plugin(mod, manager)
    target = _target(mod)
    record, _ = await manager.create_task_record(
        _base_record(manager, "img_ack", target),
        reservation=1,
    )
    event = _Event()
    event.set_extra("_gitee_bg_ack_task_id", record["task_id"])
    await plugin.decorate_background_task_result(event)
    decorated = await manager.get_task(record["task_id"])
    assert decorated["ack_state"] == "decorated"
    assert any(isinstance(item, mod.Plain) for item in event.get_result().chain)

    await plugin.arm_background_task_result_transport(event)
    event.get_result().chain[0].text = "downstream TTS rewrote this reply"
    await event.send(event.get_result())
    await plugin.confirm_background_task_result(event)
    sent = await manager.get_task(record["task_id"])
    assert sent["ack_state"] == "sent"
    await manager.cancel_task(record["task_id"], "test cleanup")
    await manager.close()


@pytest.mark.asyncio
async def test_completion_transport_probe_survives_downstream_chain_rewrite(tmp_path):
    mod, _ = _load_module()
    manager = mod.BackgroundImageTaskManager(tmp_path, heartbeat_seconds=60)
    await manager.start()
    plugin = _plugin(mod, manager)
    target = _target(mod)
    record, _ = await manager.create_task_record(
        _base_record(manager, "img_completion_probe", target),
        reservation=1,
    )
    terminal = await manager.transition(
        record["task_id"],
        "completed",
        {
            "image_generated": True,
            "image_sent": True,
            "delivery_state": "confirmed",
        },
        queue_notification=True,
    )
    token = terminal["notification_token"]
    attempt_id = "notify-probe"
    claimed = await manager.claim_notification(token, attempt_id)
    assert claimed is not None
    await manager.mark_notification(token, "queued", attempt_id=attempt_id)

    event = _Event()
    event.set_extra("_gitee_bg_task_id", record["task_id"])
    event.set_extra("_gitee_bg_notification_token", token)
    event.set_extra("_gitee_bg_notification_attempt", attempt_id)
    event.set_result(_Result([mod.Plain("照片发出来了。")]))
    await plugin.decorate_background_task_result(event)
    await plugin.arm_background_task_result_transport(event)
    event.get_result().chain = [types.SimpleNamespace(file="voice.wav")]
    await event.send(event.get_result())
    await plugin.confirm_background_task_result(event)

    updated = await manager.get_task(record["task_id"])
    assert updated["notification_state"] == "sent"
    assert event.send == event._gitee_bg_original_send
    await manager.close()


@pytest.mark.asyncio
async def test_transport_arm_suppresses_agent_reply_after_watchdog_takeover(tmp_path):
    mod, _ = _load_module()
    manager = mod.BackgroundImageTaskManager(tmp_path, heartbeat_seconds=60)
    await manager.start()
    plugin = _plugin(mod, manager)
    target = _target(mod)
    record, _ = await manager.create_task_record(
        _base_record(manager, "img_stale_completion", target),
        reservation=1,
    )
    terminal = await manager.transition(
        record["task_id"],
        "completed",
        {
            "image_generated": True,
            "image_sent": True,
            "delivery_state": "confirmed",
        },
        queue_notification=True,
    )
    token = terminal["notification_token"]
    agent_attempt = "notify-agent"
    claimed = await manager.claim_notification(token, agent_attempt)
    assert claimed is not None
    await manager.mark_notification(token, "queued", attempt_id=agent_attempt)
    watchdog_attempt = "notify-watchdog"
    watchdog_claim = await manager.claim_notification(token, watchdog_attempt)
    assert watchdog_claim is not None

    event = _Event()
    event.set_extra("_gitee_bg_task_id", record["task_id"])
    event.set_extra("_gitee_bg_notification_token", token)
    event.set_extra("_gitee_bg_notification_attempt", agent_attempt)
    event.set_result(_Result([mod.Plain("照片发出来了。")]))
    await plugin.arm_background_task_result_transport(event)

    assert event.is_stopped()
    assert not event.get_extra("_gitee_bg_transport_probe_armed", False)
    await manager.mark_notification(token, "sent", attempt_id=watchdog_attempt)
    await manager.close()


@pytest.mark.asyncio
async def test_transport_probe_does_not_confirm_failed_send(tmp_path):
    mod, _ = _load_module()
    manager = mod.BackgroundImageTaskManager(tmp_path, heartbeat_seconds=60)
    await manager.start()
    plugin = _plugin(mod, manager)
    target = _target(mod)
    record, _ = await manager.create_task_record(
        _base_record(manager, "img_ack_failed_send", target),
        reservation=1,
    )

    class _FailedEvent(_Event):
        async def send(self, result):
            raise RuntimeError("transport failed")

    event = _FailedEvent()
    event.set_extra("_gitee_bg_ack_task_id", record["task_id"])
    await plugin.decorate_background_task_result(event)
    await plugin.arm_background_task_result_transport(event)
    with pytest.raises(RuntimeError, match="transport failed"):
        await event.send(event.get_result())
    await plugin.confirm_background_task_result(event)

    updated = await manager.get_task(record["task_id"])
    assert updated["ack_state"] == "unknown"
    await manager.cancel_task(record["task_id"], "test cleanup")
    await manager.close()


@pytest.mark.asyncio
async def test_transport_probe_marks_split_send_partial_failure_unknown(tmp_path):
    mod, _ = _load_module()
    manager = mod.BackgroundImageTaskManager(tmp_path, heartbeat_seconds=60)
    await manager.start()
    plugin = _plugin(mod, manager)
    target = _target(mod)
    record, _ = await manager.create_task_record(
        _base_record(manager, "img_ack_partial_send", target),
        reservation=1,
    )

    class _PartialEvent(_Event):
        def __init__(self):
            super().__init__()
            self.send_calls = 0

        async def send(self, result):
            self.send_calls += 1
            if self.send_calls == 2:
                raise RuntimeError("second split send failed")
            self._has_send_oper = True

    event = _PartialEvent()
    event.set_extra("_gitee_bg_ack_task_id", record["task_id"])
    await plugin.decorate_background_task_result(event)
    await plugin.arm_background_task_result_transport(event)
    await event.send(event.get_result())
    with pytest.raises(RuntimeError, match="second split send failed"):
        await event.send(event.get_result())
    await plugin.confirm_background_task_result(event)

    updated = await manager.get_task(record["task_id"])
    assert updated["ack_state"] == "unknown"
    await manager.cancel_task(record["task_id"], "test cleanup")
    await manager.close()


@pytest.mark.asyncio
async def test_contextaware_session_gate_prevents_old_context_reentry(tmp_path):
    mod, _ = _load_module()
    manager = mod.BackgroundImageTaskManager(tmp_path, heartbeat_seconds=60)
    await manager.start()
    plugin = _plugin(mod, manager)
    target = _target(mod)
    plugin.context.context_aware = types.SimpleNamespace(has_session=lambda umo: False)
    safe, conversation = await plugin._background_context_is_safe(target)
    assert safe is False
    assert conversation is None

    plugin.context.context_aware = types.SimpleNamespace(has_session=lambda umo: True)
    safe, conversation = await plugin._background_context_is_safe(target)
    assert safe is True
    assert conversation.cid == "conversation"
    await manager.close()


@pytest.mark.asyncio
async def test_astrbot_loaded_drains_recovery_notifications_once(tmp_path):
    mod, _ = _load_module()
    manager = mod.BackgroundImageTaskManager(tmp_path, heartbeat_seconds=60)
    await manager.start()
    plugin = _plugin(mod, manager)
    target = _target(mod)
    record, _ = await manager.create_task_record(
        _base_record(manager, "img_recovery_hook", target),
        reservation=1,
    )
    terminal = await manager.transition(
        record["task_id"],
        "interrupted",
        {
            "error_code": "process_restarted",
            "terminal_reason": "process_restarted",
        },
        queue_notification=True,
    )
    plugin._background_astrbot_loaded = False
    plugin._background_recovery_records = [terminal]
    dispatched = []

    async def capture_dispatch(self, current_manager, current_record, current_target):
        dispatched.append((current_record["task_id"], current_target.umo))

    plugin._dispatch_background_completion = types.MethodType(
        capture_dispatch,
        plugin,
    )

    await plugin.on_background_astrbot_loaded()
    await plugin.on_background_astrbot_loaded()

    assert plugin._background_astrbot_loaded is True
    assert plugin._background_recovery_records == []
    assert dispatched == [(record["task_id"], target.umo)]
    await manager.close()


@pytest.mark.asyncio
async def test_synthetic_completion_uses_non_content_input_chain(tmp_path):
    mod, _ = _load_module()
    manager = mod.BackgroundImageTaskManager(tmp_path, heartbeat_seconds=60)
    await manager.start()
    plugin = _plugin(mod, manager)
    target = _target(mod)
    record, _ = await manager.create_task_record(
        _base_record(manager, "img_complete", target),
        reservation=1,
    )
    terminal = await manager.transition(
        record["task_id"],
        "completed",
        {
            "image_generated": True,
            "image_sent": True,
            "delivery_state": "confirmed",
        },
        queue_notification=True,
    )
    captured = {}

    async def safe_context(self, current_target):
        return True, types.SimpleNamespace(cid="conversation")

    class _Synthetic(_Event):
        def request_llm(self, **kwargs):
            return types.SimpleNamespace(**kwargs)

    async def rebuild(self, current_target, *, message=None, message_str=""):
        captured["message"] = list(message or [])
        return _Synthetic()

    plugin._background_context_is_safe = types.MethodType(safe_context, plugin)
    plugin._rebuild_background_event = types.MethodType(rebuild, plugin)
    await plugin._dispatch_background_completion(manager, terminal, target)
    assert len(plugin.context.adapter.committed) == 1
    assert captured["message"]
    assert not any(
        isinstance(component, mod.Plain) for component in captured["message"]
    )
    synthetic = plugin.context.adapter.committed[0]
    synthetic.set_result(_Result([mod.Plain("照片发出来了。")]))
    await plugin.decorate_background_task_result(synthetic)
    await plugin.arm_background_task_result_transport(synthetic)
    await synthetic.send(synthetic.get_result())
    await plugin.confirm_background_task_result(synthetic)

    updated = await manager.get_task(record["task_id"])
    assert updated["notification_state"] == "sent"
    assert synthetic.send == synthetic._gitee_bg_original_send
    await manager.close()


@pytest.mark.asyncio
async def test_deterministic_notification_without_transport_confirmation_is_unknown(
    tmp_path,
):
    mod, _ = _load_module()
    manager = mod.BackgroundImageTaskManager(tmp_path, heartbeat_seconds=60)
    await manager.start()
    plugin = _plugin(mod, manager)
    target = _target(mod)
    record, _ = await manager.create_task_record(
        _base_record(manager, "img_deterministic_unknown", target),
        reservation=1,
    )
    terminal = await manager.transition(
        record["task_id"],
        "failed",
        queue_notification=True,
    )
    token = terminal["notification_token"]
    attempt_id = "deterministic-sender"
    claimed = await manager.claim_notification(token, attempt_id)

    class _DirectEvent(_Event):
        async def send(self, result):
            self.sent_result = result

    async def rebuild(self, current_target, *, message=None, message_str=""):
        return _DirectEvent()

    plugin._rebuild_background_event = types.MethodType(rebuild, plugin)
    await plugin._send_deterministic_background_notification(
        manager,
        claimed,
        target,
        attempt_id=attempt_id,
    )

    updated = await manager.get_task(record["task_id"])
    assert updated["notification_state"] == "unknown"
    await manager.close()


@pytest.mark.asyncio
async def test_task_status_limits_batch_prompt_page_to_eight(tmp_path):
    mod, _ = _load_module()
    manager = mod.BackgroundImageTaskManager(tmp_path, heartbeat_seconds=60)
    await manager.start()
    plugin = _plugin(mod, manager)
    target = _target(mod)
    record = _base_record(
        manager,
        "batch_status",
        target,
        kind="batch",
        state="planning",
    )
    record["requested_count"] = 10
    stored, _ = await manager.create_task_record(record, reservation=10)
    items = [
        {
            "item_id": f"item-{index}",
            "index": index,
            "state": "queued",
            "effective_prompt": f"prompt-{index}",
        }
        for index in range(10)
    ]
    await manager.transition(
        stored["task_id"],
        "queued",
        {"items": items, "planned_count": 10},
    )

    async def build_target(self, current_event):
        return target

    plugin._build_background_delivery_target = types.MethodType(build_target, plugin)
    result = await plugin.aiimg_task_status(
        _Event(),
        task_id=stored["task_id"],
        include_prompts=True,
        offset=0,
        limit=99,
    )
    payload = json.loads(result.content[0].text)
    assert len(payload["items"]) == 8
    assert payload["next_offset"] == 8
    assert payload["items"][0]["effective_prompt"] == "prompt-0"

    hidden_result = await plugin.aiimg_task_status(
        _Event(),
        task_id=stored["task_id"],
        include_prompts=False,
        offset=8,
        limit=0,
    )
    hidden_payload = json.loads(hidden_result.content[0].text)
    assert hidden_payload["limit"] == 1
    assert hidden_payload["offset"] == 8
    assert hidden_payload["user_prompt"] is None
    assert hidden_payload["effective_prompt"] is None
    assert "effective_prompt" not in hidden_payload["items"][0]
    await manager.cancel_task(stored["task_id"], "test cleanup")
    await manager.close()


@pytest.mark.asyncio
async def test_task_status_rejects_explicit_task_from_another_conversation(tmp_path):
    mod, _ = _load_module()
    manager = mod.BackgroundImageTaskManager(tmp_path, heartbeat_seconds=60)
    await manager.start()
    plugin = _plugin(mod, manager)
    target = _target(mod)
    record = _base_record(manager, "img_private_prompt", target)
    record["conversation_id"] = "old-conversation"
    record["scope_hash"] = manager.scope_hash(
        target.umo,
        target.self_id,
        target.sender_id,
        "old-conversation",
    )
    stored, _ = await manager.create_task_record(record, reservation=1)

    async def build_target(self, current_event):
        return target

    plugin._build_background_delivery_target = types.MethodType(build_target, plugin)
    result = await plugin.aiimg_task_status(
        _Event(),
        task_id=stored["task_id"],
        include_prompts=True,
    )
    payload = json.loads(result.content[0].text)
    assert payload["status"] == "forbidden"
    assert "effective_prompt" not in payload
    await manager.cancel_task(stored["task_id"], "test cleanup")
    await manager.close()


@pytest.mark.asyncio
async def test_background_batch_rejects_planner_count_mismatch(tmp_path):
    mod, _ = _load_module()
    manager = mod.BackgroundImageTaskManager(tmp_path, heartbeat_seconds=60)
    await manager.start()
    plugin = _plugin(mod, manager)
    target = _target(mod)
    record = _base_record(
        manager,
        "batch_short_plan",
        target,
        kind="batch",
        state="planning",
    )
    record["requested_count"] = 2
    stored, _ = await manager.create_task_record(record, reservation=2)

    async def short_plan(self, **kwargs):
        return [
            mod.PlannedPromptItem(
                title="only one",
                prompt="single planned prompt",
                variation_focus="one",
                aspect_ratio="3:4",
            )
        ]

    dispatched = []

    async def capture_dispatch(self, current_manager, current_record, current_target):
        dispatched.append(current_record)

    plugin._plan_batch_prompt_items = types.MethodType(short_plan, plugin)
    plugin._dispatch_background_completion = types.MethodType(
        capture_dispatch,
        plugin,
    )
    job = mod.PreparedBatchJob(
        mode="draw",
        user_prompt="two portraits",
        requested_count=2,
        backend=None,
        output={"exact_size": None, "aspect_ratio": "3:4", "resolution": "2K"},
    )

    await plugin._run_background_batch(manager, stored["task_id"], job, target)

    current = await manager.get_task(stored["task_id"])
    assert current["state"] == "failed"
    assert "reserved slots" in current["error_message"]
    assert len(dispatched) == 1
    await manager.close()


@pytest.mark.asyncio
async def test_background_batch_sends_successes_in_index_order_after_out_of_order_generation(
    tmp_path,
):
    mod, _ = _load_module()
    manager = mod.BackgroundImageTaskManager(
        tmp_path,
        max_running=2,
        heartbeat_seconds=60,
    )
    await manager.start()
    plugin = _plugin(mod, manager)
    plugin.config["features"]["draw"]["batch_concurrency"] = 2
    target = _target(mod)
    record = _base_record(
        manager,
        "batch_ordered_delivery",
        target,
        kind="batch",
        state="planning",
    )
    record.update({"requested_count": 3, "ack_state": "sent"})
    stored, _ = await manager.create_task_record(record, reservation=3)
    third_generated = asyncio.Event()
    release_first = asyncio.Event()
    generation_order = []
    send_order = []
    dispatched = []

    async def plan(self, **kwargs):
        return [
            mod.PlannedPromptItem(
                title=f"item-{index}",
                prompt=f"prompt-{index}",
                variation_focus=[f"focus-{index}"],
                aspect_ratio="3:4",
            )
            for index in range(1, 4)
        ]

    async def execute(self, current_manager, job):
        index = int(job.user_prompt.rsplit("-", 1)[1])
        if index == 1:
            await release_first.wait()
        elif index == 2:
            raise RuntimeError("controlled child failure")
        else:
            third_generated.set()
        generation_order.append(index)
        image_path = tmp_path / f"image-{index}.png"
        image_path.write_bytes(f"image-{index}".encode())
        return image_path, {"index": index}

    async def send_once(self, current_target, image_path):
        send_order.append(int(image_path.stem.rsplit("-", 1)[1]))
        return _Event()

    def remember(self, delivery_event, image_path):
        return None

    async def save_meta(self, delivery_event, task_meta):
        return None

    async def capture_dispatch(self, current_manager, current_record, current_target):
        dispatched.append(current_record)

    plugin._plan_batch_prompt_items = types.MethodType(plan, plugin)
    plugin._execute_prepared_image_job = types.MethodType(execute, plugin)
    plugin._send_background_image_once = types.MethodType(send_once, plugin)
    plugin._remember_last_image = types.MethodType(remember, plugin)
    plugin._save_last_image_task_meta = types.MethodType(save_meta, plugin)
    plugin._dispatch_background_completion = types.MethodType(
        capture_dispatch,
        plugin,
    )
    job = mod.PreparedBatchJob(
        mode="draw",
        user_prompt="three portraits",
        requested_count=3,
        backend=None,
        output={"exact_size": None, "aspect_ratio": "3:4", "resolution": "2K"},
    )

    batch_task = asyncio.create_task(
        plugin._run_background_batch(manager, stored["task_id"], job, target)
    )
    await asyncio.wait_for(third_generated.wait(), timeout=1)
    release_first.set()
    await asyncio.wait_for(batch_task, timeout=2)

    current = await manager.get_task(stored["task_id"])
    assert generation_order == [3, 1]
    assert send_order == [1, 3], json.dumps(current, ensure_ascii=False, indent=2)
    assert current["state"] == "partial"
    assert current["sent_count"] == 2
    assert current["failed_count"] == 1
    assert [item["state"] for item in current["items"]] == [
        "completed",
        "failed",
        "completed",
    ]
    assert len(dispatched) == 1
    await manager.close()


@pytest.mark.parametrize(
    ("global_limit", "child_slots", "expected_peak"),
    [(2, 1, 1), (1, 2, 1), (2, 2, 2)],
)
@pytest.mark.asyncio
async def test_background_batch_child_respects_parent_and_global_limits(
    tmp_path,
    global_limit,
    child_slots,
    expected_peak,
):
    mod, _ = _load_module()
    manager = mod.BackgroundImageTaskManager(
        tmp_path,
        max_running=global_limit,
        heartbeat_seconds=60,
    )
    await manager.start()
    plugin = _plugin(mod, manager)
    target = _target(mod)
    record = _base_record(
        manager,
        "batch_combined_limits",
        target,
        kind="batch",
        state="queued",
    )
    items = [
        {
            "item_id": f"item-{index}",
            "index": index,
            "state": "queued",
            "image_generated": False,
            "image_sent": False,
            "delivery_state": "not_started",
        }
        for index in range(1, 5)
    ]
    record.update({"requested_count": len(items), "items": items})
    stored, _ = await manager.create_task_record(record, reservation=len(items))
    active = 0
    peak = 0

    async def execute(self, current_manager, job):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        try:
            await asyncio.sleep(0.02)
            return tmp_path / f"{job.user_prompt}.png", {}
        finally:
            active -= 1

    plugin._execute_prepared_image_job = types.MethodType(execute, plugin)
    child_limit = asyncio.Semaphore(child_slots)
    generated = {}
    jobs = [
        mod.PreparedImageJob(
            mode="draw",
            user_prompt=f"prompt-{index}",
            effective_prompt=f"prompt-{index}",
            backend=None,
            output={"exact_size": None, "aspect_ratio": "3:4", "resolution": "2K"},
        )
        for index in range(1, 5)
    ]

    await asyncio.gather(
        *(
            plugin._run_background_batch_child(
                manager,
                stored["task_id"],
                item,
                job,
                child_limit,
                generated,
            )
            for item, job in zip(items, jobs, strict=True)
        )
    )

    assert peak == expected_peak
    assert len(generated) == len(items)
    current = await manager.get_task(stored["task_id"])
    assert [item["state"] for item in current["items"]] == ["generated"] * len(items)
    await manager.cancel_task(stored["task_id"], "test cleanup")
    await manager.close()


@pytest.mark.asyncio
async def test_background_single_provider_failure_persists_and_dispatches(tmp_path):
    mod, _ = _load_module()
    manager = mod.BackgroundImageTaskManager(tmp_path, heartbeat_seconds=60)
    await manager.start()
    plugin = _plugin(mod, manager)
    target = _target(mod)
    record, _ = await manager.create_task_record(
        _base_record(manager, "img_provider_failure", target),
        reservation=1,
    )
    dispatched = []

    async def fail_provider(self, current_manager, job):
        raise RuntimeError("controlled provider failure")

    async def capture_dispatch(self, current_manager, current_record, current_target):
        dispatched.append(current_record)

    plugin._execute_prepared_image_job = types.MethodType(fail_provider, plugin)
    plugin._dispatch_background_completion = types.MethodType(
        capture_dispatch,
        plugin,
    )
    job = mod.PreparedImageJob(
        mode="text",
        user_prompt="failure test",
        effective_prompt="controlled failure prompt",
        backend=None,
        output={"exact_size": None, "aspect_ratio": "3:4", "resolution": "2K"},
    )

    await plugin._run_background_single(manager, record["task_id"], job, target)

    current = await manager.get_task(record["task_id"])
    assert current["state"] == "failed"
    assert current["error_code"] == "provider_failed"
    assert current["notification_state"] == "pending"
    assert len(dispatched) == 1
    await manager.close()


@pytest.mark.asyncio
async def test_background_single_timeout_marks_delivery_unknown_without_retry(tmp_path):
    mod, _ = _load_module()
    manager = mod.BackgroundImageTaskManager(tmp_path, heartbeat_seconds=60)
    await manager.start()
    plugin = _plugin(mod, manager)
    target = _target(mod)
    record = _base_record(manager, "img_delivery_timeout", target)
    record["ack_state"] = "sent"
    stored, _ = await manager.create_task_record(record, reservation=1)
    provider_calls = 0
    send_calls = 0
    dispatched = []
    image_path = tmp_path / "timeout-image.png"
    image_path.write_bytes(b"image")

    async def execute(self, current_manager, job):
        nonlocal provider_calls
        provider_calls += 1
        return image_path, {"mode": "draw"}

    async def timeout_send(self, current_target, current_path):
        nonlocal send_calls
        send_calls += 1
        raise TimeoutError("controlled image send timeout")

    async def capture_dispatch(self, current_manager, current_record, current_target):
        dispatched.append(current_record)

    plugin._execute_prepared_image_job = types.MethodType(execute, plugin)
    plugin._send_background_image_once = types.MethodType(timeout_send, plugin)
    plugin._dispatch_background_completion = types.MethodType(
        capture_dispatch,
        plugin,
    )
    job = mod.PreparedImageJob(
        mode="text",
        user_prompt="timeout test",
        effective_prompt="timeout test prompt",
        backend=None,
        output={"exact_size": None, "aspect_ratio": "3:4", "resolution": "2K"},
    )

    await plugin._run_background_single(manager, stored["task_id"], job, target)

    current = await manager.get_task(stored["task_id"])
    assert provider_calls == 1
    assert send_calls == 1
    assert current["state"] == "interrupted"
    assert current["delivery_state"] == "unknown"
    assert current["error_code"] == "delivery_unknown"
    assert current["notification_state"] == "pending"
    with sqlite3.connect(manager.db_path) as conn:
        receipts = conn.execute(
            "SELECT delivery_state FROM receipts WHERE task_id=?",
            (stored["task_id"],),
        ).fetchall()
    assert receipts == [("unknown",)]
    assert len(dispatched) == 1
    await manager.close()


@pytest.mark.asyncio
async def test_failed_reset_releases_send_gate_without_cancelling_task(tmp_path):
    mod, _ = _load_module()
    manager = mod.BackgroundImageTaskManager(tmp_path, heartbeat_seconds=60)
    await manager.start()
    plugin = _plugin(mod, manager)
    plugin._background_send_gates = {}
    target = _target(mod)
    record, _ = await manager.create_task_record(
        _base_record(manager, "img_reset_guard", target),
        reservation=1,
    )
    event = _Event()
    event.message_str = "/reset"
    await plugin.handle_background_session_commands(event)
    gate = plugin._background_send_gates[target.umo]
    assert not gate.is_set()

    waiter = asyncio.create_task(plugin._wait_background_send_gate(target.umo))
    await asyncio.sleep(0)
    assert not waiter.done()
    await plugin.confirm_background_task_result(event)
    await asyncio.wait_for(waiter, timeout=1)
    current = await manager.get_task(record["task_id"])
    assert current["state"] == "queued"
    await manager.cancel_task(record["task_id"], "test cleanup")
    await manager.close()


@pytest.mark.asyncio
async def test_successful_new_cancels_and_hides_old_conversation_task(tmp_path):
    mod, _ = _load_module()
    manager = mod.BackgroundImageTaskManager(tmp_path, heartbeat_seconds=60)
    await manager.start()
    plugin = _plugin(mod, manager)
    plugin._background_send_gates = {}
    target = _target(mod)
    record, _ = await manager.create_task_record(
        _base_record(manager, "img_new_conversation", target),
        reservation=1,
    )
    dispatched = []

    async def capture_dispatch(self, current_manager, current_record, current_target):
        dispatched.append(current_record)

    plugin._dispatch_background_completion = types.MethodType(
        capture_dispatch,
        plugin,
    )
    event = _Event()
    event.message_str = "/new"
    await plugin.handle_background_session_commands(event)
    event.set_extra("_clean_group_context_session", True)
    await plugin.confirm_background_task_result(event)

    current = await manager.get_task(record["task_id"])
    assert current["state"] == "cancelled"
    assert current["suppress_future_injection"] is True
    assert len(dispatched) == 1
    assert target.umo not in plugin._background_send_gates

    plugin.context.conversation_manager = _ConversationManager("new-conversation")
    req = types.SimpleNamespace(
        conversation=None,
        extra_user_content_parts=[],
        system_prompt="",
        func_tool=None,
    )
    await plugin.inject_background_image_tasks(_Event(), req)
    assert req.conversation.cid == "new-conversation"
    assert req.extra_user_content_parts == []
    await manager.close()


@pytest.mark.asyncio
async def test_shutdown_persists_interrupted_task_for_next_owner(tmp_path):
    mod, _ = _load_module()
    manager = mod.BackgroundImageTaskManager(tmp_path, heartbeat_seconds=60)
    await manager.start()
    plugin = _plugin(mod, manager)
    event = _Event()
    target = _target(mod)
    provider_started = asyncio.Event()

    async def build_target(self, current_event):
        return target

    async def no_images(self, current_event):
        return False

    async def blocked_provider(self, current_manager, job):
        provider_started.set()
        await asyncio.sleep(60)

    async def no_dispatch(self, current_manager, record, current_target):
        return None

    plugin._build_background_delivery_target = types.MethodType(build_target, plugin)
    plugin._has_message_images = types.MethodType(no_images, plugin)
    plugin._execute_prepared_image_job = types.MethodType(blocked_provider, plugin)
    plugin._dispatch_background_completion = types.MethodType(no_dispatch, plugin)
    result = await plugin._accept_background_single(
        event,
        prompt="shutdown test",
        mode="text",
        backend="auto",
        output="",
        aspect_ratio="3:4",
        resolution="2K",
    )
    task_id = json.loads(result.content[0].text)["task_id"]
    await asyncio.wait_for(provider_started.wait(), timeout=1)
    await manager.close()

    next_owner = mod.BackgroundImageTaskManager(tmp_path, heartbeat_seconds=60)
    recovered = await next_owner.start()
    assert any(record["task_id"] == task_id for record in recovered)
    record = await next_owner.get_task(task_id)
    assert record["state"] == "interrupted"
    assert record["notification_state"] == "pending"
    await next_owner.close()
