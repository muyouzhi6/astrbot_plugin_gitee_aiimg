import asyncio
import json
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
    assert injected["_no_save"] is True
    assert "cinematic portrait with window light" in injected["text"]
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

    event._has_send_oper = True
    await plugin.confirm_background_task_result(event)
    sent = await manager.get_task(record["task_id"])
    assert sent["ack_state"] == "sent"
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
    updated = await manager.get_task(record["task_id"])
    assert updated["notification_state"] == "queued"
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
