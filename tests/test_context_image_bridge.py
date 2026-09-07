import asyncio
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from test_main_initialize_request_mode import _load_module
from test_background_pipeline_event import _Event, _plugin, _target


class Peer:
    image_reference_api_version = 1

    def __init__(self):
        self.rows = {"ca_cat": b"cat-reference", "ca_output": b"generated-result"}
        self.registered = []

    async def resolve_reference_images(self, event, ids):
        return [
            {
                "id": i,
                "data": self.rows[i],
                "sha256": hashlib.sha256(self.rows[i]).hexdigest(),
                "message_id": "source",
                "quality": "retained_input",
            }
            for i in ids
        ]

    def capture_image_request(self, event, conversation_id=""):
        return {"sender": event.get_sender_id(), "cid": conversation_id}

    async def register_generated_image(self, token, **kwargs):
        self.registered.append((token, kwargs))
        return "ca_new_result"


def setup_plugin():
    mod, _ = _load_module()
    plugin = _plugin(mod, None)
    peer = Peer()
    plugin.context.context_aware = peer
    plugin.debouncer = SimpleNamespace(
        llm_tool_is_duplicate=lambda *a: False, hit=lambda *a: False
    )
    plugin._signal_llm_tool_failure = AsyncMock()
    return mod, plugin, peer


@pytest.mark.asyncio
async def test_invalid_explicit_reference_stops_before_any_generation():
    mod, plugin, peer = setup_plugin()
    plugin._accept_background_single = AsyncMock()
    result = await plugin.aiimg_generate(
        _Event(),
        prompt="edit",
        mode="edit",
        reference_image_ids=["ca_missing"],
        reference_roles=["subject"],
    )
    assert "no generation started" in result.content[0].text
    plugin._accept_background_single.assert_not_awaited()


@pytest.mark.asyncio
async def test_explicit_text_mode_never_silently_drops_reference():
    mod, plugin, peer = setup_plugin()
    result = await plugin.aiimg_generate(
        _Event(),
        prompt="cat",
        mode="text",
        reference_image_ids=["ca_cat"],
        reference_roles=["object"],
    )
    assert "text-only mode" in result.content[0].text


@pytest.mark.asyncio
async def test_auto_with_references_resolves_to_edit_before_background_dispatch():
    mod, plugin, peer = setup_plugin()
    plugin._background_manager_for_event = lambda e: object()
    plugin._should_auto_selfie_ref = AsyncMock(return_value=False)
    plugin._accept_background_single = AsyncMock(return_value="accepted")
    result = await plugin.aiimg_generate(
        _Event(),
        prompt="change background",
        mode="auto",
        reference_image_ids=["ca_output"],
        reference_roles=["subject"],
    )
    assert result == "accepted"
    assert plugin._accept_background_single.call_args.kwargs["mode"] == "edit"


@pytest.mark.asyncio
async def test_background_edit_spools_selected_bytes_and_roles_before_source_disappears(
    tmp_path,
):
    mod, plugin, peer = setup_plugin()
    manager = mod.BackgroundImageTaskManager(tmp_path, heartbeat_seconds=60)
    await manager.start()
    captured = []
    try:
        plugin._background_manager_for_event = lambda e: manager
        plugin._build_background_delivery_target = AsyncMock(return_value=_target(mod))
        plugin._has_message_images = AsyncMock(return_value=False)
        mod.get_images_from_event = AsyncMock(return_value=[])
        plugin._image_segs_to_bytes = AsyncMock(return_value=[])

        # Capture the actual immutable job without starting a provider.
        async def worker(m, task_id, job, target, **kwargs):
            captured.append(job)

        plugin._run_background_single = worker
        manager.start_worker = lambda task_id, factory: captured.append(factory)
        event = _Event()
        await plugin._prepare_context_reference_request(event, ["ca_cat"], ["subject"])
        result = await plugin._accept_background_single(
            event,
            prompt="snow background",
            mode="edit",
            backend="auto",
            output="",
            aspect_ratio="4:3",
            resolution="4K",
        )
        record = await manager.get_task(json.loads(result.content[0].text)["task_id"])
        factory = captured.pop()
        await factory()
        job = captured[0]
        peer.rows.clear()
        event._extras.clear()
        images = await manager.read_spooled_inputs(
            job.input_paths, job.options["input_manifest"]
        )
        assert images == [b"cat-reference"]
        assert job.task_meta["reference_sources"][0]["id"] == "ca_cat"
        assert "待编辑的主体" in job.effective_prompt
        assert job.output["resolution"] == "4K"
        assert record["input_manifest"]
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_selfie_preparation_keeps_identity_before_cat_and_native_clothing():
    mod, plugin, peer = setup_plugin()
    plugin._get_selfie_conf = lambda: {}
    plugin._is_selfie_enabled = lambda: True
    plugin._get_selfie_reference_paths = AsyncMock(
        return_value=([Path("identity")], "webui")
    )
    plugin._read_paths_bytes = AsyncMock(return_value=[b"identity-photo"])
    mod.get_images_from_event = AsyncMock(return_value=["native-clothing"])
    plugin._image_segs_to_bytes = AsyncMock(return_value=[b"clothing-photo"])
    plugin._get_life_context_without_llm = AsyncMock(return_value={})
    plugin._get_selfie_default_output = lambda: ""
    event = _Event()
    await plugin._prepare_context_reference_request(event, ["ca_cat"], ["object"])
    images, prompt, options, meta = await plugin._prepare_background_selfie(
        event, "抱着图里的猫拍照", None
    )
    assert images == [b"identity-photo", b"cat-reference", b"clothing-photo"]
    assert "参考图 2" in prompt and "动物或物体" in prompt
    assert options["reference_count"] == 1
    assert options["extra_reference_count"] == 2


@pytest.mark.asyncio
async def test_registration_only_after_successful_delivery_and_carries_parents(
    tmp_path,
):
    mod, plugin, peer = setup_plugin()
    event = _Event()
    await plugin._prepare_context_reference_request(event, ["ca_cat"], ["subject"])
    path = tmp_path / "result.png"
    path.write_bytes(b"result-bytes")
    plugin._send_image_with_fallback = AsyncMock(
        return_value=SimpleNamespace(ok=False, reason="failed")
    )
    # SendImageResult supplies bool semantics; use its actual type.
    plugin._send_image_with_fallback.return_value = mod.SendImageResult(False, "failed")
    meta = {
        "mode": "edit",
        "continue_with": "edit",
        "reference_sources": [{"id": "ca_cat", "role": "subject"}],
    }
    await plugin._finalize_llm_tool_image(event, path, task_meta=meta)
    assert not peer.registered
    plugin._send_image_with_fallback.return_value = mod.SendImageResult(True, "ok")
    plugin._save_last_image_task_meta = AsyncMock()
    mod.mark_success = AsyncMock()
    result = await plugin._finalize_llm_tool_image(event, path, task_meta=meta)
    assert peer.registered[0][1]["parent_ids"] == ["ca_cat"]
    assert "ca_new_result" in result.content[0].text
    assert "reference_sources" in meta


@pytest.mark.asyncio
async def test_reference_role_validation_and_native_dedup():
    mod, plugin, peer = setup_plugin()
    event = _Event()
    for ids, roles in [
        (["ca_cat"], []),
        (["ca_cat"], ["identity"]),
        ([{}], ["subject"]),
        (["ca_cat"], [{}]),
    ]:
        with pytest.raises(ValueError):
            await mod.resolve_selection(peer, event, ids, roles)
    selection = await mod.resolve_selection(peer, event, ["ca_cat"], ["object"])
    images, prompt = mod.compose_inputs(
        selection, [b"cat-reference", b"other"], identity_images=[b"face"]
    )
    assert images == [b"face", b"cat-reference", b"other"]
    assert "不能被后面的动物" in prompt


@pytest.mark.asyncio
async def test_optional_peer_failure_does_not_break_plain_generation():
    mod, plugin, peer = setup_plugin()
    plugin.context.context_aware = None
    selection = await plugin._prepare_context_reference_request(_Event(), [], [])
    assert selection is None


@pytest.mark.asyncio
async def test_background_receipt_survives_original_event_release(tmp_path):
    mod, plugin, peer = setup_plugin()
    event = _Event()
    await plugin._prepare_context_reference_request(event, [], [])
    receipt = event.get_extra(mod.RECEIPT_KEY)
    event._extras.clear()
    path = tmp_path / "result.png"
    path.write_bytes(b"result")
    meta = {"reference_sources": [{"id": "ca_cat", "role": "object"}]}
    await plugin._publish_context_result(
        _Event(), path, meta, receipt=receipt, task_id="task-A"
    )
    assert peer.registered[0][1]["task_id"] == "task-A"
    assert meta["result_image_id"] == "ca_new_result"


@pytest.mark.asyncio
async def test_same_event_concurrent_call_cannot_replace_selection():
    mod, plugin, peer = setup_plugin()
    entered, release = asyncio.Event(), asyncio.Event()
    original = peer.resolve_reference_images

    async def resolve(event, ids):
        entered.set()
        await release.wait()
        return await original(event, ids)

    peer.resolve_reference_images = resolve
    plugin._background_manager_for_event = lambda e: object()
    plugin._accept_background_single = AsyncMock(return_value="accepted")
    event = _Event()
    first = asyncio.create_task(
        plugin.aiimg_generate(
            event,
            prompt="cat",
            mode="edit",
            reference_image_ids=["ca_cat"],
            reference_roles=["subject"],
        )
    )
    await entered.wait()
    second = await plugin.aiimg_generate(
        event,
        prompt="other",
        mode="edit",
        reference_image_ids=["ca_output"],
        reference_roles=["subject"],
    )
    assert "concurrently" in second.content[0].text
    release.set()
    assert await first == "accepted"
    assert plugin._context_selection(event).sources[0]["id"] == "ca_cat"


@pytest.mark.asyncio
async def test_batch_history_references_require_background_before_dispatch():
    mod, plugin, peer = setup_plugin()
    plugin._accept_background_batch = AsyncMock()
    result = await plugin.aiimg_batch_generate(
        _Event(),
        prompt="make 4 variants",
        count=4,
        reference_image_ids=["ca_cat"],
        reference_roles=["subject"],
    )
    assert "require background task mode" in result.content[0].text
    plugin._accept_background_batch.assert_not_awaited()


@pytest.mark.asyncio
async def test_result_without_receipt_has_explicit_registration_state(tmp_path):
    mod, plugin, peer = setup_plugin()
    meta = {}
    await plugin._publish_context_result(_Event(), tmp_path / "irrelevant", meta)
    assert meta["result_registration"] == "unavailable_receipt"


@pytest.mark.asyncio
async def test_identical_explicit_bytes_keep_distinct_roles_and_indices():
    mod, plugin, peer = setup_plugin()
    peer.rows["ca_output"] = peer.rows["ca_cat"]
    selection = await mod.resolve_selection(
        peer, _Event(), ["ca_cat", "ca_output"], ["subject", "style"]
    )
    images, prompt = mod.compose_inputs(
        selection, [b"cat-reference"], identity_images=[b"cat-reference"]
    )
    assert len(images) == 3
    assert "参考图 2" in prompt and "参考图 3" in prompt
    with pytest.raises(ValueError):
        mod.compose_inputs(selection, [], identity_images=[b"face"] * 7)


def test_selfie_object_reference_preserves_specific_stylized_subject_without_conflicting_defaults():
    mod, plugin, peer = setup_plugin()
    plugin._get_selfie_conf = lambda: {}
    # Production role text: don't substitute a cat-shaped generic concept.
    from importlib import import_module

    bridge = import_module(mod.__package__ + ".core.image_reference_bridge")
    note = bridge.ROLE_TEXT["object"]
    prompt = plugin._build_selfie_prompt(
        "抱着图里的猫", extra_refs=1, reference_note=note
    )
    assert "仅作为服装/姿势/构图/场景" not in prompt
    assert "不自动替换成普通写实动物" in prompt
    assert "用户明确要求改变主体造型或风格时" in prompt
    assert prompt.index("本次参考图用途") < prompt.index("用户要求（最高优先级）")
    ordinary = plugin._build_selfie_prompt("自拍", extra_refs=1)
    assert "仅作为服装/姿势/构图/场景" in ordinary


@pytest.mark.asyncio
async def test_batch_auto_references_dispatches_edit_and_retains_count():
    mod, plugin, peer = setup_plugin()
    plugin._background_manager_for_event = lambda e: object()
    plugin._accept_background_batch = AsyncMock(return_value="accepted")
    result = await plugin.aiimg_batch_generate(
        _Event(),
        prompt="four edits",
        count=4,
        reference_image_ids=["ca_cat"],
        reference_roles=["subject"],
    )
    assert result == "accepted"
    assert plugin._accept_background_batch.call_args.kwargs["count"] == 4
    assert plugin._accept_background_batch.call_args.kwargs["mode"] == "edit"


@pytest.mark.asyncio
async def test_batch_text_mode_cannot_discard_explicit_reference():
    mod, plugin, peer = setup_plugin()
    plugin._accept_background_batch = AsyncMock()
    result = await plugin.aiimg_batch_generate(
        _Event(),
        prompt="four images",
        mode="text",
        reference_image_ids=["ca_cat"],
        reference_roles=["object"],
    )
    assert "no batch started" in result.content[0].text
    plugin._accept_background_batch.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode,role", [("selfie_ref", "object"), ("edit", "subject")])
async def test_batch_reference_lifecycle_spools_roles_and_registers_only_delivered_children(
    tmp_path, mode, role
):
    mod, plugin, peer = setup_plugin()
    manager = mod.BackgroundImageTaskManager(tmp_path, heartbeat_seconds=60)
    await manager.start()
    captured = []
    try:
        plugin._background_manager_for_event = lambda e: manager
        plugin._build_background_delivery_target = AsyncMock(return_value=_target(mod))
        plugin._get_selfie_conf = lambda: {}
        plugin._is_selfie_enabled = lambda: True
        plugin._is_selfie_llm_enabled = lambda: True
        plugin._get_selfie_reference_paths = AsyncMock(
            return_value=([Path("identity")], "webui")
        )
        plugin._read_paths_bytes = AsyncMock(return_value=[b"identity-photo"])
        mod.get_images_from_event = AsyncMock(return_value=[])
        plugin._image_segs_to_bytes = AsyncMock(return_value=[])
        plugin._get_life_context_without_llm = AsyncMock(return_value={})
        plugin._get_selfie_default_output = lambda: ""
        manager.start_worker = lambda task_id, factory: captured.append(factory)
        event = _Event()
        result = await plugin.aiimg_batch_generate(
            event,
            prompt="three variants",
            count=3,
            mode=mode,
            reference_image_ids=["ca_cat"],
            reference_roles=[role],
        )
        task_id = json.loads(result.content[0].text)["task_id"]
        record = await manager.get_task(task_id)
        assert record["reference_sources"][0]["id"] == "ca_cat"
        assert len(record["input_manifest"]) == (2 if mode == "selfie_ref" else 1)
        # Original platform event and recall cache are gone before planner/provider.
        event._extras.clear()
        peer.rows.clear()
        plugin._plan_batch_prompt_items = AsyncMock(
            return_value=[
                mod.PlannedPromptItem(
                    title=str(i),
                    prompt=f"variant-{i}",
                    variation_focus=["pose"],
                    aspect_ratio="3:4",
                )
                for i in range(3)
            ]
        )
        plugin._describe_spooled_objects = AsyncMock(
            return_value=("参考图可见特征：大黑眼睛", "described")
        )
        seen = []

        async def edit_call(prompt, images, **kwargs):
            assert images == (
                [b"identity-photo", b"cat-reference"]
                if mode == "selfie_ref"
                else [b"cat-reference"]
            )
            assert kwargs["require_ordered_references"] is True
            assert "参考图" in prompt
            assert "大黑眼睛" in prompt
            if mode == "selfie_ref":
                assert "仅作为服装/姿势/构图/场景" not in prompt
                assert "主体抠图合成" in prompt
            seen.append(prompt)
            path = tmp_path / (f"result-{len(seen)}.png")
            path.write_bytes(b"result-bytes")
            return path

        plugin.edit = SimpleNamespace(edit=edit_call)
        send_count = 0

        async def send_once(target, path):
            nonlocal send_count
            send_count += 1
            if send_count == 2:
                raise RuntimeError("controlled delivery failure")
            return _Event()

        plugin._send_background_image_once = send_once
        plugin._wait_for_background_ack = AsyncMock()
        plugin._wait_background_send_gate = AsyncMock()
        plugin._dispatch_background_completion = AsyncMock()
        plugin._save_last_image_task_meta = AsyncMock()
        await captured[0]()
        record = await manager.get_task(task_id)
        plugin._describe_spooled_objects.assert_awaited_once()
        assert len(seen) == 3
        assert record["sent_count"] == 2
        assert record["state"] == "partial"
        assert len(peer.registered) == 2
        assert [r[1]["task_id"] for r in peer.registered] == [
            task_id + "_01",
            task_id + "_03",
        ]
        assert all(r[1]["parent_ids"] == ["ca_cat"] for r in peer.registered)
        assert record["items"][0]["task_meta"]["result_image_id"] == "ca_new_result"
        assert "result_image_id" not in record["items"][1]["task_meta"]
    finally:
        await manager.close()


def test_bound_selfie_preserves_custom_prefix_and_explicit_user_transform():
    mod, plugin, peer = setup_plugin()
    plugin._get_selfie_conf = lambda: {"prompt_prefix": "自定义人物风格规则"}
    prompt = plugin._build_selfie_prompt(
        "把猫改成水彩画风", 1, reference_note="object role"
    )
    assert "自定义人物风格规则" in prompt
    assert "用户明确修改要求仍优先" in prompt
    assert "用户要求（最高优先级）：把猫改成水彩画风" in prompt
    assert "object role" in prompt
