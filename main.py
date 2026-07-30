"""
Gitee AI 图像生成插件

功能:
- 文生图 (z-image-turbo)
- 图生图/改图 (Gemini / Gitee 千问，可切换)
- Bot 自拍（参考照）：上传参考人像后用改图模型生成自拍
- 视频生成 (Grok imagine, 参考图 + 提示词)
- 预设提示词
- 智能降级
"""

import asyncio
import base64
import copy
import hashlib
import html
import io
import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mcp
from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import (
    At,
    AtAll,
    File,
    Image,
    Plain,
    Reply,
    Video,
)
from astrbot.api.platform import MessageMember
from astrbot.api.star import Context, Star, StarTools
from astrbot.core.utils.astrbot_path import get_astrbot_temp_path

from .core.background_tasks import (
    ACTIVE_STATES,
    TERMINAL_STATES,
    BackgroundImageTaskManager,
    BackgroundTaskCapacityError,
    BackgroundTaskError,
    PreparedBatchJob,
    PreparedImageJob,
    TaskDeliveryTarget,
)
from .core.batch_executor import BatchRunResult, run_batch
from .core.debouncer import Debouncer
from .core.draw_service import ImageDrawService
from .core.edit_router import EditRouter
from .core.emoji_feedback import mark_failed, mark_processing, mark_success
from .core.gitee_sizes import (
    GITEE_SUPPORTED_RATIOS,
    normalize_size_text,
    resolve_ratio_size,
)
from .core.image_format import decode_base64_image_payload, guess_image_mime_and_ext
from .core.image_manager import ImageManager
from .core.image_task_parser import (
    ImageTaskSpec,
    ParsedImageRequest,
    parse_image_request,
)
from .core.llm_batch_planner import (
    PlannedPromptItem,
    build_batch_planning_prompt,
    parse_planned_prompt_items,
    validate_planned_prompt_items,
)
from .core.nanobanana import NanoBananaService
from .core.output_spec import (
    OutputIntent,
    aspect_ratio_from_size,
    format_output_intent,
    merge_output_intents,
    normalize_aspect_ratio,
    parse_output_intent,
    resolve_llm_output_intent,
    split_prompt_output_suffix,
)
from .core.provider_registry import ProviderRegistry
from .core.ref_store import ReferenceStore
from .core.utils import close_session, collect_at_user_ids, get_images_from_event
from .core.video_manager import VideoManager

try:
    from astrbot.core.agent.message import TextPart
except ImportError:
    TextPart = None

_async_pause = asyncio.sleep


class ImageCommandWakePrefixFilter(filter.CustomFilter):
    """Require AstrBot's real wake behavior for regex-based plugin commands."""

    @staticmethod
    def _wake_prefixes(cfg: object) -> tuple[str, ...]:
        try:
            raw = cfg.get("wake_prefix", ["/"])
        except Exception:
            raw = ["/"]
        if isinstance(raw, str):
            return (raw,) if raw else ("/",)
        if isinstance(raw, (list, tuple, set)):
            return tuple(str(item) for item in raw if str(item)) or ("/",)
        return ("/",)

    @staticmethod
    def _is_private_chat(event: AstrMessageEvent) -> bool:
        try:
            return bool(event.is_private_chat())
        except Exception:
            message_obj = getattr(event, "message_obj", None)
            return not bool(getattr(message_obj, "group", None))

    @staticmethod
    def _plain_has_configured_prefix(text: str, prefixes: tuple[str, ...]) -> bool:
        """Check whether a raw text segment starts with a configured prefix.

        Args:
            text: Raw plain-text segment from the platform message chain.
            prefixes: Wake prefixes from AstrBot's active configuration.

        Returns:
            Whether the segment begins with a non-empty prefixed token.
        """
        plain = str(text or "").lstrip()
        for prefix in prefixes:
            if not plain.startswith(prefix):
                continue
            end = len(prefix)
            if end < len(plain) and not plain[end].isspace():
                return True
        return False

    def filter(self, event: AstrMessageEvent, cfg: object) -> bool:
        """Apply wake-prefix gating before a regex handler can wake the event.

        Args:
            event: Current AstrBot message event.
            cfg: Active AstrBot base configuration.

        Returns:
            Whether the event is allowed to reach the plugin handler.
        """
        if self._is_private_chat(event):
            return bool(getattr(event, "is_at_or_wake_command", False))
        prefixes = self._wake_prefixes(cfg)
        try:
            chain = event.get_messages()
        except Exception:
            chain = []
        return any(
            isinstance(seg, Plain)
            and self._plain_has_configured_prefix(
                str(getattr(seg, "text", "") or ""),
                prefixes,
            )
            for seg in chain or []
        )


@dataclass(slots=True)
class SendImageResult:
    ok: bool
    reason: str = ""
    cached_path: Path | None = None
    used_fallback: bool = False
    last_error: str = ""

    def __bool__(self) -> bool:
        return self.ok


@dataclass(slots=True)
class ExecutedImageTask:
    spec: ImageTaskSpec
    image_path: Path
    task_meta: dict[str, Any]


class GiteeAIImagePlugin(Star):
    """Gitee AI 图像生成插件"""

    # Gitee AI 支持的图片比例
    SUPPORTED_RATIOS: dict[str, list[str]] = GITEE_SUPPORTED_RATIOS
    IMAGE_AS_FILE_THRESHOLD_BYTES: int = 20 * 1024 * 1024
    WEIXIN_SEND_TEMP_PATTERN: str = "weixin_send_*.jpg"
    WEIXIN_SEND_TEMP_MAX_FILES: int = 64
    WEIXIN_SEND_TEMP_TTL_SECONDS: int = 24 * 60 * 60

    def __init__(self, context: Context, config: dict):
        super().__init__(context)
        self.config = config
        self.data_dir = StarTools.get_data_dir("astrbot_plugin_gitee_aiimg")
        self._last_image_by_user: dict[str, Path] = {}
        self._last_image_task_meta_cache: dict[str, dict[str, Any]] = {}
        self.background_tasks: BackgroundImageTaskManager | None = None
        self._background_recovery_records: list[dict[str, Any]] = []
        self._background_send_gates: dict[str, asyncio.Event] = {}

    async def _call_native_poke(self, event: AstrMessageEvent, target_id: str) -> bool:
        bot = getattr(event, "bot", None)
        if bot is None or not hasattr(bot, "call_action"):
            return False

        user_id: int | str = int(target_id) if target_id.isdigit() else target_id
        try:
            await bot.call_action("friend_poke", user_id=user_id)
            return True
        except Exception as exc:
            logger.warning(
                "[GiteeAIImagePlugin] friend_poke failed: target=%s err=%s",
                target_id,
                exc,
            )

        try:
            await bot.call_action("send_poke", user_id=user_id)
            return True
        except Exception as exc:
            logger.warning(
                "[GiteeAIImagePlugin] send_poke failed: target=%s err=%s",
                target_id,
                exc,
            )
            return False

    async def _signal_llm_tool_failure(self, event: AstrMessageEvent) -> None:
        if event.is_private_chat():
            target_id = str(event.get_sender_id() or "").strip()
            if target_id:
                if await self._call_native_poke(event, target_id):
                    return
        await mark_failed(event)

    @staticmethod
    def _llm_tool_text_result(message: str) -> mcp.types.CallToolResult:
        text = str(message or "").strip()
        if not text:
            text = "The tool completed without additional details."
        return mcp.types.CallToolResult(
            content=[mcp.types.TextContent(type="text", text=text)]
        )

    @staticmethod
    def _summarize_status_text(
        value: Exception | str | None,
        *,
        fallback: str,
        limit: int = 180,
    ) -> str:
        text = " ".join(str(value or "").split())
        if not text:
            return fallback
        if len(text) <= limit:
            return text
        return f"{text[: limit - 3].rstrip()}..."

    @staticmethod
    def _truncate_text(value: Any, *, limit: int = 320) -> str:
        text = " ".join(str(value or "").split())
        if len(text) <= limit:
            return text
        return f"{text[: limit - 3].rstrip()}..."

    @staticmethod
    def _get_event_conversation_id(event: AstrMessageEvent) -> str:
        provider_request = event.get_extra("provider_request")
        conversation = getattr(provider_request, "conversation", None)
        return str(getattr(conversation, "cid", "") or "").strip()

    @staticmethod
    def _get_event_self_id(event: AstrMessageEvent) -> str:
        try:
            return str(event.get_self_id() or "").strip()
        except Exception:
            return ""

    def _image_task_store_key(
        self,
        event: AstrMessageEvent,
        *,
        conversation_id: str = "",
    ) -> str:
        umo = str(getattr(event, "unified_msg_origin", "") or "").strip() or "unknown"
        self_id = self._get_event_self_id(event) or "unknown_bot"
        sender_id = str(event.get_sender_id() or "").strip() or "unknown"
        conversation_scope = (
            str(conversation_id or "").strip()
            or self._get_event_conversation_id(event)
            or "default"
        )
        return f"last_image_task::{umo}::{self_id}::{sender_id}::{conversation_scope}"

    async def _resolve_image_task_store_key(self, event: AstrMessageEvent) -> str:
        conversation_id = self._get_event_conversation_id(event)
        if not conversation_id:
            conversation = await self._resolve_plugin_conversation(event)
            conversation_id = str(getattr(conversation, "cid", "") or "").strip()
        return self._image_task_store_key(event, conversation_id=conversation_id)

    @staticmethod
    def _normalize_image_task_meta(meta: Any) -> dict[str, Any] | None:
        if not isinstance(meta, dict):
            return None
        mode = str(meta.get("mode") or "").strip()
        if not mode:
            return None
        try:
            reference_count = int(meta.get("reference_count") or 0)
            extra_reference_count = int(meta.get("extra_reference_count") or 0)
            created_at = float(meta.get("created_at") or time.time())
        except (TypeError, ValueError, OverflowError) as exc:
            logger.warning(
                "[GiteeAIImagePlugin] discard malformed last-image-task meta: %s",
                exc,
            )
            return None
        if (
            reference_count < 0
            or extra_reference_count < 0
            or not math.isfinite(created_at)
            or created_at < 0
        ):
            logger.warning(
                "[GiteeAIImagePlugin] discard invalid last-image-task meta values: %s",
                meta,
            )
            return None
        normalized = {
            "mode": mode,
            "user_prompt": str(meta.get("user_prompt") or "").strip(),
            "effective_user_prompt": str(
                meta.get("effective_user_prompt") or ""
            ).strip(),
            "effective_prompt": str(meta.get("effective_prompt") or "").strip(),
            "reference_source": str(meta.get("reference_source") or "").strip(),
            "reference_count": reference_count,
            "extra_reference_count": extra_reference_count,
            "continue_with": str(meta.get("continue_with") or mode).strip() or mode,
            "follow_up": bool(meta.get("follow_up", False)),
            "backend": str(meta.get("backend") or "").strip(),
            "created_at": created_at,
        }
        return normalized

    async def _save_last_image_task_meta(
        self, event: AstrMessageEvent, meta: dict[str, Any]
    ) -> None:
        normalized = self._normalize_image_task_meta(meta)
        if normalized is None:
            return

        store_key = await self._resolve_image_task_store_key(event)
        self._last_image_task_meta_cache[store_key] = normalized

        try:
            await self.put_kv_data(store_key, normalized)
        except Exception as exc:
            logger.debug(
                "[GiteeAIImagePlugin] skip persistent last-image-task save: %s",
                exc,
            )

    async def _load_last_image_task_meta(
        self, event: AstrMessageEvent
    ) -> dict[str, Any] | None:
        store_key = await self._resolve_image_task_store_key(event)
        cached_raw = self._last_image_task_meta_cache.get(store_key)
        cached = self._normalize_image_task_meta(cached_raw)
        if cached is not None:
            return cached
        if cached_raw is not None:
            self._last_image_task_meta_cache.pop(store_key, None)

        try:
            stored = await self.get_kv_data(store_key, None)
        except Exception as exc:
            logger.debug(
                "[GiteeAIImagePlugin] skip persistent last-image-task load: %s",
                exc,
            )
            return None

        normalized = self._normalize_image_task_meta(stored)
        if normalized is not None:
            self._last_image_task_meta_cache[store_key] = normalized
            return normalized
        if stored is not None:
            try:
                await self.delete_kv_data(store_key)
            except Exception as exc:
                logger.debug(
                    "[GiteeAIImagePlugin] skip cleanup malformed last-image-task meta: %s",
                    exc,
                )
        return None

    @staticmethod
    def _looks_like_image_follow_up(prompt: str) -> bool:
        text = str(prompt or "").strip()
        if not text:
            return False
        lowered = text.lower()
        keywords = (
            "不满意",
            "不太满意",
            "重新",
            "重来",
            "再来",
            "再拍",
            "换个",
            "换成",
            "换一下",
            "改一下",
            "改改",
            "调整",
            "重拍",
            "再生成",
            "重新拍",
            "重新来",
            "pose",
            "again",
            "redo",
            "adjust",
            "change",
        )
        return any(keyword in text or keyword in lowered for keyword in keywords)

    async def _match_selfie_follow_up(
        self, event: AstrMessageEvent, prompt: str
    ) -> dict[str, Any] | None:
        if self._is_auto_selfie_prompt(prompt):
            return None
        if not self._looks_like_image_follow_up(prompt):
            return None

        last_meta = await self._load_last_image_task_meta(event)
        if last_meta is None:
            return None
        if str(last_meta.get("continue_with") or "") != "selfie_ref":
            return None

        created_at = float(last_meta.get("created_at") or 0)
        if created_at > 0 and time.time() - created_at > 1800:
            return None

        ref_paths, ref_source = await self._get_selfie_reference_paths(event)
        if not ref_paths:
            return None

        meta = dict(last_meta)
        meta["reference_source"] = ref_source
        meta["reference_count"] = len(ref_paths)
        return meta

    def _build_selfie_follow_up_prompt(
        self, prompt: str, last_meta: dict[str, Any] | None
    ) -> str:
        current_prompt = str(prompt or "").strip()
        if last_meta is None:
            return current_prompt

        previous_prompt = (
            str(last_meta.get("effective_user_prompt") or "").strip()
            or str(last_meta.get("user_prompt") or "").strip()
        )
        if not previous_prompt:
            return current_prompt
        if not current_prompt:
            return f"延续上一张自拍要求：{previous_prompt}"
        return f"延续上一张自拍要求：{previous_prompt}；本次新增要求：{current_prompt}"

    def _build_image_task_meta(
        self,
        *,
        mode: str,
        user_prompt: str,
        effective_prompt: str,
        effective_user_prompt: str | None = None,
        reference_source: str = "",
        reference_count: int = 0,
        extra_reference_count: int = 0,
        continue_with: str | None = None,
        follow_up: bool = False,
        backend: str | None = None,
    ) -> dict[str, Any]:
        return {
            "mode": str(mode or "").strip(),
            "user_prompt": str(user_prompt or "").strip(),
            "effective_user_prompt": str(
                effective_user_prompt
                if effective_user_prompt is not None
                else user_prompt
            ).strip(),
            "effective_prompt": str(effective_prompt or "").strip(),
            "reference_source": str(reference_source or "").strip(),
            "reference_count": max(0, int(reference_count or 0)),
            "extra_reference_count": max(0, int(extra_reference_count or 0)),
            "continue_with": str(continue_with or mode or "").strip()
            or str(mode or "").strip(),
            "follow_up": bool(follow_up),
            "backend": str(backend or "").strip(),
            "created_at": time.time(),
        }

    def _build_image_task_completion_result(
        self, task_meta: dict[str, Any]
    ) -> mcp.types.CallToolResult:
        mode = str(task_meta.get("mode") or "image").strip() or "image"
        summary = {
            "status": "completed",
            "mode": mode,
            "continue_with": str(task_meta.get("continue_with") or mode).strip()
            or mode,
            "user_prompt": self._truncate_text(task_meta.get("user_prompt"), limit=180),
            "effective_prompt": self._truncate_text(
                task_meta.get("effective_prompt"), limit=260
            ),
            "reference_source": str(task_meta.get("reference_source") or "").strip(),
            "reference_count": int(task_meta.get("reference_count") or 0),
            "extra_reference_count": int(task_meta.get("extra_reference_count") or 0),
            "follow_up": bool(task_meta.get("follow_up", False)),
        }
        if task_meta.get("backend"):
            summary["backend"] = str(task_meta.get("backend"))

        hint = (
            "If the user asks to redo or adjust this selfie, continue with selfie_ref and reuse the same reference images unless the user explicitly changes them."
            if summary["continue_with"] == "selfie_ref"
            else "If the user asks for changes, continue from this completed image task instead of guessing a brand-new request."
        )
        return self._llm_tool_text_result(
            "The image has already been generated and sent to the user. Do not send another confirmation message to the user. "
            f"Store this task summary for follow-ups: {json.dumps(summary, ensure_ascii=False)} "
            + hint
        )

    async def _resolve_plugin_conversation(self, event: AstrMessageEvent) -> Any | None:
        provider_request = event.get_extra("provider_request")
        conversation = getattr(provider_request, "conversation", None)
        if conversation is not None:
            return conversation

        conv_mgr = getattr(self.context, "conversation_manager", None)
        if conv_mgr is None:
            return None

        umo = str(getattr(event, "unified_msg_origin", "") or "").strip()
        if not umo:
            return None

        try:
            conversation_id = await conv_mgr.get_curr_conversation_id(umo)
            if not conversation_id:
                return None
            conversation = await conv_mgr.get_conversation(umo, conversation_id)
        except Exception as exc:
            logger.warning(
                "[GiteeAIImagePlugin] failed to resolve conversation for plugin note: %s",
                exc,
            )
            return None

        if conversation is not None and provider_request is not None:
            try:
                provider_request.conversation = conversation
            except Exception:
                pass
        return conversation

    async def _append_plugin_conversation_note(
        self, event: AstrMessageEvent, note: str
    ) -> None:
        note = str(note or "").strip()
        if not note:
            return

        conv_mgr = getattr(self.context, "conversation_manager", None)
        if conv_mgr is None:
            return

        conversation = await self._resolve_plugin_conversation(event)
        if conversation is None:
            return

        history_raw = getattr(conversation, "history", "[]")
        if isinstance(history_raw, list):
            history = list(history_raw)
        else:
            try:
                parsed_history = json.loads(history_raw or "[]")
                history = (
                    list(parsed_history) if isinstance(parsed_history, list) else []
                )
            except Exception as exc:
                logger.warning(
                    "[GiteeAIImagePlugin] failed to parse conversation history for plugin note: %s",
                    exc,
                )
                history = []

        history.append(
            {"role": "user", "content": "Output your last task result below."}
        )
        history.append({"role": "assistant", "content": note})

        try:
            await conv_mgr.update_conversation(
                event.unified_msg_origin,
                getattr(conversation, "cid", None),
                history=history,
            )
        except Exception as exc:
            logger.warning(
                "[GiteeAIImagePlugin] failed to persist plugin conversation note: %s",
                exc,
            )
            return

        try:
            conversation.history = json.dumps(history, ensure_ascii=False)
        except Exception:
            pass

    async def initialize(self):
        self.debouncer = Debouncer(self.config)
        self.imgr = ImageManager(self.config, self.data_dir)
        self.registry = ProviderRegistry(
            self.config, imgr=self.imgr, data_dir=self.data_dir
        )
        for err in self.registry.validate():
            logger.warning("[GiteeAIImagePlugin][config] %s", err)

        self.draw = ImageDrawService(
            self.config, self.imgr, self.data_dir, registry=self.registry
        )
        self.edit = EditRouter(
            self.config, self.imgr, self.data_dir, registry=self.registry
        )
        self.nb = NanoBananaService(self.config, self.imgr)
        self.refs = ReferenceStore(self.data_dir)
        self.videomgr = VideoManager(self.config, self.data_dir)

        self._concurrency_lock = asyncio.Lock()
        self._image_inflight: dict[str, int] = {}
        self._video_inflight: dict[str, int] = {}
        self._video_tasks: set[asyncio.Task] = set()

        background_conf = self._get_feature("background_llm_image")
        if self._as_bool(background_conf.get("enabled", False), default=False):
            manager = BackgroundImageTaskManager(
                Path(self.data_dir),
                max_running=self._as_int(
                    background_conf.get("max_running", 2), default=2
                ),
                max_queued=self._as_int(
                    background_conf.get("max_queued", 16), default=16
                ),
                log=logger,
            )
            try:
                self._background_recovery_records = await manager.start()
            except Exception as exc:
                logger.error(
                    "[background-image] disabled after startup check failed: %s",
                    BackgroundImageTaskManager.sanitize_error(exc),
                )
            else:
                self.background_tasks = manager

        # 读取 AstrBot 配置的唤醒前缀，用于限制命令匹配范围
        try:
            if hasattr(self.context, "get_config"):
                base_config = self.context.get_config()
            else:
                base_config = getattr(self.context, "base_config", {})
            raw = base_config.get("wake_prefix", ["/"])
            if isinstance(raw, str):
                self._wake_prefixes: tuple[str, ...] = (raw,) if raw else ("/",)
            elif isinstance(raw, list):
                self._wake_prefixes = tuple(str(p) for p in raw if p) or ("/",)
            else:
                self._wake_prefixes = ("/",)
        except Exception:
            self._wake_prefixes = ("/",)

        self._patch_tool_image_cache_runtime()

        # 动态注册预设命令 (方案C: /手办化 直接触发)
        self._register_preset_commands()

        logger.info(
            f"[GiteeAIImagePlugin] 插件初始化完成: "
            f"改图后端={self.edit.get_available_backends()}, "
            f"文生图预设={len(self._get_draw_presets())}个, "
            f"改图预设={len(self.edit.get_preset_names())}个, "
            f"视频启用={bool(self._get_feature('video').get('enabled', False))}, "
            f"视频预设={len(self._get_video_presets())}个, "
            f"LLM后台生图={self.background_tasks is not None}"
        )

    @filter.on_astrbot_loaded()
    async def on_background_astrbot_loaded(self) -> None:
        """Drain restart recovery notifications after platforms are available."""

        manager = getattr(self, "background_tasks", None)
        if manager is None:
            return
        records = list(self._background_recovery_records)
        self._background_recovery_records.clear()
        for record in records:
            target = self._delivery_target_from_record(record)
            await self._dispatch_background_completion(manager, record, target)

    @filter.on_llm_request(priority=-20)
    async def inject_background_image_tasks(self, event: AstrMessageEvent, req) -> None:
        """Inject authoritative background image facts into the current LLM turn.

        Args:
            event: Current pipeline event.
            req: Mutable AstrBot provider request.
        """

        manager = getattr(self, "background_tasks", None)
        if manager is None:
            return
        exact_task_id = str(event.get_extra("_gitee_bg_task_id", "") or "")
        records: list[dict[str, Any]] = []
        if exact_task_id:
            record = await manager.get_task(exact_task_id)
            if record is not None:
                records.append(record)
        else:
            try:
                umo = str(event.unified_msg_origin or "")
                sender_id = str(event.get_sender_id() or "")
                self_id = str(event.get_self_id() or "")
            except Exception:
                return
            conversation = getattr(req, "conversation", None)
            conversation_id = str(getattr(conversation, "cid", "") or "")
            if not conversation_id:
                conversation_manager = getattr(
                    self.context, "conversation_manager", None
                )
                if conversation_manager is None:
                    return
                try:
                    current_id = await conversation_manager.get_curr_conversation_id(
                        umo
                    )
                    if not current_id:
                        return
                    conversation = await conversation_manager.get_conversation(
                        umo,
                        current_id,
                    )
                except Exception as exc:
                    logger.warning(
                        "[background-image] skipped task injection because the conversation could not be resolved: %s",
                        BackgroundImageTaskManager.sanitize_error(exc),
                    )
                    return
                if conversation is None:
                    return
                conversation_id = str(getattr(conversation, "cid", "") or current_id)
                req.conversation = conversation
            scope = manager.scope_hash(umo, self_id, sender_id, conversation_id)
            own_records = await manager.list_scope_tasks(scope, limit=6)
            now = manager.now_ms()
            for record in own_records:
                if record.get("state") in ACTIVE_STATES:
                    records.append(record)
                elif now - int(record.get("finished_at") or 0) <= 30 * 60 * 1000:
                    records.append(record)
                if len(records) >= 3:
                    break

            for record in await manager.list_active_for_umo(umo):
                if str(record.get("sender_id") or "") == sender_id:
                    continue
                if str(record.get("self_id") or "") != self_id:
                    continue
                if str(record.get("conversation_id") or "") != conversation_id:
                    continue
                records.append(
                    {
                        "task_id": record.get("task_id"),
                        "task_kind": record.get("task_kind"),
                        "state": record.get("state"),
                        "requester": record.get("sender_name")
                        or record.get("sender_id"),
                        "created_at": record.get("created_at"),
                    }
                )
                if len(records) >= 3:
                    break
        if not records:
            return

        now = manager.now_ms()
        summaries: list[dict[str, Any]] = []
        for record in records[:3]:
            if "user_prompt" not in record:
                summary = dict(record)
                summary["elapsed_seconds"] = max(
                    0,
                    (now - int(record.get("created_at") or now)) // 1000,
                )
                summaries.append(summary)
                continue
            summary = {
                "task_id": record.get("task_id"),
                "task_kind": record.get("task_kind"),
                "state": record.get("state"),
                "mode": record.get("mode"),
                "requester": record.get("sender_name") or record.get("sender_id"),
                "elapsed_seconds": max(
                    0,
                    (now - int(record.get("created_at") or now)) // 1000,
                ),
                "user_prompt": record.get("user_prompt"),
                "effective_prompt": record.get("effective_prompt"),
                "image_generated": bool(record.get("image_generated")),
                "image_sent": bool(record.get("image_sent")),
                "delivery_state": record.get("delivery_state"),
                "notification_state": record.get("notification_state"),
                "requested_count": record.get("requested_count"),
                "planned_count": record.get("planned_count"),
                "generated_count": record.get("generated_count"),
                "sent_count": record.get("sent_count"),
                "failed_count": record.get("failed_count"),
                "cancelled_count": record.get("cancelled_count"),
                "unknown_count": record.get("unknown_count"),
                "error_message": record.get("error_message"),
            }
            items = record.get("items")
            if isinstance(items, list):
                summary["items"] = [
                    {
                        "index": item.get("index"),
                        "state": item.get("state"),
                        "effective_prompt": item.get("effective_prompt"),
                        "aspect_ratio": item.get("aspect_ratio"),
                        "image_generated": bool(item.get("image_generated")),
                        "image_sent": bool(item.get("image_sent")),
                        "delivery_state": item.get("delivery_state"),
                        "error_message": item.get("error_message"),
                    }
                    for item in items[:8]
                ]
                summary["prompts_truncated"] = len(items) > 8
            summaries.append(summary)

        serialized = json.dumps(summaries, ensure_ascii=False, separators=(",", ":"))
        if len(serialized) > 6000:
            serialized = serialized[:5900] + '..."prompts_truncated":true'
        block = (
            "<background_image_tasks_json>"
            + html.escape(serialized)
            + "</background_image_tasks_json>\n"
            "These are authoritative live task facts. If an image is still planning, "
            "queued, running, or sending, say it is in progress. If image_sent is true "
            "and delivery_state is confirmed, the adapter accepted the image send. Never "
            "invent progress percentages or prompts."
        )
        extra_parts = getattr(req, "extra_user_content_parts", None)
        if isinstance(extra_parts, list) and TextPart is not None:
            extra_parts.append(TextPart(text=block).mark_as_temp())
        else:
            req.system_prompt = (
                str(getattr(req, "system_prompt", "") or "") + "\n" + block
            )

        if exact_task_id:
            tool_set = getattr(req, "func_tool", None)
            if tool_set is not None and isinstance(
                getattr(tool_set, "tools", None), list
            ):
                filtered = copy.copy(tool_set)
                blocked = {
                    "aiimg_generate",
                    "gitee_draw_image",
                    "gitee_edit_image",
                    "aiimg_batch_generate",
                    "send_message_to_user",
                }
                filtered.tools = [
                    tool
                    for tool in tool_set.tools
                    if str(getattr(tool, "name", "") or "") not in blocked
                ]
                req.func_tool = filtered

    @filter.event_message_type(filter.EventMessageType.ALL, priority=10)
    async def handle_background_session_commands(self, event: AstrMessageEvent) -> None:
        """Observe stop/reset/new without replacing AstrBot's command handlers.

        Args:
            event: Current platform event.
        """

        manager = getattr(self, "background_tasks", None)
        if manager is None or event.get_extra("_gitee_bg_completion", False):
            return
        text = str(getattr(event, "message_str", "") or "").strip()
        match = re.match(r"^[\s/!！.。．]*(stop|reset|new)(?:\s|$)", text, re.I)
        if match is None:
            return
        command = match.group(1).lower()
        if command == "stop":
            cancelled = await self._cancel_background_scope_with_notifications(
                manager,
                umo=str(event.unified_msg_origin or ""),
                sender_id=str(event.get_sender_id() or ""),
                reason="user requested /stop",
            )
            event.set_extra("_gitee_bg_stop_cancelled", cancelled)
            return
        umo = str(event.unified_msg_origin or "")
        gate = self._background_send_gates.get(umo)
        if gate is None or gate.is_set():
            gate = asyncio.Event()
            self._background_send_gates[umo] = gate
            manager.start_managed(
                self._expire_background_send_gate(umo, gate),
                name=f"background-reset-gate-{hashlib.sha256(umo.encode()).hexdigest()[:12]}",
            )
        event.set_extra("_gitee_bg_reset_candidate", command)

    @staticmethod
    def _background_plain_digest(event: AstrMessageEvent) -> tuple[str, str]:
        result = event.get_result()
        chain = getattr(result, "chain", None) if result is not None else None
        plain = "".join(
            str(getattr(component, "text", "") or "")
            for component in chain or []
            if isinstance(component, Plain)
        ).strip()
        digest = hashlib.sha256(plain.encode("utf-8")).hexdigest() if plain else ""
        return plain, digest

    @filter.on_decorating_result()
    async def decorate_background_task_result(self, event: AstrMessageEvent) -> None:
        """Ensure accepted and terminal events contain a sendable natural reply.

        Args:
            event: Event whose final result is about to be sent.
        """

        manager = getattr(self, "background_tasks", None)
        if manager is None:
            return
        ack_task_id = str(event.get_extra("_gitee_bg_ack_task_id", "") or "")
        completion_task_id = str(event.get_extra("_gitee_bg_task_id", "") or "")
        if not ack_task_id and not completion_task_id:
            return
        plain, _ = self._background_plain_digest(event)
        if not plain:
            result = event.get_result()
            if result is None:
                result = event.plain_result("")
                event.set_result(result)
            record = await manager.get_task(completion_task_id or ack_task_id)
            if completion_task_id and record is not None:
                fallback = self._background_notification_text(record)
            elif record and record.get("task_kind") == "batch":
                fallback = "我开始准备这组照片了，你们可以继续聊，拍好后我会发出来。"
            else:
                fallback = "我开始准备这张照片了，你们可以继续聊，拍好后我会发出来。"
            result.chain.append(Plain(fallback))
        plain, digest = self._background_plain_digest(event)
        event.set_extra("_gitee_bg_final_plain_digest", digest)
        if ack_task_id:
            await manager.mark_ack(ack_task_id, "decorated")
        if completion_task_id:
            event.set_extra("_gitee_bg_completion_plain", plain)

    @filter.on_decorating_result(priority=-1000000)
    async def arm_background_task_result_transport(
        self, event: AstrMessageEvent
    ) -> None:
        """Track the exact final result send after all normal decorators run.

        Args:
            event: Event whose final result will enter AstrBot's respond stage.
        """

        ack_task_id = str(event.get_extra("_gitee_bg_ack_task_id", "") or "")
        token = str(event.get_extra("_gitee_bg_notification_token", "") or "")
        if not ack_task_id and not token:
            return
        if bool(event.get_extra("_gitee_bg_transport_probe_armed", False)):
            return

        original_send = event.send

        async def tracked_send(*args, **kwargs):
            """Forward a final send and record only a successful return.

            Args:
                *args: Positional arguments accepted by the adapter send method.
                **kwargs: Keyword arguments accepted by the adapter send method.
            """

            try:
                await original_send(*args, **kwargs)
            except BaseException:
                failures = int(
                    event.get_extra("_gitee_bg_result_send_failures", 0) or 0
                )
                event.set_extra("_gitee_bg_result_send_failures", failures + 1)
                raise
            successes = int(event.get_extra("_gitee_bg_result_send_successes", 0) or 0)
            event.set_extra("_gitee_bg_result_send_successes", successes + 1)

        event.set_extra("_gitee_bg_transport_probe_armed", True)
        setattr(event, "_gitee_bg_original_send", original_send)
        setattr(event, "send", tracked_send)

    @filter.after_message_sent(priority=200)
    async def confirm_background_task_result(self, event: AstrMessageEvent) -> None:
        """Confirm acceptance and terminal text only after adapter send returns.

        Args:
            event: Event after AstrBot's respond stage.
        """

        manager = getattr(self, "background_tasks", None)
        if manager is None:
            return
        expected = str(event.get_extra("_gitee_bg_final_plain_digest", "") or "")
        plain, actual = self._background_plain_digest(event)
        probe_armed = bool(event.get_extra("_gitee_bg_transport_probe_armed", False))
        probe_successes = int(
            event.get_extra("_gitee_bg_result_send_successes", 0) or 0
        )
        probe_failures = int(event.get_extra("_gitee_bg_result_send_failures", 0) or 0)
        probe_ok = probe_successes > 0 and probe_failures == 0
        transport_ok = bool(getattr(event, "_has_send_oper", False))
        digest_ok = bool(plain and expected and actual == expected)
        confirmed = probe_ok if probe_armed else transport_ok and digest_ok
        original_send = getattr(event, "_gitee_bg_original_send", None)
        if original_send is not None:
            setattr(event, "send", original_send)

        ack_task_id = str(event.get_extra("_gitee_bg_ack_task_id", "") or "")
        if ack_task_id:
            try:
                await manager.mark_ack(
                    ack_task_id,
                    "sent" if confirmed else "unknown",
                )
            except BackgroundTaskError:
                pass

        token = str(event.get_extra("_gitee_bg_notification_token", "") or "")
        attempt_id = str(event.get_extra("_gitee_bg_notification_attempt", "") or "")
        if token and attempt_id:
            marked = await manager.mark_notification(
                token,
                "sent" if confirmed else "unknown",
                attempt_id=attempt_id,
            )
            if not marked:
                logger.warning(
                    "[background-image] notification confirmation lost its claim: token_hash=%s",
                    hashlib.sha256(token.encode()).hexdigest()[:12],
                )

        if ack_task_id or token:
            logger.info(
                "[background-image] result confirmation: kind=%s task=%s "
                "probe_armed=%s probe_ok=%s probe_successes=%s probe_failures=%s "
                "transport_flag=%s digest_ok=%s "
                "plain_len=%s expected_hash=%s actual_hash=%s state=%s",
                "completion" if token else "acceptance",
                str(event.get_extra("_gitee_bg_task_id", "") or ack_task_id),
                probe_armed,
                probe_ok,
                probe_successes,
                probe_failures,
                transport_ok,
                digest_ok,
                len(plain),
                expected[:12],
                actual[:12],
                "sent" if confirmed else "unknown",
            )

        reset_command = str(event.get_extra("_gitee_bg_reset_candidate", "") or "")
        if reset_command and bool(
            event.get_extra("_clean_group_context_session", False)
        ):
            await self._cancel_background_scope_with_notifications(
                manager,
                umo=str(event.unified_msg_origin or ""),
                sender_id=str(event.get_sender_id() or ""),
                reason=f"session_reset:{reset_command}",
                suppress_future_injection=True,
            )
        if reset_command:
            umo = str(event.unified_msg_origin or "")
            gate = self._background_send_gates.pop(umo, None)
            if gate is not None:
                gate.set()

    def _remember_last_image(self, event: AstrMessageEvent, image_path: Path) -> None:
        try:
            user_id = str(event.get_sender_id() or "")
            umo = str(getattr(event, "unified_msg_origin", "") or "")
        except Exception:
            user_id = ""
            umo = ""
        if not user_id or not umo:
            return
        self._last_image_by_user[f"{umo}::{user_id}"] = Path(image_path)

    @staticmethod
    def _as_int(value: Any, *, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _as_bool(value: Any, *, default: bool) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return default
        if isinstance(value, (int, float)):
            return value != 0
        if isinstance(value, str):
            v = value.strip().lower()
            if v in {"1", "true", "yes", "y", "on", "enable", "enabled"}:
                return True
            if v in {"0", "false", "no", "n", "off", "disable", "disabled", ""}:
                return False
        return default

    def _patch_tool_image_cache_runtime(self) -> None:
        try:
            from astrbot.core.agent import tool_image_cache as cache_module
        except Exception as exc:
            logger.debug(
                "[GiteeAIImagePlugin] skip tool image cache runtime patch: %s", exc
            )
            return

        cache_cls = getattr(cache_module, "ToolImageCache", None)
        cache_obj = getattr(cache_module, "tool_image_cache", None)
        cached_image_cls = getattr(cache_module, "CachedImage", None)
        if cache_cls is None or cache_obj is None or cached_image_cls is None:
            return
        if getattr(cache_cls, "_gitee_aiimg_runtime_patch", False):
            return

        def _patched_save_image(
            cache_self,
            base64_data: str,
            tool_call_id: str,
            tool_name: str,
            index: int = 0,
            mime_type: str = "image/png",
        ):
            ext = cache_self._get_file_extension(mime_type)
            cache_dir_value = str(getattr(cache_self, "_cache_dir", "") or "").strip()
            cache_dir = (
                Path(cache_dir_value)
                if cache_dir_value
                else Path(get_astrbot_temp_path())
                / getattr(cache_self, "CACHE_DIR_NAME", "tool_images")
            )
            file_path = cache_dir / f"{tool_call_id}_{index}{ext}"

            try:
                cache_dir.mkdir(parents=True, exist_ok=True)
                image_bytes = base64.b64decode(base64_data)
                file_path.write_bytes(image_bytes)
            except Exception as exc:
                logger.error(f"Failed to save tool image: {exc}")
                raise

            cache_self._cache_dir = str(cache_dir)
            logger.debug(
                "[GiteeAIImagePlugin] tool image cache runtime patch wrote: %s",
                file_path,
            )
            return cached_image_cls(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                file_path=str(file_path),
                mime_type=mime_type,
            )

        cache_cls.save_image = _patched_save_image
        cache_cls._gitee_aiimg_runtime_patch = True
        cache_obj._cache_dir = str(
            Path(get_astrbot_temp_path())
            / getattr(cache_cls, "CACHE_DIR_NAME", "tool_images")
        )
        Path(cache_obj._cache_dir).mkdir(parents=True, exist_ok=True)
        logger.info(
            "[GiteeAIImagePlugin] tool image cache runtime patch active: %s",
            cache_obj._cache_dir,
        )

    def _get_max_user_concurrency(self) -> int:
        v = self._as_int(self.config.get("max_user_concurrency", 2), default=2)
        return max(1, min(10, v))

    def _get_max_user_video_concurrency(self) -> int:
        v = self._as_int(self.config.get("max_user_video_concurrency", 1), default=1)
        return max(1, min(5, v))

    def _debounce_key(self, event: AstrMessageEvent, prefix: str, user_id: str) -> str:
        """尽量用消息维度去重，避免同用户短时间内无法并发提交多条任务。"""
        mid = str(
            getattr(getattr(event, "message_obj", None), "message_id", "") or ""
        ).strip()
        origin = str(getattr(event, "unified_msg_origin", "") or "").strip()
        if mid and origin:
            return f"{prefix}:{origin}:{mid}"
        return f"{prefix}:{user_id}"

    async def _begin_user_job(self, user_id: str, *, kind: str) -> bool:
        user_id = str(user_id or "").strip()
        if not user_id:
            return True

        if kind == "video":
            limit = self._get_max_user_video_concurrency()
            store = self._video_inflight
        else:
            limit = self._get_max_user_concurrency()
            store = self._image_inflight

        async with self._concurrency_lock:
            cur = int(store.get(user_id, 0) or 0)
            if cur >= limit:
                return False
            store[user_id] = cur + 1
            return True

    async def _end_user_job(self, user_id: str, *, kind: str) -> None:
        user_id = str(user_id or "").strip()
        if not user_id:
            return

        store = self._video_inflight if kind == "video" else self._image_inflight
        async with self._concurrency_lock:
            cur = int(store.get(user_id, 0) or 0)
            if cur <= 1:
                store.pop(user_id, None)
            else:
                store[user_id] = cur - 1

    @staticmethod
    def _is_rich_media_transfer_failed(exc: Exception | None) -> bool:
        if exc is None:
            return False
        msg = f"{exc!r} {exc}".lower()
        return "rich media transfer failed" in msg

    def _get_send_conf(self) -> dict[str, Any]:
        conf = self.config.get("send", {}) if isinstance(self.config, dict) else {}
        return conf if isinstance(conf, dict) else {}

    def _is_weixin_event(self, event: AstrMessageEvent | None) -> bool:
        if not event:
            return False
        names: list[str] = []
        try:
            names.append(str(event.get_platform_name() or ""))
        except Exception:
            pass

        platform_inst = getattr(event, "platform", None)
        if platform_inst is not None:
            try:
                meta = platform_inst.meta() if hasattr(platform_inst, "meta") else None
                if meta:
                    names.append(str(getattr(meta, "name", "") or ""))
                    names.append(str(getattr(meta, "id", "") or ""))
            except Exception:
                pass

        return any(name.strip().lower() == "weixin_oc" for name in names)

    def _get_weixin_timeout_ms(self) -> int:
        conf = self._get_send_conf()
        timeout_seconds = self._as_int(
            conf.get("weixin_api_timeout_seconds", 60), default=60
        )
        timeout_ms = timeout_seconds * 1000
        return max(15000, min(timeout_ms, 300000))

    def _apply_weixin_timeout(self, platform_inst: Any) -> None:
        if not platform_inst:
            return
        timeout_ms = self._get_weixin_timeout_ms()
        try:
            old_timeout = getattr(platform_inst, "api_timeout_ms", None)
            if old_timeout != timeout_ms:
                setattr(platform_inst, "api_timeout_ms", timeout_ms)

            client = getattr(platform_inst, "client", None)
            if client and getattr(client, "api_timeout_ms", None) != timeout_ms:
                setattr(client, "api_timeout_ms", timeout_ms)
        except Exception as exc:
            logger.debug("[GiteeAIImagePlugin] 设置 weixin_oc 超时失败: %s", exc)

    def _get_weixin_send_temp_dir(self) -> Path:
        return Path(self.data_dir) / "Temp"

    def _is_weixin_send_temp_path(self, image_path: Path) -> bool:
        try:
            p = Path(image_path).resolve(strict=False)
            temp_dir = self._get_weixin_send_temp_dir().resolve(strict=False)
            return (
                p.parent == temp_dir
                and p.name.startswith("weixin_send_")
                and p.suffix.lower() == ".jpg"
            )
        except Exception:
            return False

    def _cleanup_weixin_send_temp_images_sync(self) -> None:
        temp_dir = self._get_weixin_send_temp_dir()
        try:
            if not temp_dir.exists():
                return

            now = time.time()
            entries: list[tuple[Path, float]] = []
            for p in temp_dir.glob(self.WEIXIN_SEND_TEMP_PATTERN):
                try:
                    if not p.is_file():
                        continue
                    st = p.stat()
                except OSError:
                    continue
                entries.append((p, st.st_mtime))

            stale = [
                p
                for p, mtime in entries
                if now - mtime > self.WEIXIN_SEND_TEMP_TTL_SECONDS
            ]
            stale_keys = {str(p.resolve(strict=False)) for p in stale}
            remaining = [
                item
                for item in entries
                if str(item[0].resolve(strict=False)) not in stale_keys
            ]
            if len(remaining) > self.WEIXIN_SEND_TEMP_MAX_FILES:
                remaining.sort(key=lambda item: item[1])
                overflow = len(remaining) - self.WEIXIN_SEND_TEMP_MAX_FILES
                stale.extend(p for p, _ in remaining[:overflow])

            seen: set[str] = set()
            for p in stale:
                try:
                    key = str(p.resolve(strict=False))
                    if key in seen:
                        continue
                    seen.add(key)
                    p.unlink(missing_ok=True)
                except Exception as exc:
                    logger.debug(
                        "[GiteeAIImagePlugin] 清理 weixin_oc 临时图片失败: %s, err=%s",
                        p,
                        exc,
                    )
        except Exception as exc:
            logger.debug("[GiteeAIImagePlugin] 扫描 weixin_oc 临时图片失败: %s", exc)

    def _remove_weixin_send_temp_image_sync(self, image_path: Path) -> None:
        p = Path(image_path)
        if not self._is_weixin_send_temp_path(p):
            return
        try:
            p.unlink(missing_ok=True)
            logger.debug("[GiteeAIImagePlugin] 已清理 weixin_oc 发送临时图片: %s", p)
        except Exception as exc:
            logger.debug(
                "[GiteeAIImagePlugin] 删除 weixin_oc 发送临时图片失败: %s, err=%s",
                p,
                exc,
            )

    def _compress_image_for_weixin_sync(self, image_path: Path) -> Path:
        conf = self._get_send_conf()
        if not self._as_bool(conf.get("weixin_compress_images", True), default=True):
            return image_path

        p = Path(image_path)
        if not p.exists():
            return p

        try:
            from PIL import Image as PILImage
            from PIL import ImageOps
        except Exception as exc:
            logger.debug(
                "[GiteeAIImagePlugin] Pillow 不可用，跳过 weixin_oc 图片优化: %s",
                exc,
            )
            return p

        max_side = self._as_int(conf.get("weixin_image_max_side", 4096), default=4096)
        max_kb = self._as_int(
            conf.get("weixin_image_max_size_kb", 10240), default=10240
        )
        max_side = max(1600, min(max_side, 8192))
        target_bytes = max(512, max_kb) * 1024

        try:
            raw_size = p.stat().st_size
            with PILImage.open(p) as im:
                im = ImageOps.exif_transpose(im)
                width, height = im.size
                if raw_size <= target_bytes and max(width, height) <= max_side:
                    return p

                if im.mode in ("RGBA", "LA") or (
                    im.mode == "P" and "transparency" in im.info
                ):
                    bg = PILImage.new("RGB", im.size, (255, 255, 255))
                    rgba = im.convert("RGBA")
                    bg.paste(rgba, mask=rgba.split()[-1])
                    im = bg
                else:
                    im = im.convert("RGB")

                if max(width, height) > max_side:
                    resampling = getattr(
                        getattr(PILImage, "Resampling", PILImage), "LANCZOS"
                    )
                    im.thumbnail((max_side, max_side), resampling)

                temp_dir = Path(self.data_dir) / "Temp"
                temp_dir.mkdir(parents=True, exist_ok=True)
                self._cleanup_weixin_send_temp_images_sync()
                digest_src = (
                    f"{p}:{raw_size}:{p.stat().st_mtime}:{max_side}:{max_kb}".encode(
                        "utf-8", errors="ignore"
                    )
                )
                digest = hashlib.md5(digest_src).hexdigest()[:12]
                out_path = temp_dir / f"weixin_send_{digest}_{time.time_ns()}.jpg"

                for quality in (95, 93, 90, 88, 85, 82, 78, 74, 70):
                    im.save(
                        out_path,
                        format="JPEG",
                        quality=quality,
                        optimize=True,
                        progressive=True,
                        subsampling=0 if quality >= 90 else -1,
                    )
                    if out_path.stat().st_size <= target_bytes:
                        break

                out_size = out_path.stat().st_size
                if out_size < raw_size:
                    logger.info(
                        "[GiteeAIImagePlugin] 已为 weixin_oc 优化图片: %.2fMB -> %.2fMB, 分辨率 %sx%s -> %sx%s",
                        raw_size / 1024 / 1024,
                        out_size / 1024 / 1024,
                        width,
                        height,
                        im.size[0],
                        im.size[1],
                    )
                    return out_path
        except Exception as exc:
            logger.warning(
                "[GiteeAIImagePlugin] weixin_oc 图片优化失败，继续发送原图: %s",
                exc,
            )

        return p

    async def _prepare_image_for_send(
        self, event: AstrMessageEvent, image_path: Path
    ) -> Path:
        if self._is_weixin_event(event):
            self._apply_weixin_timeout(getattr(event, "platform", None))
            return await asyncio.to_thread(
                self._compress_image_for_weixin_sync, image_path
            )
        return Path(image_path)

    @staticmethod
    def _build_compact_image_bytes(
        image_path: Path, *, max_side: int = 2048, target_bytes: int = 3_500_000
    ) -> bytes | None:
        """Build a smaller JPEG variant for platforms that reject large rich-media upload."""
        try:
            from PIL import Image as PILImage
        except Exception:
            return None

        try:
            with PILImage.open(image_path) as im:
                if im.mode not in {"RGB", "L"}:
                    im = im.convert("RGB")
                elif im.mode == "L":
                    im = im.convert("RGB")

                w, h = im.size
                if max(w, h) > max_side:
                    ratio = float(max_side) / float(max(w, h))
                    nw = max(1, int(w * ratio))
                    nh = max(1, int(h * ratio))
                    resampling = getattr(
                        getattr(PILImage, "Resampling", PILImage), "LANCZOS"
                    )
                    im = im.resize((nw, nh), resampling)

                for q in (88, 82, 76, 70, 64):
                    buf = io.BytesIO()
                    im.save(
                        buf,
                        format="JPEG",
                        quality=q,
                        optimize=True,
                        progressive=True,
                    )
                    data = buf.getvalue()
                    if data and (len(data) <= target_bytes or q == 64):
                        return data
        except Exception:
            return None
        return None

    def _is_selfie_enabled(self) -> bool:
        conf = self._get_feature("selfie")
        return self._as_bool(conf.get("enabled", True), default=True)

    def _is_selfie_llm_enabled(self) -> bool:
        conf = self._get_feature("selfie")
        return self._as_bool(conf.get("llm_tool_enabled", True), default=True)

    @staticmethod
    def _selfie_disabled_message() -> str:
        return "自拍参考图模式已关闭（features.selfie.enabled=false）"

    async def _send_image_with_fallback(
        self, event: AstrMessageEvent, image_path: Path, *, max_attempts: int = 5
    ) -> SendImageResult:
        """Send image with retries and fallback to base64 bytes.

        Avoids wasting generation credits when platform send fails transiently.
        """
        original_path = Path(image_path)
        p = await self._prepare_image_for_send(event, original_path)
        should_cleanup_temp = self._is_weixin_send_temp_path(p) and (
            p.resolve(strict=False) != original_path.resolve(strict=False)
        )

        async def finish(result: SendImageResult) -> SendImageResult:
            if should_cleanup_temp:
                await asyncio.to_thread(self._remove_weixin_send_temp_image_sync, p)
                if result.cached_path == p:
                    result.cached_path = original_path
            return result

        if not p.exists():
            logger.warning("[send_image] file not found: %s", p)
            return await finish(
                SendImageResult(ok=False, reason="file_not_found", cached_path=p)
            )

        # Large original images (e.g. 4K 20MB+) are likely to fail rich-media upload.
        # Prefer sending as a normal file first so the original bytes are preserved.
        try:
            size_bytes = int(p.stat().st_size)
        except Exception:
            size_bytes = 0

        file_send_tries = 0

        async def try_send_as_file(trigger: str) -> bool:
            nonlocal file_send_tries
            if file_send_tries >= 2:
                return False
            file_send_tries += 1
            try:
                await event.send(event.chain_result([File(name=p.name, file=str(p))]))
                logger.info(
                    "[send_image][file-fallback-v2] file send success: %s (%s bytes), trigger=%s, try=%s",
                    p.name,
                    size_bytes,
                    trigger,
                    file_send_tries,
                )
                return True
            except Exception as e:
                logger.warning(
                    "[send_image][file-fallback-v2] file send failed: trigger=%s, try=%s, err=%s",
                    trigger,
                    file_send_tries,
                    e,
                )
                return False

        if size_bytes > self.IMAGE_AS_FILE_THRESHOLD_BYTES:
            if await try_send_as_file("size_threshold"):
                return await finish(
                    SendImageResult(ok=True, cached_path=p, used_fallback=True)
                )

        delay = 1.5
        last_exc: Exception | None = None
        attempts = max(1, int(max_attempts))
        rich_media_failures = 0
        compact_bytes: bytes | None = None
        compact_prepared = False
        for attempt in range(1, attempts + 1):
            fs_exc: Exception | None = None
            bytes_exc: Exception | None = None
            compact_exc: Exception | None = None
            fs_failed_by_rich_media = False

            try:
                await event.send(event.chain_result([Image.fromFileSystem(str(p))]))
                return await finish(
                    SendImageResult(ok=True, cached_path=p, used_fallback=False)
                )
            except Exception as e:
                fs_exc = e
                last_exc = e
                if self._is_rich_media_transfer_failed(e):
                    fs_failed_by_rich_media = True
                logger.debug(
                    "[send_image] fromFileSystem failed (attempt=%s/%s): %s",
                    attempt,
                    attempts,
                    e,
                )

            try:
                data = await asyncio.to_thread(p.read_bytes)
                await event.send(event.chain_result([Image.fromBytes(data)]))
                if fs_exc is not None:
                    logger.info(
                        "[send_image] fromBytes fallback succeeded (attempt=%s/%s).",
                        attempt,
                        attempts,
                    )
                return await finish(
                    SendImageResult(ok=True, cached_path=p, used_fallback=True)
                )
            except Exception as e:
                bytes_exc = e
                last_exc = e
                logger.debug(
                    "[send_image] fromBytes failed (attempt=%s/%s): %s",
                    attempt,
                    attempts,
                    e,
                )

            # If rich-media channel is failing, immediately try original-file sending.
            if self._is_rich_media_transfer_failed(
                fs_exc
            ) or self._is_rich_media_transfer_failed(bytes_exc):
                if await try_send_as_file("rich_media_transfer_failed"):
                    return await finish(
                        SendImageResult(ok=True, cached_path=p, used_fallback=True)
                    )

            # Extra fallback for repeated rich-media failures: compress and retry by bytes.
            if self._is_rich_media_transfer_failed(
                fs_exc
            ) or self._is_rich_media_transfer_failed(bytes_exc):
                if not compact_prepared:
                    compact_prepared = True
                    compact_bytes = await asyncio.to_thread(
                        self._build_compact_image_bytes, p
                    )
                    if compact_bytes:
                        logger.info(
                            "[send_image] prepared compact fallback image: %s -> %s bytes",
                            p,
                            len(compact_bytes),
                        )
                if compact_bytes:
                    try:
                        await event.send(
                            event.chain_result([Image.fromBytes(compact_bytes)])
                        )
                        logger.info(
                            "[send_image] compact fromBytes fallback succeeded (attempt=%s/%s).",
                            attempt,
                            attempts,
                        )
                        return await finish(
                            SendImageResult(ok=True, cached_path=p, used_fallback=True)
                        )
                    except Exception as e:
                        compact_exc = e
                        last_exc = e
                        logger.debug(
                            "[send_image] compact fromBytes failed (attempt=%s/%s): %s",
                            attempt,
                            attempts,
                            e,
                        )

            attempt_has_rich_media = (
                self._is_rich_media_transfer_failed(fs_exc)
                or self._is_rich_media_transfer_failed(bytes_exc)
                or self._is_rich_media_transfer_failed(compact_exc)
            )
            if attempt_has_rich_media:
                rich_media_failures += 1

            if fs_exc is not None and bytes_exc is not None and compact_exc is not None:
                logger.debug(
                    "[send_image] attempt=%s/%s failed on all channels.",
                    attempt,
                    attempts,
                )
            elif fs_exc is not None and bytes_exc is not None:
                logger.debug(
                    "[send_image] attempt=%s/%s failed on both channels.",
                    attempt,
                    attempts,
                )
            elif fs_exc is not None and fs_failed_by_rich_media:
                logger.debug(
                    "[send_image] attempt=%s/%s failed by rich media transfer.",
                    attempt,
                    attempts,
                )
            else:
                logger.debug(
                    "[send_image] attempt=%s/%s failed to send image.",
                    attempt,
                    attempts,
                )

            if rich_media_failures >= 2:
                logger.info(
                    "[send_image] detected repeated rich media transfer failures, stop retrying early."
                )
                break

            if attempt < attempts:
                await _async_pause(delay)
                delay = min(delay * 1.8, 8.0)

        reason = (
            "rich_media_transfer_failed"
            if self._is_rich_media_transfer_failed(last_exc)
            else "send_failed"
        )
        logger.error(
            "[send_image] failed after retries: reason=%s, err=%s", reason, last_exc
        )
        return await finish(
            SendImageResult(
                ok=False,
                reason=reason,
                cached_path=p,
                last_error=str(last_exc or ""),
            )
        )

    def _register_preset_commands(self):
        """动态注册预设命令

        为每个预设创建对应的命令，如 /手办化, /Q版化 等
        """
        preset_names = self.edit.get_preset_names()
        if not preset_names:
            return

        for preset_name in preset_names:
            # 创建闭包捕获 preset_name
            self._create_and_register_preset_handler(preset_name)

        logger.info(f"[GiteeAIImagePlugin] 已注册 {len(preset_names)} 个预设命令")

    def _create_and_register_preset_handler(self, preset_name: str):
        """为单个预设创建并注册命令处理器

        支持: /手办化 [额外提示词]
        例如: /手办化 加点金色元素
        """

        # 默认后端命令: /手办化
        async def preset_handler(event: AstrMessageEvent):
            # 提取命令后的额外提示词
            extra_prompt = self._extract_extra_prompt(event, preset_name)
            await self._do_edit_direct(event, extra_prompt, preset=preset_name)

        preset_handler.__name__ = f"preset_{preset_name}"
        preset_handler.__doc__ = f"预设改图: {preset_name} [额外提示词]"

        self.context.register_commands(
            star_name="astrbot_plugin_gitee_aiimg",
            command_name=preset_name,
            desc=f"预设改图: {preset_name}",
            priority=5,
            awaitable=preset_handler,
        )

    def _extract_extra_prompt(self, event: AstrMessageEvent, command_name: str) -> str:
        """从消息中提取命令后的额外提示词

        支持格式:
        - /手办化 加点金色元素 -> "加点金色元素"
        - /手办化@张三 背景是星空 -> "背景是星空"
        - /手办化@张三@李四 背景是星空 -> "背景是星空"

        注意: message_str 中 @用户 会被替换为空格或移除
        """
        msg = event.message_str.strip()
        # 移除命令前缀 (/, !, ., 等)
        # 兼容唤醒前缀：.视频 / 。视频 / ．视频
        if msg and msg[0] in "/!！.。．":
            msg = msg[1:]
        # 移除命令名
        if msg.startswith(command_name):
            msg = msg[len(command_name) :]
        msg = re.sub(r"@\S+\(\d+\)", " ", msg)
        # 清理多余空格
        return msg.strip()

    def _cmd_prefixes(self) -> tuple[str, ...]:
        """Return configured AstrBot wake prefixes for command matching."""
        return getattr(self, "_wake_prefixes", None) or ("/",)

    @staticmethod
    def _extract_command_arg_anywhere(
        message: str, command_name: str, *, prefixes: tuple[str, ...] | None = None
    ) -> str:
        """从任意位置提取"/命令 参数"，用于图片在前导致 @filter.command 不触发的场景。"""
        _found, arg = GiteeAIImagePlugin._find_command_arg_anywhere(
            message,
            command_name,
            prefixes=prefixes,
        )
        return arg

    @staticmethod
    def _find_command_arg_anywhere(
        message: str,
        command_name: str,
        *,
        allow_bare: bool = False,
        prefixes: tuple[str, ...] | None = None,
    ) -> tuple[bool, str]:
        """查找带前缀命令；已被 AstrBot wake_prefix 剥离时允许裸命令。
        prefixes 为 None 时回退到全量集合（兼容静态调用场景）。
        """
        msg = (message or "").strip()
        if not msg:
            return False, ""
        effective_prefixes = (
            prefixes if prefixes is not None else ("/", "!", "！", ".", "。", "．")
        )
        for prefix in effective_prefixes:
            token = f"{prefix}{command_name}"
            idx = msg.find(token)
            while idx >= 0:
                end = idx + len(token)
                if end == len(msg) or msg[end].isspace():
                    return True, msg[end:].strip()
                idx = msg.find(token, idx + 1)
        if allow_bare and msg.startswith(command_name):
            end = len(command_name)
            if end == len(msg) or msg[end].isspace():
                return True, msg[end:].strip()
        return False, ""

    def _extract_command_arg_from_chain(
        self, event: AstrMessageEvent, command_name: str
    ) -> tuple[bool, str]:
        """从消息链中提取命令后的提示词。

        用于修复"/命令 + 图片 + 文本"时，平台把文本段无空格拼接到 `message_str`
        导致 command filter 和字符串提取都失效的问题。
        """
        try:
            chain = event.get_messages()
        except Exception:
            return False, ""

        found = False
        parts: list[str] = []
        for seg in chain:
            if isinstance(seg, (At, AtAll, Reply)):
                continue

            if not found:
                if not isinstance(seg, Plain):
                    continue
                plain = str(getattr(seg, "text", "") or "").lstrip()
                if not plain:
                    continue
                matched_prefix = next(
                    (
                        prefix
                        for prefix in self._cmd_prefixes()
                        if plain.startswith(f"{prefix}{command_name}")
                    ),
                    None,
                )
                if matched_prefix is None:
                    continue
                plain = plain[len(matched_prefix) :]
                end = len(command_name)
                if len(plain) > end and not plain[end].isspace():
                    continue
                found = True
                tail = plain[len(command_name) :].strip()
                if tail:
                    parts.append(tail)
                continue

            if isinstance(seg, Plain):
                text = str(getattr(seg, "text", "") or "").strip()
                if text:
                    parts.append(text)

        return found, " ".join(parts).strip()

    def _extract_chain_provider_id(self, item: object) -> str:
        if isinstance(item, str):
            return item.strip()
        if not isinstance(item, dict):
            return ""
        return str(
            item.get("provider_id")
            or item.get("id")
            or item.get("provider")
            or item.get("backend")
            or ""
        ).strip()

    def _normalize_chain_item(self, item: object) -> dict | None:
        pid = self._extract_chain_provider_id(item)
        if not pid:
            return None
        out = ""
        if isinstance(item, dict):
            out = str(item.get("output") or item.get("default_output") or "").strip()
        return {"provider_id": pid, "output": out} if out else {"provider_id": pid}

    def _parse_provider_override_prefix(self, text: str) -> tuple[str | None, str]:
        """仅当 @token 命中已配置 provider_id 时，才作为 provider 覆盖。"""
        s = (text or "").strip()
        if not s.startswith("@"):
            return None, s
        first, _, rest = s.partition(" ")
        candidate = first.lstrip("@").strip()
        if not candidate:
            return None, s
        if candidate in set(self.registry.provider_ids()):
            return candidate, rest.strip()
        logger.debug(
            "[provider_override] 忽略未知 @token，继续走自动链路: token=%s",
            candidate,
        )
        return None, s

    @staticmethod
    def _plain_starts_with_command(
        text: str, command_name: str, *, prefixes: tuple[str, ...] | None = None
    ) -> bool:
        plain = (text or "").lstrip()
        if not plain:
            return False
        effective_prefixes = (
            prefixes if prefixes is not None else ("/", "!", "！", ".", "。", "．")
        )
        for prefix in effective_prefixes:
            if plain.startswith(f"{prefix}{command_name}"):
                return True
        return False

    def _is_direct_command_message(
        self, event: AstrMessageEvent, command_names: tuple[str, ...]
    ) -> bool:
        """仅当"首个有效文本段"直接是命令时返回 True。

        用于 regex 兜底去重：避免正常 /命令 被重复处理；
        同时允许"图片在前、命令在后"的消息继续走兜底逻辑。
        """
        try:
            chain = event.get_messages()
        except Exception:
            return False
        if not chain:
            return False

        first_plain = ""
        for seg in chain:
            if isinstance(seg, (At, AtAll, Reply)):
                continue
            if isinstance(seg, Plain):
                first_plain = str(getattr(seg, "text", "") or "")
            break

        if not first_plain:
            return False
        return any(
            self._plain_starts_with_command(
                first_plain, name, prefixes=self._cmd_prefixes()
            )
            for name in command_names
        )

    @staticmethod
    def _is_framework_direct_command_text(
        message: str, command_names: tuple[str, ...], *, allow_bare: bool = True
    ) -> bool:
        """按 AstrBot CommandFilter 的文本规则判断是否可直接命中 command handler。"""
        plain = " ".join(str(message or "").strip().split())
        if not plain:
            return False
        if plain[0] in "/!！.。．":
            plain = plain[1:].lstrip()
        return any(
            (plain == name if allow_bare else False) or plain.startswith(f"{name} ")
            for name in command_names
        )

    @staticmethod
    def _has_activated_handler(event: AstrMessageEvent, handler_name: str) -> bool:
        """检查本轮事件是否已经激活指定 handler，用于 regex fallback 去重。"""
        try:
            handlers = event.get_extra("activated_handlers", [])
        except Exception:
            return False
        for handler in handlers or []:
            if str(getattr(handler, "handler_name", "") or "") == handler_name:
                return True
            raw_handler = getattr(handler, "handler", None)
            if str(getattr(raw_handler, "__name__", "") or "") == handler_name:
                return True
        return False

    async def terminate(self):
        self.debouncer.clear_all()
        manager = getattr(self, "background_tasks", None)
        if manager is not None:
            try:
                await manager.close()
            except Exception as exc:
                logger.error(
                    "[background-image] manager shutdown failed: %s",
                    BackgroundImageTaskManager.sanitize_error(exc),
                )
            finally:
                self.background_tasks = None
        try:
            tasks = list(getattr(self, "_video_tasks", []))
            for t in tasks:
                t.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
        except Exception:
            pass
        await self.imgr.close()
        await self.draw.close()
        await self.edit.close()
        await self.nb.close()
        await close_session()  # 关闭 utils.py 的 HTTP 会话

    # ==================== 文生图 ====================

    @filter.command("文生图")
    @filter.custom_filter(ImageCommandWakePrefixFilter)
    async def generate_image_with_presets(self, event: AstrMessageEvent):
        """支持文生图预设的图片生成命令。"""
        event.should_call_llm(True)
        parsed = self._parse_structured_image_request(event.message_str)
        if parsed is None or parsed.spec.source_command != "文生图":
            await mark_failed(event)
            return

        spec = parsed.spec
        if not str(spec.effective_prompt or "").strip():
            await mark_failed(event)
            return

        user_id = str(event.get_sender_id() or "")
        request_id = self._debounce_key(event, "draw_preset", user_id)
        if self.debouncer.hit(request_id):
            await mark_failed(event)
            return
        if not await self._begin_user_job(user_id, kind="image"):
            await mark_failed(event)
            return

        try:
            await mark_processing(event)
            executed = await self._execute_image_task_spec(event, spec)
            self._remember_last_image(event, executed.image_path)
            sent = await self._send_image_with_fallback(event, executed.image_path)
            if not sent:
                await mark_failed(event)
                return
            await self._save_last_image_task_meta(event, executed.task_meta)
            await mark_success(event)
        except Exception as exc:
            logger.error("[文生图预设] 失败: %s", exc, exc_info=True)
            await mark_failed(event)
        finally:
            await self._end_user_job(user_id, kind="image")

    @filter.command("aiimg", alias={"生图", "画图", "绘图", "出图"})
    @filter.custom_filter(ImageCommandWakePrefixFilter)
    async def generate_image_command(self, event: AstrMessageEvent, prompt: str):
        """生成图片指令

        用法: /aiimg [@provider_id] <提示词> [比例]
        示例: /aiimg 一个女孩 9:16
        支持比例: 1:1, 4:3, 3:4, 3:2, 2:3, 16:9, 9:16
        """
        event.should_call_llm(True)
        # 解析参数
        arg = event.message_str.partition(" ")[2]
        if not arg:
            await mark_failed(event)
            return
        provider_override: str | None = None
        provider_override, arg = self._parse_provider_override_prefix(arg)
        if not arg:
            await mark_failed(event)
            return

        try:
            prompt, output_intent = split_prompt_output_suffix(arg)
        except ValueError as exc:
            logger.warning("[aiimg] 输出参数无效: %s", exc)
            await mark_failed(event)
            return

        if not prompt:
            await mark_failed(event)
            return

        user_id = str(event.get_sender_id() or "")
        request_id = self._debounce_key(event, "generate", user_id)

        # 防抖检查
        if self.debouncer.hit(request_id):
            await mark_failed(event)
            return

        if not await self._begin_user_job(user_id, kind="image"):
            await mark_failed(event)
            return

        try:
            # 标记处理中
            await mark_processing(event)
            t_start = time.perf_counter()
            image_path = await self.draw.generate(
                prompt,
                output_intent=output_intent,
                provider_id=provider_override,
            )
            t_end = time.perf_counter()

            self._remember_last_image(event, image_path)
            sent = await self._send_image_with_fallback(event, image_path)
            if not sent:
                await mark_failed(event)
                logger.warning(
                    "[文生图] 图片发送失败，已仅使用表情标注: reason=%s", sent.reason
                )
                return

            # 标记成功
            await mark_success(event)
            logger.info(
                f"[文生图] 完成: {prompt[:30] if prompt else '文生图'}..., 耗时={t_end - t_start:.2f}s"
            )

        except Exception as e:
            logger.error(f"[文生图] 失败: {e}")
            await mark_failed(event)
        finally:
            await self._end_user_job(user_id, kind="image")

    @filter.regex(r"[/!！.。．]批量(?:\s*\d+|\d+)(?:\s|$)", priority=-10)
    @filter.custom_filter(ImageCommandWakePrefixFilter)
    async def batch_image_command(self, event: AstrMessageEvent):
        """批量图片任务入口。"""
        fragment = self._extract_batch_command_fragment(event)
        if not fragment:
            return
        event.should_call_llm(True)
        parsed = self._parse_structured_image_request(fragment)
        if parsed is None or parsed.batch_count <= 1:
            await mark_failed(event)
            return
        if parsed.batch_count > self._get_batch_max_count():
            await event.send(
                event.plain_result(
                    f"批量数量过大，当前上限为 {self._get_batch_max_count()}。"
                )
            )
            await mark_failed(event)
            return

        user_id = str(event.get_sender_id() or "")
        request_id = self._debounce_key(event, "batch_image", user_id)
        if self.debouncer.hit(request_id):
            await mark_failed(event)
            return
        if not await self._begin_user_job(user_id, kind="image"):
            await mark_failed(event)
            return

        try:
            await mark_processing(event)
            specs = [parsed.spec for _ in range(parsed.batch_count)]
            results = await self._run_batch_specs(event, specs)
            title = f"{self._batch_mode_label(parsed.spec)} x{parsed.batch_count}"
            await self._send_batch_results(event, results, title=title)
            if any(result.success and result.value for result in results):
                await self._remember_batch_success(event, results)
                await mark_success(event)
            else:
                await mark_failed(event)
        except Exception as exc:
            logger.error("[批量图片] 失败: %s", exc, exc_info=True)
            await mark_failed(event)
        finally:
            await self._end_user_job(user_id, kind="image")

    # ==================== 图生图/改图 ====================

    @filter.command("aiedit", alias={"图生图", "改图", "修图"})
    @filter.custom_filter(ImageCommandWakePrefixFilter)
    async def edit_image_default(self, event: AstrMessageEvent, prompt: str):
        """使用默认后端改图

        用法: /aiedit <提示词>
        需要同时发送或引用图片
        """
        event.should_call_llm(True)
        await self._do_edit(event, prompt, backend=None)

    @filter.command("重发图片")
    async def resend_last_image(self, event: AstrMessageEvent):
        """重发最近一次生成/改图的图片（不重新生成，不消耗次数）。"""
        user_id = str(event.get_sender_id() or "")
        umo = str(getattr(event, "unified_msg_origin", "") or "")
        p = self._last_image_by_user.get(f"{umo}::{user_id}")
        if not p:
            await mark_failed(event)
            return
        if not Path(p).exists():
            await mark_failed(event)
            return
        ok = await self._send_image_with_fallback(event, p)
        if ok:
            await mark_success(event)
        else:
            await mark_failed(event)

    @filter.regex(r".*[/!！.。．](改图|图生图|修图|aiedit)(?:\s|$)", priority=-10)
    @filter.custom_filter(ImageCommandWakePrefixFilter)
    async def edit_image_regex_fallback(self, event: AstrMessageEvent):
        """兼容"图片在前、文字在后"的消息：确保 /改图 能触发。"""
        msg = (event.message_str or "").strip()
        command_names = ("改图", "图生图", "修图", "aiedit")
        if self._is_framework_direct_command_text(msg, command_names, allow_bare=False):
            return
        try:
            if not await self._has_message_images_or_avatar_mentions(event):
                return
        except Exception:
            return

        prompt = ""
        matched = False
        for name in command_names:
            found_in_chain, prompt = self._extract_command_arg_from_chain(event, name)
            if found_in_chain:
                matched = True
                break
        if matched:
            event.should_call_llm(True)
            await self._do_edit(event, prompt, backend=None)
            event.stop_event()

    @filter.regex(r".*[/!！.。．][^\s]+", priority=-10)
    @filter.custom_filter(ImageCommandWakePrefixFilter)
    async def preset_regex_fallback(self, event: AstrMessageEvent):
        """兼容"图片在前、预设命令在后"的消息：确保 /<预设名> 能触发。"""
        preset_names = self.edit.get_preset_names()
        if not preset_names:
            return

        # 如果首段文本本来就是 /预设，则交给 command handler，避免重复处理
        try:
            if self._is_direct_command_message(event, tuple(preset_names)):
                return
        except Exception:
            pass

        # 仅当消息/引用里有图或有效 @ 头像目标时才兜底，避免误伤其它插件命令
        try:
            if not await self._has_message_images_or_avatar_mentions(event):
                return
        except Exception:
            return

        # Match only a complete preset command at the start of a raw text segment.
        used_preset: str | None = None
        extra_prompt = ""
        for name in preset_names:
            found, prompt = self._extract_command_arg_from_chain(event, name)
            if found:
                used_preset = name
                extra_prompt = prompt
                break

        if not used_preset:
            return

        await self._do_edit_direct(event, extra_prompt, preset=used_preset)
        event.stop_event()

    # ==================== Bot 自拍（参考照） ====================

    @filter.command("自拍")
    @filter.custom_filter(ImageCommandWakePrefixFilter)
    async def selfie_command(self, event: AstrMessageEvent):
        """使用"自拍参考照"生成 Bot 自拍。

        用法:
        - /自拍 <提示词>
        - 可附带多张参考图（衣服/姿势/场景）作为额外参考
        """
        if not self._is_selfie_enabled():
            await mark_failed(event)
            return
        event.should_call_llm(True)
        prompt = self._extract_extra_prompt(event, "自拍")
        await self._do_selfie(event, prompt, backend=None)

    @filter.regex(r".*[/!！.。．]自拍(?:\s|$)", priority=-10)
    @filter.custom_filter(ImageCommandWakePrefixFilter)
    async def selfie_regex_fallback(self, event: AstrMessageEvent):
        """兼容"图片在前、文字在后"的消息：确保 /自拍 能触发。"""
        if self._has_activated_handler(event, "selfie_command"):
            return
        found, prompt = self._extract_command_arg_from_chain(event, "自拍")
        if found:
            event.should_call_llm(True)
            if not self._is_selfie_enabled():
                await mark_failed(event)
                event.stop_event()
                return
            await self._do_selfie(event, prompt, backend=None)
            event.stop_event()

    @filter.command("自拍参考")
    @filter.custom_filter(ImageCommandWakePrefixFilter)
    async def selfie_reference_command(self, event: AstrMessageEvent):
        """管理自拍参考照（建议仅管理员使用）。

        用法:
        - 发送图片 + /自拍参考 设置
        - /自拍参考 查看
        - /自拍参考 删除
        """
        event.should_call_llm(True)
        if not self._is_selfie_enabled():
            await mark_failed(event)
            return
        arg = self._extract_extra_prompt(event, "自拍参考")
        action, _, _rest = (arg or "").strip().partition(" ")
        action = action.strip().lower()

        if not action or action in {"帮助", "help", "h"}:
            msg = (
                "📸 自拍参考照\n"
                "━━━━━━━━━━━━━━\n"
                "设置：发送图片 + /自拍参考 设置\n"
                "查看：/自拍参考 查看\n"
                "删除：/自拍参考 删除\n"
                "━━━━━━━━━━━━━━\n"
                "生成自拍：/自拍 <提示词>\n"
                "可附带额外参考图（衣服/姿势/场景）"
            )
            yield event.plain_result(msg)
            return

        if action in {"设置", "set"}:
            await self._set_selfie_reference(event)
            return

        if action in {"查看", "show", "看"}:
            async for result in self._show_selfie_reference(event):
                yield result
            return

        if action in {"删除", "del", "delete"}:
            await self._delete_selfie_reference(event)
            return

        await mark_failed(event)

    @filter.regex(r".*[/!！.。．]自拍参考(?:\s|$)", priority=-10)
    @filter.custom_filter(ImageCommandWakePrefixFilter)
    async def selfie_reference_regex_fallback(self, event: AstrMessageEvent):
        """兼容"图片在前、文字在后"的消息：确保 /自拍参考 能触发。"""
        if self._has_activated_handler(event, "selfie_reference_command"):
            return
        found, arg = self._extract_command_arg_from_chain(event, "自拍参考")
        if not found:
            return
        event.should_call_llm(True)
        if not self._is_selfie_enabled():
            await mark_failed(event)
            event.stop_event()
            return
        action, _, _rest = (arg or "").strip().partition(" ")
        action = action.strip().lower()

        if not action or action in {"帮助", "help", "h"}:
            yield event.plain_result(
                "📸 自拍参考照\n"
                "━━━━━━━━━━━━━━\n"
                "设置：发送图片 + /自拍参考 设置\n"
                "查看：/自拍参考 查看\n"
                "删除：/自拍参考 删除\n"
                "━━━━━━━━━━━━━━\n"
                "生成自拍：/自拍 <提示词>\n"
                "可附带额外参考图（衣服/姿势/场景）"
            )
            event.stop_event()
            return

        if action in {"设置", "set"}:
            await self._set_selfie_reference(event)
            event.stop_event()
            return

        if action in {"查看", "show", "看"}:
            async for r in self._show_selfie_reference(event):
                yield r
            event.stop_event()
            return

        if action in {"删除", "del", "delete"}:
            await self._delete_selfie_reference(event)
            event.stop_event()
            return

        await mark_failed(event)
        event.stop_event()

    # ==================== 视频生成 ====================

    @filter.command("视频")
    async def generate_video_command(self, event: AstrMessageEvent):
        """生成视频

        用法:
        - /视频 [@provider_id] <提示词>
        - /视频 [@provider_id] <预设名> [额外提示词]
        """
        event.should_call_llm(True)
        if not bool(self._get_feature("video").get("enabled", False)):
            await mark_failed(event)
            return
        arg = self._extract_extra_prompt(event, "视频")
        if not arg:
            await mark_failed(event)
            return

        provider_override, arg = self._parse_provider_override_prefix(arg)
        if not arg:
            await mark_failed(event)
            return

        preset, prompt = self._parse_video_args(arg)
        presets = self._get_video_presets()
        if preset and preset in presets:
            preset_prompt = presets[preset]
            prompt = f"{preset_prompt}, {prompt}" if prompt else preset_prompt

        user_id = str(event.get_sender_id() or "")
        request_id = self._debounce_key(event, "video", user_id)

        if self.debouncer.hit(request_id):
            await mark_failed(event)
            return

        if not await self._video_begin(user_id):
            await mark_failed(event)
            return

        try:
            await mark_processing(event)
        except Exception:
            await self._video_end(user_id)
            await mark_failed(event)
            return

        try:
            task = asyncio.create_task(
                self._async_generate_video(
                    event, prompt, user_id, provider_id=provider_override
                )
            )
        except Exception:
            await self._video_end(user_id)
            await mark_failed(event)
            return

        self._video_tasks.add(task)
        task.add_done_callback(lambda t: self._video_tasks.discard(t))
        return

    @filter.regex(r"[/!！.。．]视频(\s|$)", priority=-10)
    @filter.custom_filter(ImageCommandWakePrefixFilter)
    async def generate_video_regex_fallback(self, event: AstrMessageEvent):
        """兼容"图片在前、文字在后"的消息：确保 /视频 能触发。"""
        if self._is_direct_command_message(event, ("视频",)):
            return

        found, arg = self._extract_command_arg_from_chain(event, "视频")
        if not found:
            return

        event.should_call_llm(True)
        if not bool(self._get_feature("video").get("enabled", False)):
            await mark_failed(event)
            event.stop_event()
            return
        if not arg:
            await mark_failed(event)
            event.stop_event()
            return

        provider_override, arg = self._parse_provider_override_prefix(arg)
        if not arg:
            await mark_failed(event)
            event.stop_event()
            return

        preset, prompt = self._parse_video_args(arg)
        presets = self._get_video_presets()
        if preset and preset in presets:
            preset_prompt = presets[preset]
            prompt = f"{preset_prompt}, {prompt}" if prompt else preset_prompt

        user_id = str(event.get_sender_id() or "")
        request_id = self._debounce_key(event, "video", user_id)

        if self.debouncer.hit(request_id):
            await mark_failed(event)
            event.stop_event()
            return

        if not await self._video_begin(user_id):
            await mark_failed(event)
            event.stop_event()
            return

        try:
            await mark_processing(event)
        except Exception:
            await self._video_end(user_id)
            await mark_failed(event)
            event.stop_event()
            return

        try:
            task = asyncio.create_task(
                self._async_generate_video(
                    event, prompt, user_id, provider_id=provider_override
                )
            )
        except Exception:
            await self._video_end(user_id)
            await mark_failed(event)
            event.stop_event()
            return

        self._video_tasks.add(task)
        task.add_done_callback(lambda t: self._video_tasks.discard(t))
        event.stop_event()
        return

    @filter.command("视频预设列表")
    async def list_video_presets(self, event: AstrMessageEvent):
        """列出所有可用视频预设"""
        event.should_call_llm(True)
        presets = self._get_video_presets()
        names = list(presets.keys())
        if not names:
            yield event.plain_result(
                "📋 视频预设列表\n暂无预设（请在配置 features.video.presets 中添加）"
            )
            return

        msg = "📋 视频预设列表\n"
        for name in names:
            msg += f"- {name}\n"
        msg += "\n用法: /视频 [@provider_id] <预设名> [额外提示词]"
        yield event.plain_result(msg)

    # ==================== 管理命令 ====================

    @filter.command("文生图预设列表")
    async def list_draw_presets(self, event: AstrMessageEvent):
        """列出所有可用文生图预设"""
        event.should_call_llm(True)
        presets = self._get_draw_presets()
        backends = self.draw._candidate_ids()
        draw_conf = self._get_feature("draw")
        chain = []
        for it in (
            draw_conf.get("chain", [])
            if isinstance(draw_conf.get("chain", []), list)
            else []
        ):
            pid = self._extract_chain_provider_id(it)
            if pid and pid not in chain:
                chain.append(pid)

        if not presets:
            msg = "📋 文生图预设列表\n"
            msg += "━━━━━━━━━━━━━━\n"
            msg += f"🔧 可用后端: {', '.join(backends)}\n"
            if chain:
                msg += f"⭐ 当前链路: {', '.join(chain)}\n"
            msg += "━━━━━━━━━━━━━━\n"
            msg += "📌 暂无预设\n"
            msg += "━━━━━━━━━━━━━━\n"
            msg += "💡 在配置 features.draw.presets 中添加:\n"
            msg += '  格式: "预设名:英文提示词"'
        else:
            msg = "📋 文生图预设列表\n"
            msg += "━━━━━━━━━━━━━━\n"
            msg += f"🔧 可用后端: {', '.join(backends)}\n"
            if chain:
                msg += f"⭐ 当前链路: {', '.join(chain)}\n"
            msg += "━━━━━━━━━━━━━━\n"
            msg += "📌 预设:\n"
            for name in presets:
                msg += f"  • {name}\n"
        msg += "━━━━━━━━━━━━━━\n"
        msg += "💡 用法: /文生图 [@provider_id] <预设名> [补充提示词]"
        yield event.plain_result(msg)

    @filter.command("预设列表")
    async def list_presets(self, event: AstrMessageEvent):
        """列出所有可用预设"""
        event.should_call_llm(True)
        presets = self.edit.get_preset_names()
        backends = self.edit.get_available_backends()
        edit_conf = self._get_feature("edit")
        chain = []
        for it in (
            edit_conf.get("chain", [])
            if isinstance(edit_conf.get("chain", []), list)
            else []
        ):
            pid = self._extract_chain_provider_id(it)
            if pid and pid not in chain:
                chain.append(pid)

        if not presets:
            msg = "📋 改图预设列表\n"
            msg += "━━━━━━━━━━━━━━\n"
            msg += f"🔧 可用后端: {', '.join(backends)}\n"
            if chain:
                msg += f"⭐ 当前链路: {', '.join(chain)}\n"
            msg += "━━━━━━━━━━━━━━\n"
            msg += "📌 暂无预设\n"
            msg += "━━━━━━━━━━━━━━\n"
            msg += "💡 在配置 features.edit.presets 中添加:\n"
            msg += '  格式: "触发词:英文提示词"'
        else:
            msg = "📋 改图预设列表\n"
            msg += "━━━━━━━━━━━━━━\n"
            msg += f"🔧 可用后端: {', '.join(backends)}\n"
            if chain:
                msg += f"⭐ 当前链路: {', '.join(chain)}\n"
            msg += "━━━━━━━━━━━━━━\n"
            msg += "📌 预设:\n"
            for name in presets:
                msg += f"  • {name}\n"
        msg += "━━━━━━━━━━━━━━\n"
        msg += "💡 用法: /aiedit [@provider_id] <提示词> [图片]"

        yield event.plain_result(msg)

    @filter.command("改图帮助")
    async def edit_help(self, event: AstrMessageEvent):
        """显示改图帮助"""
        event.should_call_llm(True)
        msg = """🎨 改图功能帮助

━━ 基础命令 ━━
/aiedit [@provider_id] <提示词>

━━ 使用方式 ━━
1. 发送图片 + 命令
2. 引用图片消息 + 命令

━━ 服务商链路 ━━
在 WebUI 配置：
- providers：添加服务商（id/url/key/model/超时/重试等）
- features.edit.chain：按顺序填写 provider_id（第一个=主用，其余=兜底）

━━ 自定义预设 ━━
查看预设：/预设列表
在 WebUI 配置 features.edit.presets 添加：
格式: 预设名:英文提示词
示例: 手办化:Transform into figurine style
"""

        yield event.plain_result(msg)

    # ==================== LLM 工具 ====================

    @filter.llm_tool(name="gitee_draw_image")
    async def gitee_draw_image(self, event: AstrMessageEvent, prompt: str):
        """（兼容旧版本）根据提示词生成图片。

        Args:
            prompt(string): 图片提示词，需要包含主体、场景、风格等描述
        """
        return await self.aiimg_generate(
            event, prompt=prompt, mode="text", backend="auto"
        )

    @filter.llm_tool(name="gitee_edit_image")
    async def gitee_edit_image(
        self,
        event: AstrMessageEvent,
        prompt: str,
        use_message_images: bool = True,
        backend: str = "auto",
    ):
        """（兼容旧版本）编辑用户发送的图片或引用的图片。

        Args:
            prompt(string): 图片编辑提示词
            use_message_images(boolean): 是否自动获取用户消息中的图片（目前仅支持 true）
            backend(string): auto=自动选择；也可填 provider_id（你在 WebUI providers 里配置的 id）
        """
        if not use_message_images:
            await self._signal_llm_tool_failure(event)
            return self._llm_tool_text_result(
                "This image editing request is invalid because message images were disabled. Use the images already attached to the current message."
            )
        return await self.aiimg_generate(
            event, prompt=prompt, mode="edit", backend=backend
        )

    @filter.llm_tool(name="aiimg_generate")
    async def aiimg_generate(
        self,
        event: AstrMessageEvent,
        prompt: str,
        mode: str = "auto",
        backend: str = "auto",
        output: str = "",
        aspect_ratio: str = "auto",
        resolution: str = "auto",
    ):
        """统一图片生成/改图/生活照（参考照）工具。

        使用建议（给 LLM 的决策规则）：
        - 用户发送/引用了图片，并要求"改图/换背景/换风格/修图/换衣服"等：用 mode=edit（或 mode=auto）
        - 用户要求"看看你/来一张你自己的生活照"，且已设置自拍参考照：用 mode=selfie_ref（或 mode=auto）
        - 纯文生图（用户没有给图片）：用 mode=text（或 mode=auto）

        当前 LLM tool 行为：
        - 成功后优先直接把图片发送给用户
        - tool result 返回文本摘要，写明本次任务的 mode、effective_prompt 和 follow-up 提示
        - 不再把 ImageContent 回传给 LLM 上下文，避免额外多模态识图耗时

        画面比例决策：
        - 用户明确比例时必须原样传入 aspect_ratio
        - 用户未指定时根据构图主动选择：人像/自拍优先 3:4、4:5、9:16，横向场景优先 4:3、3:2、16:9
        - 除非方形构图确实合适，否则不要省略 aspect_ratio，也不要默认使用 1:1

        Args:
            prompt(string): 提示词
            mode(string): auto=自动判断, text=文生图, edit=改图, selfie_ref=参考照
            backend(string): auto=自动选择；也可填 provider_id（你在 WebUI providers 里配置的 id）
            output(string): 兼容输出参数。只有用户明确指定时才传；不得自行填入 1:1 或正方形默认值
            aspect_ratio(string): 图片比例。用户明确要求时照办；未指定时根据构图主动选择 16:9、9:16、4:3、3:4 等
            resolution(string): 图片分辨率。默认 auto；用户明确要求时传 1K、2K 或 4K
        """
        prompt = (prompt or "").strip()
        m = (mode or "auto").strip().lower()

        # === TTL 去重检查（防止 ToolLoop 重复调用）===
        message_id = (
            getattr(getattr(event, "message_obj", None), "message_id", "") or ""
        )
        origin = getattr(event, "unified_msg_origin", "") or ""
        if message_id and origin:
            if self.debouncer.llm_tool_is_duplicate(message_id, origin):
                logger.debug(f"[aiimg_generate] 重复调用已拦截: msg_id={message_id}")
                await mark_success(event)
                return self._llm_tool_text_result(
                    "This image request was already handled for the same message. Do not run it again."
                )

        user_id = str(event.get_sender_id() or "")
        request_id = self._debounce_key(event, "aiimg", user_id)
        if self.debouncer.hit(request_id):
            await mark_success(event)
            return self._llm_tool_text_result(
                "This image request is already being handled or was just handled. Do not submit it again unless the user explicitly asks for a new image."
            )

        if self._background_manager_for_event(event) is not None:
            try:
                return await self._accept_background_single(
                    event,
                    prompt=prompt,
                    mode=m,
                    backend=backend,
                    output=output,
                    aspect_ratio=aspect_ratio,
                    resolution=resolution,
                )
            except BackgroundTaskCapacityError as exc:
                await self._signal_llm_tool_failure(event)
                return self._llm_tool_text_result(
                    "The background image queue is currently full. The request was not started. Reason: "
                    + BackgroundImageTaskManager.sanitize_error(exc)
                )
            except Exception as exc:
                logger.error(
                    "[background-image] task preparation failed: %s",
                    BackgroundImageTaskManager.sanitize_error(exc),
                    exc_info=True,
                )
                await self._signal_llm_tool_failure(event)
                return self._llm_tool_text_result(
                    "The image request could not be prepared and was not started. Reason: "
                    + BackgroundImageTaskManager.sanitize_error(exc)
                )

        if not await self._begin_user_job(user_id, kind="image"):
            await mark_success(event)
            return self._llm_tool_text_result(
                "An image request for this user is already in progress. Do not resubmit unless the user asks for a new request."
            )

        b_raw = (backend or "auto").strip()
        known_provider_ids = set(self.registry.provider_ids())
        if not b_raw or b_raw.lower() == "auto":
            target_backend = None
        elif b_raw in known_provider_ids:
            target_backend = b_raw
        else:
            logger.warning(
                "[aiimg_generate] 忽略未知 backend 覆盖，回退自动链路: backend=%s",
                b_raw,
            )
            target_backend = None

        output = (output or "").strip()

        try:
            output_intent = resolve_llm_output_intent(
                prompt,
                output=output,
                aspect_ratio=aspect_ratio,
                resolution=resolution,
            )
            logger.info(
                "[aiimg_generate] resolved output intent: %s",
                output_intent,
            )
            await mark_processing(event)

            if m in {"selfie_ref", "selfie", "ref"}:
                logger.info("[aiimg_generate] route=selfie_ref (explicit)")
                if not self._is_selfie_enabled():
                    logger.warning(
                        "[aiimg_generate] selfie blocked: features.selfie.enabled=false"
                    )
                    await self._signal_llm_tool_failure(event)
                    return self._llm_tool_text_result(
                        "The requested selfie image tool is disabled by plugin configuration."
                    )
                if not self._is_selfie_llm_enabled():
                    logger.warning(
                        "[aiimg_generate] selfie blocked: features.selfie.llm_tool_enabled=false"
                    )
                    await self._signal_llm_tool_failure(event)
                    return self._llm_tool_text_result(
                        "The requested selfie image tool is disabled by plugin configuration."
                    )
                image_path, task_meta = await self._generate_selfie_image_with_meta(
                    event,
                    prompt,
                    target_backend,
                    output_intent=output_intent,
                )
                return await self._finalize_llm_tool_image(
                    event, image_path, task_meta=task_meta
                )

            # 自动模式：优先识别"自拍"语义 + 已配置参考照
            if m == "auto" and await self._should_auto_selfie_ref(event, prompt):
                if not self._is_selfie_enabled():
                    logger.info(
                        "[aiimg_generate] auto-selfie skipped: features.selfie.enabled=false"
                    )
                elif not self._is_selfie_llm_enabled():
                    logger.info(
                        "[aiimg_generate] auto-selfie skipped: features.selfie.llm_tool_enabled=false"
                    )
                else:
                    try:
                        logger.info("[aiimg_generate] route=auto->selfie_ref")
                        (
                            image_path,
                            task_meta,
                        ) = await self._generate_selfie_image_with_meta(
                            event,
                            prompt,
                            target_backend,
                            output_intent=output_intent,
                        )
                    except Exception as e:
                        logger.warning(
                            "[aiimg_generate] auto-selfie failed, fallback to draw/edit: %s",
                            e,
                        )
                    else:
                        return await self._finalize_llm_tool_image(
                            event, image_path, task_meta=task_meta
                        )

            if m == "auto":
                follow_up_selfie_meta = await self._match_selfie_follow_up(
                    event, prompt
                )
                if follow_up_selfie_meta is not None:
                    try:
                        logger.info(
                            "[aiimg_generate] route=auto->selfie_ref (follow-up)"
                        )
                        (
                            image_path,
                            task_meta,
                        ) = await self._generate_selfie_image_with_meta(
                            event,
                            prompt,
                            target_backend,
                            output_intent=output_intent,
                            follow_up_meta=follow_up_selfie_meta,
                        )
                    except Exception as e:
                        logger.warning(
                            "[aiimg_generate] selfie follow-up failed, fallback to draw/edit: %s",
                            e,
                        )
                    else:
                        return await self._finalize_llm_tool_image(
                            event, image_path, task_meta=task_meta
                        )

            # 改图：用户消息中有图片（不含头像兜底）或显式指定
            has_msg_images = await self._has_message_images(event)
            prefetched_edit_image_segs = None
            has_at_avatar_refs = False
            if m == "auto" and not has_msg_images:
                prefetched_edit_image_segs = await get_images_from_event(
                    event,
                    include_avatar=True,
                    include_sender_avatar_fallback=False,
                )
                has_at_avatar_refs = bool(prefetched_edit_image_segs)

            if m in {"edit", "img2img", "aiedit"} or (
                m == "auto" and (has_msg_images or has_at_avatar_refs)
            ):
                logger.info("[aiimg_generate] route=edit")
                edit_conf = self._get_feature("edit")
                if not bool(edit_conf.get("enabled", True)):
                    await self._signal_llm_tool_failure(event)
                    return self._llm_tool_text_result(
                        "The requested image editing tool is disabled by plugin configuration."
                    )
                if not bool(edit_conf.get("llm_tool_enabled", True)):
                    await self._signal_llm_tool_failure(event)
                    return self._llm_tool_text_result(
                        "The requested image editing tool is disabled by plugin configuration."
                    )
                image_segs = prefetched_edit_image_segs
                if image_segs is None:
                    image_segs = await get_images_from_event(
                        event,
                        include_avatar=True,
                        include_sender_avatar_fallback=False,
                    )
                bytes_images = await self._image_segs_to_bytes(image_segs)
                if not bytes_images:
                    await self._signal_llm_tool_failure(event)
                    return self._llm_tool_text_result(
                        "Image editing could not continue because no usable input image was found in the current message. This request has ended."
                    )

                image_path = await self.edit.edit(
                    prompt=prompt,
                    images=bytes_images,
                    backend=target_backend,
                    output_intent=output_intent,
                )
                task_meta = self._build_image_task_meta(
                    mode="edit",
                    user_prompt=prompt,
                    effective_prompt=prompt,
                    continue_with="edit",
                    backend=target_backend,
                )
                return await self._finalize_llm_tool_image(
                    event, image_path, task_meta=task_meta
                )

            # 默认：文生图
            draw_conf = self._get_feature("draw")
            if not bool(draw_conf.get("enabled", True)):
                await self._signal_llm_tool_failure(event)
                return self._llm_tool_text_result(
                    "The requested image generation tool is disabled by plugin configuration."
                )
            if not bool(draw_conf.get("llm_tool_enabled", True)):
                await self._signal_llm_tool_failure(event)
                return self._llm_tool_text_result(
                    "The requested image generation tool is disabled by plugin configuration."
                )
            if not prompt:
                prompt = "a selfie photo"

            logger.info("[aiimg_generate] route=draw")
            image_path = await self.draw.generate(
                prompt,
                provider_id=target_backend,
                output_intent=output_intent,
            )
            task_meta = self._build_image_task_meta(
                mode="text",
                user_prompt=prompt,
                effective_prompt=prompt,
                continue_with="text",
                backend=target_backend,
            )
            return await self._finalize_llm_tool_image(
                event, image_path, task_meta=task_meta
            )

        except Exception as e:
            logger.error(f"[aiimg_generate] 失败: {e}", exc_info=True)
            await self._signal_llm_tool_failure(event)
            return self._llm_tool_text_result(
                "The image request failed and has ended. Reason: "
                + self._summarize_status_text(
                    e,
                    fallback="unknown error",
                )
                + ". Do not retry automatically unless the user explicitly asks."
            )
        finally:
            await self._end_user_job(user_id, kind="image")

    @filter.llm_tool(name="aiimg_batch_generate")
    async def aiimg_batch_generate(
        self,
        event: AstrMessageEvent,
        prompt: str,
        count: int = 4,
        mode: str = "auto",
        backend: str = "auto",
        output: str = "",
        aspect_ratio: str = "auto",
        resolution: str = "auto",
    ):
        """规划并批量生成一组图片。

        使用建议（给 LLM 的决策规则）：
        - 当用户明确想要一组不重复但同主题的图片时，优先调用这个工具。
        - 先规划多条不同 prompt，再批量执行，不要自己重复调用单图工具。
        - 用户未指定比例时保持 aspect_ratio=auto，内部 planner 会为每张图按构图选择不同的合适比例。

        Args:
            prompt(string): 用户的总要求。应包含整组图片共同要满足的条件。
            count(number): 目标数量。建议 2-8。
            mode(string): auto=自动判断, text=文生图, edit=改图, selfie_ref=参考照自拍
            backend(string): auto=自动选择；也可填 provider_id（你在 WebUI providers 里配置的 id）
            output(string): 兼容输出参数。只有用户明确指定时才传；不得自行填入 1:1 或正方形默认值
            aspect_ratio(string): 整组固定比例。默认 auto，由内部 planner 逐图选择；用户明确要求时传 16:9、9:16、4:3 等
            resolution(string): 整组图片分辨率。默认 auto；用户明确要求时传 1K、2K 或 4K
        """
        prompt = str(prompt or "").strip()
        if not prompt:
            await self._signal_llm_tool_failure(event)
            return self._llm_tool_text_result(
                "Batch image planning failed because no prompt was provided."
            )

        target_count = self._as_int(count, default=4)
        target_count = max(1, min(self._get_batch_max_count(), target_count))
        resolved_mode = await self._resolve_llm_batch_mode(event, mode, prompt)
        target_backend = self._resolve_target_backend(backend)

        output = (output or "").strip()

        if resolved_mode == "draw":
            draw_conf = self._get_feature("draw")
            if not bool(draw_conf.get("enabled", True)) or not bool(
                draw_conf.get("llm_tool_enabled", True)
            ):
                await self._signal_llm_tool_failure(event)
                return self._llm_tool_text_result(
                    "The requested batch text-to-image tool is disabled by plugin configuration."
                )
        elif resolved_mode == "edit":
            edit_conf = self._get_feature("edit")
            if not bool(edit_conf.get("enabled", True)) or not bool(
                edit_conf.get("llm_tool_enabled", True)
            ):
                await self._signal_llm_tool_failure(event)
                return self._llm_tool_text_result(
                    "The requested batch image editing tool is disabled by plugin configuration."
                )
        elif resolved_mode == "selfie_ref":
            if not self._is_selfie_enabled() or not self._is_selfie_llm_enabled():
                await self._signal_llm_tool_failure(event)
                return self._llm_tool_text_result(
                    "The requested batch selfie image tool is disabled by plugin configuration."
                )

        message_id = (
            getattr(getattr(event, "message_obj", None), "message_id", "") or ""
        )
        origin = getattr(event, "unified_msg_origin", "") or ""
        if (
            message_id
            and origin
            and self.debouncer.llm_tool_is_duplicate(message_id, origin)
        ):
            await mark_success(event)
            return self._llm_tool_text_result(
                "This batch image request was already handled for the same message. Do not run it again."
            )

        user_id = str(event.get_sender_id() or "")
        request_id = self._debounce_key(event, "aiimg_batch", user_id)
        if self.debouncer.hit(request_id):
            await mark_success(event)
            return self._llm_tool_text_result(
                "This batch image request is already being handled or was just handled. Do not resubmit unless the user explicitly asks for a new batch."
            )

        if self._background_manager_for_event(event) is not None:
            try:
                return await self._accept_background_batch(
                    event,
                    prompt=prompt,
                    count=target_count,
                    mode=resolved_mode,
                    backend=backend,
                    output=output,
                    aspect_ratio=aspect_ratio,
                    resolution=resolution,
                )
            except BackgroundTaskCapacityError as exc:
                await self._signal_llm_tool_failure(event)
                return self._llm_tool_text_result(
                    "The background image queue does not have enough capacity for this batch. The request was not started. Reason: "
                    + BackgroundImageTaskManager.sanitize_error(exc)
                )
            except Exception as exc:
                logger.error(
                    "[background-image] batch preparation failed: %s",
                    BackgroundImageTaskManager.sanitize_error(exc),
                    exc_info=True,
                )
                await self._signal_llm_tool_failure(event)
                return self._llm_tool_text_result(
                    "The batch image request could not be prepared and was not started. Reason: "
                    + BackgroundImageTaskManager.sanitize_error(exc)
                )

        if not await self._begin_user_job(user_id, kind="image"):
            await mark_success(event)
            return self._llm_tool_text_result(
                "A batch image request for this user is already in progress. Do not resubmit unless the user asks for a new request."
            )

        try:
            output_intent = resolve_llm_output_intent(
                prompt,
                output=output,
                aspect_ratio=aspect_ratio,
                resolution=resolution,
            )
            logger.info(
                "[aiimg_batch_generate] resolved output intent: %s",
                output_intent,
            )
            await mark_processing(event)
            planned_items = await self._plan_batch_prompt_items(
                mode=resolved_mode,
                user_prompt=prompt,
                count=target_count,
                fixed_aspect_ratio=(
                    output_intent.aspect_ratio
                    or aspect_ratio_from_size(output_intent.exact_size)
                ),
            )
            specs = [
                ImageTaskSpec(
                    mode=resolved_mode,
                    provider_id=target_backend,
                    preset_name=None,
                    user_prompt=item.prompt,
                    effective_prompt=item.prompt,
                    source_command="llm_batch",
                    variant_title=item.title,
                    output=format_output_intent(
                        OutputIntent(aspect_ratio=item.aspect_ratio)
                    ),
                )
                for item in planned_items
            ]
            results = await self._run_batch_specs(
                event,
                specs,
                output_intent=output_intent,
            )
            await self._send_batch_results(
                event,
                results,
                title=f"LLM 批量{self._batch_mode_label(specs[0])} x{len(specs)}",
            )
            success_count = sum(
                1 for result in results if result.success and result.value
            )
            failed_count = len(results) - success_count
            if success_count > 0:
                await self._remember_batch_success(event, results)
                await mark_success(event)
            else:
                await self._signal_llm_tool_failure(event)
            return self._llm_tool_text_result(
                "The batch image set has already been generated and sent to the user. "
                f"Mode={resolved_mode}, success={success_count}, failed={failed_count}. "
                "Do not send another confirmation message to the user."
            )
        except Exception as e:
            logger.error("[aiimg_batch_generate] 失败: %s", e, exc_info=True)
            await self._signal_llm_tool_failure(event)
            return self._llm_tool_text_result(
                "The batch image request failed and has ended. Reason: "
                + self._summarize_status_text(e, fallback="unknown error")
            )
        finally:
            await self._end_user_job(user_id, kind="image")

    @filter.llm_tool(name="aiimg_task_status")
    async def aiimg_task_status(
        self,
        event: AstrMessageEvent,
        task_id: str = "",
        include_prompts: bool = True,
        offset: int = 0,
        limit: int = 8,
    ) -> mcp.types.CallToolResult:
        """Query durable image task facts without starting provider work.

        Args:
            task_id(string): Task ID. Leave empty for the latest task in this conversation.
            include_prompts(boolean): Include complete effective prompts for the requester.
            offset(number): Batch child pagination offset.
            limit(number): Batch child page size, limited to 1-8.
        """

        manager = getattr(self, "background_tasks", None)
        if manager is None:
            return self._llm_tool_text_result(
                json.dumps(
                    {
                        "status": "unavailable",
                        "message": "Background LLM image tasks are not enabled.",
                    }
                )
            )
        target = await self._build_background_delivery_target(event)
        requested_id = str(task_id or "").strip()
        record: dict[str, Any] | None
        if requested_id:
            record = await manager.get_task(requested_id)
        else:
            scope = manager.scope_hash(
                target.umo,
                target.self_id,
                target.sender_id,
                target.conversation_id,
            )
            records = await manager.list_scope_tasks(scope, limit=1)
            record = records[0] if records else None
        if record is None:
            return self._llm_tool_text_result(
                json.dumps({"status": "not_found", "task_id": requested_id})
            )

        is_owner = (
            str(record.get("umo") or "") == target.umo
            and str(record.get("self_id") or "") == target.self_id
            and str(record.get("sender_id") or "") == target.sender_id
            and str(record.get("conversation_id") or "") == target.conversation_id
        )
        if not is_owner:
            public = {
                "status": "forbidden",
                "task_id": record.get("task_id"),
                "state": record.get("state"),
                "requester": record.get("sender_name") or record.get("sender_id"),
                "message": "Detailed prompts are only available to the original requester in the original conversation.",
            }
            return self._llm_tool_text_result(json.dumps(public, ensure_ascii=False))

        start = max(0, self._as_int(offset, default=0))
        page_size = max(1, min(8, self._as_int(limit, default=8)))
        items = record.get("items") if isinstance(record.get("items"), list) else []
        page = []
        for item in items[start : start + page_size]:
            child = {
                "item_id": item.get("item_id"),
                "index": item.get("index"),
                "state": item.get("state"),
                "aspect_ratio": item.get("aspect_ratio"),
                "image_generated": bool(item.get("image_generated")),
                "image_sent": bool(item.get("image_sent")),
                "delivery_state": item.get("delivery_state"),
                "error_message": item.get("error_message"),
            }
            if include_prompts:
                child["user_prompt"] = item.get("user_prompt")
                child["effective_prompt"] = item.get("effective_prompt")
            page.append(child)
        payload = {
            "status": "ok",
            "task_id": record.get("task_id"),
            "task_kind": record.get("task_kind"),
            "state": record.get("state"),
            "mode": record.get("mode"),
            "image_generated": bool(record.get("image_generated")),
            "image_sent": bool(record.get("image_sent")),
            "delivery_state": record.get("delivery_state"),
            "requested_count": record.get("requested_count"),
            "planned_count": record.get("planned_count"),
            "generated_count": record.get("generated_count"),
            "sent_count": record.get("sent_count"),
            "failed_count": record.get("failed_count"),
            "cancelled_count": record.get("cancelled_count"),
            "unknown_count": record.get("unknown_count"),
            "user_prompt": record.get("user_prompt") if include_prompts else None,
            "effective_prompt": record.get("effective_prompt")
            if include_prompts
            else None,
            "error_message": record.get("error_message"),
            "items": page,
            "offset": start,
            "limit": page_size,
            "total": len(items),
            "next_offset": start + page_size
            if start + page_size < len(items)
            else None,
        }
        return self._llm_tool_text_result(json.dumps(payload, ensure_ascii=False))

    @filter.llm_tool()
    async def grok_generate_video(self, event: AstrMessageEvent, prompt: str):
        """根据用户发送/引用的图片生成视频。

        Args:
            prompt(string): 视频提示词。支持 "预设名 额外提示词"（与 `/视频 预设名 额外提示词` 一致）
        """
        vconf = self._get_feature("video")
        if not bool(vconf.get("enabled", False)):
            await self._signal_llm_tool_failure(event)
            return self._llm_tool_text_result(
                "The requested video tool is disabled by plugin configuration."
            )
        if not bool(vconf.get("llm_tool_enabled", True)):
            await self._signal_llm_tool_failure(event)
            return self._llm_tool_text_result(
                "The requested video tool is disabled by plugin configuration."
            )

        arg = (prompt or "").strip()
        if not arg:
            await self._signal_llm_tool_failure(event)
            return self._llm_tool_text_result(
                "The video request failed because no prompt was provided. This request has ended."
            )

        provider_override, arg = self._parse_provider_override_prefix(arg)
        if not arg:
            await self._signal_llm_tool_failure(event)
            return self._llm_tool_text_result(
                "The video request failed because no usable prompt remained after parsing provider overrides. This request has ended."
            )

        preset, extra_prompt = self._parse_video_args(arg)
        presets = self._get_video_presets()
        if preset and preset in presets:
            preset_prompt = presets[preset]
            extra_prompt = (
                f"{preset_prompt}, {extra_prompt}" if extra_prompt else preset_prompt
            )

        user_id = str(event.get_sender_id() or "")
        request_id = self._debounce_key(event, "video", user_id)

        if self.debouncer.hit(request_id):
            await mark_success(event)
            return self._llm_tool_text_result(
                "This video request is already being handled or was just handled. Do not submit it again unless the user explicitly asks for a new video."
            )

        if not await self._video_begin(user_id):
            await mark_success(event)
            return self._llm_tool_text_result(
                "A video request for this user is already in progress. Do not resubmit unless the user asks for a new request."
            )

        try:
            await mark_processing(event)
            task = asyncio.create_task(
                self._async_generate_video(
                    event,
                    extra_prompt,
                    user_id,
                    provider_id=provider_override,
                    llm_tool_failure=True,
                )
            )
        except Exception:
            await self._video_end(user_id)
            await self._signal_llm_tool_failure(event)
            return self._llm_tool_text_result(
                "The video request failed before background execution could start. This request has ended."
            )

        self._video_tasks.add(task)
        task.add_done_callback(lambda t: self._video_tasks.discard(t))

        return self._llm_tool_text_result(
            "Video generation has been accepted and is running in the background. The result will be sent to the user automatically when ready. Do not submit the same request again unless the user explicitly asks."
        )

    # ==================== 内部方法 ====================

    def _get_feature(self, name: str) -> dict:
        feats = self.config.get("features", {}) if isinstance(self.config, dict) else {}
        feats = feats if isinstance(feats, dict) else {}
        conf = feats.get(name, {})
        return conf if isinstance(conf, dict) else {}

    def _background_manager_for_event(
        self, event: AstrMessageEvent
    ) -> BackgroundImageTaskManager | None:
        """Return the active manager when this event supports safe background mode.

        Args:
            event: Current LLM tool event.

        Returns:
            Active manager, or ``None`` to preserve the synchronous path.
        """

        manager = getattr(self, "background_tasks", None)
        if manager is None or not manager.accepting:
            return None
        try:
            platform_name = str(event.get_platform_name() or "").strip()
            platform_id = str(event.get_platform_id() or "").strip()
            umo = str(event.unified_msg_origin or "").strip()
        except Exception:
            return None
        if (
            platform_name not in {"aiocqhttp", "weixin_oc"}
            or not platform_id
            or not umo
        ):
            return None
        if not callable(getattr(StarTools, "create_message", None)):
            return None
        adapter = self.context.get_platform_inst(platform_id)
        if adapter is None or not callable(getattr(adapter, "create_event", None)):
            return None
        try:
            config = self.context.get_config(umo=umo)
            streaming = bool(
                config.get("provider_settings", {}).get("streaming_response", False)
            )
        except Exception:
            return None
        if streaming:
            logger.warning(
                "[background-image] synchronous fallback because streaming_response is enabled"
            )
            return None
        return manager

    async def _expire_background_send_gate(
        self,
        umo: str,
        gate: asyncio.Event,
    ) -> None:
        """Release a reset/new send gate if the command pipeline never resolves.

        Args:
            umo: Unified message origin guarded by the gate.
            gate: Exact gate instance to avoid deleting a newer barrier.
        """

        await asyncio.sleep(30)
        current = self._background_send_gates.get(umo)
        if current is gate:
            self._background_send_gates.pop(umo, None)
            gate.set()

    async def _wait_background_send_gate(self, umo: str) -> None:
        """Wait for a pending reset/new command before sending an image.

        Args:
            umo: Unified message origin about to receive an image.
        """

        gate = self._background_send_gates.get(umo)
        if gate is None or gate.is_set():
            return
        try:
            await asyncio.wait_for(gate.wait(), timeout=35)
        except TimeoutError:
            current = self._background_send_gates.get(umo)
            if current is gate:
                self._background_send_gates.pop(umo, None)
                gate.set()

    async def _build_background_delivery_target(
        self, event: AstrMessageEvent
    ) -> TaskDeliveryTarget:
        """Capture the persistable routing data needed after the event ends.

        Args:
            event: Current LLM tool event.

        Returns:
            Immutable delivery target.
        """

        conversation = await self._resolve_plugin_conversation(event)
        conversation_id = str(getattr(conversation, "cid", "") or "").strip()
        message_type = event.get_message_type()
        message_type_text = str(getattr(message_type, "value", message_type) or "")
        return TaskDeliveryTarget(
            platform_id=str(event.get_platform_id() or "").strip(),
            platform_name=str(event.get_platform_name() or "").strip(),
            message_type=message_type_text,
            umo=str(event.unified_msg_origin or "").strip(),
            session_id=str(event.get_session_id() or "").strip(),
            group_id=str(event.get_group_id() or "").strip(),
            self_id=str(event.get_self_id() or "").strip(),
            sender_id=str(event.get_sender_id() or "").strip(),
            sender_name=str(event.get_sender_name() or "").strip(),
            source_message_id=str(
                getattr(getattr(event, "message_obj", None), "message_id", "") or ""
            ).strip(),
            conversation_id=conversation_id,
        )

    @staticmethod
    def _output_intent_dict(intent: OutputIntent) -> dict[str, Any]:
        return {
            "exact_size": intent.exact_size,
            "aspect_ratio": intent.aspect_ratio,
            "resolution": intent.resolution,
        }

    async def _prepare_background_selfie(
        self,
        event: AstrMessageEvent,
        prompt: str,
        backend: str | None,
        *,
        follow_up_meta: dict[str, Any] | None = None,
    ) -> tuple[list[bytes], str, dict[str, Any], dict[str, Any]]:
        """Prepare selfie inputs and provider options without calling a provider.

        Args:
            event: Current tool event used only during input preparation.
            prompt: User selfie request.
            backend: Requested provider ID.
            follow_up_meta: Optional previous selfie metadata.

        Returns:
            Input bytes, effective prompt, provider options, and task metadata.

        Raises:
            RuntimeError: Selfie configuration or reference inputs are invalid.
        """

        conf = self._get_selfie_conf()
        if not self._is_selfie_enabled():
            raise RuntimeError(self._selfie_disabled_message())
        ref_paths, ref_source = await self._get_selfie_reference_paths(event)
        ref_images = await self._read_paths_bytes(ref_paths)
        if not ref_images:
            raise RuntimeError(
                "未设置自拍参考照。请先发送图片并设置自拍参考，或在 WebUI 上传参考图。"
            )
        extra_segs = await get_images_from_event(event, include_avatar=False)
        extra_bytes = await self._image_segs_to_bytes(extra_segs)
        effective_user_prompt = self._build_selfie_follow_up_prompt(
            prompt, follow_up_meta
        )
        effective_prompt = self._build_selfie_prompt(
            effective_user_prompt,
            extra_refs=len(extra_bytes),
        )

        chain_override: list[dict] | None = None
        raw_chain = conf.get("chain", [])
        if isinstance(raw_chain, list):
            normalized_chain = [
                normalized
                for normalized in (
                    self._normalize_chain_item(item) for item in raw_chain
                )
                if normalized is not None
            ]
            if normalized_chain:
                chain_override = normalized_chain
        use_edit_chain = bool(conf.get("use_edit_chain_when_empty", True))
        if backend is None:
            if chain_override is None and not use_edit_chain:
                raise RuntimeError(
                    "No selfie provider chain configured. Please configure features.selfie.chain."
                )
            if chain_override is not None and use_edit_chain:
                chain_override = self._merge_selfie_chain_with_edit_chain(
                    chain_override
                )

        raw_task_types = conf.get("gitee_task_types")
        task_types = (
            [str(item).strip() for item in raw_task_types if str(item).strip()]
            if isinstance(raw_task_types, list) and raw_task_types
            else ["id", "background", "style"]
        )
        options = {
            "task_types": task_types,
            "default_output": self._get_selfie_default_output(),
            "chain_override": chain_override,
            "infer_source_aspect": False,
            "reference_source": ref_source,
            "reference_count": len(ref_images),
            "extra_reference_count": len(extra_bytes),
        }
        task_meta = self._build_image_task_meta(
            mode="selfie_ref",
            user_prompt=prompt,
            effective_user_prompt=effective_user_prompt,
            effective_prompt=effective_prompt,
            reference_source=ref_source,
            reference_count=len(ref_images),
            extra_reference_count=len(extra_bytes),
            continue_with="selfie_ref",
            follow_up=follow_up_meta is not None,
            backend=backend,
        )
        return [*ref_images, *extra_bytes], effective_prompt, options, task_meta

    async def _accept_background_single(
        self,
        event: AstrMessageEvent,
        *,
        prompt: str,
        mode: str,
        backend: str,
        output: str,
        aspect_ratio: str,
        resolution: str,
    ) -> mcp.types.CallToolResult:
        """Prepare and accept one LLM image task without provider blocking.

        Args:
            event: Current LLM tool event.
            prompt: User image request.
            mode: Requested routing mode.
            backend: Requested provider override.
            output: Legacy output specification.
            aspect_ratio: Requested aspect ratio.
            resolution: Requested resolution.

        Returns:
            Accepted task result or a deterministic preparation failure.
        """

        manager = self._background_manager_for_event(event)
        if manager is None:
            raise BackgroundTaskError("Background mode is unavailable for this event")

        task_id = manager.new_task_id("img")
        spooled = False
        try:
            target = await self._build_background_delivery_target(event)
            output_intent = resolve_llm_output_intent(
                prompt,
                output=output,
                aspect_ratio=aspect_ratio,
                resolution=resolution,
            )
            resolved_mode = str(mode or "auto").strip().lower()
            target_backend = self._resolve_target_backend(backend)
            input_bytes: list[bytes] = []
            options: dict[str, Any] = {}
            task_meta: dict[str, Any]
            effective_prompt = prompt

            explicit_selfie = resolved_mode in {"selfie_ref", "selfie", "ref"}
            follow_up_meta: dict[str, Any] | None = None
            if explicit_selfie:
                if not self._is_selfie_llm_enabled():
                    raise RuntimeError("The requested selfie image tool is disabled.")
                (
                    input_bytes,
                    effective_prompt,
                    options,
                    task_meta,
                ) = await self._prepare_background_selfie(
                    event,
                    prompt,
                    target_backend,
                )
                resolved_mode = "selfie_ref"
            else:
                prepared_selfie = False
                if (
                    resolved_mode == "auto"
                    and self._is_selfie_enabled()
                    and self._is_selfie_llm_enabled()
                    and await self._should_auto_selfie_ref(event, prompt)
                ):
                    try:
                        (
                            input_bytes,
                            effective_prompt,
                            options,
                            task_meta,
                        ) = await self._prepare_background_selfie(
                            event,
                            prompt,
                            target_backend,
                        )
                    except Exception as exc:
                        logger.warning(
                            "[background-image] auto selfie preparation fell back: %s",
                            BackgroundImageTaskManager.sanitize_error(exc),
                        )
                    else:
                        resolved_mode = "selfie_ref"
                        prepared_selfie = True

                if resolved_mode == "auto" and not prepared_selfie:
                    follow_up_meta = await self._match_selfie_follow_up(event, prompt)
                    if follow_up_meta is not None:
                        try:
                            (
                                input_bytes,
                                effective_prompt,
                                options,
                                task_meta,
                            ) = await self._prepare_background_selfie(
                                event,
                                prompt,
                                target_backend,
                                follow_up_meta=follow_up_meta,
                            )
                        except Exception as exc:
                            logger.warning(
                                "[background-image] selfie follow-up preparation fell back: %s",
                                BackgroundImageTaskManager.sanitize_error(exc),
                            )
                        else:
                            resolved_mode = "selfie_ref"
                            prepared_selfie = True

                if not prepared_selfie:
                    has_message_images = await self._has_message_images(event)
                    image_segs = None
                    if resolved_mode == "auto" and not has_message_images:
                        image_segs = await get_images_from_event(
                            event,
                            include_avatar=True,
                            include_sender_avatar_fallback=False,
                        )
                    use_edit = resolved_mode in {"edit", "img2img", "aiedit"} or (
                        resolved_mode == "auto"
                        and (has_message_images or bool(image_segs))
                    )
                    if use_edit:
                        edit_conf = self._get_feature("edit")
                        if not bool(edit_conf.get("enabled", True)) or not bool(
                            edit_conf.get("llm_tool_enabled", True)
                        ):
                            raise RuntimeError(
                                "The requested image editing tool is disabled."
                            )
                        if image_segs is None:
                            image_segs = await get_images_from_event(
                                event,
                                include_avatar=True,
                                include_sender_avatar_fallback=False,
                            )
                        input_bytes = await self._image_segs_to_bytes(image_segs)
                        if not input_bytes:
                            raise RuntimeError(
                                "No usable input image was found in the current message."
                            )
                        resolved_mode = "edit"
                        effective_prompt = prompt
                        task_meta = self._build_image_task_meta(
                            mode="edit",
                            user_prompt=prompt,
                            effective_prompt=prompt,
                            continue_with="edit",
                            backend=target_backend,
                        )
                    else:
                        draw_conf = self._get_feature("draw")
                        if not bool(draw_conf.get("enabled", True)) or not bool(
                            draw_conf.get("llm_tool_enabled", True)
                        ):
                            raise RuntimeError(
                                "The requested image generation tool is disabled."
                            )
                        resolved_mode = "draw"
                        effective_prompt = prompt or "a selfie photo"
                        task_meta = self._build_image_task_meta(
                            mode="text",
                            user_prompt=effective_prompt,
                            effective_prompt=effective_prompt,
                            continue_with="text",
                            backend=target_backend,
                        )

            input_paths, manifest = await manager.spool_inputs(task_id, input_bytes)
            spooled = bool(input_paths)
            scope = manager.scope_hash(
                target.umo,
                target.self_id,
                target.sender_id,
                target.conversation_id,
            )
            normalized_args = {
                "prompt": prompt,
                "mode": resolved_mode,
                "backend": target_backend or "auto",
                "output": self._output_intent_dict(output_intent),
            }
            fingerprint = manager.request_fingerprint(
                scope,
                target.source_message_id,
                normalized_args,
            )
            record = {
                "task_id": task_id,
                "task_kind": "single",
                "state": "queued",
                "scope_hash": scope,
                "request_fingerprint": fingerprint,
                **BackgroundImageTaskManager.dataclass_dict(target),
                "mode": resolved_mode,
                "backend_requested": target_backend or "auto",
                "aspect_ratio": output_intent.aspect_ratio or "",
                "resolution": output_intent.resolution or "",
                "user_prompt": prompt,
                "effective_prompt": effective_prompt,
                "attempts": [
                    {
                        "attempt": 1,
                        "mode": resolved_mode,
                        "effective_prompt": effective_prompt,
                        "state": "queued",
                        "error_code": "",
                    }
                ],
                "current_attempt": 1,
                "input_manifest": manifest,
                "reference_source": options.get("reference_source", ""),
                "reference_count": options.get("reference_count", 0),
                "extra_reference_count": options.get("extra_reference_count", 0),
                "image_generated": False,
                "image_sent": False,
                "delivery_state": "not_started",
                "items": [],
                "reset_epoch": 0,
            }
            stored, created = await manager.create_task_record(record, reservation=1)
            if not created:
                await manager.cleanup_task_files(task_id)
                task_id = str(stored["task_id"])
            else:
                job = PreparedImageJob(
                    mode=resolved_mode,
                    user_prompt=prompt,
                    effective_prompt=effective_prompt,
                    backend=target_backend,
                    output=self._output_intent_dict(output_intent),
                    input_paths=input_paths,
                    task_meta=task_meta,
                    options={**options, "input_manifest": manifest},
                )
                manager.start_worker(
                    task_id,
                    lambda: self._run_background_single(
                        manager,
                        task_id,
                        job,
                        target,
                    ),
                )

            event.set_extra("_gitee_bg_ack_task_id", task_id)
            event.set_extra("_gitee_bg_ack_token", stored.get("ack_token", ""))
            response = {
                "status": "accepted",
                "task_id": task_id,
                "task_kind": "single",
                "state": str(stored.get("state") or "queued"),
                "mode": str(stored.get("mode") or resolved_mode),
                "user_prompt": str(stored.get("user_prompt") or prompt),
                "effective_prompt": str(
                    stored.get("effective_prompt") or effective_prompt
                ),
                "message": (
                    "The image task is running in the background. Respond naturally "
                    "as yourself, tell the user it is underway, and say they can "
                    "continue chatting. Do not imply that the image is finished."
                ),
            }
            return self._llm_tool_text_result(json.dumps(response, ensure_ascii=False))
        except Exception:
            if spooled:
                await manager.cleanup_task_files(task_id)
            raise

    async def _execute_prepared_image_job(
        self,
        manager: BackgroundImageTaskManager,
        job: PreparedImageJob,
    ) -> tuple[Path, dict[str, Any]]:
        """Execute one prepared provider call without accessing an old event.

        Args:
            manager: Owning task manager.
            job: Immutable provider input.

        Returns:
            Generated image path and follow-up metadata.
        """

        intent = OutputIntent(**job.output)
        images = await manager.read_spooled_inputs(
            job.input_paths,
            job.options.get("input_manifest"),
        )
        if job.mode == "draw":
            image_path = await self.draw.generate(
                job.effective_prompt,
                provider_id=job.backend,
                output_intent=intent,
            )
            return image_path, dict(job.task_meta)
        if job.mode == "edit":
            image_path = await self.edit.edit(
                prompt=job.effective_prompt,
                images=images,
                backend=job.backend,
                output_intent=intent,
            )
            return image_path, dict(job.task_meta)
        if job.mode == "selfie_ref":
            image_path = await self.edit.edit(
                prompt=job.effective_prompt,
                images=images,
                backend=job.backend,
                task_types=job.options.get("task_types"),
                output_intent=intent,
                default_output=job.options.get("default_output"),
                chain_override=job.options.get("chain_override"),
                infer_source_aspect=False,
            )
            return image_path, dict(job.task_meta)
        raise RuntimeError(f"Unsupported prepared image mode: {job.mode}")

    async def _rebuild_background_event(
        self,
        target: TaskDeliveryTarget,
        *,
        message: list[Any] | None = None,
        message_str: str = "",
    ) -> AstrMessageEvent:
        """Build a fresh event from a persistable delivery target.

        Args:
            target: Persisted platform route.
            message: Synthetic input chain.
            message_str: Synthetic plain message string.

        Returns:
            Fresh adapter event using the currently active adapter instance.
        """

        adapter = self.context.get_platform_inst(target.platform_id)
        if adapter is None:
            raise RuntimeError(f"Platform is no longer available: {target.platform_id}")
        abm = await StarTools.create_message(
            type=target.message_type,
            self_id=target.self_id,
            session_id=target.session_id,
            sender=MessageMember(
                user_id=target.sender_id,
                nickname=target.sender_name or None,
            ),
            message=list(message or []),
            message_str=message_str,
            message_id=f"gitee-bg-{BackgroundImageTaskManager.new_task_id('event')}",
            raw_message=None,
            group_id=target.group_id,
        )
        rebuilt = adapter.create_event(abm)
        rebuilt.is_wake = True
        rebuilt.is_at_or_wake_command = True
        return rebuilt

    @staticmethod
    def _is_unknown_delivery_error(exc: BaseException) -> bool:
        if isinstance(exc, (TimeoutError, ConnectionError, asyncio.TimeoutError)):
            return True
        text = str(exc or "").lower()
        return any(
            marker in text
            for marker in (
                "timeout",
                "timed out",
                "connection reset",
                "broken pipe",
                "server disconnected",
                "connection closed",
            )
        )

    async def _send_background_image_once(
        self,
        target: TaskDeliveryTarget,
        image_path: Path,
    ) -> AstrMessageEvent:
        """Send an image-only chain exactly once through the current adapter.

        Args:
            target: Persisted delivery route.
            image_path: Generated image path.

        Returns:
            Fresh event whose send await returned successfully.
        """

        event = await self._rebuild_background_event(target)
        original_path = Path(image_path)
        send_path = await self._prepare_image_for_send(event, original_path)
        cleanup = self._is_weixin_send_temp_path(send_path) and (
            send_path.resolve(strict=False) != original_path.resolve(strict=False)
        )
        try:
            timeout = self._as_int(
                self._get_send_conf().get("background_send_timeout_seconds", 120),
                default=120,
            )
            try:
                send_as_file = (
                    send_path.stat().st_size > self.IMAGE_AS_FILE_THRESHOLD_BYTES
                )
            except OSError:
                send_as_file = False
            component = (
                File(name=send_path.name, file=str(send_path))
                if send_as_file
                else Image.fromFileSystem(str(send_path))
            )
            await asyncio.wait_for(
                event.send(event.chain_result([component])),
                timeout=float(max(15, min(300, timeout))),
            )
            return event
        finally:
            if cleanup:
                await asyncio.to_thread(
                    self._remove_weixin_send_temp_image_sync,
                    send_path,
                )

    async def _wait_for_background_ack(
        self,
        manager: BackgroundImageTaskManager,
        task_id: str,
    ) -> None:
        for _ in range(40):
            record = await manager.get_task(task_id)
            if record is None or record.get("ack_state") != "pending":
                return
            await asyncio.sleep(0.25)

    async def _run_background_single(
        self,
        manager: BackgroundImageTaskManager,
        task_id: str,
        job: PreparedImageJob,
        target: TaskDeliveryTarget,
    ) -> None:
        """Run one durable image task to generation, delivery, and notification.

        Args:
            manager: Owning task manager.
            task_id: Durable task ID.
            job: Event-independent provider input.
            target: Persisted platform route.
        """

        try:

            async def provider_call() -> tuple[Path, dict[str, Any]]:
                await manager.transition(
                    task_id,
                    "running",
                    {
                        "attempts": [
                            {
                                "attempt": 1,
                                "mode": job.mode,
                                "effective_prompt": job.effective_prompt,
                                "state": "running",
                                "error_code": "",
                            }
                        ]
                    },
                )
                return await self._execute_prepared_image_job(manager, job)

            image_path, task_meta = await asyncio.wait_for(
                manager.run_provider(task_id, provider_call),
                timeout=2 * 60 * 60,
            )
            if manager.is_cancelled(task_id):
                raise asyncio.CancelledError
            await self._wait_for_background_ack(manager, task_id)
            await self._wait_background_send_gate(target.umo)
            if manager.is_cancelled(task_id):
                raise asyncio.CancelledError
            send_attempt_id = manager.new_task_id("send")
            await manager.transition(
                task_id,
                "sending",
                {
                    "image_generated": True,
                    "delivery_state": "attempting",
                    "send_attempt_id": send_attempt_id,
                },
            )
            await manager.record_receipt(
                task_id,
                send_attempt_id=send_attempt_id,
                kind="image",
                delivery_state="attempting",
                transport=target.platform_name,
            )
            try:
                delivery_event = await self._send_background_image_once(
                    target,
                    image_path,
                )
            except Exception as exc:
                unknown = self._is_unknown_delivery_error(exc)
                delivery_state = "unknown" if unknown else "not_started"
                await manager.record_receipt(
                    task_id,
                    send_attempt_id=send_attempt_id,
                    kind="image",
                    delivery_state="unknown" if unknown else "not_started",
                    transport=target.platform_name,
                    response_digest=hashlib.sha256(str(exc).encode()).hexdigest(),
                )
                terminal_state = "interrupted" if unknown else "failed"
                record = await manager.transition(
                    task_id,
                    terminal_state,
                    {
                        "image_generated": True,
                        "image_sent": False,
                        "delivery_state": delivery_state,
                        "error_code": "delivery_unknown"
                        if unknown
                        else "delivery_failed",
                        "error_message": manager.sanitize_error(exc),
                        "terminal_reason": "image_delivery",
                    },
                    queue_notification=True,
                )
                await self._dispatch_background_completion(manager, record, target)
                return

            await manager.record_receipt(
                task_id,
                send_attempt_id=send_attempt_id,
                kind="image",
                delivery_state="confirmed",
                transport=target.platform_name,
                response_digest=hashlib.sha256(str(image_path).encode()).hexdigest(),
            )
            self._remember_last_image(delivery_event, image_path)
            await self._save_last_image_task_meta(delivery_event, task_meta)
            record = await manager.transition(
                task_id,
                "completed",
                {
                    "image_generated": True,
                    "image_sent": True,
                    "delivery_state": "confirmed",
                    "task_meta": task_meta,
                    "terminal_reason": "completed",
                },
                queue_notification=True,
            )
            await self._dispatch_background_completion(manager, record, target)
        except asyncio.CancelledError:
            record = await manager.get_task(task_id)
            if record and record.get("state") not in TERMINAL_STATES:
                try:
                    record = await asyncio.shield(
                        manager.transition(
                            task_id,
                            "interrupted",
                            {
                                "error_code": "plugin_shutdown",
                                "error_message": "The image task was interrupted before completion.",
                                "terminal_reason": "plugin_shutdown",
                            },
                            queue_notification=True,
                        )
                    )
                except Exception:
                    pass
                else:
                    await self._dispatch_background_completion(manager, record, target)
            raise
        except Exception as exc:
            logger.error(
                "[background-image] single task failed: task=%s err=%s",
                task_id,
                manager.sanitize_error(exc),
                exc_info=True,
            )
            record = await manager.get_task(task_id)
            if record and record.get("state") not in TERMINAL_STATES:
                record = await manager.transition(
                    task_id,
                    "failed",
                    {
                        "error_code": "provider_failed",
                        "error_message": manager.sanitize_error(exc),
                        "terminal_reason": "provider_failed",
                    },
                    queue_notification=True,
                )
                await self._dispatch_background_completion(manager, record, target)
        finally:
            await manager.cleanup_task_files(task_id)

    @staticmethod
    def _delivery_target_from_record(record: dict[str, Any]) -> TaskDeliveryTarget:
        return TaskDeliveryTarget(
            platform_id=str(record.get("platform_id") or ""),
            platform_name=str(record.get("platform_name") or ""),
            message_type=str(record.get("message_type") or "FriendMessage"),
            umo=str(record.get("umo") or ""),
            session_id=str(record.get("session_id") or ""),
            group_id=str(record.get("group_id") or ""),
            self_id=str(record.get("self_id") or ""),
            sender_id=str(record.get("sender_id") or ""),
            sender_name=str(record.get("sender_name") or ""),
            source_message_id=str(record.get("source_message_id") or ""),
            conversation_id=str(record.get("conversation_id") or ""),
        )

    async def _background_context_is_safe(
        self,
        target: TaskDeliveryTarget,
    ) -> tuple[bool, Any | None]:
        """Check conversation and ContextAware state before synthetic re-entry.

        Args:
            target: Original task route and conversation binding.

        Returns:
            ``(safe, conversation)`` for normal Agent completion.
        """

        conversation_manager = getattr(self.context, "conversation_manager", None)
        if conversation_manager is None or not target.conversation_id:
            return False, None
        try:
            current_id = await conversation_manager.get_curr_conversation_id(target.umo)
            if str(current_id or "") != target.conversation_id:
                return False, None
            conversation = await conversation_manager.get_conversation(
                target.umo,
                target.conversation_id,
            )
        except Exception as exc:
            logger.warning(
                "[background-image] conversation gate failed: %s",
                BackgroundImageTaskManager.sanitize_error(exc),
            )
            return False, None
        if conversation is None:
            return False, None

        if target.group_id:
            try:
                metadata = self.context.get_registered_star(
                    "astrbot_plugin_context_aware"
                )
                context_aware = getattr(metadata, "star_cls", None)
                if context_aware is not None:
                    has_session = getattr(context_aware, "has_session", None)
                    if not callable(has_session) or not bool(has_session(target.umo)):
                        return False, None
            except Exception as exc:
                logger.warning(
                    "[background-image] ContextAware gate failed: %s",
                    BackgroundImageTaskManager.sanitize_error(exc),
                )
                return False, None
        return True, conversation

    async def _cancel_background_scope_with_notifications(
        self,
        manager: BackgroundImageTaskManager,
        *,
        umo: str,
        sender_id: str,
        reason: str,
        suppress_future_injection: bool = False,
    ) -> int:
        """Cancel scoped tasks and dispatch each cancellation outbox.

        Args:
            manager: Owning task manager.
            umo: Unified message origin.
            sender_id: Requester whose active tasks should stop.
            reason: Durable cancellation reason.
            suppress_future_injection: Hide records after reset/new.

        Returns:
            Number of tasks cancelled.
        """

        records = [
            record
            for record in await manager.list_active_for_umo(umo)
            if str(record.get("sender_id") or "") == str(sender_id or "")
        ]
        cancelled = 0
        for record in records:
            task_id = str(record.get("task_id") or "")
            if not task_id:
                continue
            if not await manager.cancel_task(
                task_id,
                reason,
                suppress_future_injection=suppress_future_injection,
            ):
                continue
            cancelled += 1
            terminal = await manager.get_task(task_id)
            if terminal is not None:
                await self._dispatch_background_completion(
                    manager,
                    terminal,
                    self._delivery_target_from_record(terminal),
                )
        return cancelled

    @staticmethod
    def _background_notification_text(record: dict[str, Any]) -> str:
        state = str(record.get("state") or "failed")
        task_kind = str(record.get("task_kind") or "single")
        if task_kind == "batch":
            requested = int(record.get("requested_count") or 0)
            sent = int(record.get("sent_count") or 0)
            failed = int(record.get("failed_count") or 0)
            cancelled = int(record.get("cancelled_count") or 0)
            unknown = int(record.get("unknown_count") or 0)
            if state == "completed":
                return f"这组照片拍完啦，计划的 {requested} 张都已经发出来了。"
            if state == "partial":
                return (
                    f"这组照片处理完了，计划 {requested} 张，已经发出 {sent} 张，"
                    f"失败 {failed} 张，取消 {cancelled} 张。"
                )
            if unknown:
                return (
                    f"这组照片在发送时遇到连接中断，其中 {unknown} 张的送达状态暂时无法确认，"
                    "我没有自动重发，免得重复刷出来。"
                )
            if state == "cancelled":
                return f"这组照片已经停下来了，已发出 {sent} 张，剩余任务没有继续。"
            return f"这组照片没能完成，计划 {requested} 张，实际发出 {sent} 张。"

        if state == "completed":
            return "照片拍好啦，我已经发出来了。"
        if state == "cancelled":
            return "刚才那张照片已经停下来了，我没有再继续生成。"
        if str(record.get("delivery_state") or "") == "unknown":
            return (
                "照片生成好了，但发送时连接断了一下，我现在无法确认你那边是否收到；"
                "我先不自动重发，避免重复。"
            )
        if state == "interrupted":
            return "刚才那张照片因为服务重启中断了，没有继续扣费或重复发送。"
        return "刚才那张照片没能生成成功，这次任务已经结束了。"

    async def _send_deterministic_background_notification(
        self,
        manager: BackgroundImageTaskManager,
        record: dict[str, Any],
        target: TaskDeliveryTarget,
        *,
        attempt_id: str,
    ) -> None:
        text = self._background_notification_text(record)
        token = str(record.get("notification_token") or "")
        try:
            event = await self._rebuild_background_event(target)
            timeout = self._as_int(
                self._get_send_conf().get("background_send_timeout_seconds", 120),
                default=120,
            )
            await asyncio.wait_for(
                event.send(event.chain_result([Plain(text)])),
                timeout=float(max(15, min(300, timeout))),
            )
        except Exception as exc:
            state = "unknown" if self._is_unknown_delivery_error(exc) else "failed"
            await manager.mark_notification(
                token,
                state,
                attempt_id=attempt_id,
            )
            logger.warning(
                "[background-image] deterministic notification failed: task=%s err=%s",
                record.get("task_id"),
                manager.sanitize_error(exc),
            )
            return
        if not bool(getattr(event, "_has_send_oper", False)):
            await manager.mark_notification(
                token,
                "unknown",
                attempt_id=attempt_id,
            )
            logger.warning(
                "[background-image] deterministic notification returned without transport confirmation: task=%s",
                record.get("task_id"),
            )
            return
        await manager.mark_notification(token, "sent", attempt_id=attempt_id)

    async def _background_notification_watchdog(
        self,
        manager: BackgroundImageTaskManager,
        task_id: str,
        token: str,
        target: TaskDeliveryTarget,
    ) -> None:
        await asyncio.sleep(90)
        attempt_id = manager.new_task_id("notify-watchdog")
        record = await manager.claim_notification(
            token,
            attempt_id,
            from_states=("pending", "queued"),
        )
        if record is None:
            return
        await self._send_deterministic_background_notification(
            manager,
            record,
            target,
            attempt_id=attempt_id,
        )

    async def _dispatch_background_completion(
        self,
        manager: BackgroundImageTaskManager,
        record: dict[str, Any],
        target: TaskDeliveryTarget,
    ) -> None:
        """Route a terminal task to Agent completion or deterministic fallback.

        Args:
            manager: Owning task manager.
            record: Terminal durable record.
            target: Original platform route.
        """

        if manager.is_closing:
            return
        token = str(record.get("notification_token") or "")
        if not token:
            return
        safe, conversation = await self._background_context_is_safe(target)
        attempt_id = manager.new_task_id("notify-agent" if safe else "notify-direct")
        claimed = await manager.claim_notification(token, attempt_id)
        if claimed is None:
            return
        if not safe:
            await self._send_deterministic_background_notification(
                manager,
                claimed,
                target,
                attempt_id=attempt_id,
            )
            return

        message_id = target.source_message_id or manager.new_task_id("source")
        synthetic_chain: list[Any] = []
        if target.group_id:
            synthetic_chain.append(At(qq=target.self_id))
        synthetic_chain.append(
            Reply(
                id=message_id,
                sender_id=target.sender_id,
                sender_nickname=target.sender_name,
                message_str="",
            )
        )
        try:
            event = await self._rebuild_background_event(
                target,
                message=synthetic_chain,
            )
            event.set_extra("_gitee_bg_completion", True)
            event.set_extra("_gitee_bg_task_id", str(record.get("task_id") or ""))
            event.set_extra("_gitee_bg_notification_token", token)
            event.set_extra("_gitee_bg_notification_attempt", attempt_id)
            request = event.request_llm(
                prompt=(
                    "An image task has reached a terminal state. Use the temporary "
                    "background task facts injected by the Gitee image plugin. Respond "
                    "once, naturally in your existing persona, to the original requester. "
                    "Do not call image tools and do not claim delivery beyond the recorded facts."
                ),
                conversation=conversation,
            )
            event.set_extra("provider_request", request)
            adapter = self.context.get_platform_inst(target.platform_id)
            if adapter is None:
                raise RuntimeError(
                    "Platform adapter disappeared before completion enqueue"
                )
            await manager.mark_notification(
                token,
                "queued",
                attempt_id=attempt_id,
            )
            adapter.commit_event(event)
        except Exception as exc:
            logger.warning(
                "[background-image] synthetic completion enqueue failed: task=%s err=%s",
                record.get("task_id"),
                manager.sanitize_error(exc),
            )
            await self._send_deterministic_background_notification(
                manager,
                claimed,
                target,
                attempt_id=attempt_id,
            )
            return
        manager.start_managed(
            self._background_notification_watchdog(
                manager,
                str(record.get("task_id") or ""),
                token,
                target,
            ),
            name=f"background-notification-watchdog-{record.get('task_id')}",
        )

    async def _accept_background_batch(
        self,
        event: AstrMessageEvent,
        *,
        prompt: str,
        count: int,
        mode: str,
        backend: str,
        output: str,
        aspect_ratio: str,
        resolution: str,
    ) -> mcp.types.CallToolResult:
        """Accept an LLM batch before planner or image provider execution.

        Args:
            event: Current LLM tool event.
            prompt: Shared user request.
            count: Requested child count.
            mode: Requested routing mode.
            backend: Requested provider override.
            output: Legacy output specification.
            aspect_ratio: Shared aspect ratio override.
            resolution: Shared resolution override.

        Returns:
            Accepted batch task result.
        """

        manager = self._background_manager_for_event(event)
        if manager is None:
            raise BackgroundTaskError("Background mode is unavailable for this event")
        task_id = manager.new_task_id("batch")
        spooled = False
        try:
            target = await self._build_background_delivery_target(event)
            requested_count = max(
                1,
                min(self._get_batch_max_count(), self._as_int(count, default=4)),
            )
            resolved_mode = await self._resolve_llm_batch_mode(event, mode, prompt)
            target_backend = self._resolve_target_backend(backend)
            output_intent = resolve_llm_output_intent(
                prompt,
                output=output,
                aspect_ratio=aspect_ratio,
                resolution=resolution,
            )
            input_bytes: list[bytes] = []
            options: dict[str, Any] = {}
            if resolved_mode == "draw":
                draw_conf = self._get_feature("draw")
                if not bool(draw_conf.get("enabled", True)) or not bool(
                    draw_conf.get("llm_tool_enabled", True)
                ):
                    raise RuntimeError(
                        "The requested batch text-to-image tool is disabled."
                    )
            elif resolved_mode == "edit":
                edit_conf = self._get_feature("edit")
                if not bool(edit_conf.get("enabled", True)) or not bool(
                    edit_conf.get("llm_tool_enabled", True)
                ):
                    raise RuntimeError(
                        "The requested batch image editing tool is disabled."
                    )
                input_bytes = await self._prepare_edit_image_bytes(event)
            elif resolved_mode == "selfie_ref":
                if not self._is_selfie_enabled() or not self._is_selfie_llm_enabled():
                    raise RuntimeError("The requested batch selfie tool is disabled.")
                input_bytes, _, options, _ = await self._prepare_background_selfie(
                    event,
                    prompt,
                    target_backend,
                )

            input_paths, manifest = await manager.spool_inputs(task_id, input_bytes)
            spooled = bool(input_paths)
            scope = manager.scope_hash(
                target.umo,
                target.self_id,
                target.sender_id,
                target.conversation_id,
            )
            normalized_args = {
                "prompt": prompt,
                "count": requested_count,
                "mode": resolved_mode,
                "backend": target_backend or "auto",
                "output": self._output_intent_dict(output_intent),
            }
            fingerprint = manager.request_fingerprint(
                scope,
                target.source_message_id,
                normalized_args,
            )
            record = {
                "task_id": task_id,
                "task_kind": "batch",
                "state": "planning",
                "scope_hash": scope,
                "request_fingerprint": fingerprint,
                **BackgroundImageTaskManager.dataclass_dict(target),
                "mode": resolved_mode,
                "backend_requested": target_backend or "auto",
                "aspect_ratio": output_intent.aspect_ratio or "",
                "resolution": output_intent.resolution or "",
                "user_prompt": prompt,
                "effective_prompt": "",
                "requested_count": requested_count,
                "planned_count": 0,
                "generated_count": 0,
                "sent_count": 0,
                "failed_count": 0,
                "cancelled_count": 0,
                "unknown_count": 0,
                "input_manifest": manifest,
                "reference_source": options.get("reference_source", ""),
                "reference_count": options.get("reference_count", 0),
                "extra_reference_count": options.get("extra_reference_count", 0),
                "image_generated": False,
                "image_sent": False,
                "delivery_state": "not_started",
                "items": [],
                "reset_epoch": 0,
            }
            stored, created = await manager.create_task_record(
                record,
                reservation=requested_count,
            )
            if not created:
                await manager.cleanup_task_files(task_id)
                task_id = str(stored["task_id"])
            else:
                job = PreparedBatchJob(
                    mode=resolved_mode,
                    user_prompt=prompt,
                    requested_count=requested_count,
                    backend=target_backend,
                    output=self._output_intent_dict(output_intent),
                    input_paths=input_paths,
                    options={
                        **options,
                        "input_manifest": manifest,
                        "fixed_aspect_ratio": output_intent.aspect_ratio
                        or aspect_ratio_from_size(output_intent.exact_size),
                    },
                )
                manager.start_worker(
                    task_id,
                    lambda: self._run_background_batch(
                        manager,
                        task_id,
                        job,
                        target,
                    ),
                )

            event.set_extra("_gitee_bg_ack_task_id", task_id)
            event.set_extra("_gitee_bg_ack_token", stored.get("ack_token", ""))
            response = {
                "status": "accepted",
                "task_id": task_id,
                "task_kind": "batch",
                "state": str(stored.get("state") or "planning"),
                "requested_count": int(
                    stored.get("requested_count") or requested_count
                ),
                "mode": str(stored.get("mode") or resolved_mode),
                "user_prompt": str(stored.get("user_prompt") or prompt),
                "message": (
                    "The batch image task was accepted before planning and is now "
                    "running in the background. Respond naturally with the accepted "
                    "count and tell the user they can continue chatting. Do not invent "
                    "child prompts until the planner has stored them."
                ),
            }
            return self._llm_tool_text_result(json.dumps(response, ensure_ascii=False))
        except Exception:
            if spooled:
                await manager.cleanup_task_files(task_id)
            raise

    async def _run_background_batch_child(
        self,
        manager: BackgroundImageTaskManager,
        task_id: str,
        item: dict[str, Any],
        job: PreparedImageJob,
        child_limit: asyncio.Semaphore,
        generated: dict[str, tuple[Path, dict[str, Any]]],
    ) -> None:
        """Generate one batch child through the shared fair scheduler.

        Args:
            manager: Owning task manager.
            task_id: Batch parent ID.
            item: Mutable child descriptor from the durable record.
            job: Prepared child provider input.
            child_limit: Per-parent concurrency limit.
            generated: In-process successful image results by item ID.
        """

        item_id = str(item["item_id"])
        try:
            async with child_limit:

                async def provider_call() -> tuple[Path, dict[str, Any]]:
                    await manager.update_item(
                        task_id,
                        item_id,
                        {"state": "running", "started_at": manager.now_ms()},
                    )
                    return await self._execute_prepared_image_job(manager, job)

                result = await asyncio.wait_for(
                    manager.run_provider(task_id, provider_call),
                    timeout=2 * 60 * 60,
                )
            generated[item_id] = result
            await manager.update_item(
                task_id,
                item_id,
                {
                    "state": "generated",
                    "image_generated": True,
                    "delivery_state": "not_started",
                },
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await manager.update_item(
                task_id,
                item_id,
                {
                    "state": "failed",
                    "error_code": "provider_failed",
                    "error_message": manager.sanitize_error(exc),
                },
                release_if_terminal=True,
            )

    async def _run_background_batch(
        self,
        manager: BackgroundImageTaskManager,
        task_id: str,
        job: PreparedBatchJob,
        target: TaskDeliveryTarget,
    ) -> None:
        """Plan, fairly generate, and sequentially deliver one image batch.

        Args:
            manager: Owning task manager.
            task_id: Durable batch parent ID.
            job: Event-independent batch input.
            target: Persisted platform route.
        """

        child_tasks: list[asyncio.Task[Any]] = []
        try:
            planned_items = await asyncio.wait_for(
                manager.run_planner(
                    lambda: self._plan_batch_prompt_items(
                        mode=job.mode,
                        user_prompt=job.user_prompt,
                        count=job.requested_count,
                        fixed_aspect_ratio=job.options.get("fixed_aspect_ratio"),
                        umo=target.umo,
                    )
                ),
                timeout=300,
            )
            if len(planned_items) != job.requested_count:
                raise RuntimeError(
                    "Batch planner returned "
                    f"{len(planned_items)} items for {job.requested_count} reserved slots"
                )
            common_intent = OutputIntent(**job.output)
            items: list[dict[str, Any]] = []
            child_jobs: list[PreparedImageJob] = []
            for index, planned in enumerate(planned_items, start=1):
                item_id = f"{task_id}_{index:02d}"
                effective_prompt = planned.prompt
                if job.mode == "selfie_ref":
                    effective_prompt = self._build_selfie_prompt(
                        planned.prompt,
                        extra_refs=int(job.options.get("extra_reference_count") or 0),
                    )
                item_intent = merge_output_intents(
                    common_intent,
                    OutputIntent(aspect_ratio=planned.aspect_ratio),
                )
                item = {
                    "item_id": item_id,
                    "index": index,
                    "state": "queued",
                    "mode": job.mode,
                    "title": planned.title,
                    "variation_focus": planned.variation_focus,
                    "user_prompt": planned.prompt,
                    "effective_prompt": effective_prompt,
                    "aspect_ratio": planned.aspect_ratio,
                    "image_generated": False,
                    "image_sent": False,
                    "delivery_state": "not_started",
                    "error_code": "",
                    "error_message": "",
                    "send_attempt_id": "",
                }
                if job.mode == "selfie_ref":
                    task_meta = self._build_image_task_meta(
                        mode="selfie_ref",
                        user_prompt=planned.prompt,
                        effective_prompt=effective_prompt,
                        reference_source=str(job.options.get("reference_source") or ""),
                        reference_count=int(job.options.get("reference_count") or 0),
                        extra_reference_count=int(
                            job.options.get("extra_reference_count") or 0
                        ),
                        continue_with="selfie_ref",
                        backend=job.backend,
                    )
                else:
                    task_meta = self._build_image_task_meta(
                        mode="text" if job.mode == "draw" else "edit",
                        user_prompt=planned.prompt,
                        effective_prompt=effective_prompt,
                        continue_with="text" if job.mode == "draw" else "edit",
                        backend=job.backend,
                    )
                items.append(item)
                child_jobs.append(
                    PreparedImageJob(
                        mode=job.mode,
                        user_prompt=planned.prompt,
                        effective_prompt=effective_prompt,
                        backend=job.backend,
                        output=self._output_intent_dict(item_intent),
                        input_paths=job.input_paths,
                        task_meta=task_meta,
                        options=job.options,
                    )
                )

            await manager.transition(
                task_id,
                "queued",
                {
                    "items": items,
                    "planned_count": len(items),
                    "effective_prompt": "",
                },
            )
            generated: dict[str, tuple[Path, dict[str, Any]]] = {}
            child_limit = asyncio.Semaphore(
                self._get_batch_concurrency_for_mode(job.mode)
            )
            for item, child_job in zip(items, child_jobs, strict=True):
                child_tasks.append(
                    manager.start_managed(
                        self._run_background_batch_child(
                            manager,
                            task_id,
                            item,
                            child_job,
                            child_limit,
                            generated,
                        ),
                        name=f"background-batch-child-{item['item_id']}",
                    )
                )
            if child_tasks:
                await asyncio.gather(*child_tasks)

            await self._wait_for_background_ack(manager, task_id)
            for item in items:
                item_id = str(item["item_id"])
                if manager.is_cancelled(task_id):
                    raise asyncio.CancelledError
                generated_result = generated.get(item_id)
                if generated_result is None:
                    continue
                image_path, task_meta = generated_result
                await self._wait_background_send_gate(target.umo)
                if manager.is_cancelled(task_id):
                    raise asyncio.CancelledError
                send_attempt_id = manager.new_task_id("send")
                await manager.update_item(
                    task_id,
                    item_id,
                    {
                        "state": "sending",
                        "send_attempt_id": send_attempt_id,
                        "delivery_state": "attempting",
                    },
                )
                await manager.record_receipt(
                    task_id,
                    send_attempt_id=send_attempt_id,
                    item_id=item_id,
                    kind="image",
                    delivery_state="attempting",
                    transport=target.platform_name,
                )
                try:
                    delivery_event = await self._send_background_image_once(
                        target,
                        image_path,
                    )
                except Exception as exc:
                    unknown = self._is_unknown_delivery_error(exc)
                    await manager.record_receipt(
                        task_id,
                        send_attempt_id=send_attempt_id,
                        item_id=item_id,
                        kind="image",
                        delivery_state="unknown" if unknown else "not_started",
                        transport=target.platform_name,
                        response_digest=hashlib.sha256(str(exc).encode()).hexdigest(),
                    )
                    await manager.update_item(
                        task_id,
                        item_id,
                        {
                            "state": "unknown" if unknown else "failed",
                            "image_generated": True,
                            "image_sent": False,
                            "delivery_state": "unknown" if unknown else "not_started",
                            "error_code": "delivery_unknown"
                            if unknown
                            else "delivery_failed",
                            "error_message": manager.sanitize_error(exc),
                        },
                        release_if_terminal=True,
                    )
                    continue
                await manager.record_receipt(
                    task_id,
                    send_attempt_id=send_attempt_id,
                    item_id=item_id,
                    kind="image",
                    delivery_state="confirmed",
                    transport=target.platform_name,
                    response_digest=hashlib.sha256(
                        str(image_path).encode()
                    ).hexdigest(),
                )
                self._remember_last_image(delivery_event, image_path)
                await self._save_last_image_task_meta(delivery_event, task_meta)
                await manager.update_item(
                    task_id,
                    item_id,
                    {
                        "state": "completed",
                        "image_generated": True,
                        "image_sent": True,
                        "delivery_state": "confirmed",
                    },
                    release_if_terminal=True,
                )

            current = await manager.get_task(task_id)
            if current is None:
                return
            unknown_count = int(current.get("unknown_count") or 0)
            sent_count = int(current.get("sent_count") or 0)
            requested_count = int(current.get("requested_count") or 0)
            if unknown_count > 0:
                terminal_state = "interrupted"
            elif sent_count == requested_count and requested_count > 0:
                terminal_state = "completed"
            elif sent_count > 0:
                terminal_state = "partial"
            else:
                terminal_state = "failed"
            record = await manager.transition(
                task_id,
                terminal_state,
                {
                    "delivery_state": "unknown"
                    if unknown_count
                    else ("confirmed" if sent_count else "not_started"),
                    "terminal_reason": "batch_finished",
                },
                queue_notification=True,
            )
            await self._dispatch_background_completion(manager, record, target)
        except asyncio.CancelledError:
            for child in child_tasks:
                child.cancel()
            if child_tasks:
                await asyncio.gather(*child_tasks, return_exceptions=True)
            record = await manager.get_task(task_id)
            if record and record.get("state") not in TERMINAL_STATES:
                try:
                    record = await asyncio.shield(
                        manager.transition(
                            task_id,
                            "interrupted",
                            {
                                "error_code": "plugin_shutdown",
                                "error_message": "The batch image task was interrupted.",
                                "terminal_reason": "plugin_shutdown",
                            },
                            queue_notification=True,
                        )
                    )
                except Exception:
                    pass
                else:
                    await self._dispatch_background_completion(manager, record, target)
            raise
        except Exception as exc:
            logger.error(
                "[background-image] batch task failed: task=%s err=%s",
                task_id,
                manager.sanitize_error(exc),
                exc_info=True,
            )
            record = await manager.get_task(task_id)
            if record and record.get("state") not in TERMINAL_STATES:
                record = await manager.transition(
                    task_id,
                    "failed",
                    {
                        "error_code": "batch_failed",
                        "error_message": manager.sanitize_error(exc),
                        "terminal_reason": "batch_failed",
                    },
                    queue_notification=True,
                )
                await self._dispatch_background_completion(manager, record, target)
        finally:
            await manager.cleanup_task_files(task_id)

    def _get_batch_feature(self) -> dict:
        return self._get_feature("batch")

    def _get_batch_max_count(self) -> int:
        value = self._as_int(self._get_batch_feature().get("max_count", 8), default=8)
        return max(1, min(32, value))

    def _get_draw_batch_concurrency(self) -> int:
        value = self._as_int(
            self._get_feature("draw").get("batch_concurrency", 2), default=2
        )
        return max(1, min(8, value))

    def _get_edit_batch_concurrency(self) -> int:
        value = self._as_int(
            self._get_feature("edit").get("batch_concurrency", 2), default=2
        )
        return max(1, min(8, value))

    def _get_draw_presets(self) -> dict[str, str]:
        presets: dict[str, str] = {}
        conf = self._get_feature("draw")
        items = conf.get("presets", [])
        if not isinstance(items, list):
            return presets
        for item in items:
            if isinstance(item, str) and ":" in item:
                key, val = item.split(":", 1)
                key = key.strip()
                val = val.strip()
                if key and val:
                    presets[key] = val
        return presets

    def _parse_structured_image_request(self, text: str) -> ParsedImageRequest | None:
        edit_presets = dict(getattr(self.edit, "presets", {}) or {})
        return parse_image_request(
            text,
            draw_presets=self._get_draw_presets(),
            edit_presets=edit_presets,
            known_provider_ids=set(self.registry.provider_ids()),
        )

    def _extract_batch_command_fragment(self, event: AstrMessageEvent) -> str:
        """Extract a batch command from the start of a raw text segment.

        Args:
            event: Current AstrBot message event.

        Returns:
            The complete batch command fragment, or an empty string when absent.
        """
        try:
            chain = event.get_messages()
        except Exception:
            return ""

        for seg in chain or []:
            if not isinstance(seg, Plain):
                continue
            plain = str(getattr(seg, "text", "") or "").lstrip()
            for prefix in self._cmd_prefixes():
                if re.match(
                    rf"{re.escape(prefix)}批量(?:\s*\d+|\d+)(?:\s|$)",
                    plain,
                ):
                    return plain
        return ""

    def _batch_mode_label(self, spec: ImageTaskSpec) -> str:
        if spec.mode == "draw":
            if spec.preset_name:
                return f"文生图预设/{spec.preset_name}"
            return "文生图"
        if spec.mode == "edit":
            if spec.preset_name:
                return f"改图预设/{spec.preset_name}"
            return "改图"
        if spec.mode == "selfie_ref":
            return "自拍"
        return spec.mode

    def _get_batch_concurrency_for_mode(self, mode: str) -> int:
        if mode == "draw":
            return self._get_draw_batch_concurrency()
        return self._get_edit_batch_concurrency()

    def _resolve_target_backend(self, backend: str | None) -> str | None:
        raw = str(backend or "auto").strip()
        known_provider_ids = set(self.registry.provider_ids())
        if not raw or raw.lower() == "auto":
            return None
        if raw in known_provider_ids:
            return raw
        logger.warning(
            "[backend_override] 忽略未知 backend 覆盖，回退自动链路: backend=%s",
            raw,
        )
        return None

    def _get_draw_ratio_default_sizes(self) -> dict[str, str]:
        conf = self._get_feature("draw")
        raw = conf.get("ratio_default_sizes", {})
        if not isinstance(raw, dict):
            return {}
        out: dict[str, str] = {}
        for ratio, size in raw.items():
            r = str(ratio or "").strip()
            s = normalize_size_text(size)
            if not r or not s:
                continue
            out[r] = s
        return out

    def _resolve_ratio_size(self, ratio: str) -> str:
        ratio = str(ratio or "").strip()
        overrides = self._get_draw_ratio_default_sizes()
        size, warning = resolve_ratio_size(
            ratio,
            overrides=overrides,
            supported_ratios=self.SUPPORTED_RATIOS,
        )
        if warning:
            logger.warning("[aiimg] %s", warning)
        return size

    def _get_video_presets(self) -> dict[str, str]:
        presets: dict[str, str] = {}
        conf = self._get_feature("video")
        items = conf.get("presets", [])
        if not isinstance(items, list):
            return presets
        for item in items:
            if isinstance(item, str) and ":" in item:
                key, val = item.split(":", 1)
                key = key.strip()
                val = val.strip()
                if key and val:
                    presets[key] = val
        return presets

    def _get_video_chain(self) -> list[str]:
        conf = self._get_feature("video")
        chain = conf.get("chain", [])
        if not isinstance(chain, list):
            return []
        out: list[str] = []
        for item in chain:
            pid = self._extract_chain_provider_id(item)
            if pid and pid not in out:
                out.append(pid)
        return out

    def _parse_video_args(self, text: str) -> tuple[str | None, str]:
        """解析 /视频 参数，返回 (preset, prompt)

        - 当第一个 token 命中预设名时：preset=该 token, prompt=剩余内容
        - 否则：preset=None, prompt=text
        """
        text = (text or "").strip()
        if not text:
            return None, ""

        first, _, rest = text.partition(" ")
        if first and first in self._get_video_presets():
            return first, rest.strip()
        return None, text

    async def _prepare_edit_image_bytes(self, event: AstrMessageEvent) -> list[bytes]:
        image_segs = await get_images_from_event(
            event,
            include_avatar=True,
            include_sender_avatar_fallback=False,
        )
        if not image_segs:
            raise RuntimeError("当前消息没有可用输入图片，无法执行改图批量任务。")
        bytes_images = await self._image_segs_to_bytes(image_segs)
        if not bytes_images:
            raise RuntimeError("当前消息图片读取失败，无法执行改图批量任务。")
        return bytes_images

    async def _execute_image_task_spec(
        self,
        event: AstrMessageEvent,
        spec: ImageTaskSpec,
        *,
        prepared_edit_images: list[bytes] | None = None,
        size: str | None = None,
        resolution: str | None = None,
        output_intent: OutputIntent | None = None,
    ) -> ExecutedImageTask:
        effective_output_intent = merge_output_intents(
            output_intent,
            parse_output_intent(spec.output) if spec.output else None,
        )

        if spec.mode == "draw":
            prompt = str(spec.effective_prompt or spec.user_prompt or "").strip()
            if not prompt:
                raise RuntimeError("文生图提示词为空。")
            image_path = await self.draw.generate(
                prompt,
                provider_id=spec.provider_id,
                size=size,
                resolution=resolution,
                output_intent=effective_output_intent,
            )
            task_meta = self._build_image_task_meta(
                mode="text",
                user_prompt=spec.user_prompt,
                effective_user_prompt=prompt if spec.preset_name else spec.user_prompt,
                effective_prompt=prompt,
                continue_with="text",
                backend=spec.provider_id,
            )
            return ExecutedImageTask(
                spec=spec, image_path=image_path, task_meta=task_meta
            )

        if spec.mode == "edit":
            bytes_images = prepared_edit_images
            if bytes_images is None:
                bytes_images = await self._prepare_edit_image_bytes(event)
            image_path = await self.edit.edit(
                prompt=spec.user_prompt,
                images=bytes_images,
                backend=spec.provider_id,
                preset=spec.preset_name,
                size=size,
                resolution=resolution,
                output_intent=effective_output_intent,
            )
            task_meta = self._build_image_task_meta(
                mode="edit",
                user_prompt=spec.user_prompt,
                effective_user_prompt=spec.effective_prompt,
                effective_prompt=spec.effective_prompt,
                continue_with="edit",
                backend=spec.provider_id,
            )
            if spec.preset_name:
                task_meta["preset_name"] = spec.preset_name
            return ExecutedImageTask(
                spec=spec, image_path=image_path, task_meta=task_meta
            )

        if spec.mode == "selfie_ref":
            if not self._is_selfie_enabled():
                raise RuntimeError(self._selfie_disabled_message())
            image_path, task_meta = await self._generate_selfie_image_with_meta(
                event,
                spec.user_prompt,
                spec.provider_id,
                size=size,
                resolution=resolution,
                output_intent=effective_output_intent,
            )
            return ExecutedImageTask(
                spec=spec, image_path=image_path, task_meta=task_meta
            )

        raise RuntimeError(f"不支持的图片任务模式: {spec.mode}")

    async def _run_batch_specs(
        self,
        event: AstrMessageEvent,
        specs: list[ImageTaskSpec],
        *,
        size: str | None = None,
        resolution: str | None = None,
        output_intent: OutputIntent | None = None,
    ) -> list[BatchRunResult[ExecutedImageTask]]:
        if not specs:
            return []

        prepared_edit_images: list[bytes] | None = None
        if any(spec.mode == "edit" for spec in specs):
            prepared_edit_images = await self._prepare_edit_image_bytes(event)

        concurrency = self._get_batch_concurrency_for_mode(specs[0].mode)

        async def _runner(index: int, spec: ImageTaskSpec) -> ExecutedImageTask:
            return await self._execute_image_task_spec(
                event,
                spec,
                prepared_edit_images=prepared_edit_images,
                size=size,
                resolution=resolution,
                output_intent=output_intent,
            )

        return await run_batch(specs, concurrency=concurrency, runner=_runner)

    async def _remember_batch_success(
        self,
        event: AstrMessageEvent,
        results: list[BatchRunResult[ExecutedImageTask]],
    ) -> None:
        for result in reversed(results):
            if not result.success or result.value is None:
                continue
            self._remember_last_image(event, result.value.image_path)
            await self._save_last_image_task_meta(event, result.value.task_meta)
            return

    async def _send_batch_results_single(
        self,
        event: AstrMessageEvent,
        results: list[BatchRunResult[ExecutedImageTask]],
        *,
        title: str,
    ) -> None:
        for result in results:
            if result.success and result.value is not None:
                await self._send_image_with_fallback(event, result.value.image_path)

    async def _send_batch_results(
        self,
        event: AstrMessageEvent,
        results: list[BatchRunResult[ExecutedImageTask]],
        *,
        title: str,
    ) -> None:
        await self._send_batch_results_single(event, results, title=title)

    async def _plan_batch_prompt_items(
        self,
        *,
        mode: str,
        user_prompt: str,
        count: int,
        fixed_aspect_ratio: str | None = None,
        umo: str | None = None,
    ) -> list[PlannedPromptItem]:
        provider = self.context.get_using_provider(umo=umo)
        if provider is None or not hasattr(provider, "text_chat"):
            raise RuntimeError("当前没有可用的 LLM 提供商，无法规划批量提示词。")

        planning_prompt = build_batch_planning_prompt(
            mode=mode,
            user_prompt=user_prompt,
            count=count,
            fixed_aspect_ratio=fixed_aspect_ratio,
        )
        last_error: Exception | None = None
        for _ in range(3):
            llm_response = await provider.text_chat(
                prompt=planning_prompt,
                contexts=[],
                image_urls=[],
                func_tool=None,
                system_prompt=(
                    "You plan image prompt sets. Output JSON only. "
                    "No markdown, no code fence, no explanation."
                ),
            )
            text = str(getattr(llm_response, "completion_text", "") or "").strip()
            if not text:
                last_error = RuntimeError("LLM returned empty planner output")
                continue
            try:
                items = parse_planned_prompt_items(text)
                validation_error = validate_planned_prompt_items(
                    items,
                    expected_count=count,
                    fixed_aspect_ratio=fixed_aspect_ratio,
                )
                if validation_error is not None:
                    raise ValueError(validation_error)
                return items
            except Exception as exc:
                last_error = exc
        raise RuntimeError(f"批量提示词规划失败: {last_error}")

    async def _resolve_llm_batch_mode(
        self, event: AstrMessageEvent, mode: str, prompt: str
    ) -> str:
        m = str(mode or "auto").strip().lower()
        if m in {"text", "draw", "aiimg"}:
            return "draw"
        if m in {"edit", "img2img", "aiedit"}:
            return "edit"
        if m in {"selfie_ref", "selfie", "ref"}:
            return "selfie_ref"
        if m != "auto":
            return "draw"

        if (
            self._is_selfie_enabled()
            and self._is_selfie_llm_enabled()
            and await self._should_auto_selfie_ref(event, prompt)
        ):
            return "selfie_ref"

        has_msg_images = await self._has_message_images(event)
        if has_msg_images:
            return "edit"

        prefetched_edit_image_segs = await get_images_from_event(
            event,
            include_avatar=True,
            include_sender_avatar_fallback=False,
        )
        if prefetched_edit_image_segs:
            return "edit"
        return "draw"

    async def _video_begin(self, user_id: str) -> bool:
        """单用户并发保护：成功占用返回 True，否则 False（上限可配置）"""
        return await self._begin_user_job(str(user_id or ""), kind="video")

    async def _video_end(self, user_id: str) -> None:
        await self._end_user_job(str(user_id or ""), kind="video")

    async def _send_video_result(self, event: AstrMessageEvent, video_url: str) -> None:
        vconf = self._get_feature("video")
        mode = str(vconf.get("send_mode", "auto")).strip().lower()
        if mode not in {"auto", "url", "file"}:
            mode = "auto"

        send_timeout = self._as_int(vconf.get("send_timeout_seconds", 90), default=90)
        send_timeout = max(10, min(send_timeout, 300))

        download_timeout = self._as_int(
            vconf.get("download_timeout_seconds", 300), default=300
        )
        download_timeout = max(1, min(download_timeout, 3600))

        async def _send_file(url: str) -> bool:
            try:
                video_path = await self.videomgr.download_video(
                    url, timeout_seconds=download_timeout
                )
                await asyncio.wait_for(
                    event.send(
                        event.chain_result([Video.fromFileSystem(str(video_path))])
                    ),
                    timeout=float(send_timeout),
                )
                return True
            except Exception as e:
                logger.warning(f"[视频] 本地文件发送失败: {e}")
                return False

        async def _send_url(url: str) -> bool:
            try:
                await asyncio.wait_for(
                    event.send(event.chain_result([Video.fromURL(url)])),
                    timeout=float(send_timeout),
                )
                return True
            except Exception as e:
                logger.warning(f"[视频] URL 发送失败: {e}")
                return False

        # file/url forced
        if mode == "file":
            if await _send_file(video_url):
                return
            await event.send(event.plain_result(video_url))
            return

        if mode == "url":
            if await _send_url(video_url):
                return
            await event.send(event.plain_result(video_url))
            return

        if await _send_url(video_url):
            return
        if await _send_file(video_url):
            return
        await event.send(event.plain_result(video_url))

    async def _async_generate_video(
        self,
        event: AstrMessageEvent,
        prompt: str,
        user_id: str,
        *,
        provider_id: str | None = None,
        llm_tool_failure: bool = False,
    ) -> None:
        try:
            image_segs = await get_images_from_event(
                event,
                include_avatar=True,
                include_sender_avatar_fallback=False,
            )
            had_image = bool(image_segs)
            image_bytes: bytes | None = None
            for i, seg in enumerate(image_segs):
                try:
                    b64 = await seg.convert_to_base64()
                    image_bytes = decode_base64_image_payload(b64)
                    break
                except Exception as e:
                    logger.warning(f"[视频] 图片 {i + 1} 转换失败，跳过: {e}")

            # 允许文生视频（无图）走支持的后端；但若用户确实发了图却读不到，则直接失败
            if had_image and not image_bytes:
                if llm_tool_failure:
                    await self._append_plugin_conversation_note(
                        event,
                        "The last video generation task failed and has ended because the source image could not be read. Do not retry automatically unless the user explicitly asks.",
                    )
                if llm_tool_failure:
                    await self._signal_llm_tool_failure(event)
                else:
                    await mark_failed(event)
                return

            t_start = time.perf_counter()
            candidates = (
                [str(provider_id).strip()] if provider_id else self._get_video_chain()
            )
            candidates = [c for c in candidates if c]
            if not candidates:
                raise RuntimeError(
                    "No video providers configured. Please set features.video.chain."
                )

            last_error: Exception | None = None
            video_url: str | None = None
            used_pid: str | None = None
            for pid in candidates:
                try:
                    backend = self.registry.get_video_backend(pid)
                    candidate_url = await backend.generate_video_url(
                        prompt=prompt, image_bytes=image_bytes
                    )
                    candidate_url = str(candidate_url or "").strip()
                    if not candidate_url:
                        raise RuntimeError("Provider returned empty video url")
                    video_url = candidate_url
                    used_pid = pid
                    break
                except Exception as e:
                    last_error = e
                    logger.warning("[视频] Provider=%s 失败: %s", pid, e)

            if not video_url:
                raise RuntimeError(f"视频生成失败: {last_error}") from last_error

            await self._send_video_result(event, video_url)
            await mark_success(event)
            if llm_tool_failure:
                await self._append_plugin_conversation_note(
                    event,
                    "The last video generation task has completed and the video was already sent to the user. Do not continue or resubmit this task unless the user explicitly asks for another video.",
                )

            t_end = time.perf_counter()
            name = used_pid or "video"
            logger.info(f"[视频] 完成: provider={name}, 耗时={t_end - t_start:.2f}s")

        except Exception as e:
            logger.error(f"[视频] 失败: {e}", exc_info=True)
            if llm_tool_failure:
                await self._append_plugin_conversation_note(
                    event,
                    "The last video generation task failed and has ended. Reason: "
                    + self._summarize_status_text(
                        e,
                        fallback="unknown error",
                    )
                    + ". Do not retry automatically unless the user explicitly asks.",
                )
            if llm_tool_failure:
                await self._signal_llm_tool_failure(event)
            else:
                await mark_failed(event)
        finally:
            await self._video_end(user_id)

    async def _do_edit_direct(
        self,
        event: AstrMessageEvent,
        prompt: str,
        backend: str | None = None,
        preset: str | None = None,
    ):
        """改图执行入口 (非 generator 版本，用于动态注册的命令)

        使用 event.send() 直接发送消息，不使用 yield
        """
        user_id = str(event.get_sender_id() or "")
        request_id = self._debounce_key(event, "edit", user_id)

        # 防抖
        if self.debouncer.hit(request_id):
            await mark_failed(event)
            return

        p = (prompt or "").strip()
        override, rest = self._parse_provider_override_prefix(p)
        if override:
            backend = override
            prompt = rest
        try:
            prompt, output_intent = split_prompt_output_suffix(prompt)
        except ValueError as exc:
            logger.warning("[改图] 输出参数无效: %s", exc)
            await mark_failed(event)
            return

        # 获取图片
        image_segs = await get_images_from_event(
            event,
            include_avatar=True,
            include_sender_avatar_fallback=False,
        )
        logger.debug(f"[改图] 获取到 {len(image_segs)} 个图片段")
        if not image_segs:
            await mark_failed(event)
            return

        bytes_images: list[bytes] = []
        for i, seg in enumerate(image_segs):
            try:
                logger.debug(f"[改图] 转换图片 {i + 1}/{len(image_segs)}...")
                b64 = await seg.convert_to_base64()
                bytes_images.append(decode_base64_image_payload(b64))
                logger.debug(
                    f"[改图] 图片 {i + 1} 转换成功, 大小={len(bytes_images[-1])} bytes"
                )
            except Exception as e:
                logger.warning(f"[改图] 图片 {i + 1} 转换失败，跳过: {e}")

        if not bytes_images:
            await mark_failed(event)
            return

        if not await self._begin_user_job(user_id, kind="image"):
            await mark_failed(event)
            return

        try:
            # 标记处理中
            await mark_processing(event)
            t_start = time.perf_counter()
            image_path = await self.edit.edit(
                prompt=prompt,
                images=bytes_images,
                backend=backend,
                preset=preset,
                output_intent=output_intent,
            )
            t_end = time.perf_counter()

            self._remember_last_image(event, image_path)
            sent = await self._send_image_with_fallback(event, image_path)
            if not sent:
                await mark_failed(event)
                logger.warning(
                    "[改图] 结果发送失败，已仅使用表情标注: reason=%s",
                    sent.reason,
                )
                return

            # 标记成功
            await mark_success(event)
            display_name = preset or (prompt[:20] if prompt else "改图")
            logger.info(f"[改图] 完成: {display_name}..., 耗时={t_end - t_start:.2f}s")

        except Exception as e:
            logger.error(f"[改图] 失败: {e}", exc_info=True)
            await mark_failed(event)
        finally:
            await self._end_user_job(user_id, kind="image")

    async def _do_edit(
        self,
        event: AstrMessageEvent,
        prompt: str,
        backend: str | None = None,
        preset: str | None = None,
    ):
        """统一改图执行入口

        预设触发逻辑:
        1. 如果 preset 参数已指定，直接使用
        2. 否则检查 prompt 是否匹配预设名，若匹配则自动转为预设
        3. 都不匹配则作为普通提示词处理
        """
        user_id = str(event.get_sender_id() or "")
        request_id = self._debounce_key(event, "edit", user_id)

        # 防抖
        if self.debouncer.hit(request_id):
            await mark_failed(event)
            return

        # Optional provider override: "/aiedit @provider_id <prompt>"
        p = (prompt or "").strip()
        override, rest = self._parse_provider_override_prefix(p)
        if override:
            backend = override
            prompt = rest
        try:
            prompt, output_intent = split_prompt_output_suffix(prompt)
        except ValueError as exc:
            logger.warning("[改图] 输出参数无效: %s", exc)
            await mark_failed(event)
            return

        # 预设自动检测: prompt 完全匹配预设名时，自动转为预设
        if not preset and prompt:
            prompt_stripped = prompt.strip()
            preset_names = self.edit.get_preset_names()
            if prompt_stripped in preset_names:
                preset = prompt_stripped
                prompt = ""  # 清空 prompt，使用预设的提示词
                logger.debug(f"[改图] 自动匹配预设: {preset}")

        # 获取图片
        image_segs = await get_images_from_event(
            event,
            include_avatar=True,
            include_sender_avatar_fallback=False,
        )
        if not image_segs:
            await mark_failed(event)
            return

        bytes_images: list[bytes] = []
        for seg in image_segs:
            try:
                b64 = await seg.convert_to_base64()
                bytes_images.append(decode_base64_image_payload(b64))
            except Exception as e:
                logger.warning(f"[改图] 图片转换失败，跳过: {e}")

        if not bytes_images:
            await mark_failed(event)
            return

        if not await self._begin_user_job(user_id, kind="image"):
            await mark_failed(event)
            return

        try:
            # 标记处理中
            await mark_processing(event)
            t_start = time.perf_counter()
            image_path = await self.edit.edit(
                prompt=prompt,
                images=bytes_images,
                backend=backend,
                preset=preset,
                output_intent=output_intent,
            )
            t_end = time.perf_counter()

            self._remember_last_image(event, image_path)
            sent = await self._send_image_with_fallback(event, image_path)
            if not sent:
                await mark_failed(event)
                logger.warning(
                    "[改图] 结果发送失败，已仅使用表情标注: reason=%s",
                    sent.reason,
                )
                return

            # 标记成功
            await mark_success(event)
            display_name = preset or (prompt[:20] if prompt else "改图")
            logger.info(f"[改图] 完成: {display_name}..., 耗时={t_end - t_start:.2f}s")

        except Exception as e:
            logger.error(f"[改图] 失败: {e}")
            await mark_failed(event)
        finally:
            await self._end_user_job(user_id, kind="image")

    # ==================== 自拍参考照：内部实现 ====================

    def _get_selfie_conf(self) -> dict:
        return self._get_feature("selfie")

    def _get_selfie_default_output(self) -> str | None:
        conf = self._get_selfie_conf()
        default_output_text = str(conf.get("default_output") or "").strip()
        default_aspect_ratio = (
            normalize_aspect_ratio(conf.get("default_aspect_ratio")) or "3:4"
        )
        default_output_intent = merge_output_intents(
            parse_output_intent(default_output_text) if default_output_text else None,
            OutputIntent(aspect_ratio=default_aspect_ratio),
        )
        return format_output_intent(default_output_intent) or None

    async def _ensure_tool_image_cache_dir(self) -> None:
        tool_image_dir = Path(get_astrbot_temp_path()) / "tool_images"
        await asyncio.to_thread(tool_image_dir.mkdir, parents=True, exist_ok=True)

    async def _build_llm_tool_image_result(
        self, image_path: Path
    ) -> mcp.types.CallToolResult | None:
        try:
            image_bytes = await asyncio.to_thread(Path(image_path).read_bytes)
        except Exception as exc:
            logger.warning(
                "[aiimg_generate] failed to read image for LLM context: path=%s err=%s",
                image_path,
                exc,
            )
            return None

        if not image_bytes:
            logger.warning(
                "[aiimg_generate] skip empty image for LLM context: path=%s",
                image_path,
            )
            return None

        mime_type, _ = guess_image_mime_and_ext(image_bytes)
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        return mcp.types.CallToolResult(
            content=[
                mcp.types.ImageContent(
                    type="image",
                    data=image_b64,
                    mimeType=mime_type,
                )
            ]
        )

    async def _finalize_llm_tool_image(
        self,
        event: AstrMessageEvent,
        image_path: Path,
        *,
        task_meta: dict[str, Any],
    ) -> mcp.types.CallToolResult:
        self._remember_last_image(event, image_path)

        sent = await self._send_image_with_fallback(event, image_path)
        if not sent:
            await self._signal_llm_tool_failure(event)
            logger.warning(
                "[aiimg_generate] image send failed, emoji fallback only: reason=%s",
                sent.reason,
            )
            return self._llm_tool_text_result(
                "Image generation finished, but sending the image to the user failed. This request has ended. Do not retry automatically unless the user explicitly asks."
            )

        await mark_success(event)
        await self._save_last_image_task_meta(event, task_meta)
        return self._build_image_task_completion_result(task_meta)

    def _get_selfie_ref_store_key(self, event: AstrMessageEvent) -> str:
        """用于 ReferenceStore 的固定 key（按 bot self_id 隔离）。"""
        self_id = ""
        try:
            if hasattr(event, "get_self_id"):
                self_id = str(event.get_self_id() or "").strip()
        except Exception:
            self_id = ""
        return f"bot_selfie_{self_id}" if self_id else "bot_selfie"

    def _resolve_data_rel_path(self, rel_path: str) -> Path | None:
        """将 data_dir 下的相对路径解析为绝对路径，并阻止路径穿越。"""
        if not isinstance(rel_path, str) or not rel_path.strip():
            return None
        rel = rel_path.replace("\\", "/").lstrip("/")
        parts = [p for p in rel.split("/") if p]
        if any(p in {".", ".."} for p in parts):
            return None
        base = Path(self.data_dir).resolve(strict=False)
        target = (base / "/".join(parts)).resolve(strict=False)
        try:
            target.relative_to(base)
        except ValueError:
            return None
        return target

    def _get_config_selfie_reference_paths(self) -> list[Path]:
        """从 WebUI file 配置项读取参考图路径。"""
        conf = self._get_selfie_conf()
        ref_list = conf.get("reference_images", [])
        if not isinstance(ref_list, list):
            return []

        paths: list[Path] = []
        for rel_path in ref_list:
            p = self._resolve_data_rel_path(str(rel_path))
            if not p:
                continue
            if p.is_file():
                paths.append(p)
        return paths

    async def _get_selfie_reference_paths(
        self, event: AstrMessageEvent
    ) -> tuple[list[Path], str]:
        """返回(路径列表, 来源)；来源=webui/store/none"""
        webui_paths = self._get_config_selfie_reference_paths()
        if webui_paths:
            return webui_paths, "webui"

        store_key = self._get_selfie_ref_store_key(event)
        store_paths = await self.refs.get_paths(store_key)
        if store_paths:
            return store_paths, "store"

        return [], "none"

    async def _read_paths_bytes(self, paths: list[Path]) -> list[bytes]:
        out: list[bytes] = []
        for p in paths:
            try:
                data = await asyncio.to_thread(p.read_bytes)
            except Exception:
                continue
            if data:
                out.append(data)
        return out

    async def _image_segs_to_bytes(self, image_segs: list) -> list[bytes]:
        """将 Image 组件列表转换为 bytes。"""
        out: list[bytes] = []
        for seg in image_segs:
            try:
                b64 = await seg.convert_to_base64()
                out.append(decode_base64_image_payload(b64))
            except Exception as e:
                logger.warning(f"[图片] 转换失败，跳过: {e}")
        return out

    async def _has_message_images(self, event: AstrMessageEvent) -> bool:
        """仅检测用户消息/引用里的图片（不含头像兜底）。"""
        image_segs = await get_images_from_event(event, include_avatar=False)
        return bool(image_segs)

    async def _has_message_images_or_avatar_mentions(
        self, event: AstrMessageEvent
    ) -> bool:
        if await self._has_message_images(event):
            return True
        return any(str(uid).isdigit() for uid in collect_at_user_ids(event))

    def _is_auto_selfie_prompt(self, prompt: str) -> bool:
        text = (prompt or "").strip()
        if not text:
            return False
        lowered = text.lower()
        if "自拍" in text or "selfie" in lowered:
            return True
        if any(
            k in text
            for k in (
                "来一张你",
                "来张你",
                "你来一张",
                "你来张",
                "看看你",
                "你自己",
                "你本人",
                "你的照片",
                "你的自拍",
                "你自己的照片",
                "你自己的自拍",
                "你长什么样",
                "看看你本人",
                "看看你自己",
                "bot自拍",
                "机器人自拍",
            )
        ):
            return True
        if any(
            k in lowered
            for k in ("your selfie", "your photo", "your picture", "your face")
        ):
            return True
        return False

    async def _should_auto_selfie_ref(
        self, event: AstrMessageEvent, prompt: str
    ) -> bool:
        if not self._is_auto_selfie_prompt(prompt):
            logger.debug("[aiimg_generate] auto-selfie skipped: prompt not selfie")
            return False
        paths, source = await self._get_selfie_reference_paths(event)
        if not paths:
            logger.info("[aiimg_generate] auto-selfie skipped: no reference images")
            return False
        logger.debug(
            "[aiimg_generate] auto-selfie candidate: refs=%s source=%s",
            len(paths),
            source,
        )
        return True

    def _build_selfie_prompt(self, prompt: str, extra_refs: int) -> str:
        conf = self._get_selfie_conf()
        prefix = str(conf.get("prompt_prefix", "") or "").strip()
        if not prefix:
            prefix = (
                "请根据参考图生成一张新的自拍照：\n"
                "1) 以第1张参考图的人脸身份为准（仅人脸身份特征），保持五官/气质一致。\n"
                "2) 如果还有其它参考图，请将它们仅作为服装/姿势/构图/场景的参考。\n"
                "3) 输出一张高质量照片风格自拍，不要拼图，不要水印。"
            )

        user_prompt = (prompt or "").strip() or "日常自拍照"
        if extra_refs > 0:
            return (
                f"{prefix}\n\n用户要求：{user_prompt}\n（额外参考图数量：{extra_refs}）"
            )
        return f"{prefix}\n\n用户要求：{user_prompt}"

    def _merge_selfie_chain_with_edit_chain(
        self, selfie_chain: list[object]
    ) -> list[dict]:
        """将自拍链路与改图链路合并（自拍优先，去重 provider_id）。"""
        merged: list[dict] = []
        seen: set[str] = set()

        def append_unique(items: list) -> None:
            for item in items:
                normalized = self._normalize_chain_item(item)
                if not normalized:
                    continue
                pid = str(normalized.get("provider_id") or "").strip()
                if not pid or pid in seen:
                    continue
                merged.append(normalized)
                seen.add(pid)

        append_unique(selfie_chain)

        edit_chain_raw = self._get_feature("edit").get("chain", [])
        if isinstance(edit_chain_raw, list):
            append_unique(edit_chain_raw)

        return merged

    async def _generate_selfie_image_with_meta(
        self,
        event: AstrMessageEvent,
        prompt: str,
        backend: str | None,
        *,
        size: str | None = None,
        resolution: str | None = None,
        output_intent: OutputIntent | None = None,
        follow_up_meta: dict[str, Any] | None = None,
    ) -> tuple[Path, dict[str, Any]]:
        conf = self._get_selfie_conf()
        if not self._is_selfie_enabled():
            raise RuntimeError(self._selfie_disabled_message())

        # 1) 读取参考照（WebUI 优先，其次命令设置的 store）
        ref_paths, ref_source = await self._get_selfie_reference_paths(event)
        ref_images = await self._read_paths_bytes(ref_paths)
        if not ref_images:
            raise RuntimeError(
                "未设置自拍参考照。请先：发送图片 + /自拍参考 设置，或在 WebUI 配置 features.selfie.reference_images 上传。"
            )

        # 2) 读取额外参考图（衣服/姿势/场景）
        extra_segs = await get_images_from_event(event, include_avatar=False)
        extra_bytes = await self._image_segs_to_bytes(extra_segs)

        # 3) 拼接输入图：参考照在前
        images = [*ref_images, *extra_bytes]

        effective_user_prompt = self._build_selfie_follow_up_prompt(
            prompt, follow_up_meta
        )
        final_prompt = self._build_selfie_prompt(
            effective_user_prompt, extra_refs=len(extra_bytes)
        )

        chain_override: list[dict] | None = None
        use_edit_chain = bool(conf.get("use_edit_chain_when_empty", True))
        raw_chain = conf.get("chain", [])
        if isinstance(raw_chain, list):
            chain_items = [
                normalized
                for normalized in (self._normalize_chain_item(x) for x in raw_chain)
                if normalized is not None
            ]
            if chain_items:
                chain_override = chain_items

        if backend is None:
            if chain_override is None:
                if not use_edit_chain:
                    raise RuntimeError(
                        "No selfie provider chain configured. Please set features.selfie.chain or enable features.selfie.use_edit_chain_when_empty."
                    )
            elif use_edit_chain:
                # 自拍链路可作为主链，改图链路作为补充兜底，避免"自拍链仅一项导致无兜底"。
                chain_override = self._merge_selfie_chain_with_edit_chain(
                    chain_override
                )

        if chain_override:
            logger.debug(
                "[selfie] effective providers=%s",
                [
                    str(x.get("provider_id") or "").strip()
                    for x in chain_override
                    if isinstance(x, dict)
                ],
            )

        # 4) 千问后端可选 task_types（仅对 gitee 生效）
        task_types = conf.get("gitee_task_types")
        if isinstance(task_types, list) and task_types:
            gitee_task_types = [str(x).strip() for x in task_types if str(x).strip()]
        else:
            gitee_task_types = ["id", "background", "style"]

        default_output = self._get_selfie_default_output()

        image_path = await self.edit.edit(
            prompt=final_prompt,
            images=images,
            backend=backend,
            task_types=gitee_task_types,
            size=size,
            resolution=resolution,
            output_intent=output_intent,
            default_output=default_output,
            chain_override=chain_override,
            infer_source_aspect=False,
        )
        task_meta = self._build_image_task_meta(
            mode="selfie_ref",
            user_prompt=prompt,
            effective_user_prompt=effective_user_prompt,
            effective_prompt=final_prompt,
            reference_source=ref_source,
            reference_count=len(ref_images),
            extra_reference_count=len(extra_bytes),
            continue_with="selfie_ref",
            follow_up=follow_up_meta is not None,
            backend=backend,
        )
        return image_path, task_meta

    async def _generate_selfie_image(
        self,
        event: AstrMessageEvent,
        prompt: str,
        backend: str | None,
        *,
        size: str | None = None,
        resolution: str | None = None,
        output_intent: OutputIntent | None = None,
    ) -> Path:
        image_path, _ = await self._generate_selfie_image_with_meta(
            event,
            prompt,
            backend,
            size=size,
            resolution=resolution,
            output_intent=output_intent,
        )
        return image_path

    async def _do_selfie(
        self,
        event: AstrMessageEvent,
        prompt: str,
        backend: str | None = None,
    ):
        """指令 /自拍 执行入口。"""
        if not self._is_selfie_enabled():
            await mark_failed(event)
            return

        user_id = str(event.get_sender_id() or "")
        request_id = self._debounce_key(event, "selfie", user_id)

        if self.debouncer.hit(request_id):
            await mark_failed(event)
            return

        p = (prompt or "").strip()
        override, rest = self._parse_provider_override_prefix(p)
        if override:
            backend = override
            prompt = rest
        try:
            prompt, output_intent = split_prompt_output_suffix(prompt)
        except ValueError as exc:
            logger.warning("[自拍] 输出参数无效: %s", exc)
            await mark_failed(event)
            return

        if not await self._begin_user_job(user_id, kind="image"):
            await mark_failed(event)
            return

        try:
            await mark_processing(event)
            image_path, task_meta = await self._generate_selfie_image_with_meta(
                event,
                prompt,
                backend,
                output_intent=output_intent,
            )
            self._remember_last_image(event, image_path)
            sent = await self._send_image_with_fallback(event, image_path)
            if not sent:
                await mark_failed(event)
                logger.warning(
                    "[自拍] 结果发送失败，已仅使用表情标注: reason=%s",
                    sent.reason,
                )
                return
            await mark_success(event)
            await self._save_last_image_task_meta(event, task_meta)
        except Exception as e:
            logger.error(f"[自拍] 失败: {e}", exc_info=True)
            await mark_failed(event)
        finally:
            await self._end_user_job(user_id, kind="image")

    async def _set_selfie_reference(self, event: AstrMessageEvent):
        if not self._is_selfie_enabled():
            await mark_failed(event)
            return

        image_segs = await get_images_from_event(event, include_avatar=False)
        if not image_segs:
            await mark_failed(event)
            return

        bytes_images = await self._image_segs_to_bytes(image_segs)
        if not bytes_images:
            await mark_failed(event)
            return

        # 限制数量，避免一次塞太多
        max_images = 8
        bytes_images = bytes_images[:max_images]

        store_key = self._get_selfie_ref_store_key(event)
        try:
            await self.refs.set(store_key, bytes_images)
        except Exception:
            await mark_failed(event)
            return

        await mark_success(event)

    async def _show_selfie_reference(self, event: AstrMessageEvent):
        if not self._is_selfie_enabled():
            await mark_failed(event)
            return

        paths, source = await self._get_selfie_reference_paths(event)
        if not paths:
            await mark_failed(event)
            return

        # 最多回显 5 张，避免刷屏
        max_show = 5
        show_paths = paths[:max_show]
        yield event.chain_result([Image.fromFileSystem(str(p)) for p in show_paths])
        yield event.plain_result(
            f"📌 当前自拍参考照来源：{source}，共 {len(paths)} 张（已展示 {len(show_paths)} 张）"
        )

    async def _delete_selfie_reference(self, event: AstrMessageEvent):
        if not self._is_selfie_enabled():
            await mark_failed(event)
            return

        store_key = self._get_selfie_ref_store_key(event)
        deleted = await self.refs.delete(store_key)

        webui_paths = self._get_config_selfie_reference_paths()
        if webui_paths:
            logger.info(
                "[自拍参考] 命令保存的参考照已删除，但 WebUI reference_images 仍生效（优先级更高）"
            )

        if deleted:
            await mark_success(event)
        else:
            await mark_failed(event)
