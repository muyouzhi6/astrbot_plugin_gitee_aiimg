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
import io
import json
import math
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
from astrbot.api.star import Context, Star, StarTools
from astrbot.core.utils.astrbot_path import get_astrbot_temp_path

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
from .core.nanobanana import NanoBananaService
from .core.provider_registry import ProviderRegistry
from .core.ref_store import ReferenceStore
from .core.utils import close_session, get_images_from_event
from .core.video_manager import VideoManager


@dataclass(slots=True)
class SendImageResult:
    ok: bool
    reason: str = ""
    cached_path: Path | None = None
    used_fallback: bool = False
    last_error: str = ""

    def __bool__(self) -> bool:
        return self.ok


class GiteeAIImagePlugin(Star):
    """Gitee AI 图像生成插件"""

    # Gitee AI 支持的图片比例
    SUPPORTED_RATIOS: dict[str, list[str]] = GITEE_SUPPORTED_RATIOS
    IMAGE_AS_FILE_THRESHOLD_BYTES: int = 20 * 1024 * 1024

    def __init__(self, context: Context, config: dict):
        super().__init__(context)
        self.config = config
        self.data_dir = StarTools.get_data_dir("astrbot_plugin_gitee_aiimg")
        self._last_image_by_user: dict[str, Path] = {}
        self._last_image_task_meta_cache: dict[str, dict[str, Any]] = {}

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
            "effective_user_prompt": str(meta.get("effective_user_prompt") or "").strip(),
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
                effective_user_prompt if effective_user_prompt is not None else user_prompt
            ).strip(),
            "effective_prompt": str(effective_prompt or "").strip(),
            "reference_source": str(reference_source or "").strip(),
            "reference_count": max(0, int(reference_count or 0)),
            "extra_reference_count": max(0, int(extra_reference_count or 0)),
            "continue_with": str(continue_with or mode or "").strip() or str(mode or "").strip(),
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
            "continue_with": str(task_meta.get("continue_with") or mode).strip() or mode,
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
                history = list(parsed_history) if isinstance(parsed_history, list) else []
            except Exception as exc:
                logger.warning(
                    "[GiteeAIImagePlugin] failed to parse conversation history for plugin note: %s",
                    exc,
                )
                history = []

        history.append({"role": "user", "content": "Output your last task result below."})
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

        self._patch_tool_image_cache_runtime()

        # 动态注册预设命令 (方案C: /手办化 直接触发)
        self._register_preset_commands()

        logger.info(
            f"[GiteeAIImagePlugin] 插件初始化完成: "
            f"改图后端={self.edit.get_available_backends()}, "
            f"改图预设={len(self.edit.get_preset_names())}个, "
            f"视频启用={bool(self._get_feature('video').get('enabled', False))}, "
            f"视频预设={len(self._get_video_presets())}个"
        )

    def _remember_last_image(self, event: AstrMessageEvent, image_path: Path) -> None:
        try:
            user_id = str(event.get_sender_id() or "")
        except Exception:
            user_id = ""
        if not user_id:
            return
        self._last_image_by_user[user_id] = Path(image_path)

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
            logger.debug("[GiteeAIImagePlugin] skip tool image cache runtime patch: %s", exc)
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
                "[GiteeAIImagePlugin] tool image cache runtime patch wrote: %s", file_path
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
        p = Path(image_path)

        if not p.exists():
            logger.warning("[send_image] file not found: %s", p)
            return SendImageResult(ok=False, reason="file_not_found", cached_path=p)

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
                return SendImageResult(ok=True, cached_path=p, used_fallback=True)

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
                return SendImageResult(ok=True, cached_path=p, used_fallback=False)
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
                return SendImageResult(ok=True, cached_path=p, used_fallback=True)
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
                    return SendImageResult(ok=True, cached_path=p, used_fallback=True)

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
                        return SendImageResult(
                            ok=True, cached_path=p, used_fallback=True
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
                await asyncio.sleep(delay)
                delay = min(delay * 1.8, 8.0)

        reason = (
            "rich_media_transfer_failed"
            if self._is_rich_media_transfer_failed(last_exc)
            else "send_failed"
        )
        logger.error(
            "[send_image] failed after retries: reason=%s, err=%s", reason, last_exc
        )
        return SendImageResult(
            ok=False,
            reason=reason,
            cached_path=p,
            last_error=str(last_exc or ""),
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
        # 清理多余空格
        return msg.strip()

    @staticmethod
    def _extract_command_arg_anywhere(message: str, command_name: str) -> str:
        """从任意位置提取“/命令 参数”，用于图片在前导致 @filter.command 不触发的场景。"""
        msg = (message or "").strip()
        if not msg:
            return ""
        for prefix in "/!！.。．":
            token = f"{prefix}{command_name}"
            idx = msg.find(token)
            if idx >= 0:
                return msg[idx + len(token) :].strip()
        return ""

    def _extract_command_arg_from_chain(
        self, event: AstrMessageEvent, command_name: str
    ) -> tuple[bool, str]:
        """从消息链中提取命令后的提示词。

        用于修复“/命令 + 图片 + 文本”时，平台把文本段无空格拼接到 `message_str`
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
                if plain[0] in "/!！.。．":
                    plain = plain[1:]
                if not plain.startswith(command_name):
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
    def _plain_starts_with_command(text: str, command_name: str) -> bool:
        plain = (text or "").lstrip()
        if not plain:
            return False
        for prefix in "/!！.。．":
            if plain.startswith(f"{prefix}{command_name}"):
                return True
        return False

    def _is_direct_command_message(
        self, event: AstrMessageEvent, command_names: tuple[str, ...]
    ) -> bool:
        """仅当“首个有效文本段”直接是命令时返回 True。

        用于 regex 兜底去重：避免正常 /命令 被重复处理；
        同时允许“图片在前、命令在后”的消息继续走兜底逻辑。
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
            self._plain_starts_with_command(first_plain, name) for name in command_names
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

    async def terminate(self):
        self.debouncer.clear_all()
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

    @filter.command("aiimg", alias={"文生图", "生图", "画图", "绘图", "出图"})
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

        prompt = arg.strip()
        size: str | None = None
        parts = arg.split()
        if parts and parts[-1] in self.SUPPORTED_RATIOS:
            ratio = parts[-1]
            prompt = " ".join(parts[:-1]).strip()
            size = self._resolve_ratio_size(ratio)

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
                prompt, size=size, provider_id=provider_override
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

    # ==================== 图生图/改图 ====================

    @filter.command("aiedit", alias={"图生图", "改图", "修图"})
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
        p = self._last_image_by_user.get(user_id)
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

    @filter.regex(r"(?:[/!！.。．])?(改图|图生图|修图|aiedit)", priority=-10)
    async def edit_image_regex_fallback(self, event: AstrMessageEvent):
        """兼容“图片在前、文字在后”的消息：确保 /改图 能触发。"""
        msg = (event.message_str or "").strip()
        command_names = ("改图", "图生图", "修图", "aiedit")
        if self._is_framework_direct_command_text(msg, command_names, allow_bare=False):
            return
        try:
            if not await self._has_message_images(event):
                return
        except Exception:
            return

        prompt = ""
        matched = False
        for name in command_names:
            prompt = self._extract_command_arg_anywhere(msg, name)
            found_in_chain, chain_prompt = self._extract_command_arg_from_chain(
                event, name
            )
            if prompt or found_in_chain:
                matched = True
                if not prompt:
                    prompt = chain_prompt
                break
        if matched:
            event.should_call_llm(True)
            await self._do_edit(event, prompt, backend=None)
            event.stop_event()

    @filter.regex(r"[/!！.。．][^\s]+", priority=-10)
    async def preset_regex_fallback(self, event: AstrMessageEvent):
        """兼容“图片在前、预设命令在后”的消息：确保 /<预设名> 能触发。"""
        msg = (event.message_str or "").strip()
        preset_names = self.edit.get_preset_names()
        if not preset_names:
            return

        # 如果首段文本本来就是 /预设，则交给 command handler，避免重复处理
        try:
            if self._is_direct_command_message(event, tuple(preset_names)):
                return
        except Exception:
            pass

        # 仅当消息/引用里确实带图（不含头像兜底）时才兜底，避免误伤其它插件命令
        try:
            if not await self._has_message_images(event):
                return
        except Exception:
            return

        # 在任意位置找到第一个匹配的预设命令
        used_preset: str | None = None
        for name in preset_names:
            for prefix in "/!！.。．":
                if f"{prefix}{name}" in msg:
                    used_preset = name
                    break
            if used_preset:
                break

        if not used_preset:
            return

        extra_prompt = self._extract_command_arg_anywhere(msg, used_preset)
        await self._do_edit_direct(event, extra_prompt, preset=used_preset)
        event.stop_event()

    # ==================== Bot 自拍（参考照） ====================

    @filter.command("自拍")
    async def selfie_command(self, event: AstrMessageEvent):
        """使用“自拍参考照”生成 Bot 自拍。

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

    @filter.regex(r"[/!！.。．]自拍(\s|$)", priority=-10)
    async def selfie_regex_fallback(self, event: AstrMessageEvent):
        """兼容“图片在前、文字在后”的消息：确保 /自拍 能触发。"""
        msg = (event.message_str or "").strip()
        # 如果本来就是“首段文本命令”，交给 command handler，避免重复回复
        if self._is_direct_command_message(event, ("自拍",)):
            return
        prompt = self._extract_command_arg_anywhere(msg, "自拍")
        if prompt or "/自拍" in msg or "自拍" in msg:
            if not self._is_selfie_enabled():
                await mark_failed(event)
                event.stop_event()
                return
            await self._do_selfie(event, prompt, backend=None)
            event.stop_event()

    @filter.command("自拍参考")
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

    @filter.regex(r"[/!！.。．]自拍参考(\s|$)", priority=-10)
    async def selfie_reference_regex_fallback(self, event: AstrMessageEvent):
        """兼容“图片在前、文字在后”的消息：确保 /自拍参考 能触发。"""
        msg = (event.message_str or "").strip()
        if self._is_direct_command_message(event, ("自拍参考",)):
            return
        if not self._is_selfie_enabled():
            await mark_failed(event)
            event.stop_event()
            return
        arg = self._extract_command_arg_anywhere(msg, "自拍参考")
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
    async def generate_video_regex_fallback(self, event: AstrMessageEvent):
        """兼容“图片在前、文字在后”的消息：确保 /视频 能触发。"""
        msg = (event.message_str or "").strip()
        if self._is_direct_command_message(event, ("视频",)):
            return

        arg = self._extract_command_arg_anywhere(msg, "视频")
        if not arg and "/视频" not in msg:
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
    ):
        """统一图片生成/改图/自拍（参考照）工具。

        使用建议（给 LLM 的决策规则）：
        - 用户发送/引用了图片，并要求"改图/换背景/换风格/修图/换衣服"等：用 mode=edit（或 mode=auto）
        - 用户要求"bot 自拍/来一张你自己的自拍"，且已设置自拍参考照：用 mode=selfie_ref（或 mode=auto）
        - 纯文生图（用户没有给图片）：用 mode=text（或 mode=auto）

        当前 LLM tool 行为：
        - 成功后优先直接把图片发送给用户
        - tool result 返回文本摘要，写明本次任务的 mode、effective_prompt 和 follow-up 提示
        - 不再把 ImageContent 回传给 LLM 上下文，避免额外多模态识图耗时

        Args:
            prompt(string): 提示词
            mode(string): auto=自动判断, text=文生图, edit=改图, selfie_ref=参考照自拍
            backend(string): auto=自动选择；也可填 provider_id（你在 WebUI providers 里配置的 id）
            output(string): 输出尺寸/分辨率。例: 2048x2048 或 4K（不同后端支持能力不同，留空用默认）
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
        size = output if output and "x" in output else None
        resolution = output if output and size is None else None

        try:
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
                    size=size,
                    resolution=resolution,
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
                        image_path, task_meta = await self._generate_selfie_image_with_meta(
                            event,
                            prompt,
                            target_backend,
                            size=size,
                            resolution=resolution,
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
                follow_up_selfie_meta = await self._match_selfie_follow_up(event, prompt)
                if follow_up_selfie_meta is not None:
                    try:
                        logger.info("[aiimg_generate] route=auto->selfie_ref (follow-up)")
                        image_path, task_meta = await self._generate_selfie_image_with_meta(
                            event,
                            prompt,
                            target_backend,
                            size=size,
                            resolution=resolution,
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
                    size=size,
                    resolution=resolution,
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
                size=size,
                resolution=resolution,
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

        send_timeout = int(vconf.get("send_timeout_seconds", 90) or 90)
        send_timeout = max(10, min(send_timeout, 300))

        download_timeout = int(vconf.get("download_timeout_seconds", 300) or 300)
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

        # auto: prefer file first (most platforms won't render URL as playable video)
        if await _send_file(video_url):
            return
        if await _send_url(video_url):
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
                # 自拍链路可作为主链，改图链路作为补充兜底，避免“自拍链仅一项导致无兜底”。
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

        default_output = str(conf.get("default_output") or "").strip() or None

        image_path = await self.edit.edit(
            prompt=final_prompt,
            images=images,
            backend=backend,
            task_types=gitee_task_types,
            size=size,
            resolution=resolution,
            default_output=default_output,
            chain_override=chain_override,
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
    ) -> Path:
        image_path, _ = await self._generate_selfie_image_with_meta(
            event,
            prompt,
            backend,
            size=size,
            resolution=resolution,
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

        if not await self._begin_user_job(user_id, kind="image"):
            await mark_failed(event)
            return

        p = (prompt or "").strip()
        override, rest = self._parse_provider_override_prefix(p)
        if override:
            backend = override
            prompt = rest

        try:
            await mark_processing(event)
            image_path, task_meta = await self._generate_selfie_image_with_meta(
                event, prompt, backend
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
