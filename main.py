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
import time
from pathlib import Path

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import At, AtAll, Image, Plain, Reply, Video
from astrbot.api.star import Context, Star, StarTools

from .core.debouncer import Debouncer
from .core.draw_service import ImageDrawService
from .core.edit_router import EditRouter
from .core.emoji_feedback import mark_failed, mark_processing, mark_success
from .core.image_manager import ImageManager
from .core.nanobanana import NanoBananaService
from .core.provider_registry import ProviderRegistry
from .core.ref_store import ReferenceStore
from .core.utils import close_session, get_images_from_event
from .core.video_manager import VideoManager


class GiteeAIImage(Star):
    """Gitee AI 图像生成插件"""

    # Gitee AI 支持的图片比例
    SUPPORTED_RATIOS: dict[str, list[str]] = {
        "1:1": ["256x256", "512x512", "1024x1024", "2048x2048"],
        "4:3": ["1152x896", "2048x1536"],
        "3:4": ["768x1024", "1536x2048"],
        "3:2": ["2048x1360"],
        "2:3": ["1360x2048"],
        "16:9": ["1024x576", "2048x1152"],
        "9:16": ["576x1024", "1152x2048"],
    }

    def __init__(self, context: Context, config: dict):
        super().__init__(context)
        self.config = config
        self.data_dir = StarTools.get_data_dir()
        self._last_image_by_user: dict[str, Path] = {}

    async def initialize(self):
        self.debouncer = Debouncer(self.config)
        self.imgr = ImageManager(self.config, self.data_dir)
        self.registry = ProviderRegistry(
            self.config, imgr=self.imgr, data_dir=self.data_dir
        )
        for err in self.registry.validate():
            logger.warning("[GiteeAIImage][config] %s", err)

        self.draw = ImageDrawService(
            self.config, self.imgr, self.data_dir, registry=self.registry
        )
        self.edit = EditRouter(
            self.config, self.imgr, self.data_dir, registry=self.registry
        )
        self.nb = NanoBananaService(self.config, self.imgr)
        self.refs = ReferenceStore(self.data_dir)
        self.videomgr = VideoManager(self.config, self.data_dir)

        self._video_lock = asyncio.Lock()
        self._video_in_progress: set[str] = set()
        self._video_tasks: set[asyncio.Task] = set()

        # 动态注册预设命令 (方案C: /手办化 直接触发)
        self._register_preset_commands()

        logger.info(
            f"[GiteeAIImage] 插件初始化完成: "
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

    async def _send_image_with_fallback(
        self, event: AstrMessageEvent, image_path: Path, *, max_attempts: int = 5
    ) -> bool:
        """Send image with retries and fallback to base64 bytes.

        Avoids wasting generation credits when platform send fails transiently.
        """
        p = Path(image_path)
        if not p.exists():
            logger.warning("[send_image] file not found: %s", p)
            return False

        delay = 1.5
        last_exc: Exception | None = None
        attempts = max(1, int(max_attempts))
        for attempt in range(1, attempts + 1):
            try:
                await event.send(event.chain_result([Image.fromFileSystem(str(p))]))
                return True
            except Exception as e:
                last_exc = e
                logger.warning(
                    "[send_image] fromFileSystem failed (attempt=%s/%s): %s",
                    attempt,
                    attempts,
                    e,
                )

            try:
                data = await asyncio.to_thread(p.read_bytes)
                await event.send(event.chain_result([Image.fromBytes(data)]))
                return True
            except Exception as e:
                last_exc = e
                logger.warning(
                    "[send_image] fromBytes failed (attempt=%s/%s): %s",
                    attempt,
                    attempts,
                    e,
                )

            if attempt < attempts:
                await asyncio.sleep(delay)
                delay = min(delay * 1.8, 8.0)

        if last_exc is not None:
            logger.error("[send_image] failed after retries: %s", last_exc)
        return False

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

        logger.info(f"[GiteeAIImage] 已注册 {len(preset_names)} 个预设命令")

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
            self._plain_starts_with_command(first_plain, name)
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
            yield event.plain_result(
                "请提供提示词！用法：/aiimg [@provider_id] <提示词> [比例]"
            )
            return
        provider_override: str | None = None
        if arg.lstrip().startswith("@"):
            first, _, rest = arg.strip().partition(" ")
            provider_override = first.lstrip("@").strip() or None
            arg = rest.strip()
        if not arg:
            yield event.plain_result(
                "请提供提示词！用法：/aiimg [@provider_id] <提示词> [比例]"
            )
            return

        prompt = arg.strip()
        size: str | None = None
        parts = arg.split()
        if parts and parts[-1] in self.SUPPORTED_RATIOS:
            ratio = parts[-1]
            prompt = " ".join(parts[:-1]).strip()
            size = self.SUPPORTED_RATIOS[ratio][0]

        if not prompt:
            yield event.plain_result(
                "请提供提示词！用法：/aiimg [@provider_id] <提示词> [比例]"
            )
            return

        user_id = event.get_sender_id()
        request_id = f"generate_{user_id}"

        # 防抖检查
        if self.debouncer.hit(request_id):
            yield event.plain_result("操作太快了，请稍后再试")
            return

        # 标记处理中
        await mark_processing(event)

        try:
            t_start = time.perf_counter()
            image_path = await self.draw.generate(
                prompt, size=size, provider_id=provider_override
            )
            t_end = time.perf_counter()

            self._remember_last_image(event, image_path)
            sent = await self._send_image_with_fallback(event, image_path)
            if not sent:
                await mark_failed(event)
                yield event.plain_result(
                    "图片已生成，但发送失败（可能是平台临时异常）。可稍后使用：/重发图片"
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
            yield event.plain_result(f"生成图片失败: {str(e)}")

    # ==================== 图生图/改图 ====================

    @filter.command("aiedit", alias={"图生图", "改图", "修图"})
    async def edit_image_default(self, event: AstrMessageEvent, prompt: str):
        """使用默认后端改图

        用法: /aiedit <提示词>
        需要同时发送或引用图片
        """
        event.should_call_llm(True)
        async for result in self._do_edit(event, prompt, backend=None):
            yield result

    @filter.command("重发图片")
    async def resend_last_image(self, event: AstrMessageEvent):
        """重发最近一次生成/改图的图片（不重新生成，不消耗次数）。"""
        user_id = str(event.get_sender_id() or "")
        p = self._last_image_by_user.get(user_id)
        if not p:
            yield event.plain_result("当前没有可重发的图片。")
            return
        if not Path(p).exists():
            yield event.plain_result("上次图片缓存已过期/被清理，无法重发。")
            return
        ok = await self._send_image_with_fallback(event, p)
        if ok:
            yield event.plain_result("已重发图片。")
        else:
            yield event.plain_result("重发失败（平台可能异常），请稍后再试。")

    @filter.regex(r"[/!！.。．](改图|图生图|修图|aiedit)(\s|$)", priority=-10)
    async def edit_image_regex_fallback(self, event: AstrMessageEvent):
        """兼容“图片在前、文字在后”的消息：确保 /改图 能触发。"""
        msg = (event.message_str or "").strip()
        if self._is_direct_command_message(event, ("改图", "图生图", "修图", "aiedit")):
            return
        prompt = ""
        for name in ("改图", "图生图", "修图", "aiedit"):
            prompt = self._extract_command_arg_anywhere(msg, name)
            if prompt:
                break
        if (
            prompt
            or "/改图" in msg
            or "/图生图" in msg
            or "/修图" in msg
            or "/aiedit" in msg
        ):
            event.should_call_llm(True)
            async for result in self._do_edit(event, prompt, backend=None):
                yield result
            event.stop_event()

    # ==================== Bot 自拍（参考照） ====================

    @filter.command("自拍")
    async def selfie_command(self, event: AstrMessageEvent):
        """使用“自拍参考照”生成 Bot 自拍。

        用法:
        - /自拍 <提示词>
        - 可附带多张参考图（衣服/姿势/场景）作为额外参考
        """
        event.should_call_llm(True)
        prompt = self._extract_extra_prompt(event, "自拍")
        async for result in self._do_selfie(event, prompt, backend=None):
            yield result

    @filter.regex(r"[/!！.。．]自拍(\s|$)", priority=-10)
    async def selfie_regex_fallback(self, event: AstrMessageEvent):
        """兼容“图片在前、文字在后”的消息：确保 /自拍 能触发。"""
        msg = (event.message_str or "").strip()
        # 如果本来就是“首段文本命令”，交给 command handler，避免重复回复
        if self._is_direct_command_message(event, ("自拍",)):
            return
        prompt = self._extract_command_arg_anywhere(msg, "自拍")
        if prompt or "/自拍" in msg or "自拍" in msg:
            async for result in self._do_selfie(event, prompt, backend=None):
                yield result
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
            async for result in self._set_selfie_reference(event):
                yield result
            return

        if action in {"查看", "show", "看"}:
            async for result in self._show_selfie_reference(event):
                yield result
            return

        if action in {"删除", "del", "delete"}:
            async for result in self._delete_selfie_reference(event):
                yield result
            return

        yield event.plain_result("未知操作。用法：/自拍参考 （查看帮助）")

    @filter.regex(r"[/!！.。．]自拍参考(\s|$)", priority=-10)
    async def selfie_reference_regex_fallback(self, event: AstrMessageEvent):
        """兼容“图片在前、文字在后”的消息：确保 /自拍参考 能触发。"""
        msg = (event.message_str or "").strip()
        if self._is_direct_command_message(event, ("自拍参考",)):
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
            async for r in self._set_selfie_reference(event):
                yield r
            event.stop_event()
            return

        if action in {"查看", "show", "看"}:
            async for r in self._show_selfie_reference(event):
                yield r
            event.stop_event()
            return

        if action in {"删除", "del", "delete"}:
            async for r in self._delete_selfie_reference(event):
                yield r
            event.stop_event()
            return

        yield event.plain_result("未知操作。用法：/自拍参考 （查看帮助）")
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
            yield event.plain_result("视频功能已关闭（features.video.enabled=false）")
            return
        arg = self._extract_extra_prompt(event, "视频")
        if not arg:
            yield event.plain_result(
                "用法: /视频 [@provider_id] <提示词> 或 /视频 [@provider_id] <预设名> [额外提示词]"
            )
            return

        provider_override: str | None = None
        if arg.lstrip().startswith("@"):
            first, _, rest = arg.strip().partition(" ")
            provider_override = first.lstrip("@").strip() or None
            arg = rest.strip()
        if not arg:
            yield event.plain_result(
                "用法: /视频 [@provider_id] <提示词> 或 /视频 [@provider_id] <预设名> [额外提示词]"
            )
            return

        preset, prompt = self._parse_video_args(arg)
        presets = self._get_video_presets()
        if preset and preset in presets:
            preset_prompt = presets[preset]
            prompt = f"{preset_prompt}, {prompt}" if prompt else preset_prompt

        user_id = str(event.get_sender_id() or "")
        request_id = f"video_{user_id}"

        if self.debouncer.hit(request_id):
            yield event.plain_result("操作太快了，请稍后再试")
            return

        if not await self._video_begin(user_id):
            yield event.plain_result("你已有一个视频任务正在进行中，请等待完成后再试")
            return

        await mark_processing(event)

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
            yield event.plain_result("视频功能已关闭（features.video.enabled=false）")
            event.stop_event()
            return
        if not arg:
            yield event.plain_result(
                "用法: /视频 [@provider_id] <提示词> 或 /视频 [@provider_id] <预设名> [额外提示词]"
            )
            event.stop_event()
            return

        provider_override: str | None = None
        if arg.lstrip().startswith("@"):
            first, _, rest = arg.strip().partition(" ")
            provider_override = first.lstrip("@").strip() or None
            arg = rest.strip()
        if not arg:
            yield event.plain_result(
                "用法: /视频 [@provider_id] <提示词> 或 /视频 [@provider_id] <预设名> [额外提示词]"
            )
            event.stop_event()
            return

        preset, prompt = self._parse_video_args(arg)
        presets = self._get_video_presets()
        if preset and preset in presets:
            preset_prompt = presets[preset]
            prompt = f"{preset_prompt}, {prompt}" if prompt else preset_prompt

        user_id = str(event.get_sender_id() or "")
        request_id = f"video_{user_id}"

        if self.debouncer.hit(request_id):
            yield event.plain_result("操作太快了，请稍后再试")
            event.stop_event()
            return

        if not await self._video_begin(user_id):
            yield event.plain_result("你已有一个视频任务正在进行中，请等待完成后再试")
            event.stop_event()
            return

        await mark_processing(event)

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
            if isinstance(it, dict) and str(it.get("provider_id") or "").strip():
                chain.append(str(it.get("provider_id") or "").strip())

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
        # 兜底：如果模型误调用了旧工具，但用户其实在要“自拍参考照”，这里自动纠正到自拍逻辑。
        if await self._should_use_selfie_ref(event, prompt):
            return await self.aiimg_generate(
                event,
                prompt=prompt,
                mode="selfie_ref",
                backend="auto",
            )
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
            return event.plain_result("当前仅支持 use_message_images=true（请附带/引用图片后再调用）")
        # 兜底：如果模型误调用了旧工具，但用户其实在要“自拍参考照”，这里自动纠正到自拍逻辑。
        if await self._should_use_selfie_ref(event, prompt):
            return await self.aiimg_generate(
                event,
                prompt=prompt,
                mode="selfie_ref",
                backend=backend,
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

        Args:
            prompt(string): 提示词
            mode(string): auto=自动判断, text=文生图, edit=改图, selfie_ref=参考照自拍
            backend(string): auto=自动选择；也可填 provider_id（你在 WebUI providers 里配置的 id）
            output(string): 输出尺寸/分辨率。例: 2048x2048 或 4K（不同后端支持能力不同，留空用默认）
        """
        prompt = (prompt or "").strip()
        m = (mode or "auto").strip().lower()

        # === TTL 去重检查（防止 ToolLoop 重复调用）===
        message_id = getattr(getattr(event, "message_obj", None), "message_id", "") or ""
        origin = getattr(event, "unified_msg_origin", "") or ""
        if message_id and origin:
            if self.debouncer.llm_tool_is_duplicate(message_id, origin):
                logger.debug(f"[aiimg_generate] 重复调用已拦截: msg_id={message_id}")
                event.set_result(event.plain_result("图片已生成，无需重复操作。"))
                return None

        user_id = event.get_sender_id()
        request_id = f"aiimg_{user_id}"
        if self.debouncer.hit(request_id):
            event.set_result(event.plain_result("操作太快了，请稍后再试"))
            return None

        b_raw = (backend or "auto").strip()
        target_backend = None if b_raw.lower() == "auto" else b_raw

        output = (output or "").strip()
        size = output if output and "x" in output else None
        resolution = output if output and size is None else None

        try:
            await mark_processing(event)

            if m in {"selfie_ref", "selfie", "ref"}:
                selfie_conf = self._get_feature("selfie")
                if not bool(selfie_conf.get("enabled", True)):
                    await mark_failed(event)
                    event.set_result(event.plain_result("自拍功能已关闭（features.selfie.enabled=false）"))
                    return None
                if not bool(selfie_conf.get("llm_tool_enabled", True)):
                    await mark_failed(event)
                    event.set_result(event.plain_result("自拍的 LLM 调用已关闭（features.selfie.llm_tool_enabled=false）"))
                    return None
                image_path = await self._generate_selfie_image(
                    event,
                    prompt,
                    target_backend,
                    size=size,
                    resolution=resolution,
                )
                self._remember_last_image(event, image_path)
                sent = await self._send_image_with_fallback(event, image_path)
                if not sent:
                    await mark_failed(event)
                    event.set_result(event.plain_result("自拍已生成，但发送异常（无需重新生成）。可让主人输入：/重发图片"))
                    return None
                await mark_success(event)
                event.set_result(event.plain_result("自拍已生成并发送。"))
                return None

            # 自动模式：优先识别"自拍"语义 + 已配置参考照
            if m == "auto" and await self._should_use_selfie_ref(event, prompt):
                selfie_conf = self._get_feature("selfie")
                if not bool(selfie_conf.get("enabled", True)):
                    await mark_failed(event)
                    event.set_result(event.plain_result("自拍功能已关闭（features.selfie.enabled=false）"))
                    return None
                if not bool(selfie_conf.get("llm_tool_enabled", True)):
                    await mark_failed(event)
                    event.set_result(event.plain_result("自拍的 LLM 调用已关闭（features.selfie.llm_tool_enabled=false）"))
                    return None
                image_path = await self._generate_selfie_image(
                    event,
                    prompt,
                    target_backend,
                    size=size,
                    resolution=resolution,
                )
                self._remember_last_image(event, image_path)
                sent = await self._send_image_with_fallback(event, image_path)
                if not sent:
                    await mark_failed(event)
                    event.set_result(event.plain_result("自拍已生成，但发送异常（无需重新生成）。可让主人输入：/重发图片"))
                    return None
                await mark_success(event)
                event.set_result(event.plain_result("自拍已生成并发送。"))
                return None

            # 改图：用户消息中有图片（不含头像兜底）或显式指定
            has_msg_images = await self._has_message_images(event)
            if m in {"edit", "img2img", "aiedit"} or (m == "auto" and has_msg_images):
                edit_conf = self._get_feature("edit")
                if not bool(edit_conf.get("enabled", True)):
                    await mark_failed(event)
                    event.set_result(event.plain_result("改图功能已关闭（features.edit.enabled=false）"))
                    return None
                if not bool(edit_conf.get("llm_tool_enabled", True)):
                    await mark_failed(event)
                    event.set_result(event.plain_result(
                        "改图的 LLM 调用已关闭（features.edit.llm_tool_enabled=false）"
                    ))
                    return None
                image_segs = await get_images_from_event(event, include_avatar=False)
                bytes_images = await self._image_segs_to_bytes(image_segs)
                if not bytes_images:
                    await mark_failed(event)
                    event.set_result(event.plain_result("请在消息中附带需要编辑的图片（可发送图片或引用图片）。"))
                    return None

                image_path = await self.edit.edit(
                    prompt=prompt,
                    images=bytes_images,
                    backend=target_backend,
                    size=size,
                    resolution=resolution,
                )
                self._remember_last_image(event, image_path)
                sent = await self._send_image_with_fallback(event, image_path)
                if not sent:
                    await mark_failed(event)
                    event.set_result(event.plain_result("图片已编辑，但发送异常（无需重新生成）。可让主人输入：/重发图片"))
                    return None
                await mark_success(event)
                event.set_result(event.plain_result("图片已编辑并发送。"))
                return None

            # 默认：文生图
            draw_conf = self._get_feature("draw")
            if not bool(draw_conf.get("enabled", True)):
                await mark_failed(event)
                event.set_result(event.plain_result("文生图功能已关闭（features.draw.enabled=false）"))
                return None
            if not bool(draw_conf.get("llm_tool_enabled", True)):
                await mark_failed(event)
                event.set_result(event.plain_result("文生图的 LLM 调用已关闭（features.draw.llm_tool_enabled=false）"))
                return None
            if not prompt:
                prompt = "a selfie photo"

            image_path = await self.draw.generate(
                prompt,
                provider_id=target_backend,
                size=size,
                resolution=resolution,
            )
            self._remember_last_image(event, image_path)
            sent = await self._send_image_with_fallback(event, image_path)
            if not sent:
                await mark_failed(event)
                event.set_result(event.plain_result("图片已生成，但发送异常（无需重新生成）。可让主人输入：/重发图片"))
                return None
            await mark_success(event)
            event.set_result(event.plain_result("图片已生成并发送。"))
            return None

        except Exception as e:
            logger.error(f"[aiimg_generate] 失败: {e}", exc_info=True)
            await mark_failed(event)
            event.set_result(event.plain_result(f"生成失败: {str(e) or type(e).__name__}（本次已停止，请稍后再试或换后端）"))
            return None

    @filter.llm_tool()
    async def grok_generate_video(self, event: AstrMessageEvent, prompt: str):
        """根据用户发送/引用的图片生成视频。

        Args:
            prompt(string): 视频提示词。支持 "预设名 额外提示词"（与 `/视频 预设名 额外提示词` 一致）
        """
        vconf = self._get_feature("video")
        if not bool(vconf.get("enabled", False)):
            return event.plain_result("视频功能已关闭（features.video.enabled=false）")
        if not bool(vconf.get("llm_tool_enabled", True)):
            return event.plain_result("视频的 LLM 调用已关闭（features.video.llm_tool_enabled=false）")

        arg = (prompt or "").strip()
        if not arg:
            return event.plain_result("需要提供视频提示词")

        provider_override: str | None = None
        if arg.lstrip().startswith("@"):
            first, _, rest = arg.strip().partition(" ")
            provider_override = first.lstrip("@").strip() or None
            arg = rest.strip()
        if not arg:
            return event.plain_result("需要提供视频提示词")

        preset, extra_prompt = self._parse_video_args(arg)
        presets = self._get_video_presets()
        if preset and preset in presets:
            preset_prompt = presets[preset]
            extra_prompt = (
                f"{preset_prompt}, {extra_prompt}" if extra_prompt else preset_prompt
            )

        user_id = str(event.get_sender_id() or "")
        request_id = f"video_{user_id}"

        if self.debouncer.hit(request_id):
            return event.plain_result("操作太快了，请稍后再试")

        if not await self._video_begin(user_id):
            return event.plain_result("你已有一个视频任务正在进行中，请等待完成后再试")

        await mark_processing(event)

        try:
            task = asyncio.create_task(
                self._async_generate_video(
                    event, extra_prompt, user_id, provider_id=provider_override
                )
            )
        except Exception:
            await self._video_end(user_id)
            await mark_failed(event)
            return event.plain_result("")

        self._video_tasks.add(task)
        task.add_done_callback(lambda t: self._video_tasks.discard(t))

        return event.plain_result("视频正在生成中，请稍候...")

    # ==================== 内部方法 ====================

    def _get_feature(self, name: str) -> dict:
        feats = self.config.get("features", {}) if isinstance(self.config, dict) else {}
        feats = feats if isinstance(feats, dict) else {}
        conf = feats.get(name, {})
        return conf if isinstance(conf, dict) else {}

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
            if not isinstance(item, dict):
                continue
            pid = str(item.get("provider_id") or "").strip()
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
        """单用户并发保护：成功占用返回 True，否则 False"""
        user_id = str(user_id or "")
        async with self._video_lock:
            if user_id in self._video_in_progress:
                return False
            self._video_in_progress.add(user_id)
            return True

    async def _video_end(self, user_id: str) -> None:
        user_id = str(user_id or "")
        async with self._video_lock:
            self._video_in_progress.discard(user_id)

    async def _send_video_result(self, event: AstrMessageEvent, video_url: str) -> None:
        vconf = self._get_feature("video")
        mode = str(vconf.get("send_mode", "auto")).strip().lower()
        if mode not in {"auto", "url", "file"}:
            mode = "auto"

        send_timeout = int(vconf.get("send_timeout_seconds", 90) or 90)
        send_timeout = max(10, min(send_timeout, 300))

        # 1) URL 发送（优先）
        if mode in {"auto", "url"}:
            try:
                await asyncio.wait_for(
                    event.send(event.chain_result([Video.fromURL(video_url)])),
                    timeout=float(send_timeout),
                )
                return
            except Exception as e:
                if mode == "url":
                    raise
                logger.warning(f"[视频] URL 发送失败，尝试本地文件降级: {e}")

        # 2) 下载 + 本地文件发送
        download_timeout = int(vconf.get("download_timeout_seconds", 300) or 300)
        download_timeout = max(1, min(download_timeout, 3600))

        if mode in {"auto", "file"}:
            try:
                video_path = await self.videomgr.download_video(
                    video_url, timeout_seconds=download_timeout
                )
                await asyncio.wait_for(
                    event.send(
                        event.chain_result([Video.fromFileSystem(str(video_path))])
                    ),
                    timeout=float(send_timeout),
                )
                return
            except Exception as e:
                if mode == "file":
                    raise
                logger.warning(f"[视频] 本地文件发送失败，回退为文本链接: {e}")

        # 3) 最终兜底：发出可点击链接
        await event.send(event.plain_result(video_url))

    async def _async_generate_video(
        self,
        event: AstrMessageEvent,
        prompt: str,
        user_id: str,
        *,
        provider_id: str | None = None,
    ) -> None:
        try:
            image_segs = await get_images_from_event(event, include_avatar=False)
            if not image_segs:
                await event.send(
                    event.plain_result("请发送或引用一张图片后再使用 /视频。")
                )
                await mark_failed(event)
                return

            image_bytes: bytes | None = None
            for i, seg in enumerate(image_segs):
                try:
                    b64 = await seg.convert_to_base64()
                    image_bytes = base64.b64decode(b64)
                    break
                except Exception as e:
                    logger.warning(f"[视频] 图片 {i + 1} 转换失败，跳过: {e}")

            if not image_bytes:
                await event.send(event.plain_result("图片读取失败，请更换图片后重试。"))
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
                    video_url = await backend.generate_video_url(
                        prompt=prompt, image_bytes=image_bytes
                    )
                    used_pid = pid
                    break
                except Exception as e:
                    last_error = e
                    logger.warning("[视频] Provider=%s 失败: %s", pid, e)

            if not video_url:
                raise RuntimeError(f"视频生成失败: {last_error}") from last_error

            await self._send_video_result(event, video_url)
            await mark_success(event)

            t_end = time.perf_counter()
            name = used_pid or "video"
            logger.info(f"[视频] 完成: provider={name}, 耗时={t_end - t_start:.2f}s")

        except Exception as e:
            logger.error(f"[视频] 失败: {e}", exc_info=True)
            await mark_failed(event)
            try:
                await event.send(
                    event.plain_result(f"视频生成失败: {str(e) or type(e).__name__}")
                )
            except Exception:
                pass
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
        user_id = event.get_sender_id()
        request_id = f"edit_{user_id}"

        # 防抖
        if self.debouncer.hit(request_id):
            await event.send(event.plain_result("操作太快了，请稍后再试"))
            return

        p = (prompt or "").strip()
        if p.startswith("@"):
            first, _, rest = p.partition(" ")
            backend = first.lstrip("@").strip() or backend
            prompt = rest.strip()

        # 获取图片
        image_segs = await get_images_from_event(event, include_avatar=False)
        logger.debug(f"[改图] 获取到 {len(image_segs)} 个图片段")
        if not image_segs:
            await event.send(
                event.plain_result(
                    "请发送或引用图片！\n用法: 发送图片 + 命令\n或: 引用图片消息 + 命令"
                )
            )
            return

        bytes_images: list[bytes] = []
        for i, seg in enumerate(image_segs):
            try:
                logger.debug(f"[改图] 转换图片 {i + 1}/{len(image_segs)}...")
                b64 = await seg.convert_to_base64()
                bytes_images.append(base64.b64decode(b64))
                logger.debug(
                    f"[改图] 图片 {i + 1} 转换成功, 大小={len(bytes_images[-1])} bytes"
                )
            except Exception as e:
                logger.warning(f"[改图] 图片 {i + 1} 转换失败，跳过: {e}")

        if not bytes_images:
            await event.send(event.plain_result("图片处理失败，请重试"))
            return

        # 标记处理中
        await mark_processing(event)

        try:
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
                await event.send(
                    event.plain_result(
                        "图片已生成，但发送失败（可能是平台临时异常）。可稍后使用：/重发图片"
                    )
                )
                return

            # 标记成功
            await mark_success(event)
            display_name = preset or (prompt[:20] if prompt else "改图")
            logger.info(f"[改图] 完成: {display_name}..., 耗时={t_end - t_start:.2f}s")

        except Exception as e:
            logger.error(f"[改图] 失败: {e}", exc_info=True)
            await mark_failed(event)
            await event.send(event.plain_result(f"改图失败: {str(e)}"))

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
        user_id = event.get_sender_id()
        request_id = f"edit_{user_id}"

        # 防抖
        if self.debouncer.hit(request_id):
            yield event.plain_result("操作太快了，请稍后再试")
            return

        # Optional provider override: "/aiedit @provider_id <prompt>"
        p = (prompt or "").strip()
        if p.startswith("@"):
            first, _, rest = p.partition(" ")
            backend = first.lstrip("@").strip() or backend
            prompt = rest.strip()

        # 预设自动检测: prompt 完全匹配预设名时，自动转为预设
        if not preset and prompt:
            prompt_stripped = prompt.strip()
            preset_names = self.edit.get_preset_names()
            if prompt_stripped in preset_names:
                preset = prompt_stripped
                prompt = ""  # 清空 prompt，使用预设的提示词
                logger.debug(f"[改图] 自动匹配预设: {preset}")

        # 获取图片
        image_segs = await get_images_from_event(event, include_avatar=False)
        if not image_segs:
            yield event.plain_result(
                "请发送或引用图片！\n"
                "用法: 发送图片 + /aiedit <提示词>\n"
                "或: 引用图片消息 + /aiedit <提示词>"
            )
            return

        bytes_images: list[bytes] = []
        for seg in image_segs:
            try:
                b64 = await seg.convert_to_base64()
                bytes_images.append(base64.b64decode(b64))
            except Exception as e:
                logger.warning(f"[改图] 图片转换失败，跳过: {e}")

        if not bytes_images:
            yield event.plain_result("图片处理失败，请重试")
            return

        # 标记处理中
        await mark_processing(event)

        try:
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
                yield event.plain_result(
                    "图片已生成，但发送失败（可能是平台临时异常）。可稍后使用：/重发图片"
                )
                return

            # 标记成功
            await mark_success(event)
            display_name = preset or (prompt[:20] if prompt else "改图")
            logger.info(f"[改图] 完成: {display_name}..., 耗时={t_end - t_start:.2f}s")

        except Exception as e:
            logger.error(f"[改图] 失败: {e}")
            await mark_failed(event)
            yield event.plain_result(f"改图失败: {str(e)}")

    # ==================== 自拍参考照：内部实现 ====================

    def _get_selfie_conf(self) -> dict:
        return self._get_feature("selfie")

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
                out.append(base64.b64decode(b64))
            except Exception as e:
                logger.warning(f"[图片] 转换失败，跳过: {e}")
        return out

    async def _has_message_images(self, event: AstrMessageEvent) -> bool:
        """仅检测用户消息/引用里的图片（不含头像兜底）。"""
        image_segs = await get_images_from_event(event, include_avatar=False)
        return bool(image_segs)

    def _is_selfie_prompt(self, prompt: str) -> bool:
        text = (prompt or "").strip()
        if not text:
            return True  # 空提示词时，/自拍 默认走自拍逻辑
        lowered = text.lower()
        if "自拍" in text or "selfie" in lowered:
            return True
        if any(
            k in text for k in ("来一张你", "来张你", "你来一张", "你来张", "看看你")
        ):
            return True
        return False

    async def _should_use_selfie_ref(
        self, event: AstrMessageEvent, prompt: str
    ) -> bool:
        if not self._is_selfie_prompt(prompt):
            return False
        paths, _ = await self._get_selfie_reference_paths(event)
        return bool(paths)

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

    def _merge_selfie_chain_with_edit_chain(self, selfie_chain: list[dict]) -> list[dict]:
        """将自拍链路与改图链路合并（自拍优先，去重 provider_id）。"""
        merged: list[dict] = []
        seen: set[str] = set()

        def append_unique(items: list[dict]) -> None:
            for item in items:
                if not isinstance(item, dict):
                    continue
                pid = str(item.get("provider_id") or "").strip()
                if not pid or pid in seen:
                    continue
                merged.append(dict(item))
                seen.add(pid)

        append_unique(selfie_chain)

        edit_chain_raw = self._get_feature("edit").get("chain", [])
        if isinstance(edit_chain_raw, list):
            append_unique([x for x in edit_chain_raw if isinstance(x, dict)])

        return merged

    async def _generate_selfie_image(
        self,
        event: AstrMessageEvent,
        prompt: str,
        backend: str | None,
        *,
        size: str | None = None,
        resolution: str | None = None,
    ) -> Path:
        conf = self._get_selfie_conf()
        if conf.get("enabled", True) is False:
            raise RuntimeError("自拍功能已关闭（features.selfie.enabled=false）")

        # 1) 读取参考照（WebUI 优先，其次命令设置的 store）
        ref_paths, _ = await self._get_selfie_reference_paths(event)
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

        final_prompt = self._build_selfie_prompt(prompt, extra_refs=len(extra_bytes))

        chain_override: list[dict] | None = None
        use_edit_chain = bool(conf.get("use_edit_chain_when_empty", True))
        raw_chain = conf.get("chain", [])
        if isinstance(raw_chain, list):
            chain_items = [
                x
                for x in raw_chain
                if isinstance(x, dict) and str(x.get("provider_id") or "").strip()
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
                chain_override = self._merge_selfie_chain_with_edit_chain(chain_override)

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

        return await self.edit.edit(
            prompt=final_prompt,
            images=images,
            backend=backend,
            task_types=gitee_task_types,
            size=size,
            resolution=resolution,
            default_output=default_output,
            chain_override=chain_override,
        )

    async def _do_selfie(
        self,
        event: AstrMessageEvent,
        prompt: str,
        backend: str | None = None,
    ):
        """指令 /自拍 执行入口（generator 版本）。"""
        user_id = event.get_sender_id()
        request_id = f"selfie_{user_id}"

        if self.debouncer.hit(request_id):
            yield event.plain_result("操作太快了，请稍后再试")
            return

        p = (prompt or "").strip()
        if p.startswith("@"):
            first, _, rest = p.partition(" ")
            backend = first.lstrip("@").strip() or backend
            prompt = rest.strip()

        await mark_processing(event)

        try:
            image_path = await self._generate_selfie_image(event, prompt, backend)
            self._remember_last_image(event, image_path)
            sent = await self._send_image_with_fallback(event, image_path)
            if not sent:
                await mark_failed(event)
                yield event.plain_result(
                    "自拍已生成，但发送失败（可能是平台临时异常）。可稍后使用：/重发图片"
                )
                return
            await mark_success(event)
        except Exception as e:
            logger.error(f"[自拍] 失败: {e}", exc_info=True)
            await mark_failed(event)
            yield event.plain_result(f"自拍失败: {str(e) or type(e).__name__}")

    async def _set_selfie_reference(self, event: AstrMessageEvent):
        image_segs = await get_images_from_event(event, include_avatar=False)
        if not image_segs:
            yield event.plain_result(
                "请发送或引用一张清晰的人像参考图，再发送：/自拍参考 设置"
            )
            return

        bytes_images = await self._image_segs_to_bytes(image_segs)
        if not bytes_images:
            yield event.plain_result("参考图处理失败，请重试")
            return

        # 限制数量，避免一次塞太多
        max_images = 8
        bytes_images = bytes_images[:max_images]

        store_key = self._get_selfie_ref_store_key(event)
        try:
            count = await self.refs.set(store_key, bytes_images)
        except Exception as e:
            yield event.plain_result(f"保存参考照失败: {str(e) or type(e).__name__}")
            return

        webui_paths = self._get_config_selfie_reference_paths()
        note = ""
        if webui_paths:
            note = "\n⚠️ 检测到 WebUI 已配置 features.selfie.reference_images，运行时会优先使用 WebUI 的参考照。"

        yield event.plain_result(
            f"✅ 已保存 {count} 张自拍参考照。\n"
            f"现在可用：/自拍 <提示词> 生成自拍。{note}"
        )

    async def _show_selfie_reference(self, event: AstrMessageEvent):
        paths, source = await self._get_selfie_reference_paths(event)
        if not paths:
            yield event.plain_result(
                "当前没有自拍参考照。\n"
                "请先：发送图片 + /自拍参考 设置\n"
                "或在 WebUI 配置 features.selfie.reference_images 上传。"
            )
            return

        # 最多回显 5 张，避免刷屏
        max_show = 5
        show_paths = paths[:max_show]
        yield event.chain_result([Image.fromFileSystem(str(p)) for p in show_paths])
        yield event.plain_result(
            f"📌 当前自拍参考照来源：{source}，共 {len(paths)} 张（已展示 {len(show_paths)} 张）"
        )

    async def _delete_selfie_reference(self, event: AstrMessageEvent):
        store_key = self._get_selfie_ref_store_key(event)
        deleted = await self.refs.delete(store_key)

        webui_paths = self._get_config_selfie_reference_paths()
        if webui_paths:
            yield event.plain_result(
                "已删除命令保存的自拍参考照。\n"
                "⚠️ 但你仍配置了 WebUI 的 features.selfie.reference_images（运行时优先使用它）。如需彻底删除，请在 WebUI 中清空该配置。"
            )
            return

        if deleted:
            yield event.plain_result("✅ 已删除自拍参考照。")
        else:
            yield event.plain_result("当前没有已保存的自拍参考照。")
