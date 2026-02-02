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
from astrbot.api.message_components import Image, Video
from astrbot.api.star import Context, Star, StarTools

from .core.debouncer import Debouncer
from .core.draw_service import ImageDrawService
from .core.edit_router import EditRouter
from .core.emoji_feedback import mark_failed, mark_processing, mark_success
from .core.grok_video_service import GrokVideoService
from .core.image_manager import ImageManager
from .core.nanobanana import NanoBananaService
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

    async def initialize(self):
        self.debouncer = Debouncer(self.config)
        self.imgr = ImageManager(self.config, self.data_dir)
        self.draw = ImageDrawService(self.config, self.imgr, self.data_dir)
        self.edit = EditRouter(self.config, self.imgr, self.data_dir)
        self.nb = NanoBananaService(self.config, self.imgr)
        self.refs = ReferenceStore(self.data_dir)
        self.videomgr = VideoManager(self.config, self.data_dir)
        self.video = GrokVideoService(self.config)

        self._video_lock = asyncio.Lock()
        self._video_in_progress: set[str] = set()
        self._video_tasks: set[asyncio.Task] = set()

        # 动态注册预设命令 (方案C: /手办化 直接触发)
        self._register_preset_commands()

        logger.info(
            f"[GiteeAIImage] 插件初始化完成: "
            f"改图后端={self.edit.get_available_backends()}, "
            f"改图预设={len(self.edit.get_preset_names())}个, "
            f"视频启用={self.video.enabled}, "
            f"视频预设={len(self.video.get_preset_names())}个"
        )

    def _register_preset_commands(self):
        """动态注册预设命令

        为每个预设创建对应的命令，如 /手办化, /Q版化 等
        同时支持 /g手办化 (强制Gemini) 和 /q手办化 (强制千问)
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
            star_name="astrbot_plugin_gitee",
            command_name=preset_name,
            desc=f"预设改图: {preset_name}",
            priority=5,
            awaitable=preset_handler,
        )

        # Gemini 强制命令: /g手办化
        async def preset_gemini_handler(event: AstrMessageEvent):
            extra_prompt = self._extract_extra_prompt(event, f"g{preset_name}")
            await self._do_edit_direct(
                event, extra_prompt, backend="gemini", preset=preset_name
            )

        preset_gemini_handler.__name__ = f"preset_g_{preset_name}"
        preset_gemini_handler.__doc__ = f"预设改图(Gemini): {preset_name} [额外提示词]"

        self.context.register_commands(
            star_name="astrbot_plugin_gitee",
            command_name=f"g{preset_name}",
            desc=f"预设改图(Gemini): {preset_name}",
            priority=5,
            awaitable=preset_gemini_handler,
        )

        # 千问强制命令: /q手办化
        async def preset_qwen_handler(event: AstrMessageEvent):
            extra_prompt = self._extract_extra_prompt(event, f"q{preset_name}")
            await self._do_edit_direct(
                event, extra_prompt, backend="gitee", preset=preset_name
            )

        preset_qwen_handler.__name__ = f"preset_q_{preset_name}"
        preset_qwen_handler.__doc__ = f"预设改图(千问): {preset_name} [额外提示词]"

        self.context.register_commands(
            star_name="astrbot_plugin_gitee",
            command_name=f"q{preset_name}",
            desc=f"预设改图(千问): {preset_name}",
            priority=5,
            awaitable=preset_qwen_handler,
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

        用法: /aiimg <提示词> [比例]
        示例: /aiimg 一个女孩 9:16
        支持比例: 1:1, 4:3, 3:4, 3:2, 2:3, 16:9, 9:16
        """
        event.should_call_llm(True)
        # 解析参数
        arg = event.message_str.partition(" ")[2]
        if not arg:
            yield event.plain_result("请提供提示词！使用方法：/aiimg <提示词> [比例]")
            return
        prompt, ratio = arg, "1:1"
        *parts, last = arg.rsplit(maxsplit=1)
        if last in self.SUPPORTED_RATIOS:
            prompt, ratio = " ".join(parts), last

        size = self.SUPPORTED_RATIOS[ratio][0]

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
            image_path = await self.draw.generate(prompt, size=size)
            t_end = time.perf_counter()

            # 发送结果图片
            yield event.chain_result(
                [
                    Image.fromFileSystem(str(image_path)),
                ]
            )

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

    @filter.command("gedit", alias={"g改图"})
    async def edit_image_gemini(self, event: AstrMessageEvent, prompt: str):
        """使用 Gemini 改图

        用法: /gedit <提示词>
        """
        event.should_call_llm(True)
        async for result in self._do_edit(event, prompt, backend="gemini"):
            yield result

    @filter.command("qedit", alias={"q改图"})
    async def edit_image_qwen(self, event: AstrMessageEvent, prompt: str):
        """使用 Gitee 千问改图

        用法: /qedit <提示词>
        """
        event.should_call_llm(True)
        async for result in self._do_edit(event, prompt, backend="gitee"):
            yield result

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
        # 如果本来就是以 /自拍 开头，交给 command handler，避免重复回复
        if msg and msg[0] in "/!！.。．" and msg[1:].startswith("自拍"):
            return
        prompt = self._extract_command_arg_anywhere(msg, "自拍")
        if prompt or "/自拍" in msg or "自拍" in msg:
            async for result in self._do_selfie(event, prompt, backend=None):
                yield result
            event.stop_event()

    @filter.command("g自拍")
    async def selfie_command_gemini(self, event: AstrMessageEvent):
        """强制使用 Gemini 生成自拍：/g自拍 <提示词>"""
        event.should_call_llm(True)
        prompt = self._extract_extra_prompt(event, "g自拍")
        async for result in self._do_selfie(event, prompt, backend="gemini"):
            yield result

    @filter.command("q自拍")
    async def selfie_command_gitee(self, event: AstrMessageEvent):
        """强制使用千问生成自拍：/q自拍 <提示词>"""
        event.should_call_llm(True)
        prompt = self._extract_extra_prompt(event, "q自拍")
        async for result in self._do_selfie(event, prompt, backend="gitee"):
            yield result

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
        if msg and msg[0] in "/!！.。．" and msg[1:].startswith("自拍参考"):
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
        - /视频 <提示词>
        - /视频 <预设名> [额外提示词]
        """
        event.should_call_llm(True)
        arg = self._extract_extra_prompt(event, "视频")
        if not arg:
            yield event.plain_result(
                "用法: /视频 <提示词> 或 /视频 <预设名> [额外提示词]"
            )
            return

        preset, prompt = self._parse_video_args(arg)

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
                self._async_generate_video(event, prompt, preset, user_id)
            )
        except Exception:
            await self._video_end(user_id)
            await mark_failed(event)
            return

        self._video_tasks.add(task)
        task.add_done_callback(lambda t: self._video_tasks.discard(t))
        return

    @filter.command("视频预设列表")
    async def list_video_presets(self, event: AstrMessageEvent):
        """列出所有可用视频预设"""
        event.should_call_llm(True)
        presets = self.video.get_preset_names()
        if not presets:
            yield event.plain_result(
                "📋 视频预设列表\n暂无预设（请在配置 video.presets 中添加）"
            )
            return

        msg = "📋 视频预设列表\n"
        for name in presets:
            msg += f"- {name}\n"
        msg += "\n用法: /视频 <预设名> [额外提示词]"
        yield event.plain_result(msg)

    # ==================== 管理命令 ====================

    @filter.command("预设列表")
    async def list_presets(self, event: AstrMessageEvent):
        """列出所有可用预设"""
        event.should_call_llm(True)
        presets = self.edit.get_preset_names()
        backends = self.edit.get_available_backends()
        default = self.config.get("edit", {}).get("default_backend", "gemini")

        if not presets:
            msg = "📋 改图预设列表\n"
            msg += "━━━━━━━━━━━━━━\n"
            msg += f"🔧 可用后端: {', '.join(backends)}\n"
            msg += f"⭐ 默认后端: {default}\n"
            msg += "━━━━━━━━━━━━━━\n"
            msg += "📌 暂无预设\n"
            msg += "━━━━━━━━━━━━━━\n"
            msg += "💡 在配置文件 edit.presets 中添加:\n"
            msg += '  格式: "触发词:英文提示词"'
        else:
            msg = "📋 改图预设列表\n"
            msg += "━━━━━━━━━━━━━━\n"
            msg += f"🔧 可用后端: {', '.join(backends)}\n"
            msg += f"⭐ 默认后端: {default}\n"
            msg += "━━━━━━━━━━━━━━\n"
            msg += "📌 预设:\n"
            for name in presets:
                msg += f"  • {name}\n"
        msg += "━━━━━━━━━━━━━━\n"
        msg += "💡 用法: /aiedit <提示词> [图片]"

        yield event.plain_result(msg)

    @filter.command("改图帮助")
    async def edit_help(self, event: AstrMessageEvent):
        """显示改图帮助"""
        event.should_call_llm(True)
        msg = """🎨 改图功能帮助

━━ 基础命令 ━━
/aiedit <提示词>  使用默认后端
/gedit <提示词>   强制 Gemini (4K)
/qedit <提示词>   强制千问

━━ 使用方式 ━━
1. 发送图片 + 命令
2. 引用图片消息 + 命令

━━ 后端说明 ━━
Gemini: 4K高清，效果好，需代理
千问: 国内直连，速度快，效果稳定

━━ 自定义预设 ━━
在配置 edit.presets 中添加:
格式: "触发词:英文提示词"
示例: "手办化:Transform into figurine style" """

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
            backend(string): auto=自动选择, gemini=Gemini, gitee=千问
        """
        if not use_message_images:
            return "当前仅支持 use_message_images=true（请附带/引用图片后再调用）"
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
        - 用户发送/引用了图片，并要求“改图/换背景/换风格/修图/换衣服”等：用 mode=edit（或 mode=auto）
        - 用户要求“bot 自拍/来一张你自己的自拍”，且已设置自拍参考照：用 mode=selfie_ref（或 mode=auto）
        - 纯文生图（用户没有给图片）：用 mode=text（或 mode=auto）

        Args:
            prompt(string): 提示词
            mode(string): auto=自动判断, text=文生图, edit=改图, selfie_ref=参考照自拍
            backend(string): auto=自动选择；也可填服务商别名（grok/gemini/gitee/jimeng/openai_compat 等）
            output(string): 输出尺寸/分辨率。例: 2048x2048 或 4K（不同后端支持能力不同，留空用默认）
        """
        prompt = (prompt or "").strip()
        m = (mode or "auto").strip().lower()

        user_id = event.get_sender_id()
        request_id = f"aiimg_{user_id}"
        if self.debouncer.hit(request_id):
            return "操作太快了，请稍后再试"

        b_raw = (backend or "auto").strip()
        target_backend = None if b_raw.lower() == "auto" else b_raw

        output = (output or "").strip()
        size = output if output and "x" in output else None
        resolution = output if output and size is None else None

        try:
            await mark_processing(event)

            if m in {"selfie_ref", "selfie", "ref"}:
                await self._do_selfie_llm(
                    event,
                    prompt=prompt,
                    backend=target_backend,
                    size=size,
                    resolution=resolution,
                )
                await mark_success(event)
                return "自拍已生成并发送。"

            # 自动模式：优先识别“自拍”语义 + 已配置参考照
            if m == "auto" and await self._should_use_selfie_ref(event, prompt):
                await self._do_selfie_llm(
                    event,
                    prompt=prompt,
                    backend=target_backend,
                    size=size,
                    resolution=resolution,
                )
                await mark_success(event)
                return "自拍已生成并发送。"

            # 改图：用户消息中有图片（不含头像兜底）或显式指定
            has_msg_images = await self._has_message_images(event)
            if m in {"edit", "img2img", "aiedit"} or (m == "auto" and has_msg_images):
                image_segs = await get_images_from_event(event, include_avatar=True)
                bytes_images = await self._image_segs_to_bytes(image_segs)
                if not bytes_images:
                    await mark_failed(event)
                    return "请在消息中附带需要编辑的图片（可发送图片或引用图片）。"

                image_path = await self.edit.edit(
                    prompt=prompt,
                    images=bytes_images,
                    backend=target_backend,
                    size=size,
                    resolution=resolution,
                )
                await event.send(
                    event.chain_result([Image.fromFileSystem(str(image_path))])
                )
                await mark_success(event)
                return "图片已编辑并发送。"

            # 默认：文生图
            if not prompt:
                prompt = "a selfie photo"

            image_path = await self.draw.generate(
                prompt,
                provider_id=target_backend,
                size=size,
                resolution=resolution,
            )
            await event.send(
                event.chain_result([Image.fromFileSystem(str(image_path))])
            )
            await mark_success(event)
            return "图片已生成并发送。"

        except Exception as e:
            logger.error(f"[aiimg_generate] 失败: {e}", exc_info=True)
            await mark_failed(event)
            return f"生成失败: {str(e) or type(e).__name__}"

    @filter.llm_tool()
    async def grok_generate_video(self, event: AstrMessageEvent, prompt: str):
        """根据用户发送/引用的图片生成视频。

        Args:
            prompt(string): 视频提示词。支持 "预设名 额外提示词"（与 `/视频 预设名 额外提示词` 一致）
        """
        arg = (prompt or "").strip()
        if not arg:
            return "需要提供视频提示词"

        preset, extra_prompt = self._parse_video_args(arg)

        user_id = str(event.get_sender_id() or "")
        request_id = f"video_{user_id}"

        if self.debouncer.hit(request_id):
            return "操作太快了，请稍后再试"

        if not await self._video_begin(user_id):
            return "你已有一个视频任务正在进行中，请等待完成后再试"

        await mark_processing(event)

        try:
            task = asyncio.create_task(
                self._async_generate_video(event, extra_prompt, preset, user_id)
            )
        except Exception:
            await self._video_end(user_id)
            await mark_failed(event)
            return ""

        self._video_tasks.add(task)
        task.add_done_callback(lambda t: self._video_tasks.discard(t))

        return ""

    # ==================== 内部方法 ====================

    def _parse_video_args(self, text: str) -> tuple[str | None, str]:
        """解析 /视频 参数，返回 (preset, prompt)

        - 当第一个 token 命中预设名时：preset=该 token, prompt=剩余内容
        - 否则：preset=None, prompt=text
        """
        text = (text or "").strip()
        if not text:
            return None, ""

        first, _, rest = text.partition(" ")
        if first and first in getattr(self.video, "presets", {}):
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
        mode = (
            str(self.config.get("video", {}).get("send_mode", "auto")).strip().lower()
        )
        if mode not in {"auto", "url", "file"}:
            mode = "auto"

        send_timeout = int(
            self.config.get("video", {}).get("send_timeout_seconds", 90) or 90
        )
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
        download_timeout = int(
            self.config.get("video", {}).get(
                "download_timeout_seconds", self.video.timeout_seconds
            )
            or self.video.timeout_seconds
        )
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
        preset: str | None,
        user_id: str,
    ) -> None:
        try:
            image_segs = await get_images_from_event(event)
            if not image_segs:
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
                await mark_failed(event)
                return

            t_start = time.perf_counter()
            video_url = await self.video.generate_video_url(
                prompt=prompt,
                image_bytes=image_bytes,
                preset=preset,
            )
            t_end = time.perf_counter()

            await self._send_video_result(event, video_url)
            await mark_success(event)

            display_name = preset or (prompt[:20] if prompt else "视频")
            logger.info(f"[视频] 完成: {display_name}..., 耗时={t_end - t_start:.2f}s")

        except Exception as e:
            logger.error(f"[视频] 失败: {e}", exc_info=True)
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
        user_id = event.get_sender_id()
        request_id = f"edit_{user_id}"

        # 防抖
        if self.debouncer.hit(request_id):
            await event.send(event.plain_result("操作太快了，请稍后再试"))
            return

        # 获取图片
        image_segs = await get_images_from_event(event)
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

            # 发送结果图片
            await event.send(
                event.chain_result(
                    [
                        Image.fromFileSystem(str(image_path)),
                    ]
                )
            )

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

        # 预设自动检测: prompt 完全匹配预设名时，自动转为预设
        if not preset and prompt:
            prompt_stripped = prompt.strip()
            preset_names = self.edit.get_preset_names()
            if prompt_stripped in preset_names:
                preset = prompt_stripped
                prompt = ""  # 清空 prompt，使用预设的提示词
                logger.debug(f"[改图] 自动匹配预设: {preset}")

        # 获取图片
        image_segs = await get_images_from_event(event)
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

            # 发送结果图片
            yield event.chain_result(
                [
                    Image.fromFileSystem(str(image_path)),
                ]
            )

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
        conf = self.config.get("selfie", {}) if isinstance(self.config, dict) else {}
        return conf if isinstance(conf, dict) else {}

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
            raise RuntimeError("自拍功能已关闭（selfie.enabled=false）")

        # 1) 读取参考照（WebUI 优先，其次命令设置的 store）
        ref_paths, _ = await self._get_selfie_reference_paths(event)
        ref_images = await self._read_paths_bytes(ref_paths)
        if not ref_images:
            raise RuntimeError(
                "未设置自拍参考照。请先：发送图片 + /自拍参考 设置，或在 WebUI 配置 selfie.reference_images 上传。"
            )

        # 2) 读取额外参考图（衣服/姿势/场景）
        extra_segs = await get_images_from_event(event, include_avatar=False)
        extra_bytes = await self._image_segs_to_bytes(extra_segs)

        # 3) 拼接输入图：参考照在前
        images = [*ref_images, *extra_bytes]

        final_prompt = self._build_selfie_prompt(prompt, extra_refs=len(extra_bytes))

        prefer_provider = str(conf.get("prefer_provider", "auto") or "auto").strip()
        if backend is None and prefer_provider and prefer_provider.lower() != "auto":
            backend = prefer_provider

        # 4) 千问后端可选 task_types（仅对 gitee 生效）
        task_types = conf.get("gitee_task_types")
        if isinstance(task_types, list) and task_types:
            gitee_task_types = [str(x).strip() for x in task_types if str(x).strip()]
        else:
            gitee_task_types = ["id", "background", "style"]

        return await self.edit.edit(
            prompt=final_prompt,
            images=images,
            backend=backend,
            task_types=gitee_task_types,
            size=size,
            resolution=resolution,
        )

    async def _do_selfie_llm(
        self,
        event: AstrMessageEvent,
        prompt: str,
        backend: str | None,
        *,
        size: str | None = None,
        resolution: str | None = None,
    ) -> None:
        image_path = await self._generate_selfie_image(
            event,
            prompt,
            backend,
            size=size,
            resolution=resolution,
        )
        await event.send(event.chain_result([Image.fromFileSystem(str(image_path))]))

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

        await mark_processing(event)

        try:
            image_path = await self._generate_selfie_image(event, prompt, backend)
            yield event.chain_result([Image.fromFileSystem(str(image_path))])
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
            note = "\n⚠️ 检测到 WebUI 已配置 selfie.reference_images，运行时会优先使用 WebUI 的参考照。"

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
                "或在 WebUI 配置 selfie.reference_images 上传。"
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
                "⚠️ 但你仍配置了 WebUI 的 selfie.reference_images（运行时优先使用它）。如需彻底删除，请在 WebUI 中清空该配置。"
            )
            return

        if deleted:
            yield event.plain_result("✅ 已删除自拍参考照。")
        else:
            yield event.plain_result("当前没有已保存的自拍参考照。")
