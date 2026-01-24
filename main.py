"""
Gitee AI 图像生成插件

功能:
- 文生图 (z-image-turbo)
- 图生图/改图 (Gemini / Gitee 千问，可切换)
- 视频生成 (Grok imagine, 参考图 + 提示词)
- 预设提示词
- 智能降级
"""

import asyncio
import base64
import time

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
from .core.video_manager import VideoManager
from .core.utils import close_session, get_images_from_event


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
        self.draw = ImageDrawService(self.config, self.imgr)
        self.edit = EditRouter(self.config, self.imgr)
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
            await self._do_edit_direct(event, extra_prompt, backend="gemini", preset=preset_name)

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
            await self._do_edit_direct(event, extra_prompt, backend="gitee", preset=preset_name)

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
            msg = msg[len(command_name):]
        # 清理多余空格
        return msg.strip()

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
        await close_session()  # 关闭 utils.py 的 HTTP 会话

    # ==================== 文生图 ====================

    @filter.llm_tool()
    async def gitee_draw_image(self, event: AstrMessageEvent, prompt: str):
        """根据提示词生成图片。

        Args:
            prompt(string): 图片提示词，需要包含主体、场景、风格等描述
        """
        if not prompt:
            return "需提供提示词prompt"

        user_id = event.get_sender_id()
        request_id = f"generate_{user_id}"

        if self.debouncer.hit(request_id):
            return "操作太快了，请稍后再试。"

        try:
            t_start = time.perf_counter()
            image_path = await self.draw.generate(prompt)
            t_end = time.perf_counter()

            await event.send(event.chain_result([Image.fromFileSystem(str(image_path))]))
            logger.info(f"[文生图] 完成: {prompt[:30]}..., 耗时={t_end - t_start:.2f}s")
            return f"图片已生成并发送。Prompt: {prompt}"

        except Exception as e:
            logger.error(f"[文生图] 失败: {e}")
            return f"生成图片时遇到问题: {str(e)}"

    @filter.command("aiimg", alias={"文生图"})
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
            yield event.chain_result([
                Image.fromFileSystem(str(image_path)),
            ])

            # 标记成功
            await mark_success(event)
            logger.info(f"[文生图] 完成: {prompt[:30] if prompt else '文生图'}..., 耗时={t_end - t_start:.2f}s")

        except Exception as e:
            logger.error(f"[文生图] 失败: {e}")
            await mark_failed(event)
            yield event.plain_result(f"生成图片失败: {str(e)}")

    # ==================== 图生图/改图 ====================

    @filter.command("aiedit", alias={"图生图", "改图"})
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
            yield event.plain_result("用法: /视频 <提示词> 或 /视频 <预设名> [额外提示词]")
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
        except Exception as e:
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
            yield event.plain_result("📋 视频预设列表\n暂无预设（请在配置 video.presets 中添加）")
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

    @filter.llm_tool()
    async def gitee_edit_image(
        self,
        event: AstrMessageEvent,
        prompt: str,
        use_message_images: bool = True,
        backend: str = "auto",
    ):
        """编辑用户发送的图片或引用的图片。当用户发送/引用了图片并希望修改、改图、换背景、换风格、换衣服、P图时调用此工具。

        获取图片的方式：
        - use_message_images=true（默认）：自动获取用户消息或引用消息中的图片

        重要提示：
        - 当消息中包含 [Image Caption: ...] 图片描述时，说明用户发送了图片，应调用此工具并设置 use_message_images=true
        - 调用成功后图片会自动发送给用户

        Args:
            prompt(string): 图片编辑提示词，描述用户希望对图片做的修改
            use_message_images(boolean): 是否自动获取用户消息中的图片，默认 true
            backend(string): 使用的后端: auto=自动选择, gemini=Gemini, gitee=千问
        """
        user_id = event.get_sender_id()
        request_id = f"edit_{user_id}"

        # 防抖检查
        if self.debouncer.hit(request_id):
            return "操作太快了，请稍后再试"

        # 提取图片
        bytes_images: list[bytes] = []
        if use_message_images:
            image_segs = await get_images_from_event(event)
            for seg in image_segs:
                try:
                    b64 = await seg.convert_to_base64()
                    bytes_images.append(base64.b64decode(b64))
                except Exception as e:
                    logger.warning(f"[LLM改图] 图片转换失败，跳过: {e}")
        if not bytes_images:
            return "请在消息中附带需要编辑的图片。提示：发送图片或引用图片后再发送修改指令。"

        try:
            t_start = time.perf_counter()

            # 确定后端
            target_backend = None if backend == "auto" else backend

            image_path = await self.edit.edit(
                prompt=prompt,
                images=bytes_images,
                backend=target_backend,
            )

            t_end = time.perf_counter()

            await event.send(
                event.chain_result([Image.fromFileSystem(str(image_path))])
            )
            logger.info(f"[LLM改图] 完成: {prompt[:30]}..., 耗时={t_end - t_start:.2f}s")
            return f"图片已编辑并发送。"

        except Exception as e:
            logger.error(f"[LLM改图] 失败: {e}", exc_info=True)
            await event.send(event.plain_result(f"编辑图片失败: {str(e) or type(e).__name__}"))
            return f"编辑失败: {e}"

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
        except Exception as e:
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
        mode = str(self.config.get("video", {}).get("send_mode", "auto")).strip().lower()
        if mode not in {"auto", "url", "file"}:
            mode = "auto"

        send_timeout = int(self.config.get("video", {}).get("send_timeout_seconds", 90) or 90)
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
            self.config.get("video", {}).get("download_timeout_seconds", self.video.timeout_seconds)
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
            await event.send(event.plain_result(
                "请发送或引用图片！\n"
                "用法: 发送图片 + 命令\n"
                "或: 引用图片消息 + 命令"
            ))
            return

        bytes_images: list[bytes] = []
        for i, seg in enumerate(image_segs):
            try:
                logger.debug(f"[改图] 转换图片 {i+1}/{len(image_segs)}...")
                b64 = await seg.convert_to_base64()
                bytes_images.append(base64.b64decode(b64))
                logger.debug(f"[改图] 图片 {i+1} 转换成功, 大小={len(bytes_images[-1])} bytes")
            except Exception as e:
                logger.warning(f"[改图] 图片 {i+1} 转换失败，跳过: {e}")

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
            await event.send(event.chain_result([
                Image.fromFileSystem(str(image_path)),
            ]))

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
            yield event.chain_result([
                Image.fromFileSystem(str(image_path)),
            ])

            # 标记成功
            await mark_success(event)
            display_name = preset or (prompt[:20] if prompt else "改图")
            logger.info(f"[改图] 完成: {display_name}..., 耗时={t_end - t_start:.2f}s")

        except Exception as e:
            logger.error(f"[改图] 失败: {e}")
            await mark_failed(event)
            yield event.plain_result(f"改图失败: {str(e)}")
