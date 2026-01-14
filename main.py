"""
Gitee AI 图像生成插件

功能:
- 文生图 (z-image-turbo)
- 图生图/改图 (Gemini / Gitee 千问，可切换)
- 预设提示词
- 智能降级
"""

import asyncio
import base64
import time

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import Image, Plain
from astrbot.api.star import Context, Star, StarTools

from .core.debouncer import Debouncer
from .core.draw_service import ImageDrawService
from .core.edit_router import EditRouter
from .core.image_manager import ImageManager
from .core.utils import get_images_from_event


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

        # 并发控制 (带锁保护)
        self.processing_users: set[str] = set()
        self._processing_lock = asyncio.Lock()

    async def initialize(self):
        self.debouncer = Debouncer(self.config)
        self.imgr = ImageManager(self.config, self.data_dir)
        self.draw = ImageDrawService(self.config, self.imgr)
        self.edit = EditRouter(self.config, self.imgr)

        logger.info(
            f"[GiteeAIImage] 插件初始化完成: "
            f"改图后端={self.edit.get_available_backends()}, "
            f"预设={len(self.edit.get_preset_names())}个"
        )

    async def terminate(self):
        self.debouncer.clear_all()
        await self.imgr.close()
        await self.draw.close()
        await self.edit.close()

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

        # 并发控制 (原子操作)
        async with self._processing_lock:
            if request_id in self.processing_users:
                return "您有正在进行的生图任务，请稍候..."
            self.processing_users.add(request_id)

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
        finally:
            self.processing_users.discard(request_id)

    @filter.command("aiimg", alias={"文生图"})
    async def generate_image_command(self, event: AstrMessageEvent, prompt: str):
        """生成图片指令

        用法: /aiimg <提示词> [比例]
        示例: /aiimg 一个女孩 9:16
        支持比例: 1:1, 4:3, 3:4, 3:2, 2:3, 16:9, 9:16
        """
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

        # 并发控制 (原子操作)
        async with self._processing_lock:
            if request_id in self.processing_users:
                yield event.plain_result("您有正在进行的生图任务，请稍候...")
                return
            self.processing_users.add(request_id)

        try:
            t_start = time.perf_counter()
            yield event.plain_result(f"🎨 正在生成图片...")

            image_path = await self.draw.generate(prompt, size=size)
            t_end = time.perf_counter()

            yield event.chain_result([
                Image.fromFileSystem(str(image_path)),
                Plain(f"\n✅ 生成完成 ({t_end - t_start:.1f}s)")
            ])

        except Exception as e:
            logger.error(f"[文生图] 失败: {e}")
            yield event.plain_result(f"生成图片失败: {str(e)}")
        finally:
            self.processing_users.discard(request_id)

    # ==================== 图生图/改图 ====================

    @filter.command("aiedit", alias={"图生图", "改图"})
    async def edit_image_default(self, event: AstrMessageEvent, prompt: str):
        """使用默认后端改图

        用法: /aiedit <提示词>
        需要同时发送或引用图片
        """
        async for result in self._do_edit(event, prompt, backend=None):
            yield result

    @filter.command("gedit", alias={"g改图"})
    async def edit_image_gemini(self, event: AstrMessageEvent, prompt: str):
        """使用 Gemini 改图

        用法: /gedit <提示词>
        """
        async for result in self._do_edit(event, prompt, backend="gemini"):
            yield result

    @filter.command("qedit", alias={"q改图"})
    async def edit_image_qwen(self, event: AstrMessageEvent, prompt: str):
        """使用 Gitee 千问改图

        用法: /qedit <提示词>
        """
        async for result in self._do_edit(event, prompt, backend="gitee"):
            yield result

    # ==================== 管理命令 ====================

    @filter.command("预设列表")
    async def list_presets(self, event: AstrMessageEvent):
        """列出所有可用预设"""
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
            b64_images = [await seg.convert_to_base64() for seg in image_segs]
            bytes_images = [base64.b64decode(b64) for b64 in b64_images]
        if not bytes_images:
            return "请在消息中附带需要编辑的图片。提示：发送图片或引用图片后再发送修改指令。"

        # 并发控制 (原子操作)
        async with self._processing_lock:
            if request_id in self.processing_users:
                return "您有正在进行的图生图任务，请稍候..."
            self.processing_users.add(request_id)

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
        finally:
            self.processing_users.discard(request_id)

    # ==================== 内部方法 ====================

    async def _do_edit(
        self,
        event: AstrMessageEvent,
        prompt: str,
        backend: str | None = None,
        preset: str | None = None,
    ):
        """统一改图执行入口"""
        user_id = event.get_sender_id()
        request_id = f"edit_{user_id}"

        # 防抖
        if self.debouncer.hit(request_id):
            yield event.plain_result("操作太快了，请稍后再试")
            return

        # 获取图片
        image_segs = await get_images_from_event(event)
        if not image_segs:
            yield event.plain_result(
                "请发送或引用图片！\n"
                "用法: 发送图片 + /aiedit <提示词>\n"
                "或: 引用图片消息 + /aiedit <提示词>"
            )
            return

        bytes_images = [
            base64.b64decode(await seg.convert_to_base64())
            for seg in image_segs
        ]

        # 并发控制 (原子操作)
        async with self._processing_lock:
            if request_id in self.processing_users:
                yield event.plain_result("您有正在进行的改图任务，请稍候...")
                return
            self.processing_users.add(request_id)

        try:
            # 确定显示名称
            backend_name = backend or self.config.get("edit", {}).get("default_backend", "gemini")
            display_name = preset or prompt[:20] or "改图"

            yield event.plain_result(f"🎨 [{backend_name}] {display_name} 处理中...")

            t_start = time.perf_counter()
            image_path = await self.edit.edit(
                prompt=prompt,
                images=bytes_images,
                backend=backend,
                preset=preset,
            )
            t_end = time.perf_counter()

            yield event.chain_result([
                Image.fromFileSystem(str(image_path)),
                Plain(f"\n✅ [{backend_name}] 完成 ({t_end - t_start:.1f}s)")
            ])

        except Exception as e:
            logger.error(f"[改图] 失败: {e}", exc_info=True)
            yield event.plain_result(f"改图失败: {str(e)}")
        finally:
            self.processing_users.discard(request_id)
