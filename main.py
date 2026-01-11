import base64

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import Image
from astrbot.api.star import Context, Star, StarTools

from .core.debouncer import Debouncer
from .core.draw_service import ImageDrawService
from .core.edit_service import ImageEditService
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

        # 并发控制
        self.processing_users: set[str] = set()

    async def initialize(self):
        self.debouncer = Debouncer(self.config)
        self.imgr = ImageManager(self.config, self.data_dir)
        self.draw = ImageDrawService(self.config, self.imgr)
        self.edit = ImageEditService(self.config, self.imgr)


    async def terminate(self):
        self.debouncer.clear_all()
        await self.imgr.close()
        await self.draw.close()
        await self.edit.close()

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

        if request_id in self.processing_users:
            return "您有正在进行的生图任务，请稍候..."
        self.processing_users.add(request_id)

        try:
            image_path = await self.draw.generate(prompt)
            await event.send(event.chain_result([Image.fromFileSystem(str(image_path))]))
            return f"图片已生成并发送。Prompt: {prompt}"

        except Exception as e:
            logger.error(f"生图失败: {e}")
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

        # 并发控制
        if request_id in self.processing_users:
            yield event.plain_result("您有正在进行的生图任务，请稍候...")
            return
        self.processing_users.add(request_id)

        try:
            image_path = await self.draw.generate(prompt, size=size)
            yield event.chain_result([Image.fromFileSystem(str(image_path))])

        except Exception as e:
            logger.error(f"生图失败: {e}")
            yield event.plain_result(f"生成图片失败: {str(e)}")
        finally:
            self.processing_users.discard(request_id)

    @filter.command("aiedit", alias={"图生图"})
    async def edit_image_command(self, event: AstrMessageEvent, prompt: str):
        """aiedit <提示词> <任务类型>
        支持任务类型: id(保持身份), style(风格迁移), subject(主体), background(背景), element(元素)
        """
        # 解析参数
        if not prompt:
            yield event.plain_result(
                "请提供提示词！使用方法：/aiedit <提示词> [任务类型]\n"
                "示例: /aiedit 把背景换成海边 background\n"
                "支持任务类型: id, style, subject, background, element"
            )
            return
        types: list[str] = ["id"]
        end_parts = prompt.rsplit(" ", 1)
        if len(end_parts) > 1:
            types = [t.strip() for t in end_parts[1].split(",")]
            prompt = end_parts[0]

        # 请求ID
        user_id = event.get_sender_id()
        request_id = f"edit_{user_id}"

        # 防抖检查
        if self.debouncer.hit(request_id):
            yield event.plain_result("操作太快了，请稍后再试。")
            return

        # 提取图片
        image_segs = await get_images_from_event(event)
        if not image_segs:
            yield event.plain_result(
                "请在消息中附带需要编辑的图片！\n"
                "使用方法：发送图片并附带 /aiedit <提示词>"
            )
            return
        b64_images = [await seg.convert_to_base64() for seg in image_segs]
        bytes_images = [base64.b64decode(b64) for b64 in b64_images]

        # 并发控制
        if request_id in self.processing_users:
            yield event.plain_result("您有正在进行的图生图任务，请稍候...")
            return
        self.processing_users.add(request_id)

        # 请求
        try:
            image_path = await self.edit.edit(
                prompt=prompt,
                images=bytes_images,
                task_types=types,
            )
            yield event.chain_result([Image.fromFileSystem(str(image_path))])

        except Exception as e:
            logger.error(f"图生图失败: {e}")
            yield event.plain_result(f"编辑图片失败: {str(e)}")
        finally:
            self.processing_users.discard(request_id)


    @filter.llm_tool()
    async def gitee_edit_image(
        self,
        event: AstrMessageEvent,
        prompt: str,
        use_message_images: bool = True,
        task_types: str = "id",
    ):
        """编辑用户发送的图片或引用的图片。当用户发送/引用了图片并希望修改、改图、换背景、换风格、换衣服、P图时调用此工具。

        获取图片的方式：
        - use_message_images=true（默认）：自动获取用户消息或引用消息中的图片

        重要提示：
        - 当消息中包含 [Image Caption: ...] 图片描述时，说明用户发送了图片，应调用此工具并设置 use_message_images=true
        - gchat.qpic.cn 等 QQ 临时链接无法直接访问，必须使用 use_message_images=true 来获取
        - 调用成功后图片会自动发送给用户

        Args:
            prompt(string): 图片编辑提示词，描述用户希望对图片做的修改，如"换成吊带裙"、"背景换成海边"、"转成动漫风格"等
            use_message_images(boolean): 是否自动获取用户消息中的图片，默认 true（推荐）
            task_types(string): 任务类型，逗号分隔。可选值: id(保持身份/默认), style(风格迁移), subject(主体替换), background(背景替换), element(元素编辑)。默认为 id
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

        # 解析任务类型
        types = [t.strip() for t in task_types.split(",") if t.strip()]

        # 并发控制
        if request_id in self.processing_users:
            return "您有正在进行的图生图任务，请稍候..."
        self.processing_users.add(request_id)

        try:
            image_path = await self.edit.edit(prompt, bytes_images, types)
            await event.send(
                event.chain_result([Image.fromFileSystem(image_path)])
            )
            logger.info(f"[edit_image] 图生图完成: {prompt[:50]}...")
        except Exception as e:
            logger.error(f"[edit_image] 图生图失败: {e}", exc_info=True)
            await event.send(event.plain_result(f"编辑图片失败: {str(e) or type(e).__name__}"))
        finally:
            self.processing_users.discard(request_id)
