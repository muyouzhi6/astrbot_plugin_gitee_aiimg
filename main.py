from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import Image
from astrbot.api.star import Context, Star, StarTools

from .core.debouncer import Debouncer
from .core.image import ImageManager
from .core.service import ImageService


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
        self.service = ImageService(self.config, self.imgr)

    async def terminate(self):
        self.debouncer.clear_all()
        await self.imgr.close()
        await self.service.close()

    @filter.llm_tool()
    async def gitee_draw_image(self, event: AstrMessageEvent, prompt: str):
        """根据提示词生成图片。

        Args:
            prompt(string): 图片提示词，需要包含主体、场景、风格等描述
        """
        request_id = event.get_sender_id()

        if self.debouncer.hit(request_id):
            return "操作太快了，请稍后再试。"

        if request_id in self.processing_users:
            return "您有正在进行的生图任务，请稍候..."

        self.processing_users.add(request_id)

        if not prompt:
            return "需提供提示词prompt"

        try:
            image_path = await self.service.generate(prompt)
            await event.send(event.chain_result([Image.fromFileSystem(str(image_path))]))
            return f"图片已生成并发送。Prompt: {prompt}"

        except Exception as e:
            logger.error(f"生图失败: {e}")
            return f"生成图片时遇到问题: {str(e)}"
        finally:
            self.processing_users.discard(request_id)

    @filter.command("aiimg")
    async def generate_image_command(self, event: AstrMessageEvent, prompt: str):
        """生成图片指令

        用法: /aiimg <提示词> [比例]
        示例: /aiimg 一个女孩 9:16
        支持比例: 1:1, 4:3, 3:4, 3:2, 2:3, 16:9, 9:16
        """
        request_id = event.get_sender_id()

        if self.debouncer.hit(request_id):
            yield event.plain_result("操作太快了，请稍后再试")
            return

        if request_id in self.processing_users:
            yield event.plain_result("您有正在进行的生图任务，请稍候...")
            return

        if not prompt:
            yield event.plain_result("请提供提示词！使用方法：/aiimg <提示词> [比例]")
            return

        self.processing_users.add(request_id)

        # 解析比例参数
        ratio = "1:1"
        prompt_parts = prompt.rsplit(" ", 1)
        if len(prompt_parts) > 1 and prompt_parts[1] in self.SUPPORTED_RATIOS:
            ratio = prompt_parts[1]
            prompt = prompt_parts[0]

        # 确定目标尺寸
        target_size = self.config["size"]
        if ratio != "1:1" or (
            ratio == "1:1" and self.config["size"] not in self.SUPPORTED_RATIOS["1:1"]
        ):
            target_size = self.SUPPORTED_RATIOS[ratio][0]

        try:
            image_path = await self.service.generate(prompt, size=target_size)
            yield event.chain_result([Image.fromFileSystem(str(image_path))])

        except Exception as e:
            logger.error(f"生图失败: {e}")
            yield event.plain_result(f"生成图片失败: {str(e)}")
        finally:
            self.processing_users.discard(request_id)
