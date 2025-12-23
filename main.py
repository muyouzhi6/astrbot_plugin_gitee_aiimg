"""AstrBot Gitee AI 图像生成插件

支持 LLM 工具调用和 /aiimg 命令调用，支持多种图片比例和多 Key 轮询。
"""

import asyncio
import base64
import os
import time
from pathlib import Path
from typing import Optional

import aiofiles
import aiohttp
from openai import AsyncOpenAI

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import Image
from astrbot.api.star import Context, Star, StarTools, register


# 配置常量
DEFAULT_BASE_URL = "https://ai.gitee.com/v1"
DEFAULT_MODEL = "z-image-turbo"
DEFAULT_SIZE = "1024x1024"
DEFAULT_INFERENCE_STEPS = 9
DEFAULT_NEGATIVE_PROMPT = (
    "low quality, bad anatomy, bad hands, text, error, missing fingers, "
    "extra digit, fewer digits, cropped, worst quality, normal quality, "
    "jpeg artifacts, signature, watermark, username, blurry"
)

# 防抖和清理配置
DEBOUNCE_SECONDS = 10.0
MAX_CACHED_IMAGES = 50
OPERATION_CACHE_TTL = 300  # 5分钟清理一次过期操作记录
CLEANUP_INTERVAL = 10  # 每 N 次生成执行一次清理


@register(
    "astrbot_plugin_gitee_aiimg",
    "木有知",
    "接入 Gitee AI 图像生成模型。支持 LLM 调用和命令调用，支持多种比例。",
    "1.2",
)
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
        self.base_url = config.get("base_url", DEFAULT_BASE_URL)

        # 解析 API Keys
        self.api_keys = self._parse_api_keys(config.get("api_key", []))
        self.current_key_index = 0

        # 模型配置
        self.model = config.get("model", DEFAULT_MODEL)
        self.default_size = config.get("size", DEFAULT_SIZE)
        self.num_inference_steps = config.get("num_inference_steps", DEFAULT_INFERENCE_STEPS)
        self.negative_prompt = config.get("negative_prompt", DEFAULT_NEGATIVE_PROMPT)

        # 并发控制
        self.processing_users: set[str] = set()
        self.last_operations: dict[str, float] = {}

        # 复用的客户端（延迟初始化）
        self._openai_clients: dict[str, AsyncOpenAI] = {}
        self._http_session: Optional[aiohttp.ClientSession] = None

        # 图片目录
        self._image_dir: Optional[Path] = None

        # 清理计数器和后台任务引用
        self._generation_count: int = 0
        self._background_tasks: set[asyncio.Task] = set()

    @staticmethod
    def _parse_api_keys(api_keys) -> list[str]:
        """解析 API Keys 配置，支持字符串和列表格式"""
        if isinstance(api_keys, str):
            if api_keys:
                return [k.strip() for k in api_keys.split(",") if k.strip()]
            return []
        elif isinstance(api_keys, list):
            return [str(k).strip() for k in api_keys if str(k).strip()]
        return []

    def _get_image_dir(self) -> Path:
        """获取图片保存目录（延迟初始化）"""
        if self._image_dir is None:
            base_dir = StarTools.get_data_dir("astrbot_plugin_gitee_aiimg")
            self._image_dir = base_dir / "images"
            self._image_dir.mkdir(exist_ok=True)
        return self._image_dir

    def _get_client(self) -> AsyncOpenAI:
        """获取复用的 AsyncOpenAI 客户端"""
        # 重新读取配置（支持热更新）
        if not self.api_keys:
            self.api_keys = self._parse_api_keys(self.config.get("api_key", []))

        if not self.api_keys:
            raise ValueError("请先配置 API Key")

        # 轮询获取 Key
        api_key = self.api_keys[self.current_key_index]
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)

        # 复用客户端
        if api_key not in self._openai_clients:
            self._openai_clients[api_key] = AsyncOpenAI(
                base_url=self.base_url,
                api_key=api_key,
            )

        return self._openai_clients[api_key]

    async def _get_http_session(self) -> aiohttp.ClientSession:
        """获取复用的 HTTP Session"""
        if self._http_session is None or self._http_session.closed:
            self._http_session = aiohttp.ClientSession()
        return self._http_session

    def _get_save_path(self, extension: str = ".jpg") -> str:
        """生成唯一的图片保存路径"""
        image_dir = self._get_image_dir()
        filename = f"{int(time.time())}_{os.urandom(4).hex()}{extension}"
        return str(image_dir / filename)

    def _sync_cleanup_old_images(self) -> None:
        """同步清理旧图片（在线程池中执行）"""
        try:
            image_dir = self._get_image_dir()
            # 收集所有支持的图片格式
            images: list[Path] = []
            for ext in ("*.jpg", "*.png", "*.webp"):
                images.extend(image_dir.glob(ext))
            
            # 按修改时间排序
            images.sort(key=lambda p: p.stat().st_mtime)

            if len(images) > MAX_CACHED_IMAGES:
                to_delete = images[: len(images) - MAX_CACHED_IMAGES]
                for img_path in to_delete:
                    try:
                        img_path.unlink()
                    except OSError:
                        pass
        except Exception as e:
            logger.warning(f"清理旧图片时出错: {e}")

    async def _cleanup_old_images(self) -> None:
        """异步清理旧图片，使用线程池执行阻塞操作"""
        await asyncio.to_thread(self._sync_cleanup_old_images)

    def _cleanup_expired_operations(self) -> None:
        """清理过期的操作记录，防止内存泄漏"""
        current_time = time.time()
        expired_keys = [
            key
            for key, timestamp in self.last_operations.items()
            if current_time - timestamp > OPERATION_CACHE_TTL
        ]
        for key in expired_keys:
            del self.last_operations[key]

    def _check_debounce(self, request_id: str) -> bool:
        """检查防抖，返回 True 表示需要拒绝请求"""
        current_time = time.time()

        # 定期清理过期记录
        if len(self.last_operations) > 100:
            self._cleanup_expired_operations()

        if request_id in self.last_operations:
            if current_time - self.last_operations[request_id] < DEBOUNCE_SECONDS:
                return True

        self.last_operations[request_id] = current_time
        return False

    async def _download_image(self, url: str) -> str:
        """下载图片并异步保存到文件"""
        session = await self._get_http_session()

        async with session.get(url) as resp:
            if resp.status != 200:
                raise Exception(f"下载图片失败: HTTP {resp.status}")
            data = await resp.read()

        filepath = self._get_save_path()
        async with aiofiles.open(filepath, "wb") as f:
            await f.write(data)

        return filepath

    async def _save_base64_image(self, b64_data: str) -> str:
        """异步保存 base64 图片到文件"""
        filepath = self._get_save_path()
        image_bytes = base64.b64decode(b64_data)

        async with aiofiles.open(filepath, "wb") as f:
            await f.write(image_bytes)

        return filepath

    async def _generate_image(self, prompt: str, size: str = "") -> str:
        """调用 Gitee AI API 生成图片，返回本地文件路径"""
        client = self._get_client()
        target_size = size if size else self.default_size

        # 构建请求参数
        kwargs = {
            "prompt": prompt,
            "model": self.model,
            "extra_body": {
                "num_inference_steps": self.num_inference_steps,
            },
        }

        if self.negative_prompt:
            kwargs["extra_body"]["negative_prompt"] = self.negative_prompt
        if target_size:
            kwargs["size"] = target_size

        try:
            response = await client.images.generate(**kwargs)  # type: ignore
        except Exception as e:
            error_msg = str(e)
            if "401" in error_msg:
                raise Exception("API Key 无效或已过期，请检查配置。")
            elif "429" in error_msg:
                raise Exception("API 调用次数超限或并发过高，请稍后再试。")
            elif "500" in error_msg:
                raise Exception("Gitee AI 服务器内部错误，请稍后再试。")
            else:
                raise Exception(f"API调用失败: {error_msg}")

        if not response.data:  # type: ignore
            raise Exception("生成图片失败：未返回数据")

        image_data = response.data[0]  # type: ignore

        if image_data.url:
            filepath = await self._download_image(image_data.url)
        elif image_data.b64_json:
            filepath = await self._save_base64_image(image_data.b64_json)
        else:
            raise Exception("生成图片失败：未返回 URL 或 Base64 数据")

        # 每 N 次生成执行一次清理
        self._generation_count += 1
        if self._generation_count >= CLEANUP_INTERVAL:
            self._generation_count = 0
            task = asyncio.create_task(self._cleanup_old_images())
            # 保存任务引用防止 GC 回收
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)

        return filepath

    @filter.llm_tool(name="draw_image")  # type: ignore
    async def draw(self, event: AstrMessageEvent, prompt: str):
        """根据提示词生成图片。

        Args:
            prompt(string): 图片提示词，需要包含主体、场景、风格等描述
        """
        user_id = event.get_sender_id()
        request_id = user_id

        # 防抖检查
        if self._check_debounce(request_id):
            return "操作太快了，请稍后再试。"

        if request_id in self.processing_users:
            return "您有正在进行的生图任务，请稍候..."

        self.processing_users.add(request_id)
        try:
            image_path = await self._generate_image(prompt)
            await event.send(event.chain_result([Image.fromFileSystem(image_path)]))  # type: ignore
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
        if not prompt:
            yield event.plain_result("请提供提示词！使用方法：/aiimg <提示词> [比例]")
            return

        user_id = event.get_sender_id()
        request_id = user_id

        # 防抖检查（统一机制）
        if self._check_debounce(request_id):
            yield event.plain_result("操作太快了，请稍后再试。")
            return

        if request_id in self.processing_users:
            yield event.plain_result("您有正在进行的生图任务，请稍候...")
            return

        self.processing_users.add(request_id)

        # 解析比例参数
        ratio = "1:1"
        prompt_parts = prompt.rsplit(" ", 1)
        if len(prompt_parts) > 1 and prompt_parts[1] in self.SUPPORTED_RATIOS:
            ratio = prompt_parts[1]
            prompt = prompt_parts[0]

        # 确定目标尺寸
        target_size = self.default_size
        if ratio != "1:1" or (ratio == "1:1" and self.default_size not in self.SUPPORTED_RATIOS["1:1"]):
            target_size = self.SUPPORTED_RATIOS[ratio][0]

        try:
            image_path = await self._generate_image(prompt, size=target_size)
            yield event.chain_result([Image.fromFileSystem(image_path)])  # type: ignore

        except Exception as e:
            logger.error(f"生图失败: {e}")
            yield event.plain_result(f"生成图片失败: {str(e)}")
        finally:
            self.processing_users.discard(request_id)

    async def close(self) -> None:
        """清理资源"""
        # 关闭 HTTP Session
        if self._http_session and not self._http_session.closed:
            await self._http_session.close()

        # 清理 OpenAI 客户端
        for client in self._openai_clients.values():
            await client.close()
        self._openai_clients.clear()

    def __del__(self):
        """析构时尝试清理资源"""
        if self._http_session and not self._http_session.closed:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._http_session.close())
            except RuntimeError:
                pass
