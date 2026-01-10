"""AstrBot Gitee AI 图像生成插件

支持 LLM 工具调用和 /aiimg 命令调用，支持多种图片比例和多 Key 轮询。
"""

import asyncio
import base64
import mimetypes
import os
import time
from pathlib import Path

import aiofiles
import aiohttp
from openai import AsyncOpenAI

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import Image
from astrbot.api.star import Context, Star, StarTools, register
from astrbot.core.message.components import Reply


# 配置常量 - 文生图
DEFAULT_BASE_URL = "https://ai.gitee.com/v1"
DEFAULT_MODEL = "z-image-turbo"
DEFAULT_SIZE = "1024x1024"
DEFAULT_INFERENCE_STEPS = 9
DEFAULT_NEGATIVE_PROMPT = (
    "low quality, bad anatomy, bad hands, text, error, missing fingers, "
    "extra digit, fewer digits, cropped, worst quality, normal quality, "
    "jpeg artifacts, signature, watermark, username, blurry"
)

# 配置常量 - 图生图
DEFAULT_EDIT_MODEL = "Qwen-Image-Edit-2511"
DEFAULT_EDIT_INFERENCE_STEPS = 4
DEFAULT_EDIT_GUIDANCE_SCALE = 1.0
DEFAULT_EDIT_POLL_INTERVAL = 5
DEFAULT_EDIT_POLL_TIMEOUT = 300

# 图生图支持的任务类型
EDIT_TASK_TYPES = ["id", "style", "subject", "background", "element"]

# 防抖和清理配置
DEBOUNCE_SECONDS = 10.0
MAX_CACHED_IMAGES = 50
OPERATION_CACHE_TTL = 300  # 5分钟清理一次过期操作记录
CLEANUP_INTERVAL = 10  # 每 N 次生成执行一次清理


@register(
    "astrbot_plugin_gitee_aiimg",
    "木有知",
    "接入 Gitee AI 图像生成模型。支持文生图和图生图，支持 LLM 调用和命令调用。",
    "2.0.0",
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
        self._http_session: aiohttp.ClientSession | None = None

        # 图片目录
        self._image_dir: Path | None = None

        # 清理计数器和后台任务引用
        self._generation_count: int = 0
        self._background_tasks: set[asyncio.Task] = set()

        # 图生图配置
        self.edit_base_url = config.get("edit_base_url") or self.base_url
        self.edit_api_keys = self._parse_api_keys(config.get("edit_api_key", []))
        if not self.edit_api_keys:
            self.edit_api_keys = self.api_keys  # 复用文生图的 Key
        self.edit_current_key_index = 0
        self.edit_model = config.get("edit_model", DEFAULT_EDIT_MODEL)
        self.edit_num_inference_steps = config.get(
            "edit_num_inference_steps", DEFAULT_EDIT_INFERENCE_STEPS
        )
        self.edit_guidance_scale = config.get(
            "edit_guidance_scale", DEFAULT_EDIT_GUIDANCE_SCALE
        )
        self.edit_poll_interval = config.get(
            "edit_poll_interval", DEFAULT_EDIT_POLL_INTERVAL
        )
        self.edit_poll_timeout = config.get(
            "edit_poll_timeout", DEFAULT_EDIT_POLL_TIMEOUT
        )

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

    # ========== 图生图功能 ==========

    def _get_edit_api_key(self) -> str:
        """获取图生图 API Key（轮询）"""
        # 重新读取配置（支持热更新）
        if not self.edit_api_keys:
            self.edit_api_keys = self._parse_api_keys(
                self.config.get("edit_api_key", [])
            )
            if not self.edit_api_keys:
                self.edit_api_keys = self.api_keys

        if not self.edit_api_keys:
            raise ValueError("请先配置图生图 API Key（或文生图 API Key）")

        # 轮询获取 Key
        api_key = self.edit_api_keys[self.edit_current_key_index]
        self.edit_current_key_index = (
            self.edit_current_key_index + 1
        ) % len(self.edit_api_keys)

        return api_key

    async def _create_edit_task(
        self,
        prompt: str,
        image_data_list: list[bytes],
        task_types: list[str],
    ) -> tuple[str, str]:
        """创建图生图异步任务，返回 (task_id, api_key)"""
        session = await self._get_http_session()
        api_key = self._get_edit_api_key()

        headers = {
            "X-Failover-Enabled": "true",
            "Authorization": f"Bearer {api_key}",
        }

        # 构建 multipart/form-data
        data = aiohttp.FormData()
        data.add_field("prompt", prompt)
        data.add_field("model", self.edit_model)
        data.add_field("num_inference_steps", str(self.edit_num_inference_steps))
        data.add_field("guidance_scale", str(self.edit_guidance_scale))

        for task_type in task_types:
            data.add_field("task_types", task_type)

        # 处理图片二进制数据
        for idx, img_bytes in enumerate(image_data_list):
            logger.debug(f"[_create_edit_task] 添加图片 {idx}: {len(img_bytes)} bytes")
            data.add_field(
                "image",
                img_bytes,
                filename=f"image_{idx}.jpg",
                content_type="image/jpeg",
            )

        api_url = f"{self.edit_base_url}/async/images/edits"

        async with session.post(api_url, headers=headers, data=data) as resp:
            result = await resp.json()
            if resp.status != 200:
                error_msg = result.get("message", str(result))
                if resp.status == 401:
                    raise Exception("图生图 API Key 无效或已过期，请检查配置。")
                elif resp.status == 429:
                    raise Exception("API 调用次数超限或并发过高，请稍后再试。")
                else:
                    raise Exception(f"创建图生图任务失败: {error_msg}")

            task_id = result.get("task_id")
            if not task_id:
                raise Exception(f"创建图生图任务失败：未返回 task_id。响应: {result}")

            return task_id, api_key

    async def _poll_edit_task(self, task_id: str, api_key: str) -> str:
        """轮询图生图任务状态，返回结果图片 URL"""
        session = await self._get_http_session()

        headers = {
            "Authorization": f"Bearer {api_key}",
        }

        status_url = f"{self.edit_base_url}/task/{task_id}"
        max_attempts = int(self.edit_poll_timeout / self.edit_poll_interval)

        for attempt in range(1, max_attempts + 1):
            async with session.get(status_url, headers=headers) as resp:
                result = await resp.json()

                if result.get("error"):
                    raise Exception(
                        f"任务出错: {result['error']}: {result.get('message', 'Unknown error')}"
                    )

                status = result.get("status", "unknown")

                if status == "success":
                    output = result.get("output", {})
                    file_url = output.get("file_url")
                    if file_url:
                        return file_url
                    else:
                        raise Exception("任务完成但未返回图片 URL")

                elif status in ["failed", "cancelled"]:
                    raise Exception(f"图生图任务 {status}")

                # 继续等待
                logger.debug(f"图生图任务 {task_id} 状态: {status} (第 {attempt} 次检查)")

            await asyncio.sleep(self.edit_poll_interval)

        raise Exception(f"图生图任务超时（等待超过 {self.edit_poll_timeout} 秒）")

    async def _edit_image(
        self,
        prompt: str,
        image_data_list: list[bytes],
        task_types: list[str] | None = None,
    ) -> str:
        """执行图生图，返回本地文件路径"""
        if not image_data_list:
            raise ValueError("请提供至少一张图片")

        # 默认任务类型
        if task_types is None:
            task_types = ["id"]  # 默认保持身份特征

        # 验证任务类型
        valid_types = [t for t in task_types if t in EDIT_TASK_TYPES]
        if not valid_types:
            valid_types = ["id"]

        # 创建任务
        task_id, api_key = await self._create_edit_task(prompt, image_data_list, valid_types)
        logger.info(f"图生图任务已创建: {task_id}")

        # 轮询等待结果
        file_url = await self._poll_edit_task(task_id, api_key)

        # 下载结果图片
        filepath = await self._download_image(file_url)

        # 触发清理
        self._generation_count += 1
        if self._generation_count >= CLEANUP_INTERVAL:
            self._generation_count = 0
            task = asyncio.create_task(self._cleanup_old_images())
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)

        return filepath

    async def _download_image_bytes(self, url: str, timeout: int = 30) -> bytes | None:
        """下载图片并返回二进制数据，失败返回 None"""
        if not url or not url.startswith(("http://", "https://")):
            return None

        session = await self._get_http_session()
        try:
            async with session.get(
                url, timeout=aiohttp.ClientTimeout(total=timeout)
            ) as resp:
                if resp.status == 200:
                    return await resp.read()
                else:
                    logger.warning(f"下载图片失败: HTTP {resp.status}, URL: {url[:60]}...")
        except Exception as e:
            logger.warning(f"下载图片异常: {type(e).__name__}, URL: {url[:60]}...")
        return None

    async def _extract_images_from_event(
        self, event: AstrMessageEvent
    ) -> list[bytes]:
        """从消息事件中提取图片二进制数据列表

        支持：
        1. 回复/引用消息中的图片（优先）
        2. 当前消息中的图片
        3. 多图输入
        4. base64 格式图片
        """
        images: list[bytes] = []

        # 获取原始消息对象
        message_obj = event.message_obj
        chain = message_obj.message

        logger.debug(f"[_extract_images] 开始提取图片, 消息链长度: {len(chain)}")

        # 1. 从回复/引用消息中提取图片（优先）
        for seg in chain:
            if isinstance(seg, Reply):
                logger.debug(f"[_extract_images] 发现 Reply 段, chain={getattr(seg, 'chain', None)}")
                if hasattr(seg, "chain") and seg.chain:
                    for chain_item in seg.chain:
                        if isinstance(chain_item, Image):
                            img_data = await self._load_image_data(chain_item)
                            if img_data:
                                images.append(img_data)
                                logger.debug(f"[_extract_images] 从回复链提取图片: {len(img_data)} bytes")

        # 2. 从当前消息中提取图片
        for seg in chain:
            if isinstance(seg, Image):
                img_data = await self._load_image_data(seg)
                if img_data:
                    images.append(img_data)
                    logger.debug(f"[_extract_images] 从当前消息提取图片: {len(img_data)} bytes")

        logger.info(f"[_extract_images] 共提取到 {len(images)} 张图片")
        return images

    async def _load_image_data(self, img: Image) -> bytes | None:
        """从 Image 对象加载图片二进制数据
        
        优先级：本地文件 > base64 > URL下载
        """
        # 1. 尝试从本地文件读取（NapCat/LLOneBot 会缓存图片到本地）
        file_path = getattr(img, "file", None)
        if file_path and not file_path.startswith(("http://", "https://")):
            # 可能是本地路径或文件名
            local_path = Path(file_path)
            if local_path.is_file():
                try:
                    return local_path.read_bytes()
                except Exception as e:
                    logger.debug(f"读取本地文件失败: {e}")

        # 2. 尝试 base64
        b64 = getattr(img, "base64", None)
        if b64:
            try:
                return base64.b64decode(b64)
            except Exception:
                pass

        # 3. 尝试从 URL 下载
        url = getattr(img, "url", None)
        if url:
            return await self._download_image_bytes(url)

        return None

    @filter.llm_tool(name="edit_image")  # type: ignore
    async def edit(
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
        if self._check_debounce(request_id):
            return "操作太快了，请稍后再试。"

        if request_id in self.processing_users:
            return "您有正在进行的图生图任务，请稍候..."

        # 提取图片
        image_data_list: list[bytes] = []
        if use_message_images:
            image_data_list = await self._extract_images_from_event(event)
            logger.debug(f"[edit_image LLM] use_message_images=True, 提取到 {len(image_data_list)} 张图片")

        if not image_data_list:
            return "请在消息中附带需要编辑的图片。提示：发送图片或引用图片后再发送修改指令。"

        self.processing_users.add(request_id)

        # 解析任务类型
        types = [t.strip() for t in task_types.split(",") if t.strip()]

        # 启动后台任务，立即返回（非阻塞）
        async def _background_edit():
            try:
                image_path = await self._edit_image(prompt, image_data_list, types)
                await event.send(
                    event.chain_result([Image.fromFileSystem(image_path)])  # type: ignore
                )
                logger.info(f"[edit_image] 图生图完成: {prompt[:50]}...")
            except Exception as e:
                logger.error(f"[edit_image] 图生图失败: {e}", exc_info=True)
                await event.send(event.plain_result(f"编辑图片失败: {str(e) or type(e).__name__}"))
            finally:
                self.processing_users.discard(request_id)

        task = asyncio.create_task(_background_edit())
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

        return f"正在编辑图片，请稍候...（提示词: {prompt[:30]}...）"

    @filter.command("aiedit")
    async def edit_image_command(self, event: AstrMessageEvent, prompt: str):
        """图生图指令（图像编辑）

        用法: /aiedit <提示词> [任务类型]
        示例: /aiedit 把背景换成海边 background
        支持任务类型: id(保持身份), style(风格迁移), subject(主体), background(背景), element(元素)
        可组合使用，如: /aiedit 生成结婚照 id,style
        """
        if not prompt:
            yield event.plain_result(
                "请提供提示词！使用方法：/aiedit <提示词> [任务类型]\n"
                "示例: /aiedit 把背景换成海边 background\n"
                "支持任务类型: id, style, subject, background, element"
            )
            return

        user_id = event.get_sender_id()
        request_id = f"edit_{user_id}"

        # 防抖检查
        if self._check_debounce(request_id):
            yield event.plain_result("操作太快了，请稍后再试。")
            return

        if request_id in self.processing_users:
            yield event.plain_result("您有正在进行的图生图任务，请稍候...")
            return

        # 提取图片
        image_data_list = await self._extract_images_from_event(event)
        if not image_data_list:
            yield event.plain_result(
                "请在消息中附带需要编辑的图片！\n"
                "使用方法：发送图片并附带 /aiedit <提示词>"
            )
            return

        self.processing_users.add(request_id)

        # 解析任务类型参数
        task_types: list[str] = ["id"]  # 默认
        prompt_parts = prompt.rsplit(" ", 1)
        if len(prompt_parts) > 1:
            potential_types = prompt_parts[1]
            # 检查是否是有效的任务类型
            parsed_types = [t.strip() for t in potential_types.split(",")]
            if all(t in EDIT_TASK_TYPES for t in parsed_types):
                task_types = parsed_types
                prompt = prompt_parts[0]

        try:
            image_path = await self._edit_image(prompt, image_data_list, task_types)
            yield event.chain_result([Image.fromFileSystem(image_path)])  # type: ignore

        except Exception as e:
            logger.error(f"图生图失败: {e}")
            yield event.plain_result(f"编辑图片失败: {str(e)}")
        finally:
            self.processing_users.discard(request_id)

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
