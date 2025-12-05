from typing import Optional
from astrbot.api.message_components import Plain, Image
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register
from astrbot.api import logger, llm_tool
from openai import AsyncOpenAI
import os
import time
import base64
import aiohttp

@register("astrbot_plugin_gitee_aiimg", "木有知", "接入 Gitee AI 图像生成模型。支持 LLM 调用和命令调用，支持多种比例。", "1.1")
class GiteeAIImage(Star):
    def __init__(self, context: Context, config: dict):
        super().__init__(context)
        self.config = config
        self.base_url = config.get("base_url", "https://ai.gitee.com/v1")
        
        # 支持多Key轮询
        self.api_keys = []
        api_keys = config.get("api_key", [])
        if isinstance(api_keys, str):
            # 兼容旧配置
            if api_keys:
                self.api_keys = [k.strip() for k in api_keys.split(",") if k.strip()]
        elif isinstance(api_keys, list):
            self.api_keys = [str(k).strip() for k in api_keys if str(k).strip()]
        self.current_key_index = 0
        
        self.model = config.get("model", "z-image-turbo")
        self.default_size = config.get("size", "1024x1024")
        self.num_inference_steps = config.get("num_inference_steps", 9)
        self.negative_prompt = config.get("negative_prompt", " low quality, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry")
        
        # Gitee AI 支持的图片比例
        self.supported_ratios = {
            "1:1": ["256x256", "512x512", "1024x1024", "2048x2048"],
            "4:3": ["1152x896", "2048x1536"],
            "3:4": ["768x1024", "1536x2048"],
            "3:2": ["2048x1360"],
            "2:3": ["1360x2048"],
            "16:9": ["1024x576", "2048x1152"],
            "9:16": ["576x1024", "1152x2048"]
        }
        
        # 记录正在生成的用户，防止重复请求
        self.processing_users = set()

    def _get_client(self):
        if not self.api_keys:
             # 尝试重新读取配置
            api_keys = self.config.get("api_key", [])
            if isinstance(api_keys, str):
                if api_keys:
                    self.api_keys = [k.strip() for k in api_keys.split(",") if k.strip()]
            elif isinstance(api_keys, list):
                self.api_keys = [str(k).strip() for k in api_keys if str(k).strip()]
        
        if not self.api_keys:
            raise ValueError("请先配置 API Key")

        # 轮询获取 Key
        api_key = self.api_keys[self.current_key_index]
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)

        return AsyncOpenAI(
            base_url=self.base_url,
            api_key=api_key,
        )

    async def _download_image(self, url: str) -> str:
        """下载图片并保存到临时文件，返回文件路径"""
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                if resp.status != 200:
                    raise Exception(f"下载图片失败: HTTP {resp.status}")
                data = await resp.read()
                
        # 保存到插件数据目录
        plugin_name = "astrbot_plugin_gitee_aiimg"
        data_dir = os.path.join("data", "plugins", plugin_name, "images")
        
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            
        filename = f"{int(time.time())}_{os.urandom(4).hex()}.jpg"
        # 使用 abspath 获取绝对路径
        filepath = os.path.abspath(os.path.join(data_dir, filename))
        
        with open(filepath, "wb") as f:
            f.write(data)
            
        return filepath

    async def _save_base64_image(self, b64_data: str) -> str:
        """保存base64图片到临时文件，返回文件路径"""
        plugin_name = "astrbot_plugin_gitee_aiimg"
        data_dir = os.path.join("data", "plugins", plugin_name, "images")

        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            
        filename = f"{int(time.time())}_{os.urandom(4).hex()}.jpg"
        filepath = os.path.abspath(os.path.join(data_dir, filename))
        
        image_bytes = base64.b64decode(b64_data)
        with open(filepath, "wb") as f:
            f.write(image_bytes)
            
        return filepath

    async def _generate_image(self, prompt: str, size: str = "") -> str:
        """调用 Gitee AI API 生成图片，返回本地文件路径"""
        client = self._get_client()
        
        target_size = size if size else self.default_size

        # 构建参数，过滤掉None或空值的参数
        kwargs = {
            "prompt": prompt,
            "model": self.model,
            "extra_body": {
                "num_inference_steps": self.num_inference_steps,
            }
        }

        if self.negative_prompt:
            kwargs["extra_body"]["negative_prompt"] = self.negative_prompt
        if target_size:
            kwargs["size"] = target_size

        # 支持使用 width 和 height 参数
        if "width" in kwargs.get("extra_body", {}) and "height" in kwargs.get("extra_body", {}):
             if "size" in kwargs:
                 del kwargs["size"]

        try:
            # 这里的调用方式与用户提供的示例一致
            response = await client.images.generate(**kwargs) # type: ignore
        except Exception as e:
            # 优化错误处理
            error_msg = str(e)
            if "401" in error_msg:
                raise Exception("API Key 无效或已过期，请检查配置。")
            elif "429" in error_msg:
                raise Exception("API 调用次数超限或并发过高，请稍后再试。")
            elif "500" in error_msg:
                raise Exception("Gitee AI 服务器内部错误，请稍后再试。")
            else:
                raise Exception(f"API调用失败: {error_msg}")

        if not response.data: # type: ignore
            raise Exception("生成图片失败：未返回数据")

        image_data = response.data[0] # type: ignore
        
        if image_data.url:
            return await self._download_image(image_data.url)
        elif image_data.b64_json:
            return await self._save_base64_image(image_data.b64_json)
        else:
            raise Exception("生成图片失败：未返回 URL 或 Base64 数据")

    @filter.llm_tool(name="draw_image") # type: ignore
    async def draw(self, event: AstrMessageEvent, prompt: str):
        '''根据提示词生成图片。

        Args:
            prompt(string): 图片提示词，需要包含主体、场景、风格等描述
        '''
        user_id = event.get_sender_id()
        
        # 使用 prompt 的 hash 作为标识的一部分
        import hashlib
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
        request_id = f"{user_id}:{prompt_hash}"

        if request_id in self.processing_users:
            return "您有正在进行的生图任务，请稍候..."

        self.processing_users.add(request_id)
        try:
            image_path = await self._generate_image(prompt)
            
            # 使用 Image.fromFileSystem 自动处理路径
            # 优先发送图片消息
            await event.send(event.chain_result([Image.fromFileSystem(image_path)])) # type: ignore
            
            return f"绘图工具调用成功。我已根据提示词生成并发送了图片。使用的提示词(Prompt)是：{prompt}"
            
        except Exception as e:
            logger.error(f"生图失败: {e}")
            return f"生成图片时遇到问题: {str(e)}"
        finally:
            if request_id in self.processing_users:
                self.processing_users.remove(request_id)

    @filter.command("aiimg")
    async def generate_image_command(self, event: AstrMessageEvent, prompt: str):
        """
        生成图片指令
        用法: /aiimg <提示词> [比例]
        示例: /aiimg 一个女孩 9:16
        支持比例: 1:1, 4:3, 3:4, 3:2, 2:3, 16:9, 9:16
        """
        if not prompt:
            yield event.plain_result("请提供提示词！使用方法：/aiimg <提示词> [比例]")
            return

        user_id = event.get_sender_id()
        # 使用 user_id + prompt 的哈希作为标识，防止同一用户并发发送相同 prompt
        import hashlib
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
        request_id = f"{user_id}:{prompt_hash}"

        if request_id in self.processing_users:
            yield event.plain_result("您有正在进行的生图任务，请稍候...")
            return

        self.processing_users.add(request_id)
        
        ratio = "1:1"
        prompt_parts = prompt.rsplit(" ", 1)
        if len(prompt_parts) > 1 and prompt_parts[1] in self.supported_ratios:
            ratio = prompt_parts[1]
            prompt = prompt_parts[0]
            
        # 确定目标尺寸
        target_size = self.default_size
        if ratio != "1:1" or (ratio == "1:1" and self.default_size not in self.supported_ratios["1:1"]):
             # 默认取该比例下的第一个分辨率
             target_size = self.supported_ratios[ratio][0]

        try:
            image_path = await self._generate_image(prompt, size=target_size)
            # 使用 Image.fromFileSystem 自动处理路径
            yield event.chain_result([Image.fromFileSystem(image_path)]) # type: ignore

        except Exception as e:
            logger.error(f"生图失败: {e}")
            yield event.plain_result(f"生成图片失败: {str(e)}") # type: ignore
        finally:
            if request_id in self.processing_users:
                self.processing_users.remove(request_id)