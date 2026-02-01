import asyncio
from pathlib import Path

import aiohttp
from openai import AsyncOpenAI
from openai.types.images_response import ImagesResponse

from astrbot.api import logger

from .image import ImageManager

# 图生图支持的任务类型
EDIT_TASK_TYPES = ["id", "style", "subject", "background", "element"]


class ImageService:
    def __init__(self, config: dict, imgr: ImageManager):
        self.imgr = imgr
        self.config = config

        self.api_keys = self._parse_api_keys(config["api_key"])
        self._key_index = 0
        self._edit_key_index = 0  # 图生图 key 轮询索引

        self._clients: dict[str, AsyncOpenAI] = {}

    async def close(self) -> None:
        """清理资源"""
        for client in self._clients.values():
            await client.close()
        self._clients.clear()

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

    def _next_key(self) -> str:
        # 支持配置热更新：如果 api_keys 为空，重新从 config 读取
        if not self.api_keys:
            self.api_keys = self._parse_api_keys(self.config.get("api_key", []))

        if not self.api_keys:
            raise Exception("没有可用的 API Key")

        # 边界检查（防止配置热更新后 keys 数量减少导致越界）
        if self._key_index >= len(self.api_keys):
            self._key_index = 0

        key = self.api_keys[self._key_index]
        self._key_index = (self._key_index + 1) % len(self.api_keys)
        return key

    def get_openai_client(self) -> AsyncOpenAI:
        key = self._next_key()

        client = self._clients.get(key)
        if client is None:
            client = AsyncOpenAI(
                base_url=self.config["base_url"],
                api_key=key,
                timeout=self.config["timeout"],
                max_retries=self.config["max_retries"],
            )
            self._clients[key] = client

        return client

    async def generate(self, prompt: str, size: str | None = None) -> Path:
        client = self.get_openai_client()

        kwargs = {
            "prompt": prompt,
            "model": self.config["model"],
            "extra_body": {
                "num_inference_steps": self.config["num_inference_steps"],
            },
        }

        if self.config.get("negative_prompt"):
            kwargs["extra_body"]["negative_prompt"] = self.config["negative_prompt"]
        if size:
            kwargs["size"] = size

        logger.debug(
            f"[generate] 调用 API: model={kwargs['model']}, size={kwargs.get('size', '未指定')}"
        )

        try:
            resp: ImagesResponse = await client.images.generate(**kwargs)
        except Exception as e:
            self._raise_api_error(e)
            raise  # Never reached, but helps type checker

        if not resp.data:
            raise RuntimeError("未返回图片数据")

        img = resp.data[0]
        if img.url:
            return await self.imgr.download_image(img.url)
        if img.b64_json:
            return await self.imgr.save_base64_image(img.b64_json)

        raise RuntimeError("返回数据不包含图片")

    @staticmethod
    def _raise_api_error(e: Exception) -> None:
        msg = str(e)
        if "401" in msg:
            raise RuntimeError("API Key 无效") from e
        if "429" in msg:
            raise RuntimeError("请求过快或额度不足") from e
        if "500" in msg:
            raise RuntimeError("服务端错误") from e
        raise RuntimeError(msg) from e

    # ========== 图生图功能 ==========

    def _get_edit_base_url(self) -> str:
        """获取图生图 API Base URL"""
        return self.config.get("edit_base_url") or self.config["base_url"]

    def _get_edit_api_keys(self) -> list[str]:
        """获取图生图 API Keys"""
        keys = self._parse_api_keys(self.config.get("edit_api_key", []))
        if keys:
            return keys
        # fallback 到文生图的 keys，先触发热更新
        if not self.api_keys:
            self.api_keys = self._parse_api_keys(self.config.get("api_key", []))
        return self.api_keys

    def _next_edit_key(self) -> str:
        """获取下一个图生图 API Key（轮询）"""
        keys = self._get_edit_api_keys()
        if not keys:
            raise RuntimeError("没有可用的图生图 API Key")

        # 边界检查（防止配置热更新后 keys 数量减少导致越界）
        if self._edit_key_index >= len(keys):
            self._edit_key_index = 0

        key = keys[self._edit_key_index]
        self._edit_key_index = (self._edit_key_index + 1) % len(keys)
        return key

    async def _create_edit_task(
        self,
        prompt: str,
        image_data_list: list[bytes],
        task_types: list[str],
    ) -> tuple[str, str]:
        """创建图生图异步任务，返回 (task_id, api_key)"""
        api_key = self._next_edit_key()
        base_url = self._get_edit_base_url()

        headers = {
            "X-Failover-Enabled": "true",
            "Authorization": f"Bearer {api_key}",
        }

        # 构建 multipart/form-data
        data = aiohttp.FormData()
        data.add_field("prompt", prompt)
        data.add_field("model", self.config.get("edit_model", "Qwen-Image-Edit-2511"))
        data.add_field(
            "num_inference_steps",
            str(self.config.get("edit_num_inference_steps", 4)),
        )
        data.add_field(
            "guidance_scale",
            str(self.config.get("edit_guidance_scale", 1.0)),
        )

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

        api_url = f"{base_url}/async/images/edits"

        async with self.imgr._session.post(api_url, headers=headers, data=data) as resp:
            result = await resp.json()
            if resp.status != 200:
                error_msg = result.get("message", str(result))
                if resp.status == 401:
                    raise RuntimeError("图生图 API Key 无效或已过期")
                elif resp.status == 429:
                    raise RuntimeError("API 调用次数超限或并发过高")
                else:
                    raise RuntimeError(f"创建图生图任务失败: {error_msg}")

            task_id = result.get("task_id")
            if not task_id:
                raise RuntimeError(
                    f"创建图生图任务失败：未返回 task_id。响应: {result}"
                )

            return task_id, api_key

    async def _poll_edit_task(self, task_id: str, api_key: str) -> str:
        """轮询图生图任务状态，返回结果图片 URL"""
        base_url = self._get_edit_base_url()
        poll_interval = self.config.get("edit_poll_interval", 5)
        poll_timeout = self.config.get("edit_poll_timeout", 300)

        headers = {"Authorization": f"Bearer {api_key}"}
        status_url = f"{base_url}/task/{task_id}"
        max_attempts = int(poll_timeout / poll_interval)

        for attempt in range(1, max_attempts + 1):
            async with self.imgr._session.get(status_url, headers=headers) as resp:
                result = await resp.json()

                if result.get("error"):
                    raise RuntimeError(
                        f"任务出错: {result['error']}: {result.get('message', 'Unknown error')}"
                    )

                status = result.get("status", "unknown")

                if status == "success":
                    output = result.get("output", {})
                    file_url = output.get("file_url")
                    if file_url:
                        return file_url
                    else:
                        raise RuntimeError("任务完成但未返回图片 URL")

                elif status in ["failed", "cancelled"]:
                    raise RuntimeError(f"图生图任务 {status}")

                logger.debug(
                    f"图生图任务 {task_id} 状态: {status} (第 {attempt} 次检查)"
                )

            await asyncio.sleep(poll_interval)

        raise RuntimeError(f"图生图任务超时（等待超过 {poll_timeout} 秒）")

    async def edit_image(
        self,
        prompt: str,
        image_data_list: list[bytes],
        task_types: list[str] | None = None,
    ) -> Path:
        """执行图生图，返回本地文件路径"""
        if not image_data_list:
            raise ValueError("请提供至少一张图片")

        # 默认任务类型
        if task_types is None:
            task_types = ["id"]

        # 验证任务类型
        valid_types = [t for t in task_types if t in EDIT_TASK_TYPES]
        if not valid_types:
            valid_types = ["id"]

        # 创建任务
        task_id, api_key = await self._create_edit_task(
            prompt, image_data_list, valid_types
        )
        logger.info(f"图生图任务已创建: {task_id}")

        # 轮询等待结果
        file_url = await self._poll_edit_task(task_id, api_key)

        # 下载结果图片
        filepath = await self.imgr.download_image(file_url)

        return filepath
