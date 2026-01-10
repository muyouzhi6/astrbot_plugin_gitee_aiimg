from pathlib import Path

from openai import AsyncOpenAI
from openai.types.images_response import ImagesResponse

from .image import ImageManager


class ImageService:
    def __init__(self, config: dict, imgr: ImageManager):
        self.imgr = imgr
        self.config = config

        self.api_keys = self._parse_api_keys(config["api_key"])
        self._key_index = 0

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
        if not self.api_keys:
            raise Exception("没有可用的 API Key")
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

        try:
            resp: ImagesResponse = await client.images.generate(**kwargs)
        except Exception as e:
            self._raise_api_error(e)

        if not resp.data:
            raise RuntimeError("未返回图片数据")

        img = resp.data[0]
        if img.url:
            return await self.imgr.download_image(img.url)
        if img.b64_json:
            return await self.imgr.save_base64_image(img.b64_json)

        raise RuntimeError("返回数据不包含图片")

    @staticmethod
    def _raise_api_error(e: Exception):
        msg = str(e)
        if "401" in msg:
            raise RuntimeError("API Key 无效")
        if "429" in msg:
            raise RuntimeError("请求过快或额度不足")
        if "500" in msg:
            raise RuntimeError("服务端错误")
        raise RuntimeError(msg)
