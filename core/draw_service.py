import time
from pathlib import Path

from openai import AsyncOpenAI
from openai.types.images_response import ImagesResponse

from astrbot.api import logger

from .image_manager import ImageManager

# 不支持 negative_prompt 的模型（会把负向提示词当正向处理，导致出图畸形）
MODELS_WITHOUT_NEGATIVE_PROMPT = frozenset(
    {
        "z-image-turbo",
        "z-image-base",
        "flux.1-dev",
        "flux.1-schnell",
    }
)


class ImageDrawService:
    def __init__(self, config: dict, imgr: ImageManager):
        self.imgr = imgr
        self.dconf = config["draw"]

        self.api_keys = [
            str(k).strip() for k in self.dconf["api_keys"] if str(k).strip()
        ]
        self._key_index = 0

        self._clients: dict[str, AsyncOpenAI] = {}

    async def close(self) -> None:
        """清理资源"""
        for client in self._clients.values():
            await client.close()
        self._clients.clear()

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
                base_url=self.dconf["base_url"],
                api_key=key,
                timeout=self.dconf["timeout"],
                max_retries=self.dconf["max_retries"],
            )
            self._clients[key] = client

        return client

    async def generate(self, prompt: str, size: str | None = None) -> Path:
        client = self.get_openai_client()

        kwargs = {
            "prompt": prompt,
            "model": self.dconf["model"],
            "extra_body": {
                "num_inference_steps": self.dconf["num_inference_steps"],
            },
        }

        if self.dconf.get("negative_prompt"):
            # 部分模型不支持 negative_prompt，会把它当正向提示词处理导致出图畸形
            model_name = self.dconf["model"].lower()
            if model_name not in MODELS_WITHOUT_NEGATIVE_PROMPT:
                kwargs["extra_body"]["negative_prompt"] = self.dconf["negative_prompt"]

        kwargs["size"] = size or self.dconf["size"]

        t0 = time.time()
        try:
            resp: ImagesResponse = await client.images.generate(**kwargs)
        except Exception as e:
            logger.error(f"[生图] API 调用失败，耗时: {time.time() - t0:.2f}s")
            self._raise_api_error(e)
            raise  # never reached, but makes type checker happy

        api_time = time.time() - t0
        logger.info(f"[生图] API 响应耗时: {api_time:.2f}s")

        if not resp.data:
            raise RuntimeError("未返回图片数据")

        img = resp.data[0]

        t1 = time.time()
        if img.url:
            result = await self.imgr.download_image(img.url)
            logger.info(
                f"[生图] 下载图片耗时: {time.time() - t1:.2f}s, URL: {img.url[:60]}..."
            )
            return result
        if img.b64_json:
            result = await self.imgr.save_base64_image(img.b64_json)
            logger.info(f"[生图] 保存 base64 图片耗时: {time.time() - t1:.2f}s")
            return result

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
