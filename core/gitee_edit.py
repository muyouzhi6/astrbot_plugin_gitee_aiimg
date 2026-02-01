"""
Gitee AI 千问改图后端

基于 Qwen-Image-Edit-2511 模型
使用异步任务 + 轮询模式
"""

import asyncio
import time
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

import aiohttp

from astrbot.api import logger

from .image_format import guess_image_mime_and_ext

if TYPE_CHECKING:
    from .image_manager import ImageManager

EDIT_TASK_TYPES = {"id", "style", "subject", "background", "element"}


class GiteeEditBackend:
    """Gitee AI 千问改图后端"""

    name = "Gitee"

    def __init__(self, config: dict, imgr: "ImageManager"):
        self.config = config
        self.imgr = imgr

        # Gitee 配置
        gitee_conf = config.get("edit", {}).get("gitee", {})
        self.base_url = gitee_conf.get("base_url", "https://ai.gitee.com/v1")
        self.model = gitee_conf.get("model", "Qwen-Image-Edit-2511")
        self.num_inference_steps = gitee_conf.get("num_inference_steps", 4)
        self.guidance_scale = gitee_conf.get("guidance_scale", 1.0)
        self.poll_interval = gitee_conf.get("poll_interval", 5)
        self.poll_timeout = gitee_conf.get("poll_timeout", 300)

        # API Key 池 - 优先用 gitee 配置，否则用 draw 配置
        gitee_keys = gitee_conf.get("api_keys", [])
        draw_keys = config.get("draw", {}).get("api_keys", [])
        raw_keys = gitee_keys or draw_keys
        self.api_keys = [str(k).strip() for k in raw_keys if str(k).strip()]
        self._key_index = 0
        self._key_lock = asyncio.Lock()

        # HTTP Session (带锁保护)
        self._session: aiohttp.ClientSession | None = None
        self._session_lock = asyncio.Lock()

    async def close(self) -> None:
        """清理资源"""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """获取或创建 HTTP Session (线程安全)"""
        if self._session is None or self._session.closed:
            async with self._session_lock:
                # Double-check pattern
                if self._session is None or self._session.closed:
                    connector = aiohttp.TCPConnector(
                        limit=10,
                        limit_per_host=5,
                        ttl_dns_cache=300,
                        enable_cleanup_closed=True,
                    )
                    timeout = aiohttp.ClientTimeout(
                        total=self.poll_timeout + 30,  # 比轮询超时多留余量
                        connect=30,
                    )
                    self._session = aiohttp.ClientSession(
                        connector=connector,
                        timeout=timeout,
                    )
        return self._session

    async def _next_key(self) -> str:
        """轮询获取下一个 API Key (线程安全)"""
        async with self._key_lock:
            if not self.api_keys:
                raise RuntimeError("Gitee API Key 未配置")
            key = self.api_keys[self._key_index]
            self._key_index = (self._key_index + 1) % len(self.api_keys)
            return key

    async def edit(
        self,
        prompt: str,
        images: list[bytes],
        task_types: Iterable[str] = ("id",),
    ) -> Path:
        """
        执行改图

        Args:
            prompt: 提示词
            images: 图片字节列表
            task_types: 任务类型 (id/style/subject/background/element)

        Returns:
            生成图片的本地路径
        """
        if not images:
            raise ValueError("至少需要一张图片")

        api_key = await self._next_key()
        t_start = time.perf_counter()

        logger.info(
            f"[Gitee] 开始改图: model={self.model}, "
            f"task_types={list(task_types)}, images={len(images)}"
        )

        # 创建任务
        task_id = await self._create_task(prompt, images, task_types, api_key)
        t_create = time.perf_counter()
        logger.debug(
            f"[Gitee] 任务创建成功: {task_id}, 耗时: {t_create - t_start:.2f}s"
        )

        # 轮询结果
        file_url = await self._poll_task(task_id, api_key)
        t_poll = time.perf_counter()
        logger.debug(f"[Gitee] 任务完成, 轮询耗时: {t_poll - t_create:.2f}s")

        # 下载图片
        result_path = await self.imgr.download_image(file_url)
        t_end = time.perf_counter()

        logger.info(
            f"[Gitee] 改图完成: 总耗时={t_end - t_start:.2f}s, "
            f"创建={t_create - t_start:.2f}s, 轮询={t_poll - t_create:.2f}s, "
            f"下载={t_end - t_poll:.2f}s"
        )

        return result_path

    async def _create_task(
        self,
        prompt: str,
        images: list[bytes],
        task_types: Iterable[str],
        api_key: str,
    ) -> str:
        """创建异步改图任务"""
        session = await self._get_session()

        data = aiohttp.FormData()
        data.add_field("prompt", prompt)
        data.add_field("model", self.model)
        data.add_field("num_inference_steps", str(self.num_inference_steps))
        data.add_field("guidance_scale", str(self.guidance_scale))

        for t in task_types:
            if t in EDIT_TASK_TYPES:
                data.add_field("task_types", t)

        for i, img in enumerate(images):
            mime, ext = guess_image_mime_and_ext(img)
            data.add_field(
                "image",
                img,
                filename=f"image_{i}.{ext}",
                content_type=mime,
            )

        try:
            async with session.post(
                f"{self.base_url}/async/images/edits",
                headers={"Authorization": f"Bearer {api_key}"},
                data=data,
            ) as resp:
                result = await resp.json()

                if resp.status != 200:
                    error_msg = result.get("message", str(result))
                    logger.error(f"[Gitee] 创建任务失败 ({resp.status}): {error_msg}")
                    raise RuntimeError(f"Gitee 创建任务失败: {error_msg}")

                task_id = result.get("task_id")
                if not task_id:
                    logger.error(f"[Gitee] 响应未包含 task_id: {result}")
                    raise RuntimeError("Gitee 未返回 task_id")

                return task_id

        except aiohttp.ClientError as e:
            logger.error(f"[Gitee] 网络错误: {e}")
            raise RuntimeError(f"Gitee 网络错误: {e}")

    async def _poll_task(self, task_id: str, api_key: str) -> str:
        """轮询任务状态直到完成"""
        session = await self._get_session()
        url = f"{self.base_url}/task/{task_id}"
        max_rounds = self.poll_timeout // self.poll_interval

        for i in range(max_rounds):
            try:
                async with session.get(
                    url,
                    headers={"Authorization": f"Bearer {api_key}"},
                ) as resp:
                    result = await resp.json()
                    status = result.get("status")

                    if status == "success":
                        file_url = result.get("output", {}).get("file_url")
                        if not file_url:
                            logger.error(f"[Gitee] 任务成功但无 file_url: {result}")
                            raise RuntimeError("Gitee 任务成功但未返回 file_url")
                        return file_url

                    if status in {"failed", "cancelled"}:
                        error_msg = result.get("message", status)
                        logger.error(f"[Gitee] 任务失败: {error_msg}")
                        raise RuntimeError(f"Gitee 任务失败: {error_msg}")

                    logger.debug(f"[Gitee] 轮询第{i + 1}轮, 状态: {status}")

            except aiohttp.ClientError as e:
                logger.warning(f"[Gitee] 轮询网络错误 (第{i + 1}轮): {e}")

            await asyncio.sleep(self.poll_interval)

        logger.error(f"[Gitee] 任务超时 (>{self.poll_timeout}s)")
        raise TimeoutError(f"Gitee 任务超时 (>{self.poll_timeout}s)")
