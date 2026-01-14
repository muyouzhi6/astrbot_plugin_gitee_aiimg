"""
改图路由器

功能:
- 后端选择 (Gemini / Gitee)
- 预设提示词管理
- 重试机制 (指数退避)
- 自动降级
- 详细日志
"""

import asyncio
import time
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Union

from astrbot.api import logger

from .gemini_edit import GeminiEditBackend
from .gitee_edit import GiteeEditBackend

if TYPE_CHECKING:
    from .image_manager import ImageManager

# 后端类型
EditBackendType = Union[GeminiEditBackend, GiteeEditBackend]


# 内置预设提示词 (留空，用户可通过配置自定义)
BUILTIN_PRESETS: dict[str, str] = {}


class EditRouter:
    """
    改图路由器

    负责:
    1. 后端选择和管理
    2. 预设提示词处理
    3. 重试和降级逻辑
    """

    def __init__(self, config: dict, imgr: "ImageManager"):
        self.config = config
        self.imgr = imgr
        self.edit_config = config.get("edit", {})

        # 初始化后端
        self.backends: dict[str, EditBackendType] = {}
        self._init_backends()

        # 加载预设
        self.presets = self._load_presets()

        logger.info(
            f"[EditRouter] 初始化完成: "
            f"后端={list(self.backends.keys())}, "
            f"预设={len(self.presets)}个"
        )

    def _init_backends(self) -> None:
        """初始化后端实例"""
        # Gemini 后端
        gemini_conf = self.edit_config.get("gemini", {})
        if gemini_conf.get("api_keys"):
            try:
                self.backends["gemini"] = GeminiEditBackend(self.config, self.imgr)
                logger.info("[EditRouter] Gemini 后端已加载")
            except Exception as e:
                logger.warning(f"[EditRouter] Gemini 后端初始化失败: {e}")

        # Gitee 后端
        gitee_conf = self.edit_config.get("gitee", {})
        draw_keys = self.config.get("draw", {}).get("api_keys", [])
        if gitee_conf.get("api_keys") or draw_keys:
            try:
                self.backends["gitee"] = GiteeEditBackend(self.config, self.imgr)
                logger.info("[EditRouter] Gitee 后端已加载")
            except Exception as e:
                logger.warning(f"[EditRouter] Gitee 后端初始化失败: {e}")

        if not self.backends:
            logger.warning("[EditRouter] 警告: 无可用后端，请检查 API Key 配置")

    def _load_presets(self) -> dict[str, str]:
        """加载预设提示词 (内置 + 配置)"""
        presets = dict(BUILTIN_PRESETS)

        # 从配置加载自定义预设
        custom_presets = self.edit_config.get("presets", [])
        for item in custom_presets:
            if isinstance(item, str) and ":" in item:
                key, val = item.split(":", 1)
                presets[key.strip()] = val.strip()

        return presets

    def get_preset_names(self) -> list[str]:
        """获取所有预设名称"""
        return list(self.presets.keys())

    def get_available_backends(self) -> list[str]:
        """获取可用后端列表"""
        return list(self.backends.keys())

    async def edit(
        self,
        prompt: str,
        images: list[bytes],
        backend: str | None = None,
        preset: str | None = None,
        task_types: Iterable[str] = ("id",),
    ) -> Path:
        """
        执行改图

        Args:
            prompt: 提示词 (如果指定 preset 则被覆盖)
            images: 图片字节列表
            backend: 指定后端 (None=使用默认)
            preset: 预设名称 (会覆盖 prompt)
            task_types: Gitee 任务类型 (仅 Gitee 后端使用)

        Returns:
            生成图片的本地路径

        Raises:
            ValueError: 参数错误
            RuntimeError: 执行失败
        """
        if not images:
            raise ValueError("至少需要一张图片")

        if not self.backends:
            raise RuntimeError("无可用改图后端，请检查 API Key 配置")

        # 1. 预设处理 (支持预设+追加提示词)
        if preset and preset in self.presets:
            preset_prompt = self.presets[preset]
            if prompt:
                # 预设 + 追加提示词: "预设内容, 用户追加内容"
                prompt = f"{preset_prompt}, {prompt}"
                logger.debug(f"[EditRouter] 使用预设 '{preset}' + 追加: {prompt[:50]}...")
            else:
                prompt = preset_prompt
                logger.debug(f"[EditRouter] 使用预设 '{preset}'")

        if not prompt:
            prompt = "Transform this image with artistic style"

        # 2. 确定后端
        default_backend = self.edit_config.get("default_backend", "gemini")
        target_backend = backend or default_backend

        # 如果指定的后端不可用，尝试使用其他后端
        if target_backend not in self.backends:
            available = list(self.backends.keys())
            if not available:
                raise RuntimeError("无可用改图后端")
            target_backend = available[0]
            logger.warning(
                f"[EditRouter] 指定后端不可用，切换到 {target_backend}"
            )

        # 3. 执行 (带重试和降级)
        return await self._execute_with_fallback(
            target_backend, prompt, images, task_types
        )

    async def _execute_with_fallback(
        self,
        backend: str,
        prompt: str,
        images: list[bytes],
        task_types: Iterable[str],
    ) -> Path:
        """带重试和降级的执行逻辑"""
        fallback_config = self.edit_config.get("fallback", {})
        max_retries = fallback_config.get("max_retries", 2)
        retry_delay = fallback_config.get("retry_delay", 2)
        fallback_enabled = fallback_config.get("enabled", True)

        t_start = time.perf_counter()
        last_error: Exception | None = None

        # 主后端重试
        for attempt in range(max_retries + 1):
            try:
                logger.info(
                    f"[EditRouter] [{backend}] 第{attempt + 1}次尝试, "
                    f"prompt={prompt[:50]}..."
                )

                # 根据后端类型调用不同参数
                if backend == "gitee":
                    gitee_backend = self.backends[backend]
                    assert isinstance(gitee_backend, GiteeEditBackend)
                    result = await gitee_backend.edit(prompt, images, task_types=task_types)
                else:
                    gemini_backend = self.backends[backend]
                    assert isinstance(gemini_backend, GeminiEditBackend)
                    result = await gemini_backend.edit(prompt, images)

                t_end = time.perf_counter()
                logger.info(
                    f"[EditRouter] [{backend}] 成功 (第{attempt + 1}次), "
                    f"总耗时={t_end - t_start:.2f}s"
                )
                return result

            except Exception as e:
                last_error = e
                logger.warning(
                    f"[EditRouter] [{backend}] 第{attempt + 1}次失败: {e}"
                )

                if attempt < max_retries:
                    # 指数退避
                    delay = retry_delay * (2 ** attempt)
                    logger.info(f"[EditRouter] {delay}s 后重试...")
                    await asyncio.sleep(delay)

        # 降级到其他后端
        if fallback_enabled and len(self.backends) > 1:
            # 找到其他可用后端
            fallback_target = None
            for name in self.backends:
                if name != backend:
                    fallback_target = name
                    break

            if fallback_target:
                logger.warning(
                    f"[EditRouter] [{backend}] {max_retries + 1}次全部失败, "
                    f"降级到 {fallback_target}"
                )

                try:
                    fallback_instance = self.backends[fallback_target]
                    if fallback_target == "gitee":
                        assert isinstance(fallback_instance, GiteeEditBackend)
                        result = await fallback_instance.edit(prompt, images, task_types=task_types)
                    else:
                        assert isinstance(fallback_instance, GeminiEditBackend)
                        result = await fallback_instance.edit(prompt, images)

                    t_end = time.perf_counter()
                    logger.info(
                        f"[EditRouter] [{fallback_target}] 降级成功, "
                        f"总耗时={t_end - t_start:.2f}s"
                    )
                    return result

                except Exception as fallback_error:
                    logger.error(
                        f"[EditRouter] [{fallback_target}] 降级也失败: {fallback_error}"
                    )
                    raise RuntimeError(
                        f"{backend} 和 {fallback_target} 均失败: "
                        f"{last_error} / {fallback_error}"
                    ) from fallback_error

        # 无法降级，抛出原始错误
        raise RuntimeError(f"[{backend}] 改图失败: {last_error}") from last_error

    async def close(self) -> None:
        """清理所有后端资源"""
        for name, backend in self.backends.items():
            try:
                await backend.close()
                logger.debug(f"[EditRouter] {name} 后端已关闭")
            except Exception as e:
                logger.warning(f"[EditRouter] 关闭 {name} 后端时出错: {e}")
