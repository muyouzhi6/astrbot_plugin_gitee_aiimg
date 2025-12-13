from astrbot.api.message_components import Plain, Image
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register, StarTools
from astrbot.api import logger
import os
import time
import base64
import asyncio
import aiohttp
from openai import AsyncOpenAI
from pathlib import Path
from typing import Optional, Tuple


@register("astrbot_plugin_gitee_aiimg", "木有知", "接入 Gitee AI 图像生成模型。支持 LLM 调用和命令调用，支持多种比例。", "1.2.0")
class GiteeAIImage(Star):
    def __init__(self, context: Context, config: dict):
        super().__init__(context)
        self.config = config
        self.base_url = config.get("base_url", "https://ai.gitee.com/v1")
        
        self.api_keys = []
        api_keys = config.get("api_key", [])
        if isinstance(api_keys, str):
            if api_keys:
                self.api_keys = [k.strip() for k in api_keys.split(",") if k.strip()]
        elif isinstance(api_keys, list):
            self.api_keys = [str(k).strip() for k in api_keys if str(k).strip()]
        self.current_key_index = 0
        
        self.model = config.get("model", "z-image-turbo")
        self.default_size = config.get("size", "1024x1024")
        self.num_inference_steps = config.get("num_inference_steps", 9)
        self.negative_prompt = config.get("negative_prompt", "low quality, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, normal quality, jpeg artifacts, signature, watermark, username, blurry")
        
        # 超时配置（秒）- 需要小于 AstrBot 的60秒限制
        self.generation_timeout = config.get("generation_timeout", 50)
        
        # 缓存清理配置
        self.cache_cleanup_enabled = config.get("cache_cleanup_enabled", True)
        self.cache_max_age_hours = config.get("cache_max_age_hours", 24)
        self.cache_max_count = config.get("cache_max_count", 200)
        self.cache_protect_minutes = config.get("cache_protect_minutes", 5)
        self.cache_cleanup_interval = config.get("cache_cleanup_interval_minutes", 30) * 60
        
        self.supported_ratios = {
            "1:1": ["256x256", "512x512", "1024x1024", "2048x2048"],
            "4:3": ["1152x896", "2048x1536"],
            "3:4": ["768x1024", "1536x2048"],
            "3:2": ["2048x1360"],
            "2:3": ["1360x2048"],
            "16:9": ["1024x576", "2048x1152"],
            "9:16": ["576x1024", "1152x2048"]
        }
        
        self.image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
        
        # 防抖机制
        self.processing_users = set()
        self.processed_message_ids = {}
        self.user_completion_times = {}
        
        # 清理任务
        self._cleanup_task: Optional[asyncio.Task] = None
        self._state_cleanup_task: Optional[asyncio.Task] = None
        self._start_cleanup_task()

    def _start_cleanup_task(self):
        """启动定时清理任务"""
        if self.cache_cleanup_enabled and self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            logger.info("[GiteeAIImage] 缓存清理任务已启动")
        
        if self._state_cleanup_task is None:
            self._state_cleanup_task = asyncio.create_task(self._state_cleanup_loop())

    async def _state_cleanup_loop(self):
        """定期清理过期的状态记录"""
        while True:
            await asyncio.sleep(300)
            try:
                current_time = time.time()
                expired_msgs = [k for k, v in self.processed_message_ids.items() if current_time - v > 600]
                for k in expired_msgs:
                    del self.processed_message_ids[k]
                
                expired_users = [k for k, v in self.user_completion_times.items() if current_time - v > 600]
                for k in expired_users:
                    del self.user_completion_times[k]
                
                if expired_msgs or expired_users:
                    logger.debug(f"[GiteeAIImage] 清理过期状态: {len(expired_msgs)} 消息, {len(expired_users)} 用户")
            except Exception as e:
                logger.error(f"[GiteeAIImage] 状态清理异常: {e}")

    async def _cleanup_loop(self):
        """定时清理循环"""
        await asyncio.sleep(10)
        
        while True:
            try:
                await self._do_cleanup()
            except Exception as e:
                logger.error(f"[GiteeAIImage] 清理任务异常: {e}")
            
            await asyncio.sleep(self.cache_cleanup_interval)

    def _get_image_dir(self) -> Path:
        """获取图片缓存目录"""
        base_dir = StarTools.get_data_dir("astrbot_plugin_gitee_aiimg")
        image_dir = base_dir / "images"
        image_dir.mkdir(exist_ok=True)
        return image_dir

    def _parse_file_timestamp(self, filename: str) -> Optional[int]:
        """从文件名解析时间戳"""
        try:
            name_part = filename.rsplit(".", 1)[0]
            timestamp_str = name_part.split("_")[0]
            return int(timestamp_str)
        except (ValueError, IndexError):
            return None

    def _get_file_age(self, filepath: Path) -> float:
        """获取文件年龄（秒），优先从文件名解析"""
        timestamp = self._parse_file_timestamp(filepath.name)
        if timestamp is not None:
            return time.time() - timestamp
        try:
            return time.time() - filepath.stat().st_mtime
        except OSError:
            return 0

    def _is_image_file(self, filepath: Path) -> bool:
        """检查是否为图片文件"""
        return filepath.suffix.lower() in self.image_extensions

    async def _do_cleanup(self) -> Tuple[int, int, float]:
        """执行清理操作"""
        image_dir = self._get_image_dir()
        
        if not image_dir.exists():
            return 0, 0, 0.0
        
        max_age_seconds = self.cache_max_age_hours * 3600
        protect_seconds = self.cache_protect_minutes * 60
        
        files_info = []
        for filepath in image_dir.iterdir():
            if filepath.is_file() and self._is_image_file(filepath):
                try:
                    age = self._get_file_age(filepath)
                    size = filepath.stat().st_size
                    files_info.append({
                        "path": filepath,
                        "age": age,
                        "size": size
                    })
                except OSError:
                    continue
        
        files_info.sort(key=lambda x: x["age"], reverse=True)
        
        to_delete = []
        freed_bytes = 0
        
        for info in files_info:
            if info["age"] > max_age_seconds and info["age"] > protect_seconds:
                to_delete.append(info)
                freed_bytes += info["size"]
        
        remaining = [f for f in files_info if f not in to_delete]
        while len(remaining) > self.cache_max_count:
            oldest = remaining[0]
            if oldest["age"] > protect_seconds:
                to_delete.append(oldest)
                freed_bytes += oldest["size"]
                remaining.pop(0)
            else:
                break
        
        deleted_count = 0
        for info in to_delete:
            try:
                info["path"].unlink()
                deleted_count += 1
            except OSError as e:
                logger.warning(f"[GiteeAIImage] 删除文件失败 {info['path'].name}: {e}")
        
        freed_mb = freed_bytes / (1024 * 1024)
        remaining_count = len(files_info) - deleted_count
        
        if deleted_count > 0:
            logger.info(f"[GiteeAIImage] 缓存清理完成: 删除 {deleted_count} 张, 剩余 {remaining_count} 张, 释放 {freed_mb:.2f} MB")
        
        return deleted_count, remaining_count, freed_mb

    def _get_cache_stats(self) -> dict:
        """获取缓存统计信息"""
        image_dir = self._get_image_dir()
        
        if not image_dir.exists():
            return {"count": 0, "size_mb": 0.0, "oldest_hours": 0.0}
        
        total_size = 0
        oldest_age = 0
        count = 0
        
        for filepath in image_dir.iterdir():
            if filepath.is_file() and self._is_image_file(filepath):
                try:
                    total_size += filepath.stat().st_size
                    age = self._get_file_age(filepath)
                    oldest_age = max(oldest_age, age)
                    count += 1
                except OSError:
                    continue
        
        return {
            "count": count,
            "size_mb": total_size / (1024 * 1024),
            "oldest_hours": oldest_age / 3600
        }

    def _get_client(self):
        if not self.api_keys:
            api_keys = self.config.get("api_key", [])
            if isinstance(api_keys, str):
                if api_keys:
                    self.api_keys = [k.strip() for k in api_keys.split(",") if k.strip()]
            elif isinstance(api_keys, list):
                self.api_keys = [str(k).strip() for k in api_keys if str(k).strip()]
        
        if not self.api_keys:
            raise ValueError("请先配置 API Key")
        api_key = self.api_keys[self.current_key_index]
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        return AsyncOpenAI(
            base_url=self.base_url,
            api_key=api_key,
            timeout=self.generation_timeout + 5,  # 给 OpenAI 客户端设置超时
        )

    def _get_save_path(self, extension: str = ".jpg") -> str:
        image_dir = self._get_image_dir()
        filename = f"{int(time.time())}_{os.urandom(4).hex()}{extension}"
        return str(image_dir / filename)

    async def _download_image(self, url: str) -> str:
        timeout = aiohttp.ClientTimeout(total=30)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url) as resp:
                if resp.status != 200:
                    raise Exception(f"下载图片失败: HTTP {resp.status}")
                data = await resp.read()
                
        filepath = self._get_save_path()
        with open(filepath, "wb") as f:
            f.write(data)
            
        return filepath

    async def _save_base64_image(self, b64_data: str) -> str:
        filepath = self._get_save_path()
        image_bytes = base64.b64decode(b64_data)
        with open(filepath, "wb") as f:
            f.write(image_bytes)
        return filepath

    async def _generate_image(self, prompt: str, size: str = "") -> str:
        """生成图片，带超时控制"""
        client = self._get_client()
        
        target_size = size if size else self.default_size
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
        
        try:
            # 使用自己的超时控制，确保在框架超时前返回
            response = await asyncio.wait_for(
                client.images.generate(**kwargs),
                timeout=self.generation_timeout
            )
        except asyncio.TimeoutError:
            raise Exception(f"生成超时（{self.generation_timeout}秒），服务器繁忙，请稍后再试")
        except asyncio.CancelledError:
            raise Exception("生成被取消，请稍后再试")
        except Exception as e:
            error_msg = str(e)
            if "401" in error_msg:
                raise Exception("API Key 无效或已过期，请检查配置")
            elif "429" in error_msg:
                raise Exception("API 调用次数超限，请稍后再试")
            elif "500" in error_msg:
                raise Exception("服务器内部错误，请稍后再试")
            else:
                raise Exception(f"API调用失败: {error_msg}")
        
        if not response.data:
            raise Exception("生成图片失败：未返回数据")
        image_data = response.data[0]
        
        if image_data.url:
            return await self._download_image(image_data.url)
        elif image_data.b64_json:
            return await self._save_base64_image(image_data.b64_json)
        else:
            raise Exception("生成图片失败：未返回 URL 或 Base64 数据")

    def _get_message_id(self, event: AstrMessageEvent) -> str:
        """获取消息的唯一标识"""
        try:
            msg_id = event.message_obj.message_id
            if msg_id:
                return str(msg_id)
        except:
            pass
        user_id = event.get_sender_id()
        msg_str = event.message_str[:100] if event.message_str else ""
        return f"{user_id}_{hash(msg_str)}"

    @filter.llm_tool(name="draw_image")
    async def draw(self, event: AstrMessageEvent, prompt: str):
        '''根据提示词生成图片。

        【重要限制】
        - 每条用户消息最多调用一次
        - 如果返回"已处理"或"已发送"，禁止再次调用
        - 调用后用文字回复用户
        
        Args:
            prompt(string): 图片提示词（使用中文），包含主体、外貌、服装、场景、风格、光线等描述
        '''
        call_id = f"{time.time():.3f}"
        user_id = event.get_sender_id()
        message_id = self._get_message_id(event)
        current_time = time.time()
        
        logger.info(f"[DEBUG-{call_id}] ========== draw_image 被调用 ==========")
        logger.info(f"[DEBUG-{call_id}] user_id: {user_id}, message_id: {message_id}")
        logger.info(f"[DEBUG-{call_id}] prompt: {prompt[:50]}...")
        
        # 检查1: 此消息是否已经处理过
        if message_id in self.processed_message_ids:
            logger.info(f"[DEBUG-{call_id}] ⚠️ 此消息已处理过，拦截重复调用")
            return "图片已生成并发送，请直接用文字回复用户，不要再调用工具。"
        
        # 检查2: 用户冷却期
        if user_id in self.user_completion_times:
            time_since = current_time - self.user_completion_times[user_id]
            logger.info(f"[DEBUG-{call_id}] 距上次完成: {time_since:.2f}秒")
            if time_since < 30.0:
                logger.info(f"[DEBUG-{call_id}] ⚠️ 用户冷却中，拦截")
                return "图片已生成并发送，请直接用文字回复用户，不要再调用工具。"
        
        # 检查3: 是否正在处理中
        if user_id in self.processing_users:
            logger.info(f"[DEBUG-{call_id}] ⚠️ 正在处理中，拦截")
            return "图片正在生成中，请等待，不要再调用工具。"
        
        # 标记状态
        self.processed_message_ids[message_id] = current_time
        self.processing_users.add(user_id)
        logger.info(f"[DEBUG-{call_id}] 开始生成图片...")
        
        try:
            image_path = await self._generate_image(prompt)
            logger.info(f"[DEBUG-{call_id}] 图片生成完成: {image_path}")
            
            # 后台发送图片
            async def send_image():
                try:
                    await event.send(event.chain_result([Image.fromFileSystem(image_path)]))
                    logger.info(f"[DEBUG-{call_id}] 图片已发送")
                except Exception as e:
                    logger.error(f"[DEBUG-{call_id}] 图片发送失败: {e}")
            
            asyncio.create_task(send_image())
            
            self.user_completion_times[user_id] = time.time()
            logger.info(f"[DEBUG-{call_id}] 已记录完成时间")
            
            return f"图片已成功生成并发送。提示词是「{prompt[:30]}」。请用文字自然地回复用户，不要再调用任何工具。"
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"[DEBUG-{call_id}] 生图失败: {error_msg}")
            # 失败时移除消息标记，允许重试
            self.processed_message_ids.pop(message_id, None)
            return f"生成失败: {error_msg}。请告诉用户生成失败了，不要再调用工具。"
        finally:
            self.processing_users.discard(user_id)
            logger.info(f"[DEBUG-{call_id}] ========== draw_image 结束 ==========")

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
        
        if user_id in self.processing_users:
            yield event.plain_result("您有正在进行的生图任务，请稍候...")
            return
        
        self.processing_users.add(user_id)
        
        # 解析比例
        ratio = "1:1"
        prompt_parts = prompt.rsplit(" ", 1)
        if len(prompt_parts) > 1 and prompt_parts[1] in self.supported_ratios:
            ratio = prompt_parts[1]
            prompt = prompt_parts[0]
            
        target_size = self.default_size
        if ratio != "1:1" or (ratio == "1:1" and self.default_size not in self.supported_ratios["1:1"]):
            target_size = self.supported_ratios[ratio][0]
        
        try:
            image_path = await self._generate_image(prompt, size=target_size)
            yield event.chain_result([Image.fromFileSystem(image_path)])
        except Exception as e:
            logger.error(f"生图失败: {e}")
            yield event.plain_result(f"生成图片失败: {str(e)}")
        finally:
            self.processing_users.discard(user_id)

    @filter.command("aiimg_clean")
    async def clean_cache_command(self, event: AstrMessageEvent):
        """清空所有图片缓存"""
        image_dir = self._get_image_dir()
        
        if not image_dir.exists():
            yield event.plain_result("缓存目录不存在")
            return
        
        before_stats = self._get_cache_stats()
        
        if before_stats["count"] == 0:
            yield event.plain_result("缓存目录为空，无需清理")
            return
        
        yield event.plain_result(
            f"开始清理缓存...\n"
            f"当前: {before_stats['count']} 张, {before_stats['size_mb']:.2f} MB"
        )
        
        deleted_count = 0
        freed_bytes = 0
        
        for filepath in image_dir.iterdir():
            if filepath.is_file() and self._is_image_file(filepath):
                try:
                    freed_bytes += filepath.stat().st_size
                    filepath.unlink()
                    deleted_count += 1
                except OSError as e:
                    logger.warning(f"[GiteeAIImage] 删除文件失败 {filepath.name}: {e}")
                    continue
        
        freed_mb = freed_bytes / (1024 * 1024)
        
        if deleted_count > 0:
            logger.info(f"[GiteeAIImage] 手动清理完成: 删除 {deleted_count} 张, 释放 {freed_mb:.2f} MB")
            yield event.plain_result(
                f"✅ 清理完成\n"
                f"删除: {deleted_count} 张\n"
                f"释放: {freed_mb:.2f} MB"
            )
        else:
            yield event.plain_result("没有成功删除任何文件")

    @filter.command("aiimg_stats")
    async def cache_stats_command(self, event: AstrMessageEvent):
        """查看缓存统计信息"""
        stats = self._get_cache_stats()
        
        cleanup_status = "已启用" if self.cache_cleanup_enabled else "已禁用"
        
        yield event.plain_result(
            f"📊 图片缓存统计\n"
            f"━━━━━━━━━━━━━━━\n"
            f"缓存数量: {stats['count']} 张\n"
            f"占用空间: {stats['size_mb']:.2f} MB\n"
            f"最旧文件: {stats['oldest_hours']:.1f} 小时前\n"
            f"━━━━━━━━━━━━━━━\n"
            f"自动清理: {cleanup_status}\n"
            f"保留时间: {self.cache_max_age_hours} 小时\n"
            f"数量上限: {self.cache_max_count} 张\n"
            f"生成超时: {self.generation_timeout} 秒"
        )

    async def terminate(self):
        """插件卸载时清理资源"""
        if self._cleanup_task is not None:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None
        
        if self._state_cleanup_task is not None:
            self._state_cleanup_task.cancel()
            try:
                await self._state_cleanup_task
            except asyncio.CancelledError:
                pass
            self._state_cleanup_task = None
        
        logger.info("[GiteeAIImage] 清理任务已停止")
