import time


class Debouncer:
    """基于时间窗口的防抖器（支持 TTL 自动清理）"""

    def __init__(self, config: dict):
        """
        Args:
            interval: 防抖时间（秒）
            ttl: 操作记录最长保留时间（秒）
            cleanup_threshold: 记录数量超过该值时触发清理
        """
        self._interval = config.get("debounce_interval", 3)
        self._ttl = 300
        self._cleanup_threshold = 100

        self._records: dict[str, float] = {}

        # LLM 工具去重缓存（防止 ToolLoop 重复调用）
        self._llm_dedup_cache: dict[str, float] = {}
        self._llm_dedup_ttl = config.get("llm_dedup_ttl", 120)  # 默认 120 秒

    def hit(self, key: str) -> bool:
        """
        记录一次操作并判断是否命中防抖

        Returns:
            True  -> 需要拒绝（命中防抖）
            False -> 允许通过
        """
        now = time.time()

        if len(self._records) >= self._cleanup_threshold:
            self._cleanup(now)

        last = self._records.get(key)
        if last is not None and now - last < self._interval:
            return True

        self._records[key] = now
        return False

    def _cleanup(self, now: float) -> None:
        """清理过期记录"""
        expired = [k for k, ts in self._records.items() if now - ts > self._ttl]
        for k in expired:
            self._records.pop(k, None)

    def clear_all(self) -> None:
        """清空所有记录"""
        self._records.clear()
        self._llm_dedup_cache.clear()

    # ==================== LLM 工具去重 ====================

    def llm_tool_is_duplicate(self, message_id: str, origin: str) -> bool:
        """检查 LLM 工具调用是否重复（用于阻止 ToolLoop 重复调用）

        Args:
            message_id: 消息 ID
            origin: 统一来源标识 (unified_msg_origin)

        Returns:
            True  -> 重复调用，应拒绝
            False -> 首次调用，允许并记录
        """
        now = time.time()
        key = f"{origin}:{message_id}"

        # 清理过期缓存
        if len(self._llm_dedup_cache) >= self._cleanup_threshold:
            self._cleanup_llm_dedup(now)

        # 检查是否存在且未过期
        ts = self._llm_dedup_cache.get(key)
        if ts is not None and now - ts < self._llm_dedup_ttl:
            return True  # 重复

        # 记录本次调用
        self._llm_dedup_cache[key] = now
        return False

    def _cleanup_llm_dedup(self, now: float) -> None:
        """清理过期的 LLM 去重缓存"""
        expired = [k for k, ts in self._llm_dedup_cache.items() if now - ts > self._llm_dedup_ttl]
        for k in expired:
            self._llm_dedup_cache.pop(k, None)
