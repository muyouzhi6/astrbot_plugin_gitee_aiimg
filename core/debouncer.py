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
        self._interval = config["debounce_interval"]
        self._ttl = 300
        self._cleanup_threshold = 100

        self._records: dict[str, float] = {}

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
