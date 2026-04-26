from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Generic, TypeVar


T = TypeVar("T")
R = TypeVar("R")


@dataclass(slots=True)
class BatchRunResult(Generic[R]):
    index: int
    success: bool
    value: R | None = None
    error: Exception | None = None


async def run_batch(
    items: list[T],
    *,
    concurrency: int,
    runner: Callable[[int, T], Awaitable[R]],
) -> list[BatchRunResult[R]]:
    if not items:
        return []

    limit = max(1, int(concurrency or 1))
    sem = asyncio.Semaphore(limit)
    results: list[BatchRunResult[R] | None] = [None] * len(items)

    async def _run_one(index: int, item: T) -> None:
        async with sem:
            try:
                value = await runner(index, item)
            except Exception as exc:
                results[index] = BatchRunResult(index=index, success=False, error=exc)
            else:
                results[index] = BatchRunResult(index=index, success=True, value=value)

    await asyncio.gather(*(_run_one(idx, item) for idx, item in enumerate(items)))
    return [result for result in results if result is not None]
