import asyncio
import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "core" / "batch_executor.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("batch_executor_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_run_batch_keeps_partial_success():
    mod = _load_module()

    async def _runner(index, item):
        if item == "bad":
            raise RuntimeError("boom")
        return f"ok-{index}-{item}"

    results = asyncio.run(
        mod.run_batch(["a", "bad", "c"], concurrency=2, runner=_runner)
    )

    assert [r.success for r in results] == [True, False, True]
    assert results[0].value == "ok-0-a"
    assert str(results[1].error) == "boom"
    assert results[2].value == "ok-2-c"


def test_run_batch_respects_concurrency_limit():
    mod = _load_module()
    state = {"active": 0, "peak": 0}

    async def _runner(index, item):
        state["active"] += 1
        state["peak"] = max(state["peak"], state["active"])
        await asyncio.sleep(0.01)
        state["active"] -= 1
        return item

    results = asyncio.run(mod.run_batch(list(range(5)), concurrency=2, runner=_runner))

    assert len(results) == 5
    assert all(r.success for r in results)
    assert state["peak"] == 2
