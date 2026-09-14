"""Record generation results without turning archive errors into paid retries."""

import asyncio
from contextvars import ContextVar

capture_context = ContextVar("aiimg_studio_capture", default={})


async def capture_result(store, path, *, prompt, provider, model, output, mode):
    metadata = {
        **{k: v for k, v in capture_context.get().items() if k != "asset_id"},
        "prompt": prompt,
        "provider": provider,
        "model": model,
        "output": str(output),
        "mode": mode,
    }

    def save():
        return store.add_image(path.read_bytes(), kind="history", metadata=metadata)

    asset = await asyncio.to_thread(save)
    context = capture_context.get()
    if context:
        context["asset_id"] = asset["id"]
    return asset
