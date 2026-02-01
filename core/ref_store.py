import asyncio
import json
import re
from pathlib import Path

import aiofiles

from .image_format import guess_image_mime_and_ext


def _sanitize_name(name: str) -> str:
    name = name.strip()
    if not name:
        return ""
    name = re.sub(r"[^\w\-\u4e00-\u9fff]+", "_", name, flags=re.UNICODE)
    return name[:64].strip("_")


class ReferenceStore:
    def __init__(self, data_dir: Path):
        self.refs_dir = data_dir / "refs"
        self.refs_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.refs_dir / "index.json"
        self._lock = asyncio.Lock()

    async def _read_index(self) -> dict[str, list[str]]:
        if not self.index_path.exists():
            return {}
        async with aiofiles.open(self.index_path, encoding="utf-8") as f:
            raw = await f.read()
        try:
            data = json.loads(raw)
        except Exception:
            return {}
        if not isinstance(data, dict):
            return {}
        out: dict[str, list[str]] = {}
        for k, v in data.items():
            if isinstance(k, str) and isinstance(v, list):
                out[k] = [str(x) for x in v if str(x)]
        return out

    async def _write_index(self, index: dict[str, list[str]]) -> None:
        async with aiofiles.open(self.index_path, "w", encoding="utf-8") as f:
            await f.write(json.dumps(index, ensure_ascii=False, indent=2))

    async def list_names(self) -> list[str]:
        async with self._lock:
            index = await self._read_index()
        return sorted(index.keys())

    async def get_paths(self, name: str) -> list[Path]:
        name = _sanitize_name(name)
        if not name:
            return []
        async with self._lock:
            index = await self._read_index()
            files = index.get(name, [])
        paths = [self.refs_dir / f for f in files]
        return [p for p in paths if p.exists()]

    async def set(self, name: str, images: list[bytes]) -> int:
        name = _sanitize_name(name)
        if not name:
            raise ValueError("name is required")
        if not images:
            raise ValueError("at least one image is required")

        async with self._lock:
            index = await self._read_index()

            old_files = index.get(name, [])
            for f in old_files:
                p = self.refs_dir / f
                if p.exists():
                    try:
                        await asyncio.to_thread(p.unlink)
                    except Exception:
                        pass

            new_files: list[str] = []
            for i, img_bytes in enumerate(images):
                _, ext = guess_image_mime_and_ext(img_bytes)
                filename = f"{name}_{i + 1}.{ext}"
                p = self.refs_dir / filename
                async with aiofiles.open(p, "wb") as f:
                    await f.write(img_bytes)
                new_files.append(filename)

            index[name] = new_files
            await self._write_index(index)

        return len(images)

    async def delete(self, name: str) -> bool:
        name = _sanitize_name(name)
        if not name:
            return False
        async with self._lock:
            index = await self._read_index()
            files = index.pop(name, None)
            if files:
                for f in files:
                    p = self.refs_dir / f
                    if p.exists():
                        try:
                            await asyncio.to_thread(p.unlink)
                        except Exception:
                            pass
            await self._write_index(index)
        return files is not None
