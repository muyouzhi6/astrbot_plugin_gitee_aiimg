import asyncio
import importlib.util
import io
import json
import os
import sys
import threading
import time
import types
from pathlib import Path

import pytest
from PIL import Image, ImageChops, ImageStat, JpegImagePlugin, features


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "image_encoding_testpkg"
CORE_PACKAGE_NAME = f"{PACKAGE_NAME}.core"
MODULE_NAME = f"{CORE_PACKAGE_NAME}.image_manager"


class _Logger:
    def debug(self, *args, **kwargs):
        return None

    info = debug
    warning = debug
    error = debug


def _load_image_manager():
    for name in list(sys.modules):
        if name == PACKAGE_NAME or name.startswith(f"{PACKAGE_NAME}."):
            sys.modules.pop(name, None)

    package = types.ModuleType(PACKAGE_NAME)
    package.__path__ = [str(ROOT)]
    sys.modules[PACKAGE_NAME] = package

    core_package = types.ModuleType(CORE_PACKAGE_NAME)
    core_package.__path__ = [str(ROOT / "core")]
    sys.modules[CORE_PACKAGE_NAME] = core_package

    astrbot = types.ModuleType("astrbot")
    api = types.ModuleType("astrbot.api")
    api.logger = _Logger()
    sys.modules["astrbot"] = astrbot
    sys.modules["astrbot.api"] = api

    spec = importlib.util.spec_from_file_location(
        MODULE_NAME,
        ROOT / "core" / "image_manager.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[MODULE_NAME] = module
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def _source_png() -> bytes:
    image = Image.effect_noise((640, 480), 24).convert("RGB")
    for x in range(0, image.width, 32):
        color = (x % 256, (x * 3) % 256, (255 - x) % 256)
        for y in range(image.height):
            image.putpixel((x, y), color)
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def _rgba_png() -> bytes:
    image = Image.new("RGBA", (32, 32))
    for y in range(image.height):
        for x in range(image.width):
            image.putpixel((x, y), (x * 7 % 256, y * 9 % 256, 123, (x + y) % 256))
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def test_output_formats_preserve_pixels_for_lossless_modes(tmp_path):
    if not features.check("webp"):
        pytest.skip("Pillow WebP encoder unavailable")

    async def run():
        mod = _load_image_manager()
        manager = mod.ImageManager(
            {
                "image_encoding": {
                    "webp_lossless_effort": 80,
                    "webp_method": 4,
                    "png_compress_level": 9,
                }
            },
            tmp_path,
        )
        source = _source_png()
        try:
            auto_path = await manager.save_image(source, output_format="auto")
            png_path = await manager.save_image(source, output_format="png")
            webp_path = await manager.save_image(source, output_format="webp_lossless")

            assert auto_path.read_bytes() == source
            assert png_path.suffix == ".png"
            assert webp_path.suffix == ".webp"

            with (
                Image.open(auto_path) as original,
                Image.open(png_path) as png,
                Image.open(webp_path) as webp,
            ):
                assert (
                    ImageChops.difference(
                        original.convert("RGB"), png.convert("RGB")
                    ).getbbox()
                    is None
                )
                assert (
                    ImageChops.difference(
                        original.convert("RGB"), webp.convert("RGB")
                    ).getbbox()
                    is None
                )
        finally:
            await manager.close()

    asyncio.run(run())


def test_lossless_webp_preserves_rgba_values(tmp_path):
    if not features.check("webp"):
        pytest.skip("Pillow WebP encoder unavailable")

    async def run():
        mod = _load_image_manager()
        manager = mod.ImageManager({}, tmp_path)
        source = _rgba_png()
        try:
            path = await manager.save_image(source, output_format="webp_lossless")
            with (
                Image.open(io.BytesIO(source)) as original,
                Image.open(path) as converted,
            ):
                assert (
                    ImageChops.difference(
                        original.convert("RGBA"), converted.convert("RGBA")
                    ).getbbox()
                    is None
                )
        finally:
            await manager.close()

    asyncio.run(run())


def test_lossless_webp_reencodes_existing_lossy_webp(tmp_path):
    if not features.check("webp"):
        pytest.skip("Pillow WebP encoder unavailable")

    async def run():
        mod = _load_image_manager()
        manager = mod.ImageManager({}, tmp_path)
        source_image = Image.effect_noise((320, 240), 32).convert("RGB")
        lossy = io.BytesIO()
        source_image.save(lossy, format="WEBP", quality=60, method=4)
        source = lossy.getvalue()
        try:
            path = await manager.save_image(source, output_format="webp_lossless")
            converted = path.read_bytes()

            assert path.suffix == ".webp"
            assert converted != source
            with (
                Image.open(io.BytesIO(source)) as decoded_source,
                Image.open(path) as decoded_converted,
            ):
                assert (
                    ImageChops.difference(
                        decoded_source.convert("RGB"),
                        decoded_converted.convert("RGB"),
                    ).getbbox()
                    is None
                )
        finally:
            await manager.close()

    asyncio.run(run())


def test_image_encoding_does_not_block_the_event_loop(tmp_path, monkeypatch):
    if not features.check("webp"):
        pytest.skip("Pillow WebP encoder unavailable")

    source = _source_png()
    original_save = Image.Image.save

    def slow_save(image, fp, format=None, **params):
        if format == "WEBP":
            time.sleep(0.25)
        return original_save(image, fp, format=format, **params)

    monkeypatch.setattr(Image.Image, "save", slow_save)

    async def run():
        mod = _load_image_manager()
        manager = mod.ImageManager({}, tmp_path)
        loop = asyncio.get_running_loop()

        async def probe_lag():
            started = loop.time()
            await asyncio.sleep(0.03)
            return loop.time() - started

        try:
            probe = asyncio.create_task(probe_lag())
            await asyncio.sleep(0)
            path = await manager.save_image(source, output_format="webp_lossless")

            assert path.suffix == ".webp"
            assert await probe < 0.12
        finally:
            await manager.close()

    asyncio.run(run())


def test_image_encoding_uses_a_bounded_executor(tmp_path, monkeypatch):
    if not features.check("webp"):
        pytest.skip("Pillow WebP encoder unavailable")

    source = _source_png()
    original_save = Image.Image.save
    counter_lock = threading.Lock()
    active = 0
    peak = 0

    def slow_save(image, fp, format=None, **params):
        nonlocal active, peak
        if format != "WEBP":
            return original_save(image, fp, format=format, **params)
        with counter_lock:
            active += 1
            peak = max(peak, active)
        try:
            time.sleep(0.15)
            return original_save(image, fp, format=format, **params)
        finally:
            with counter_lock:
                active -= 1

    monkeypatch.setattr(Image.Image, "save", slow_save)

    async def run():
        mod = _load_image_manager()
        manager = mod.ImageManager({}, tmp_path)
        try:
            paths = await asyncio.gather(
                *(
                    manager.save_image(source, output_format="webp_lossless")
                    for _ in range(4)
                )
            )

            assert all(path.suffix == ".webp" for path in paths)
            assert peak == max(1, min(2, os.cpu_count() or 1))
        finally:
            await manager.close()

    asyncio.run(run())


def test_png_compress_level_changes_size_without_changing_pixels(tmp_path):
    async def run():
        mod = _load_image_manager()
        source = _source_png()
        fast_manager = mod.ImageManager(
            {"image_encoding": {"png_compress_level": 0}},
            tmp_path / "fast",
        )
        compact_manager = mod.ImageManager(
            {"image_encoding": {"png_compress_level": 9}},
            tmp_path / "compact",
        )
        try:
            fast_path = await fast_manager.save_image(source, output_format="png")
            compact_path = await compact_manager.save_image(
                source,
                output_format="png",
            )

            assert compact_path.stat().st_size < fast_path.stat().st_size
            with Image.open(fast_path) as fast, Image.open(compact_path) as compact:
                assert (
                    ImageChops.difference(
                        fast.convert("RGB"), compact.convert("RGB")
                    ).getbbox()
                    is None
                )
        finally:
            await fast_manager.close()
            await compact_manager.close()

    asyncio.run(run())


def test_jpeg_uses_full_chroma_and_webp_quality(tmp_path):
    if not features.check("webp"):
        pytest.skip("Pillow WebP encoder unavailable")

    async def run():
        mod = _load_image_manager()
        manager = mod.ImageManager(
            {
                "image_encoding": {
                    "jpeg_quality": 95,
                    "jpeg_subsampling": "4:4:4",
                    "webp_quality": 97,
                    "webp_method": 4,
                }
            },
            tmp_path,
        )
        source = _source_png()
        try:
            jpeg_path = await manager.save_image(source, output_format="jpeg")
            webp_path = await manager.save_image(source, output_format="webp")

            assert jpeg_path.suffix == ".jpg"
            assert webp_path.suffix == ".webp"
            with Image.open(jpeg_path) as jpeg, Image.open(webp_path) as webp:
                assert jpeg.size == (640, 480)
                assert webp.size == (640, 480)
                assert (
                    ImageStat.Stat(
                        ImageChops.difference(jpeg, webp.convert("RGB"))
                    ).mean[0]
                    > 0
                )
                assert JpegImagePlugin.get_sampling(jpeg) == 0
        finally:
            await manager.close()

    asyncio.run(run())


def test_output_format_aliases_and_schema_options():
    mod = _load_image_manager()
    assert mod.normalize_output_format("jpg") == "jpeg"
    assert mod.normalize_output_format("lossless-webp") == "webp_lossless"
    assert mod.normalize_output_format("not-a-format") == "auto"

    schema = json.loads((ROOT / "_conf_schema.json").read_text(encoding="utf-8"))
    encoding = schema["image_encoding"]["items"]
    assert encoding["jpeg_quality"]["default"] == 95
    assert encoding["jpeg_subsampling"]["default"] == "4:4:4"

    output_format_blocks = []

    def walk(value):
        if isinstance(value, dict):
            if "output_format" in value and isinstance(value["output_format"], dict):
                output_format_blocks.append(value["output_format"])
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(schema)
    assert len(output_format_blocks) >= 18
    for block in output_format_blocks:
        assert block["options"] == ["jpeg", "webp", "webp_lossless", "png", "auto"]
