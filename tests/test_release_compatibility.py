import importlib.metadata
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path
from urllib.parse import unquote

import pytest
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet


@pytest.mark.parametrize("version", ["4.26.8", "4.27.3", "4.28.2", "4.29.1", "4.29.3"])
def test_metadata_allows_deployment_astrbot_version(version):
    metadata = (Path(__file__).parents[1] / "metadata.yaml").read_text(encoding="utf-8")
    declaration = next(
        line.split(":", 1)[1].strip().strip("\"'")
        for line in metadata.splitlines()
        if line.startswith("astrbot_version:")
    )
    supported = SpecifierSet(declaration)
    assert supported.contains(version, prereleases=True)
    assert not supported.contains("5.0.0")


def test_readme_astrbot_requirement_matches_metadata():
    root = Path(__file__).parents[1]
    metadata = (root / "metadata.yaml").read_text(encoding="utf-8")
    declaration = next(
        line.split(":", 1)[1].strip().strip("\"'")
        for line in metadata.splitlines()
        if line.startswith("astrbot_version:")
    )
    minimum = next(
        spec.version for spec in SpecifierSet(declaration) if spec.operator == ">="
    )
    readme = (root / "README.md").read_text(encoding="utf-8")
    badge = re.search(r"\[!\[AstrBot\]\(([^)]+)\)", readme)
    assert badge is not None
    assert f"AstrBot->={minimum} <5" in unquote(badge.group(1))
    assert f"AstrBot >={minimum} 且 <5" in readme


def test_real_astrbot_load_and_studio_api(tmp_path):
    try:
        importlib.metadata.version("AstrBot")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("Install AstrBot for the runtime smoke test")
    root = Path(__file__).parents[1].resolve()
    script = textwrap.dedent("""\
        import asyncio
        import importlib
        import json
        import os
        import sys
        from pathlib import Path
        from types import SimpleNamespace

        os.umask(0o077)
        import astrbot
        from astrbot.api.star import Context
        from astrbot.api.web import PluginRequest, bind_request_context
        from astrbot.core.agent.message import TextPart
        from astrbot.core.star.star_manager import PluginManager
        from starlette.requests import Request

        root = Path(sys.argv[1])
        sys.path.insert(0, str(root.parent))
        main = importlib.import_module(root.name + ".main")
        assert TextPart(text="test").mark_as_temp()._no_save
        spec = next(line.split(":", 1)[1].strip().strip('"')
                    for line in (root / "metadata.yaml").read_text(encoding="utf-8").splitlines()
                    if line.startswith("astrbot_version:"))

        from packaging.specifiers import SpecifierSet
        assert SpecifierSet(spec).contains(astrbot.__version__), astrbot.__version__

        async def check():
            context = object.__new__(Context)
            context.registered_web_apis = []
            context.provider_manager = SimpleNamespace(provider_insts=[])
            context.get_config = lambda: {"wake_prefix": ["/"]}
            plugin = main.GiteeAIImagePlugin(context, {})
            try:
                await plugin.initialize()
                assert plugin.studio is not None
                route, handler, methods, _ = next(
                    api for api in context.registered_web_apis
                    if api[0].endswith("/studio/state"))
                request = Request({"type": "http", "method": "GET", "path": route,
                                   "headers": [], "query_string": b""})
                with bind_request_context(PluginRequest(request)):
                    response = await handler()
                assert response.status_code == 200
                assert json.loads(response.body)["data"]["ok"] is True
            finally:
                await plugin.terminate()
            await asyncio.sleep(0)
            assert not [task for task in asyncio.all_tasks()
                        if task is not asyncio.current_task() and not task.done()]

        asyncio.run(check())
        supported, reason = PluginManager._validate_astrbot_version_specifier(spec)
        assert supported, reason
        print(f"AstrBot {astrbot.__version__}: import, initialize, Web API, shutdown, version gate passed")
    """)
    result = subprocess.run(
        [sys.executable, "-c", script, str(root)],
        cwd=tmp_path,
        env={**os.environ, "ASTRBOT_ROOT": str(tmp_path)},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_real_astrbot_reference_http_contract(tmp_path):
    try:
        importlib.metadata.version("AstrBot")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("Install AstrBot 4.26.8 for the native HTTP reference tests")
    root = Path(__file__).parents[1].resolve()
    result = subprocess.run(
        [sys.executable, str(root / "tests" / "native_studio_references.py")],
        cwd=tmp_path,
        env={**os.environ, "ASTRBOT_ROOT": str(tmp_path)},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_real_astrbot_provider_commands(tmp_path):
    try:
        importlib.metadata.version("AstrBot")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("Install AstrBot for native channel command tests")
    root = Path(__file__).parents[1].resolve()
    result = subprocess.run(
        [sys.executable, str(root / "tests" / "native_provider_commands.py")],
        cwd=tmp_path,
        env={**os.environ, "ASTRBOT_ROOT": str(tmp_path)},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("version", ["12.3.0", "13.0.0"])
def test_pillow_requirement_accepts_protected_core_version(version):
    root = Path(__file__).parents[1]
    requirements = [
        Requirement(line.strip())
        for line in (root / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    pillow = next(req for req in requirements if req.name.lower() == "pillow")
    assert pillow.specifier.contains(version), (str(pillow), version)
