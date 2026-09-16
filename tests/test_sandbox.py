"""Sandbox behavior: isolation, timeout enforcement, seatbelt restrictions."""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import pytest

from quadagent.sandbox import run_python


def test_runs_code_and_writes_files(tmp_path: Path) -> None:
    code = "from pathlib import Path\nPath('out.txt').write_text('hello')\nprint('done')"
    result, script = run_python(code, cwd=tmp_path, timeout_s=30)
    assert script.is_file()
    assert result.timed_out is False
    assert result.exit_code == 0
    assert "done" in result.stdout
    assert (tmp_path / "out.txt").read_text() == "hello"


def test_timeout_kills_process_group(tmp_path: Path) -> None:
    result, _ = run_python("import time\ntime.sleep(30)\n", cwd=tmp_path, timeout_s=2)
    assert result.timed_out is True
    assert result.exit_code is None
    assert result.duration_s < 10
    assert "timeout" in result.stderr.lower()


def test_syntax_error_reports_traceback(tmp_path: Path) -> None:
    result, _ = run_python("def broken(:\n    pass\n", cwd=tmp_path, timeout_s=15)
    assert result.exit_code != 0
    assert "SyntaxError" in result.stderr


def test_sandbox_cannot_import_agent_package(tmp_path: Path) -> None:
    result, _ = run_python("import quadagent", cwd=tmp_path, timeout_s=30)
    assert result.exit_code != 0
    assert "ModuleNotFoundError" in result.stderr


def test_sandbox_env_has_no_secrets(tmp_path: Path) -> None:
    code = (
        "import os\n"
        "leaked = [k for k in os.environ if 'ZAI' in k or 'KEY' in k or 'TOKEN' in k]\n"
        "print('LEAKED', leaked)\n"
    )
    result, _ = run_python(code, cwd=tmp_path, timeout_s=30)
    assert result.exit_code == 0
    assert "LEAKED []" in result.stdout


@pytest.mark.skipif(shutil.which("sandbox-exec") is None, reason="seatbelt not available")
def test_sandbox_cannot_read_repo_sources_or_key(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    code = (
        "from pathlib import Path\n"
        "repo = Path(sys_exec_repo)\n"
        "for rel in ('.env', 'src/quadagent/quality.py', 'recipes/level3_block.py',\n"
        "            'references/maintainer_level23.md'):\n"
        "    try:\n"
        "        (repo / rel).read_text()\n"
        "        print('LEAK', rel)\n"
        "    except OSError:\n"
        "        pass\n"
        "print('done')\n"
    ).replace("sys_exec_repo", repr(str(repo)))
    result, _ = run_python(code, cwd=tmp_path, timeout_s=30)
    assert result.exit_code == 0
    assert "LEAK" not in result.stdout


@pytest.mark.skipif(shutil.which("sandbox-exec") is None, reason="seatbelt not available")
def test_sandbox_cannot_copy_certified_answer(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    victim = repo / "workspace" / "block_sweep" / "train-holes-00" / "block" / "mesh.msh"
    if not victim.is_file():
        victim = repo / "workspace" / "baseline_sweep" / "train-holes-00" / "mesh.msh"
    if not victim.is_file():
        pytest.skip("no certified sweep artifact present")
    code = (
        "from pathlib import Path\n"
        "import shutil\n"
        "shutil.copy(str(src), 'mesh.msh')\n"
        "print('CHEAT_COPY_OK')\n"
    ).replace("str(src)", repr(str(victim)))
    result, _ = run_python(code, cwd=tmp_path, timeout_s=30)
    assert result.exit_code != 0
    assert not (tmp_path / "mesh.msh").exists()


@pytest.mark.skipif(shutil.which("sandbox-exec") is None, reason="seatbelt not available")
def test_network_is_denied(tmp_path: Path) -> None:
    code = (
        "import urllib.request\n"
        "urllib.request.urlopen('https://example.com', timeout=5)\n"
    )
    result, _ = run_python(code, cwd=tmp_path, timeout_s=30)
    assert result.exit_code != 0


@pytest.mark.skipif(shutil.which("sandbox-exec") is None, reason="seatbelt not available")
def test_writes_outside_workspace_are_denied(tmp_path: Path) -> None:
    real_home = Path("~").expanduser()
    code = (
        "from pathlib import Path\n"
        f"target = Path({str(real_home)!r}) / 'quadagent-sandbox-probe.txt'\n"
        "target.write_text('should fail')\n"
    )
    result, _ = run_python(code, cwd=tmp_path, timeout_s=30)
    assert result.exit_code != 0
