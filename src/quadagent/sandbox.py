"""Sandboxed execution of agent-written Python (Gmsh) code.

The agent only ever produces source text; execution is a subprocess of a
dedicated interpreter (.venv-sandbox: gmsh/numpy/matplotlib, no quadagent)
with:

- cwd pinned to the session workspace (files persist across calls);
- a hard wall-clock timeout enforced by killing the whole process group;
- POSIX rlimits (CPU, file size, open files, address space on Linux);
- an env allowlist with a minimal PATH and HOME remapped into the session
  (no secrets, no repo paths, no user home);
- on macOS, a Seatbelt profile that denies all networking, all writes
  outside the workspace/tmp, and READS of repo sources (src/, recipes/,
  maintainer notes), the API key (.env), sibling sessions/sweeps
  (workspace/), and ~/.ssh, ~/.gnupg, ~/.aws. Reads of everything else
  remain allowed — treat this as accident containment, not a security
  boundary; use a container for hostile settings.
"""

from __future__ import annotations

import os
import resource
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

SEATBELT_PROFILE = """(version 1)
(allow default)
(deny network*)
(deny file-write* (subpath "/Users"))
(allow file-write* (subpath "{workspace}"))
(allow file-write* (subpath "/private/tmp"))
(allow file-write* (literal "/tmp"))
(deny file-read* (subpath "{repo}/src"))
(deny file-read* (subpath "{repo}/recipes"))
(deny file-read* (subpath "{repo}/references/maintainer_level23.md"))
(deny file-read* (subpath "{repo}/.env"))
(deny file-read* (subpath "{repo}/workspace"))
(deny file-read* (subpath "{home}/.ssh"))
(deny file-read* (subpath "{home}/.gnupg"))
(deny file-read* (subpath "{home}/.aws"))
(allow file-read* (subpath "{workspace}"))
"""

_SANDBOX_VENV = Path(__file__).resolve().parents[2] / ".venv-sandbox" / "bin" / "python"
_purity_cache: dict[str, bool] = {}


def sandbox_interpreter() -> str:
    """Interpreter used for agent code: a dedicated venv WITHOUT quadagent.

    Resolution order: QUADAGENT_SANDBOX_PYTHON, the repo's .venv-sandbox,
    then the host interpreter. Anything but the dedicated venv is a recipe
    leak unless explicitly allowed (see verify_isolation).
    """
    override = os.environ.get("QUADAGENT_SANDBOX_PYTHON")
    if override:
        return override
    if _SANDBOX_VENV.is_file():
        return str(_SANDBOX_VENV)
    return sys.executable


def verify_isolation() -> str:
    """Fail closed unless the sandbox interpreter cannot import quadagent.

    The certified recipes (recipes/*.py) and the whole quadagent package
    must be unreachable from agent-executed code. Opt out explicitly with
    QUADAGENT_ALLOW_RECIPE_LEAK=1 (documented as cheating).
    """
    interpreter = sandbox_interpreter()
    if os.environ.get("QUADAGENT_ALLOW_RECIPE_LEAK") == "1":
        return interpreter
    if interpreter not in _purity_cache:
        probe = subprocess.run(
            [interpreter, "-c", "import quadagent"],
            capture_output=True,
            timeout=60,
        )
        _purity_cache[interpreter] = probe.returncode != 0
    if not _purity_cache[interpreter]:
        msg = (
            f"sandbox interpreter {interpreter!r} can import quadagent: the "
            "certified recipes would leak to the agent. Create a clean "
            "sandbox venv (see README) or set QUADAGENT_SANDBOX_PYTHON. To "
            "deliberately allow the leak set QUADAGENT_ALLOW_RECIPE_LEAK=1."
        )
        raise RuntimeError(msg)
    return interpreter


@dataclass
class RunResult:
    """Outcome of one sandboxed execution."""

    exit_code: int | None
    timed_out: bool
    duration_s: float
    stdout: str
    stderr: str


def _limits() -> None:  # pragma: no cover - runs in the forked child
    """Apply rlimits in the child process before exec (POSIX only)."""
    cpu_s = 90
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s + 10))
    fsize = 512 * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_FSIZE, (fsize, fsize))
    resource.setrlimit(resource.RLIMIT_NOFILE, (256, 256))
    if sys.platform == "linux":
        as_bytes = 4 * 1024 * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (as_bytes, as_bytes))


def _seatbelt_command(profile: str, interpreter: str, py_script: Path) -> list[str] | None:
    """Wrap the sandbox interpreter with sandbox-exec when available."""
    if shutil.which("sandbox-exec") is None:
        return None
    return ["sandbox-exec", "-p", profile, interpreter, str(py_script)]


def run_python(
    code: str,
    cwd: Path,
    timeout_s: int = 120,
    mode: str = "auto",
    call_id: int = 0,
) -> tuple[RunResult, Path]:
    """Run `code` in a subprocess rooted at `cwd`; returns the result and script path.

    The script is kept at `cwd/_sandbox/run_<id>.py` so every attempt stays
    inspectable in the session workspace.
    """
    cwd = Path(cwd).resolve()
    sandbox_dir = cwd / "_sandbox"
    sandbox_dir.mkdir(parents=True, exist_ok=True)
    script_path = sandbox_dir / f"run_{call_id:04d}.py"
    script_path.write_text(code)

    repo = Path(__file__).resolve().parents[2]
    home = Path.home()
    profile = SEATBELT_PROFILE.format(
        workspace=cwd.as_posix(), repo=repo.as_posix(), home=home.as_posix()
    )
    interpreter = verify_isolation()
    command = [interpreter, str(script_path)]
    if mode in {"auto", "seatbelt"}:
        wrapped = _seatbelt_command(profile, interpreter, script_path)
        if wrapped is not None:
            command = wrapped

    allow = {"TMPDIR", "LANG", "LC_ALL", "MPLCONFIGDIR"}
    env = {k: v for k, v in os.environ.items() if k in allow}
    env["PATH"] = "/usr/bin:/bin:/usr/sbin:/sbin"
    env["HOME"] = cwd.as_posix()
    env["MPLBACKEND"] = "Agg"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["MPLCONFIGDIR"] = os.environ.get("MPLCONFIGDIR") or str(cwd / "_cache")

    start = time.monotonic()
    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
        preexec_fn=_limits if os.name == "posix" else None,
    )
    timed_out = False
    try:
        stdout, stderr = process.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        stdout, stderr = process.communicate()
        stderr = (stderr or "") + f"\n[sandbox] killed: wall-clock timeout of {timeout_s}s exceeded"
    duration = time.monotonic() - start
    result = RunResult(
        exit_code=None if timed_out else process.returncode,
        timed_out=timed_out,
        duration_s=duration,
        stdout=stdout or "",
        stderr=stderr or "",
    )
    return result, script_path
