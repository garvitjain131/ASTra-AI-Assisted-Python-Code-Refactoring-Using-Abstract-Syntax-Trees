"""Test helper: run original and refactored code and compare their output."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RunResult:
    stdout: str
    stderr: str
    returncode: int | None
    timed_out: bool


def run_code(source: str, timeout: float = 5.0, stdin: str = "") -> RunResult:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "program.py"
        path.write_text(source, encoding="utf-8")
        try:
            proc = subprocess.run([sys.executable, "-I", str(path)], input=stdin, capture_output=True,
                                  text=True, timeout=timeout, cwd=tmp, encoding="utf-8", errors="replace")
        except subprocess.TimeoutExpired as exc:
            out = exc.stdout.decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            return RunResult(out, "Timed out", None, True)
        stderr = proc.stderr.replace(str(path), "program.py")
        return RunResult(proc.stdout, stderr, proc.returncode, False)


def compare_behaviour(original: str, refactored: str, timeout: float = 5.0, stdin: str = "") -> dict:
    a = run_code(original, timeout, stdin)
    b = run_code(refactored, timeout, stdin)
    last_err = lambda r: (r.stderr.strip().splitlines() or [""])[-1]  # noqa: E731
    same = (a.stdout == b.stdout and a.returncode == b.returncode and last_err(a) == last_err(b)
            and not a.timed_out and not b.timed_out)
    return {"same": same, "original": a, "refactored": b}
