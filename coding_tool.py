"""The executor's tool writes a Python file, then actually runs it."""

import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
from time import perf_counter
from langchain_core.tools import tool


def write_and_run(run_directory, support_directory, filename, code, timeout_seconds=300):
    """Preserve attempts and process output; terminate timed-out process groups.

    Working-directory isolation organizes artifacts. It is not an OS security
    sandbox: generated Python runs with the launching user's permissions.
    """
    if not re.fullmatch(r"[A-Za-z0-9_-]+\.py", filename):
        raise ValueError("Use a simple Python basename, such as fit.py")
    if not code.strip() or not 0 < timeout_seconds <= 300:
        raise ValueError("Supply code and a positive execution limit <=300 seconds")
    root = Path(run_directory).resolve()
    root.mkdir(parents=True, exist_ok=True)
    attempt = len(list(root.glob("attempt_*"))) + 1
    directory = root / f"attempt_{attempt:02d}"
    directory.mkdir(exist_ok=False)
    script = directory / filename
    script.write_text(code)
    environment = os.environ.copy()
    helper_path = str(Path(support_directory).resolve() / "src")
    environment["PYTHONPATH"] = helper_path + os.pathsep + environment.get("PYTHONPATH", "")
    environment["MPLBACKEND"] = "Agg"
    environment["MPLCONFIGDIR"] = str(directory / ".mplcache")
    environment["PYTHONUNBUFFERED"] = "1"
    started = perf_counter()
    process = subprocess.Popen([sys.executable, str(script)], cwd=directory, env=environment,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
    timed_out = False
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate()
        stderr += f"\nExecution exceeded {timeout_seconds} seconds and was terminated.\n"
    result = {"script_path": str(script), "working_directory": str(directory),
        "exit_code": process.returncode, "stdout": stdout, "stderr": stderr,
        "timed_out": timed_out, "duration_seconds": perf_counter() - started}
    (directory / "stdout.txt").write_text(stdout)
    (directory / "stderr.txt").write_text(stderr)
    (directory / "execution.json").write_text(json.dumps(result, indent=2))
    return result


def make_coding_tool(run_directory, support_directory):
    @tool
    def write_and_run_python(filename: str, code: str) -> dict:
        """Write a new Python file and run it; return exit_code, stdout, stderr.

        Args:
            filename: A simple .py filename for this attempt.
            code: The complete Python source implementing the scientist's plan.
        """
        return write_and_run(run_directory, support_directory, filename, code)
    return write_and_run_python
