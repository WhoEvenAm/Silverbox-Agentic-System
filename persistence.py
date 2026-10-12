"""Save complete checkpoints before replacing the previous version."""

import json
from pathlib import Path
import tempfile
import os
import fcntl


def write_json(path, value):
    """An interrupted write leaves the previous JSON file readable."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        # The temporary file must be on the same filesystem for atomic replace.
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=path.name + ".", suffix=".tmp", delete=False) as file:
            temporary = Path(file.name)
            json.dump(value, file, indent=2)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_run(directory):
    """Prefer the single authoritative checkpoint; support earlier saved runs."""
    directory = Path(directory)
    checkpoint = directory / "checkpoint.json"
    if checkpoint.exists():
        saved = json.loads(checkpoint.read_text())
        return saved["state"], saved["summary"]
    state = json.loads((directory / "dialogue.json").read_text())
    summary_path = directory / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    return state, summary


class RunLock:
    """Prevent two processes from appending to the same saved conversation."""

    def __init__(self, directory):
        self.path = Path(directory) / ".run.lock"

    def __enter__(self):
        self.file = self.path.open("a+")
        try:
            # The OS releases this lock when the process exits, even after a crash.
            fcntl.flock(self.file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.file.close()
            raise ValueError("This run is already active; stop it before resuming") from None
        return self

    def __exit__(self, *_):
        self.file.close()
