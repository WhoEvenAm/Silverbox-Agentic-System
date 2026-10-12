"""Recovery checks exercise actual process interruption and checkpoint writes."""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from graph import build_graph, resume_node
from persistence import load_run, RunLock, write_json
from run import save_progress


class RecoveryTests(unittest.TestCase):
    def test_interruption_stops_real_child_and_preserves_output(self):
        with tempfile.TemporaryDirectory() as root:
            code = ("import os, time\nfrom pathlib import Path\n"
                    "Path('child.pid').write_text(str(os.getpid()))\n"
                    "print('training started', flush=True)\ntime.sleep(30)\n")
            launcher = ("import sys\nfrom coding_tool import write_and_run\n"
                        "write_and_run(sys.argv[1], sys.argv[1], 'train.py', sys.argv[2])\n")
            process = subprocess.Popen([sys.executable, "-c", launcher, root, code],
                cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
            pid_path = Path(root) / "attempt_01/child.pid"
            child_pid = None
            try:
                deadline = time.monotonic() + 10
                while not pid_path.exists() and time.monotonic() < deadline:
                    time.sleep(.02)
                self.assertTrue(pid_path.exists(), "Child did not start")
                child_pid = int(pid_path.read_text())
                process.send_signal(signal.SIGINT)
                process.communicate(timeout=5)
                result = json.loads((pid_path.parent / "execution.json").read_text())
                self.assertTrue(result["interrupted"])
                self.assertFalse(result["timed_out"])
                self.assertNotEqual(result["exit_code"], 0)
                self.assertIn("training started", result["stdout"])
                with self.assertRaises(ProcessLookupError):
                    os.kill(child_pid, 0)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.communicate()
                if child_pid is not None:
                    try:
                        os.killpg(child_pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass

    def test_failed_atomic_replace_preserves_previous_checkpoint(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "checkpoint.json"
            write_json(path, {"decision_count": 3})
            with patch("persistence.os.replace", side_effect=OSError("interrupted write")):
                with self.assertRaises(OSError):
                    write_json(path, {"decision_count": 4})
            self.assertEqual(json.loads(path.read_text()), {"decision_count": 3})
            self.assertEqual(list(Path(root).glob("*.tmp")), [])

    def test_checkpoint_is_authoritative_and_success_pointer_survives_failure(self):
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root) / "output/runs/example"
            directory.mkdir(parents=True)
            state = {"dialogue": [{"role": "user", "text": "example"}], "decision_count": 0}
            save_progress(directory, state, {"status": "completed"})
            save_progress(directory, state, {"status": "error"})
            (directory / "dialogue.json").write_text("partially written export")
            self.assertEqual(load_run(directory), (state, {"status": "error"}))
            latest = json.loads((Path(root) / "output/latest_completed_run.json").read_text())
            self.assertEqual(latest["status"], "completed")

    def test_pending_eighth_specialist_resumes_without_ninth_decision(self):
        state = {"dialogue": [{"role": "orchestrator", "text": "explain"}],
            "decision_count": 8,
            "decision": {"next_role": "general", "instruction": "explain", "reason": "question"}}
        def forbidden(_):
            self.fail("Resume must not make another orchestrator decision")
        def general(_):
            return {"dialogue": [{"role": "general", "text": "answer"}]}
        self.assertEqual(resume_node(state), "general")
        graph = build_graph(dict(orchestrator=forbidden, scientist=forbidden,
                                 executor=forbidden, general=general), resume_node(state))
        result = graph.invoke(state)
        self.assertEqual(result["decision_count"], 8)
        self.assertEqual(result["dialogue"][-1]["role"], "general")
        self.assertEqual(resume_node(result), "__end__")

    def test_only_one_process_can_own_a_run(self):
        with tempfile.TemporaryDirectory() as root:
            with RunLock(root):
                with self.assertRaisesRegex(ValueError, "already active"):
                    with RunLock(root):
                        self.fail("Second lock was accepted")
            with RunLock(root):
                pass  # The first owner released its lock.

    def test_legacy_run_without_summary_can_still_be_loaded(self):
        with tempfile.TemporaryDirectory() as root:
            state = {"dialogue": [], "decision_count": 0}
            (Path(root) / "dialogue.json").write_text(json.dumps(state))
            self.assertEqual(load_run(root), (state, {}))
