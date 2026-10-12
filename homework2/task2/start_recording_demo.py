"""User-operated launcher: open the live view, then start an autonomous run."""

from datetime import datetime
from pathlib import Path
import socket
import subprocess
import sys
import time
import webbrowser
from zoneinfo import ZoneInfo


def main():
    root = Path(__file__).resolve().parent
    name = datetime.now(ZoneInfo("Asia/Yerevan")).strftime("recording_%Y%m%d_%H%M%S_%f")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    viewer = subprocess.Popen([sys.executable, str(root / "live_view.py"), "--run-name", name,
                               "--port", str(port)], cwd=root, stdout=subprocess.DEVNULL)
    try:
        for _ in range(100):
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=.1):
                    break
            except OSError:
                time.sleep(.05)
        else:
            raise RuntimeError("The local live view did not start")
        # This launcher is operated by the person on their Mac, before the
        # initial agent request. No experiment settings are entered by the person.
        webbrowser.open(f"http://127.0.0.1:{port}/")
        print("Start macOS screen recording (Shift+Command+5) of the live-view window.")
        print("The video should show a decision, the focus_beam call, and camera progress.")
        input("When recording is ready, press Enter to submit the focusing request: ")
        completed = subprocess.run([sys.executable, str(root / "run.py"), "--run-name", name], cwd=root)
        print("Run folder:", root / "output/runs" / name)
        print("Agent exit code:", completed.returncode)
        input("Stop your screen recording, then press Enter to close the live view: ")
    finally:
        viewer.terminate()
        viewer.wait(timeout=5)


if __name__ == "__main__":
    main()
