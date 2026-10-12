"""Ask the autonomous focusing agent: .venv/bin/python run.py"""

import argparse
from datetime import datetime
import json
from pathlib import Path
from time import perf_counter
import traceback
from zoneinfo import ZoneInfo

from langchain_ollama import ChatOllama
from langgraph.checkpoint.sqlite import SqliteSaver

from agents import make_nodes, verified_results
from graph import build_graph
from persistence import RunLock, write_json
from prompts import REQUEST

PROJECT = Path(__file__).resolve().parent


class Events:
    """Expose actual decisions, tool calls, and camera progress for recording."""
    def __init__(self, directory):
        self.directory = Path(directory)
        self.rows = []
        path = self.directory / "events.json"
        if path.exists():
            self.rows = json.loads(path.read_text())

    def __call__(self, kind, data):
        row = {"time": datetime.now(ZoneInfo("Asia/Yerevan")).isoformat(), "kind": kind, "data": data}
        self.rows.append(row)
        write_json(self.directory / "events.json", self.rows)
        if kind == "decision":
            print(f"[orchestrator] Decision {data['number']}: {data['next_role']} | {data['reason']}", flush=True)
        elif kind == "tool_call":
            print("[executor] Calling focus_beam with model-selected settings:", json.dumps(data["arguments"]), flush=True)
        elif kind == "experiment_progress":
            print(f"[focus_beam] {data['iteration']}/{data['iterations']} | enhancement {data['retained_measurement']:.4f}", flush=True)
        else:
            print(f"[{kind}] " + json.dumps(data)[:350], flush=True)


def save_state(directory, state, summary):
    write_json(directory / "checkpoint.json", {"state": state, "summary": summary})
    write_json(directory / "dialogue.json", state)
    write_json(directory / "summary.json", summary)
    paragraphs = ["# Actual focusing-agent dialogue", ""]
    for turn in state["dialogue"]:
        paragraphs += ["## " + turn["role"], "", turn["text"], ""]
    (directory / "dialogue.md").write_text("\n".join(paragraphs))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="qwen2.5:7b")
    parser.add_argument("--executor-model", default="qwen2.5-coder:7b")
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--request", default=REQUEST)
    parser.add_argument("--run-name", help="Optional local run-folder name")
    parser.add_argument("--resume-run", help="Resume an unfinished native thread")
    args = parser.parse_args()
    name = args.resume_run or args.run_name or datetime.now(ZoneInfo("Asia/Yerevan")).strftime("%Y%m%d_%H%M%S")
    if not name or not all(character.isalnum() or character in "_-" for character in name):
        parser.error("Use a simple run-folder basename")
    directory = PROJECT / "output/runs" / name
    if args.resume_run and not directory.is_dir():
        parser.error("Saved run does not exist")
    directory.mkdir(parents=True, exist_ok=bool(args.resume_run))
    state = {"dialogue": [{"role": "user", "text": args.request}], "decision_count": 0,
             "decision": {"next_role": "scientist", "instruction": "", "reason": ""}}
    started = perf_counter()
    status, error = "running", None
    event = Events(directory)
    options = dict(base_url=args.base_url, temperature=0, num_ctx=12288, num_predict=5000,
                   client_kwargs={"timeout": 600})
    print("Run directory:", directory, flush=True)
    event("request", {"text": args.request})

    def summary():
        return {"status": status, "error": error, "decision_count": state["decision_count"],
                "model": args.model, "executor_model": args.executor_model, "thread_id": name,
                "run_directory": str(directory), "experiments": verified_results(state["dialogue"]),
                "session_seconds": perf_counter() - started}

    with RunLock(directory), SqliteSaver.from_conn_string(str(directory / "checkpoints.sqlite")) as saver:
        graph = build_graph(make_nodes(ChatOllama(model=args.model, **options),
            ChatOllama(model=args.executor_model, **options), directory, event), checkpointer=saver)
        config = {"configurable": {"thread_id": name}, "recursion_limit": 40}
        graph_input = state
        if args.resume_run:
            snapshot = graph.get_state(config)
            state = snapshot.values
            if not snapshot.next:
                parser.error("This run already ended")
            graph_input = None
        save_state(directory, state, summary())
        try:
            for values in graph.stream(graph_input, config, stream_mode="values"):
                state = values
                save_state(directory, state, summary())
            status = "completed" if state["dialogue"][-1]["role"] == "finish" else "decision_limit"
        except KeyboardInterrupt:
            status, error = "interrupted", "Stopped by the user"
        except Exception:
            status, error = "error", traceback.format_exc()
        finally:
            snapshot = graph.get_state(config)
            if snapshot.values:
                state = snapshot.values
            save_state(directory, state, summary())
            event("run_ended", summary())
    print("Status:", status, flush=True)
    raise SystemExit(0 if status == "completed" else 1)


if __name__ == "__main__":
    main()
