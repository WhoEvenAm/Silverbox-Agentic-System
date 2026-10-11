"""Run the system: python run.py --model qwen2.5:7b"""

import argparse
from datetime import datetime
import json
from pathlib import Path
from time import perf_counter
import traceback
from zoneinfo import ZoneInfo
from langchain_ollama import ChatOllama

from agents import make_nodes, verified_result
from graph import build_graph
from prompts import TASK5_REQUEST

PROJECT = Path(__file__).resolve().parent


def save_state(directory, state):
    # JSON lets us resume; Markdown makes the same conversation easy to read.
    (directory / "dialogue.json").write_text(json.dumps(state, indent=2))
    paragraphs = ["# Agent dialogue", ""]
    for number, turn in enumerate(state["dialogue"], 1):
        paragraphs.extend([f"## {number}. {turn['role']}", "", turn["text"], ""])
    (directory / "dialogue.md").write_text("\n".join(paragraphs))


def main():
    # Command-line options select local models, a new request, or a saved run.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="qwen2.5:7b")
    parser.add_argument("--executor-model", default="qwen2.5-coder:7b")
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--request-file", type=Path, help="Use another user request")
    parser.add_argument("--resume-run", help="Resume an interrupted run by its folder name")
    args = parser.parse_args()
    stamp = datetime.now(ZoneInfo("Asia/Yerevan")).strftime("%Y%m%d_%H%M%S_%f")
    run_root = PROJECT / "output" / "runs"
    start_at, previous_seconds, resumptions = "orchestrator", 0.0, []
    if args.resume_run:
        if Path(args.resume_run).name != args.resume_run or args.request_file:
            parser.error("Use a run folder basename and no new request when resuming")
        directory = run_root / args.resume_run
        state = json.loads((directory / "dialogue.json").read_text())
        previous = json.loads((directory / "summary.json").read_text())
        if previous["status"] == "completed" or state["decision_count"] >= 8:
            parser.error("Only interrupted runs with a remaining decision budget can resume")
        previous_seconds = previous["total_seconds"]
        resumptions = previous.get("resumptions", []) + [{"at": stamp,
            "previous_model": previous["model"],
            "previous_executor_model": previous.get("executor_model", previous["model"])}]
        (directory / f"resume_{len(resumptions):02d}_previous_summary.json").write_text(json.dumps(previous, indent=2))
        # A saved orchestrator turn has already selected the next specialist.
        if state["dialogue"][-1]["role"] == "orchestrator":
            choice = state["decision"]["next_role"]
            if choice in ("scientist", "general", "executor"):
                start_at = choice
    else:
        directory = run_root / stamp
        directory.mkdir(parents=True, exist_ok=False)
        request = args.request_file.read_text() if args.request_file else TASK5_REQUEST
        state = {"dialogue": [{"role": "user", "text": request}], "decision_count": 0,
                 "decision": {"next_role": "scientist", "instruction": "", "reason": ""}}
        save_state(directory, state)
    # Placeholder is not a routing decision: the first model call replaces it.
    # Three roles use the general model; only the executor uses the coding model.
    # temperature=0 reduces randomness, but does not guarantee identical answers.
    model = ChatOllama(model=args.model, base_url=args.base_url, temperature=0,
                       num_ctx=12288, num_predict=4096, client_kwargs={"timeout": 600})
    executor_model = ChatOllama(model=args.executor_model, base_url=args.base_url, temperature=0,
                              num_ctx=12288, num_predict=4096, client_kwargs={"timeout": 600})
    graph = build_graph(make_nodes(model, directory, PROJECT / "silverbox", executor_model), start_at)
    (directory / "graph.mmd").write_text(graph.get_graph().draw_mermaid())
    print("Run directory:", directory, flush=True)
    started = perf_counter()
    error = None
    try:
        # stream yields each completed node's changes so we can save immediately.
        for update in graph.stream(state, config={"recursion_limit": 32}, stream_mode="updates"):
            for node, change in update.items():
                state["dialogue"].extend(change.get("dialogue", []))
                for key, value in change.items():
                    if key != "dialogue":
                        state[key] = value
                save_state(directory, state)
                print(f"[{node}] Turn recorded; {len(state['dialogue'])} messages", flush=True)
    except (Exception, KeyboardInterrupt):
        error = traceback.format_exc()
        (directory / "error.txt").write_text(error)
    # A model saying "done" is insufficient: actual execution must be verified.
    evidence = verified_result(state["dialogue"])
    status = "completed" if state["decision"]["next_role"] == "finish" and evidence else "decision_limit"
    if error:
        status = "error"
    summary = {"status": status, "model": args.model, "executor_model": args.executor_model,
        "decision_count": state["decision_count"],
        "total_seconds": previous_seconds + perf_counter() - started,
        "resumptions": resumptions, "run_directory": str(directory),
        "execution": evidence, "error": error}
    (directory / "summary.json").write_text(json.dumps(summary, indent=2))
    (PROJECT / "output" / "latest_run.json").write_text(json.dumps(summary, indent=2))
    save_state(directory, state)
    print("Status:", status, flush=True)
    if evidence:
        print("Verified test RMSE:", evidence["verification"]["metrics"]["test_rmse"], "V", flush=True)
    if error:
        print(error, flush=True)
    print("Transcript:", directory / "dialogue.md", flush=True)
    if status != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
