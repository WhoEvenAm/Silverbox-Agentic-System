"""Run the system: python run.py --model qwen2.5:7b"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import re
import signal
from time import perf_counter
import traceback
from zoneinfo import ZoneInfo
from langchain_ollama import ChatOllama
from langgraph.graph import END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from agents import make_nodes, verified_result
from graph import build_graph, resume_node
from persistence import load_run, RunLock, write_json
from prompts import TASK5_REQUEST

PROJECT = Path(__file__).resolve().parent


def save_progress(directory, state, summary):
    # Native LangGraph checkpoints own execution state. Keep this atomic JSON
    # export for model/session metadata and importing pre-checkpointer runs.
    write_json(directory / "checkpoint.json", {"state": state, "summary": summary})
    write_json(directory / "dialogue.json", state)
    write_json(directory / "summary.json", summary)
    write_json(directory.parents[1] / "latest_run.json", summary)
    if summary["status"] == "completed":
        # A later failed run must not hide the last successful run from the report.
        write_json(directory.parents[1] / "latest_completed_run.json", summary)
    paragraphs = ["# Agent dialogue", ""]
    for number, turn in enumerate(state["dialogue"], 1):
        paragraphs.extend([f"## {number}. {turn['role']}", "", turn["text"], ""])
    (directory / "dialogue.md").write_text("\n".join(paragraphs))


def pending_interrupts(snapshot):
    # This workflow is sequential, so only one human question can be pending.
    return [item for task in snapshot.tasks for item in task.interrupts]


def read_human_reply(request):
    print("\n[human] " + request["question"], flush=True)
    if request["kind"] == "approval":
        action = request["proposed_action"]
        print("Proposed action:", action["description"], flush=True)
        print("Exact instruction:", action["instruction"], flush=True)
        print("Reply yes/approve to accept; other replies grant no permission.", flush=True)
    while True:
        try:
            # Preserve the person's actual text; never supply a default answer.
            reply = input("Your reply: ")
        except EOFError:
            return None  # Keep the native interrupt saved for --resume-run.
        if reply.strip():
            return reply
        print("Please enter a nonempty reply.", flush=True)


def drive_graph(graph, graph_input, config, record_state, ask=read_human_reply):
    """Stream full state, then resume any question on the same thread."""
    while True:
        last_count = None
        for values in graph.stream(graph_input, config=config, stream_mode="values"):
            if "dialogue" not in values:
                continue  # A __interrupt__ marker is not a conversation state.
            # A pause can attach runtime Interrupt objects to the full values.
            # They live in SQLite tasks, not in the JSON conversation export.
            values = {key: value for key, value in values.items() if key != "__interrupt__"}
            # Values already include the reducer's complete dialogue. Appending
            # them again would duplicate messages; reading SQLite here can lag.
            record_state(values)
            count = len(values["dialogue"])
            if last_count is not None and count > last_count:
                role = values["dialogue"][-1]["role"]
                print(f"[{role}] Turn recorded; {count} messages", flush=True)
            last_count = count
        snapshot = graph.get_state(config)
        record_state(snapshot.values)
        pending = pending_interrupts(snapshot)
        if not pending:
            return False
        reply = ask(pending[0].value)
        if reply is None:
            return True
        graph_input = Command(resume=reply)


def run_session(args, directory, stamp):
    start_at, previous, resumptions = "orchestrator", {}, []
    native_exists = (directory / "checkpoints.sqlite").exists()
    if args.resume_run:
        state, previous = load_run(directory)
        # Old JSON runs can be imported once. Native runs resume saved tasks
        # instead of guessing the next node from the transcript's last speaker.
        if not native_exists:
            start_at = resume_node(state)
        if previous.get("status") == "completed" or start_at == END:
            raise ValueError("This run has finished; start a new run instead")
        # Preserve model choices unless the user explicitly supplies an override.
        resumptions = previous.get("resumptions", []) + [{"at": stamp,
            "previous_model": previous.get("model", "qwen2.5:7b"),
            "previous_executor_model": previous.get("executor_model", previous.get("model", "qwen2.5:7b"))}]
        write_json(directory / f"resume_{len(resumptions):02d}_previous_summary.json", previous)
    else:
        request = args.request_file.read_text() if args.request_file else TASK5_REQUEST
        state = {"dialogue": [{"role": "user", "text": request}], "decision_count": 0,
                 "decision": {"next_role": "scientist", "instruction": "", "reason": ""},
                 "approved_action": None}
        # This placeholder is replaced by the first actual model routing decision.
    model_name = args.model or previous.get("model", "qwen2.5:7b")
    executor_name = args.executor_model or previous.get("executor_model", "qwen2.5-coder:7b")
    base_url = args.base_url or previous.get("base_url", "http://localhost:11434")
    started = perf_counter()

    def summary(status, error=None):
        return {"status": status, "model": model_name, "executor_model": executor_name,
            "thread_id": directory.name,
            "base_url": base_url, "decision_count": state["decision_count"],
            "total_seconds": previous.get("total_seconds", 0.0) + perf_counter() - started,
            "resumptions": resumptions, "run_directory": str(directory),
            "execution": verified_result(state["dialogue"]), "error": error}

    print("Run directory:", directory, flush=True)
    print("Starting node:", start_at, "| decisions already used:", state["decision_count"], flush=True)
    error, status = None, "running"
    try:
        # Three roles use the general model; the executor uses the coding model.
        # temperature=0 reduces variation but does not guarantee identical replies.
        options = dict(base_url=base_url, temperature=0, num_ctx=12288,
                       num_predict=4096, client_kwargs={"timeout": 600})
        model = ChatOllama(model=model_name, **options)
        executor_model = ChatOllama(model=executor_name, **options)
        # The directory name is a stable thread ID across process restarts.
        config = {"configurable": {"thread_id": directory.name}, "recursion_limit": 40}
        with SqliteSaver.from_conn_string(str(directory / "checkpoints.sqlite")) as checkpointer:
            graph = build_graph(make_nodes(model, directory, PROJECT / "silverbox", executor_model),
                                start_at, checkpointer=checkpointer)
            (directory / "graph.mmd").write_text(graph.get_graph().draw_mermaid())
            snapshot = graph.get_state(config)
            if args.resume_run and snapshot.values:
                state = snapshot.values
                if not snapshot.next:
                    raise ValueError("This native thread has finished; start a new run instead")
                # None continues saved tasks (including errors); Command supplies
                # a reply only when this thread actually has a pending question.
                graph_input = None
                pending = pending_interrupts(snapshot)
                if pending:
                    reply = read_human_reply(pending[0].value)
                    if reply is None:
                        save_progress(directory, state, summary("waiting_for_human"))
                        print("Status: waiting_for_human", flush=True)
                        return 0
                    graph_input = Command(resume=reply)
            else:
                graph_input = state

            def record_state(values):
                nonlocal state
                state = values
                save_progress(directory, state, summary("running"))

            # Save before the first model call, even for newly imported runs.
            save_progress(directory, state, summary("running"))
            try:
                waiting = drive_graph(graph, graph_input, config, record_state)
            finally:
                # Refresh even on Ctrl+C/error before the SQLite connection closes.
                saved = graph.get_state(config).values
                if saved:
                    state = saved
            status = ("waiting_for_human" if waiting else "completed"
                      if state["dialogue"] and state["dialogue"][-1]["role"] == "finish"
                      else "decision_limit")
    except KeyboardInterrupt:
        status, error = "interrupted", "Stopped by user; saved progress can be resumed."
    except Exception:
        status, error = "error", traceback.format_exc()
    final = summary(status, error)
    if error:
        # Preserve each session's diagnostic instead of overwriting an older one.
        (directory / f"session_{len(resumptions) + 1:02d}_error.txt").write_text(error)
    save_progress(directory, state, final)
    print("Status:", status, flush=True)
    if status == "completed" and final["execution"]:
        print("Verified test RMSE:", final["execution"]["verification"]["metrics"]["test_rmse"], "V", flush=True)
    if error:
        print(error, flush=True)
    print("Transcript:", directory / "dialogue.md", flush=True)
    return 0 if status in {"completed", "waiting_for_human"} else 130 if status == "interrupted" else 1


def request_stop(*_):
    # Treat SIGTERM like Ctrl+C so an executing child is stopped and reaped too.
    raise KeyboardInterrupt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="General model (default: qwen2.5:7b)")
    parser.add_argument("--executor-model", help="Coding model (default: qwen2.5-coder:7b)")
    parser.add_argument("--base-url", help="Ollama endpoint (default: http://localhost:11434)")
    parser.add_argument("--request-file", type=Path, help="Use another user request")
    parser.add_argument("--resume-run", help="Resume an interrupted run by its folder name")
    args = parser.parse_args()
    stamp = datetime.now(ZoneInfo("Asia/Yerevan")).strftime("%Y%m%d_%H%M%S_%f")
    run_root = PROJECT / "output/runs"
    if args.resume_run:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", args.resume_run) or args.request_file:
            parser.error("Use a run folder basename and no new request when resuming")
        directory = run_root / args.resume_run
        if not directory.is_dir():
            parser.error("Saved run folder does not exist")
    else:
        directory = run_root / stamp
        directory.mkdir(parents=True, exist_ok=False)
    signal.signal(signal.SIGTERM, request_stop)
    try:
        # One process owns a run at a time; interrupted process locks release automatically.
        with RunLock(directory):
            return_code = run_session(args, directory, stamp)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    raise SystemExit(return_code)


if __name__ == "__main__":
    main()
