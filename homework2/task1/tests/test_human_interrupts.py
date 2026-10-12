"""Exercise real interrupts and checkpoints using deterministic test nodes."""

from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from agents import make_nodes
from coding_tool import write_and_run
from graph import build_graph, route_decision
from human import claim_execution_approval
from run import drive_graph, pending_interrupts, read_human_reply, run_session
from schemas import ScientificPlan


ACTION = {"next_role": "executor", "instruction": "Run the planned study once",
          "description": "Execute a new script without replacing earlier results"}
PLAN = ScientificPlan(differential_equation="y'' + a*y' + b*y + c*y**3 = g*u",
    state_equation="y'=v; v'=alpha", network_and_scaling="Training RMS scales",
    architectures=[[8], [16]], settings=[
        dict(optimizer="Adam", learning_rate=.001, steps=500, rollout_length=16, batch_size=1),
        dict(optimizer="AdamW", learning_rate=.002, steps=600, rollout_length=20, batch_size=1)],
    training_method="Training windows", free_run_test="Continuous rollout",
    implementation_steps=["Load", "Build", "Train", "Evaluate", "Report"])
EVIDENCE = {"exit_code": 0, "timed_out": False,
            "verification": {"verified": True, "metrics": {"test_rmse": .038}}}


def initial(count=7):
    return {"dialogue": [{"role": "user", "text": "Run the study"},
                         {"role": "scientist", "text": PLAN.model_dump_json()}],
            "decision_count": count, "approved_action": None,
            "decision": {"next_role": "general", "instruction": "", "reason": ""}}


def ask_node(kind="approval", action=ACTION):
    def orchestrator(state):
        return {"decision_count": state["decision_count"] + 1, "approved_action": None,
            "decision": {"next_role": "human", "instruction": "Ask the person",
                "reason": "Human input needed", "human_request": {
                    "kind": kind, "question": "Run this action?" if kind == "approval" else "Which error?",
                    "proposed_action": action if kind == "approval" else None}},
            "dialogue": [{"role": "orchestrator", "text": "Human input needed"}]}
    return orchestrator


def forbidden(_):
    raise AssertionError("No specialist should run in this test")


def nodes(orchestrator=None, executor=forbidden, general=forbidden):
    return {"orchestrator": orchestrator or ask_node(), "executor": executor,
            "scientist": forbidden, "general": general}


class HumanInterruptTests(unittest.TestCase):
    def setUp(self):
        self.config = {"configurable": {"thread_id": "test-thread"}, "recursion_limit": 40}

    def test_refusal_does_not_start_a_real_process(self):
        with tempfile.TemporaryDirectory() as root:
            def execute(_):
                return {"dialogue": [{"role": "executor", "text": str(
                    write_and_run(root, root, "example.py", "print('ran')"))}]}
            graph = build_graph(nodes(executor=execute), checkpointer=InMemorySaver())
            paused = graph.invoke(initial(), self.config)
            self.assertEqual(paused["__interrupt__"][0].value["proposed_action"], ACTION)
            self.assertFalse(list(Path(root).glob("attempt_*")))
            final = graph.invoke(Command(resume="no"), self.config)
            self.assertEqual(final["dialogue"][-2], {"role": "human", "text": "no"})
            self.assertEqual(final["dialogue"][-1]["role"], "system")
            self.assertIn("step did not run", final["dialogue"][-1]["text"])
            self.assertIsNone(final["approved_action"])
            self.assertFalse(list(Path(root).glob("attempt_*")))
            self.assertEqual(final["decision_count"], 8)

    def test_refusal_returns_to_orchestrator_before_fresh_acceptance(self):
        executed, observed = [], []
        def orchestrator(state):
            replies = [turn["text"] for turn in state["dialogue"] if turn["role"] == "human"]
            if replies:
                observed.append(replies[-1])
            return ask_node()(state)
        def execute(state):
            executed.append(state["decision"]["instruction"])
            return {"dialogue": [{"role": "executor", "text": "ran once"}]}
        graph = build_graph(nodes(orchestrator, executor=execute), checkpointer=InMemorySaver())
        graph.invoke(initial(6), self.config)
        refused = graph.invoke(Command(resume="no, discuss first"), self.config)
        self.assertIn("__interrupt__", refused)
        self.assertEqual(observed, ["no, discuss first"])
        self.assertEqual(executed, [])
        final = graph.invoke(Command(resume="approve"), self.config)
        self.assertEqual(executed, [ACTION["instruction"]])
        self.assertIsNone(final["approved_action"])
        self.assertEqual(final["decision_count"], 8)
        graph.invoke(None, self.config)
        self.assertEqual(len(executed), 1)

    def test_clarification_reply_is_shared_and_never_grants_approval(self):
        observed = []
        def orchestrator(state):
            if state["decision_count"] == 6:
                return ask_node("clarification")(state)
            observed.extend(state["dialogue"])
            return {"decision_count": 8, "decision": {"next_role": "general",
                    "instruction": "Explain test RMSE", "reason": "Human clarified"}}
        def general(_):
            return {"dialogue": [{"role": "general", "text": "Explanation"}]}
        graph = build_graph(nodes(orchestrator, general=general), checkpointer=InMemorySaver())
        graph.invoke(initial(6), self.config)
        final = graph.invoke(Command(resume="yes"), self.config)
        self.assertIn({"role": "human", "text": "yes"}, observed)
        self.assertIsNone(final["approved_action"])
        self.assertEqual(final["dialogue"][-1]["role"], "general")

    def test_sqlite_pause_survives_reopening_and_records_reply_once(self):
        with tempfile.TemporaryDirectory() as root:
            database = str(Path(root) / "checkpoints.sqlite")
            with SqliteSaver.from_conn_string(database) as saver:
                graph = build_graph(nodes(ask_node("clarification")), checkpointer=saver)
                graph.invoke(initial(), self.config)
            # Rebuild everything after closing the first connection.
            with SqliteSaver.from_conn_string(database) as saver:
                graph = build_graph(nodes(ask_node("clarification")), checkpointer=saver)
                self.assertEqual(len(pending_interrupts(graph.get_state(self.config))), 1)
                final = graph.invoke(Command(resume="Test RMSE, please"), self.config)
                self.assertEqual([turn for turn in final["dialogue"] if turn["role"] == "human"],
                                 [{"role": "human", "text": "Test RMSE, please"}])
                self.assertEqual(final["decision_count"], 8)
                self.assertFalse(graph.get_state(self.config).next)

    def test_changed_action_and_direct_routes_are_blocked(self):
        state = initial()
        state["dialogue"].append({"role": "human", "text": "yes"})
        state["approved_action"] = deepcopy(ACTION)
        state["decision"] = {**ACTION, "instruction": "Run a different study"}
        self.assertEqual(route_decision(state), "orchestrator")
        state["decision"] = {**ACTION, "next_role": "finish"}
        self.assertEqual(route_decision(state), "orchestrator")
        state["approved_action"] = None
        state["decision"] = ACTION
        self.assertEqual(route_decision(state), "orchestrator")

    def test_executor_cannot_be_called_without_approval(self):
        class Model:
            def with_structured_output(self, *args, **kwargs):
                return self
            def invoke(self, messages):
                raise AssertionError("Approval must be checked before the model call")
        with tempfile.TemporaryDirectory() as root:
            executor = make_nodes(Model(), root, root)["executor"]
            with self.assertRaisesRegex(ValueError, "requires approval"):
                executor(initial())
            self.assertFalse(list(Path(root).glob("attempt_*")))

    def test_consumed_permission_cannot_be_replayed_after_a_crash(self):
        state = initial()
        state["approved_action"] = ACTION
        with tempfile.TemporaryDirectory() as root:
            claim_execution_approval(root, state)
            with self.assertRaisesRegex(ValueError, "already consumed"):
                claim_execution_approval(root, state)
            # Only a new human reply can create a new approval identity.
            state["dialogue"].append({"role": "human", "text": "approve"})
            claim_execution_approval(root, state)

    def test_finish_needs_both_approval_and_verified_evidence(self):
        action = {**ACTION, "next_role": "finish", "instruction": "Report final test RMSE"}
        graph = build_graph(nodes(ask_node(action=action)), checkpointer=InMemorySaver())
        state = initial()
        evidence = deepcopy(EVIDENCE)
        evidence["verification"]["metrics"].update(train_rmse=.02, zero_test_rmse=.06)
        evidence["verification"]["best_config"] = {"hidden_sizes": [8], "optimizer": "Adam"}
        state["dialogue"].append({"role": "executor", "text": "actual output", "execution": evidence})
        graph.invoke(state, self.config)
        final = graph.invoke(Command(resume="yes"), self.config)
        self.assertEqual(final["dialogue"][-1]["role"], "finish")
        self.assertIn("0.038 V", final["dialogue"][-1]["text"])
        self.assertIn("Train free-run RMSE: 0.02 V", final["dialogue"][-1]["text"])
        self.assertIn("baseline RMSE: 0.06 V", final["dialogue"][-1]["text"])
        self.assertIn('"hidden_sizes": [8]', final["dialogue"][-1]["text"])
        self.assertIsNone(final["approved_action"])
        other = build_graph(nodes(ask_node(action=action)), checkpointer=InMemorySaver())
        other.invoke(initial(), self.config)
        rejected = other.invoke(Command(resume="yes"), self.config)
        self.assertEqual(rejected["dialogue"][-1]["role"], "human")

    def test_runner_saves_pause_on_eof_then_resumes_without_duplicate_dialogue(self):
        graph = build_graph(nodes(ask_node("clarification")), checkpointer=InMemorySaver())
        recorded = []
        self.assertTrue(drive_graph(graph, initial(), self.config, recorded.append, ask=lambda _: None))
        self.assertEqual(len(recorded[1]["dialogue"]), 3)
        self.assertEqual(recorded[-1]["decision_count"], 8)
        self.assertFalse(drive_graph(graph, Command(resume="Test error"), self.config,
                                    recorded.append, ask=lambda _: self.fail("Unexpected second question")))
        human = [turn for turn in recorded[-1]["dialogue"] if turn["role"] == "human"]
        self.assertEqual(human, [{"role": "human", "text": "Test error"}])

    def test_input_never_invents_an_eof_reply(self):
        with patch("builtins.input", side_effect=EOFError):
            self.assertIsNone(read_human_reply({"kind": "clarification", "question": "Which error?"}))

    def test_runner_reopens_native_thread_and_retains_actual_reply(self):
        def orchestrator(state):
            if state["decision_count"] == 0:
                return ask_node("clarification")(state)
            self.assertEqual(state["dialogue"][-1], {"role": "human", "text": "Test RMSE"})
            return {"decision_count": 8, "decision": {"next_role": "general",
                    "instruction": "Explain test RMSE", "reason": "Human clarified"}}
        def general(_):
            return {"dialogue": [{"role": "general", "text": "Test RMSE explanation"}]}
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root) / "output/runs/example"
            directory.mkdir(parents=True)
            request = Path(root) / "request.txt"
            request.write_text("Explain the error")
            args = SimpleNamespace(resume_run=None, request_file=request,
                                   model=None, executor_model=None, base_url=None)
            with patch("run.ChatOllama"), patch("run.make_nodes", return_value=nodes(orchestrator, general=general)):
                with patch("builtins.input", side_effect=EOFError):
                    self.assertEqual(run_session(args, directory, "first"), 0)
                saved = json.loads((directory / "checkpoint.json").read_text())
                self.assertEqual(saved["summary"]["status"], "waiting_for_human")
                self.assertEqual(saved["summary"]["thread_id"], "example")
                args.resume_run, args.request_file = "example", None
                with patch("builtins.input", return_value="Test RMSE") as user_input:
                    self.assertEqual(run_session(args, directory, "second"), 1)
                    user_input.assert_called_once()
            saved = json.loads((directory / "checkpoint.json").read_text())
            self.assertEqual(saved["summary"]["status"], "decision_limit")
            self.assertEqual([turn for turn in saved["state"]["dialogue"] if turn["role"] == "human"],
                             [{"role": "human", "text": "Test RMSE"}])
            self.assertEqual(len(saved["summary"]["resumptions"]), 1)


if __name__ == "__main__":
    unittest.main()
