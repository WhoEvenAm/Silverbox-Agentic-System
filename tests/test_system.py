"""Focused checks for real process execution and graph termination rules."""

from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents import make_nodes, verified_result
from coding_tool import write_and_run
from graph import build_graph
from verification import printed_number
from schemas import CodeSubmission, Decision, ScientificPlan


class SystemTests(unittest.TestCase):
    def test_checked_result_is_visible_but_model_still_selects_role(self):
        class Model:
            def with_structured_output(self, schema, **kwargs):
                return self

            def invoke(self, messages):
                self.messages = messages
                return Decision(next_role="general", instruction="Explain the units",
                                reason="The user still needs an explanation.")

        model = Model()
        evidence = {"exit_code": 0, "timed_out": False,
                    "verification": {"verified": True, "metrics": {"test_rmse": .038}}}
        state = {"dialogue": [{"role": "user", "text": "Explain the units"},
                             {"role": "executor", "text": "actual output", "execution": evidence}],
                 "decision_count": 3}
        with tempfile.TemporaryDirectory() as root:
            result = make_nodes(model, root, root)["orchestrator"](state)
        self.assertEqual(result["decision"]["next_role"], "general")
        self.assertIn("Explain the units", model.messages[1].content)
        self.assertIn("actual output", model.messages[1].content)
        self.assertIn("ALREADY printed TEST_RMSE_V = 0.038", model.messages[-1].content)

    def test_executor_structured_arguments_reach_real_process(self):
        class Model:
            def with_structured_output(self, schema, **kwargs):
                return self

            def invoke(self, messages):
                return CodeSubmission(filename="generated.py", code=
                    "print('generated code ran', flush=True)\nraise RuntimeError('repair needed')\n")

        plan = ScientificPlan(differential_equation="y'' + a*y' + b*y + c*y**3 = g*u",
            state_equation="y'=v; v'=alpha", network_and_scaling="Training RMS scales",
            architectures=[[8], [16]], settings=[
                dict(optimizer="Adam", learning_rate=.001, steps=500, rollout_length=16, batch_size=1),
                dict(optimizer="AdamW", learning_rate=.002, steps=600, rollout_length=20, batch_size=1)],
            training_method="Training windows", free_run_test="Continuous rollout",
            implementation_steps=["Load", "Build", "Train", "Evaluate", "Report"])
        with tempfile.TemporaryDirectory() as root:
            nodes = make_nodes(Model(), root, root)
            result = nodes["executor"]({"dialogue": [{"role": "scientist", "text": plan.model_dump_json()}]})
            evidence = result["dialogue"][0]["execution"]
            self.assertEqual(evidence["exit_code"], 1)
            self.assertIn("generated code ran", evidence["stdout"])
            self.assertIn("repair needed", evidence["stderr"])
            self.assertFalse(evidence["verification"]["verified"])
            self.assertIsNone(verified_result(result["dialogue"]))

    def test_tool_preserves_exit_code_and_traceback(self):
        with tempfile.TemporaryDirectory() as root:
            result = write_and_run(root, root, "failure.py",
                "print('before failure', flush=True)\nraise ValueError('example error')\n")
            self.assertEqual(result["exit_code"], 1)
            self.assertIn("before failure", result["stdout"])
            self.assertIn("ValueError: example error", result["stderr"])
            self.assertTrue(Path(result["script_path"]).is_file())

    def test_tool_terminates_timed_out_script(self):
        with tempfile.TemporaryDirectory() as root:
            result = write_and_run(root, root, "wait.py", "import time\ntime.sleep(30)\n", 0.1)
            self.assertTrue(result["timed_out"])
            self.assertNotEqual(result["exit_code"], 0)
            self.assertLess(result["duration_seconds"], 5)

    def test_tool_rejects_path_escape(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(ValueError):
                write_and_run(root, root, "../outside.py", "print(1)")

    def test_claimed_rmse_cannot_finish_and_budget_is_exactly_eight(self):
        calls = []

        def orchestrator(state):
            calls.append(state["decision_count"])
            return {"decision_count": state["decision_count"] + 1,
                "decision": {"next_role": "finish", "instruction": "done", "reason": "claimed score"},
                "dialogue": [{"role": "orchestrator", "text": "finish"}]}

        def unexpected(state):
            self.fail("An unsupported finish must return to the orchestrator")

        graph = build_graph(dict(orchestrator=orchestrator, scientist=unexpected,
                                 general=unexpected, executor=unexpected))
        initial = {"dialogue": [{"role": "executor", "text": "TEST_RMSE_V = 0.01"}],
            "decision_count": 0, "decision": {"next_role": "finish", "instruction": "", "reason": ""}}
        final = graph.invoke(initial, config={"recursion_limit": 32})
        self.assertEqual(len(calls), 8)
        self.assertEqual(final["decision_count"], 8)
        self.assertEqual(len(final["dialogue"]), 9)
        self.assertIsNone(verified_result(final["dialogue"]))

    def test_specialist_can_take_eighth_turn_without_ninth_decision(self):
        def orchestrator(state):
            return {"decision_count": state["decision_count"] + 1,
                "decision": {"next_role": "general", "instruction": "explain", "reason": "question"},
                "dialogue": [{"role": "orchestrator", "text": "general"}]}

        def general(state):
            self.assertEqual(state["dialogue"][-1]["role"], "orchestrator")
            return {"dialogue": [{"role": "general", "text": "explanation"}]}

        graph = build_graph(dict(orchestrator=orchestrator, general=general,
                                 scientist=general, executor=general))
        final = graph.invoke({"dialogue": [], "decision_count": 0,
            "decision": {"next_role": "general", "instruction": "", "reason": ""}},
            config={"recursion_limit": 32})
        self.assertEqual(final["decision_count"], 8)
        self.assertEqual(final["dialogue"][-1]["role"], "general")
        self.assertEqual(len(final["dialogue"]), 16)

    def test_duplicate_or_nonfinite_scores_are_rejected(self):
        self.assertEqual(printed_number("TEST_RMSE_V = 0.012\n", "TEST_RMSE_V"), 0.012)
        for text in ("TEST_RMSE_V = 0.01\nTEST_RMSE_V = 0.02\n", "TEST_RMSE_V = 1e999\n"):
            with self.assertRaises(ValueError):
                printed_number(text, "TEST_RMSE_V")


if __name__ == "__main__":
    unittest.main()
