"""Exercise actual bench calls through four-role graph control with test models."""

from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agents import make_nodes, verified_results
from graph import build_graph
from schemas import Decision, FocusPlan, FocusSettings
from test_bench import settings


class Model:
    def __init__(self):
        self.decisions = 0
        self.executions = 0
        self.router_messages = []
        self.experiments = [settings(iterations=10), settings(iterations=10, macropixel_size=8)]

    def with_structured_output(self, schema, **kwargs):
        parent = self
        class Structured:
            def invoke(self, messages):
                if schema is Decision:
                    parent.router_messages.append(messages)
                    parent.decisions += 1
                    # Start with one mistaken route to exercise runtime feedback.
                    choice = ("executor", "scientist", "executor", "executor", "finish")[parent.decisions - 1]
                    return Decision(next_role=choice, instruction="Perform the selected step", reason="Test routing")
                if schema is FocusPlan:
                    return FocusPlan(physics="Amplitude-only scattering", enhancement_definition="Focus/background",
                                     experiments=parent.experiments, rationale="Compare two macro sizes")
                if schema is FocusSettings:
                    result = parent.experiments[parent.executions]
                    parent.executions += 1
                    return result
                raise AssertionError("Unexpected schema")
        return Structured()


class AgentTests(unittest.TestCase):
    def test_executor_calls_bench_and_finish_keeps_returned_measurement(self):
        events = []
        model = Model()
        with tempfile.TemporaryDirectory() as root:
            nodes = make_nodes(model, model, root, lambda kind, data: events.append((kind, data)))
            self.assertEqual(set(nodes["executor"].tools), {"focus_beam", "write_and_run_python"})
            graph = build_graph(nodes)
            final = graph.invoke({"dialogue": [{"role": "user", "text": "Focus the beam"}],
                "decision_count": 0, "decision": {}}, {"recursion_limit": 32})
            results = verified_results(final["dialogue"])
            self.assertEqual(len(results), 2)
            self.assertEqual(final["decision_count"], 5)
            self.assertIn('"validated_scientist_plan_exists": false', model.router_messages[0][-1].content)
            self.assertIn("executor blocked", model.router_messages[1][1].content)
            self.assertEqual(final["dialogue"][-1]["role"], "finish")
            best = max(result["final"]["enhancement"] for result in results)
            self.assertIn(f"{best:.6f}", final["dialogue"][-1]["text"])
            calls = [data for kind, data in events if kind == "tool_call"]
            self.assertEqual(len(calls), 2)
            self.assertTrue(all(call["role"] == "executor" for call in calls))
            self.assertEqual(calls[0]["arguments"], model.experiments[0].model_dump())

    def test_claimed_enhancement_cannot_finish_and_budget_stays_eight(self):
        def orchestrator(state):
            return {"decision_count": state["decision_count"] + 1,
                "decision": {"next_role": "finish", "instruction": "Finish", "reason": "Claimed score"},
                "dialogue": [{"role": "orchestrator", "text": "finish"}]}
        def unexpected(_):
            self.fail("No unverified measurement may run a specialist or finish")
        graph = build_graph(dict(orchestrator=orchestrator, scientist=unexpected,
                                 general=unexpected, executor=unexpected))
        final = graph.invoke({"dialogue": [{"role": "executor", "text": "Enhancement: 9999"}],
            "decision_count": 0, "decision": {}}, {"recursion_limit": 32})
        self.assertEqual(final["decision_count"], 8)
        self.assertEqual(verified_results(final["dialogue"]), [])


if __name__ == "__main__":
    unittest.main()
