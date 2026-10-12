"""Check that clarification cannot be mistaken for action approval."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pydantic import ValidationError
from schemas import Decision, HumanRequest


ACTION = {"next_role": "executor", "instruction": "Run the planned Silverbox study",
          "description": "Execute a new Python script to train four models"}


class HumanSchemaTests(unittest.TestCase):
    def test_model_schema_requires_an_action_in_the_approval_branch(self):
        schema = Decision.model_json_schema()
        approval = schema["$defs"]["ApprovalRequest"]
        self.assertIn("proposed_action", approval["required"])
        self.assertEqual(approval["properties"]["kind"]["const"], "approval")
        self.assertNotIn("anyOf", approval["properties"]["proposed_action"])

    def test_native_schema_cannot_mix_a_specialist_route_with_a_human_question(self):
        ordinary, human = Decision.model_json_schema()["anyOf"]
        self.assertNotIn("human", ordinary["properties"]["next_role"]["enum"])
        self.assertEqual(ordinary["properties"]["human_request"], {"type": "null"})
        self.assertEqual(human["properties"]["next_role"]["const"], "human")
        self.assertTrue(all(branch.get("type") != "null" for branch in
                            human["properties"]["human_request"]["anyOf"]))

    def test_clarification_has_no_action(self):
        request = HumanRequest(kind="clarification", question="Training MSE or test RMSE?")
        self.assertIsNone(request.proposed_action)
        with self.assertRaises(ValidationError):
            HumanRequest(kind="clarification", question="Which error?", proposed_action=ACTION)

    def test_approval_requires_a_concrete_action(self):
        with self.assertRaises(ValidationError):
            HumanRequest(kind="approval", question="Proceed?")
        request = HumanRequest(kind="approval", question="Run the four-model study?",
                               proposed_action=ACTION)
        self.assertEqual(request.proposed_action.next_role, "executor")
        self.assertEqual(request.proposed_action.description, ACTION["description"])

    def test_human_routing_requires_a_question(self):
        with self.assertRaises(ValidationError):
            Decision(next_role="human", instruction="Ask the user", reason="Ambiguous request")
        with self.assertRaises(ValidationError):
            HumanRequest(kind="clarification", question="   ")

    def test_question_cannot_be_attached_to_an_agent_destination(self):
        with self.assertRaises(ValidationError):
            Decision(next_role="executor", instruction="Run study", reason="Ready",
                     human_request={"kind": "approval", "question": "Proceed?",
                                    "proposed_action": ACTION})

    def test_human_approval_survives_json_round_trip(self):
        decision = Decision(next_role="human", instruction="Ask permission to run the study",
            reason="The next action would execute Python",
            human_request={"kind": "approval", "question": "Run the study?",
                           "proposed_action": ACTION})
        restored = Decision.model_validate_json(decision.model_dump_json())
        self.assertEqual(restored, decision)

    def test_existing_role_decisions_remain_valid(self):
        decision = Decision(next_role="general", instruction="Explain RMSE", reason="User asks about units")
        self.assertIsNone(decision.human_request)


if __name__ == "__main__":
    unittest.main()
