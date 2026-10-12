"""Pause for a real reply and keep each approval specific and single-use."""

import hashlib
import json
import os
from pathlib import Path

from langgraph.types import interrupt

from schemas import HumanRequest


def human_node(state):
    request = HumanRequest.model_validate(state["decision"]["human_request"])
    # LangGraph restarts this node on resume. Do not write anything before
    # interrupt(): only its returned value is an actual human reply.
    reply = interrupt(request.model_dump())
    if not isinstance(reply, str) or not reply.strip():
        raise ValueError("A human reply must be a nonempty string")
    update = {"dialogue": [{"role": "human", "text": reply}], "approved_action": None}
    # Clarification changes our understanding, but never grants permission.
    if request.kind == "approval" and reply.strip().casefold() in {"yes", "y", "approve", "accept"}:
        action = request.proposed_action.model_dump()
        update["approved_action"] = action
        # Go directly to the accepted action without spending another decision.
        update["decision"] = {"next_role": action["next_role"],
                              "instruction": action["instruction"],
                              "reason": "The human accepted this specific action.",
                              "human_request": None}
    elif request.kind == "approval":
        # This branch never routes to the proposed action or invokes a tool.
        # Keep explicit refusal evidence in the dialogue for the assignment.
        action = request.proposed_action
        update["dialogue"].append({"role": "system", "text":
            f"Approval was not granted. Proposed {action.next_role} step did not run: "
            f"{action.instruction}. Returning control to the orchestrator."})
    return update


def action_is_approved(state, role):
    """An approval cannot be reused for a different destination or instruction."""
    action = state.get("approved_action")
    decision = state.get("decision", {})
    dialogue = state.get("dialogue", [])
    return bool(action and dialogue and dialogue[-1]["role"] == "human"
                and action["next_role"] == role == decision.get("next_role")
                and action["instruction"] == decision.get("instruction"))


def claim_execution_approval(directory, state):
    """Consume permission durably before invoking the coding tool."""
    # The same checkpoint has the same identity; a new human reply has a new
    # transcript. Exclusive creation blocks replay after a crash during execution.
    identity = json.dumps({"action": state["approved_action"],
                           "dialogue": state["dialogue"]}, sort_keys=True)
    key = hashlib.sha256(identity.encode()).hexdigest()
    root = Path(directory) / "consumed_approvals"
    root.mkdir(exist_ok=True)
    try:
        with (root / f"{key}.json").open("x") as file:
            json.dump(state["approved_action"], file)
            file.flush()
            os.fsync(file.fileno())
    except FileExistsError:
        raise ValueError("This approval was already consumed; ask for fresh approval") from None
