"""The conversation and routing information shared by all graph nodes."""

from operator import add
from typing import Annotated, Literal, NotRequired, TypedDict


class DialogueTurn(TypedDict):
    """A normal dictionary describing one role's message."""

    # The person's reply is appended with role="human", just like agent turns.
    role: str
    text: str
    # Only the executor's Python node attaches genuine tool evidence here.
    execution: NotRequired[dict]


class ProposedAction(TypedDict):
    """The exact next action described in an approval question."""

    # executor runs code; finish ends the dialogue and reports the measurement.
    next_role: Literal["executor", "finish"]
    instruction: str
    # Explain the action and any result it would overwrite in plain language.
    description: str


class HumanRequest(TypedDict):
    """What the orchestrator asks the person while the graph is paused."""

    kind: Literal["clarification", "approval"]
    question: str
    # Clarification has no action attached: choosing a meaning grants no approval.
    proposed_action: ProposedAction | None


class RoutingDecision(TypedDict):
    """The model's next destination, instruction, reason, and optional question."""

    next_role: Literal["scientist", "general", "executor", "human", "finish"]
    instruction: str
    reason: str
    # Only a decision addressed to human includes a human_request.
    human_request: NotRequired[HumanRequest | None]


class AgentState(TypedDict):
    # add is the reducer: old conversation + newly returned messages.
    # Without it, a new node response would replace the previous dialogue.
    dialogue: Annotated[list[DialogueTurn], add]
    # No reducer on these fields: each update replaces their previous values.
    decision_count: int
    decision: RoutingDecision
    # The human handler fills this only after acceptance, then the graph
    # consumes and clears it when performing that specific approved action.
    # None means that no action currently has permission to run.
    approved_action: NotRequired[ProposedAction | None]
