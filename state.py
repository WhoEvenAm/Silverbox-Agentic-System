"""The conversation and routing information shared by all graph nodes."""

from operator import add
from typing import Annotated, Literal, NotRequired, TypedDict


class DialogueTurn(TypedDict):
    """A normal dictionary describing one role's message."""

    role: str
    text: str
    # Only the executor's Python node attaches genuine tool evidence here.
    execution: NotRequired[dict]


class RoutingDecision(TypedDict):
    """The orchestrator's three model-generated decision fields."""

    next_role: Literal["scientist", "general", "executor", "finish"]
    instruction: str
    reason: str


class AgentState(TypedDict):
    # add is the reducer: old conversation + newly returned messages.
    # Without it, a new node response would replace the previous dialogue.
    dialogue: Annotated[list[DialogueTurn], add]
    # No reducer on these fields: each update replaces their previous values.
    decision_count: int
    decision: RoutingDecision
