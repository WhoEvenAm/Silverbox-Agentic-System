"""The original four-role shared dialogue, adapted to focusing experiments."""

from operator import add
from typing import Annotated, NotRequired, TypedDict


class Turn(TypedDict):
    role: str
    text: str
    # Only the Python executor attaches actual bench-tool evidence.
    tool_result: NotRequired[dict]


class AgentState(TypedDict):
    dialogue: Annotated[list[Turn], add]
    decision_count: int
    decision: dict
