"""StateGraph edges follow model decisions and enforce the two stop rules."""

from langgraph.graph import END, START, StateGraph
from agents import latest_plan, verified_result
from state import AgentState


def route_decision(state):
    choice = state["decision"]["next_role"]
    if choice == "finish":
        if verified_result(state["dialogue"]) is not None:
            return END
        return END if state["decision_count"] >= 8 else "orchestrator"
    if choice == "executor" and latest_plan(state["dialogue"]) is None:
        return END if state["decision_count"] >= 8 else "orchestrator"
    return choice


def return_turn(state):
    # Let decision eight's selected specialist run, then stop without decision nine.
    return END if state["decision_count"] >= 8 else "orchestrator"


def resume_node(state):
    """Resume the saved decision, including an uncompleted specialist turn eight."""
    if not 0 <= state["decision_count"] <= 8:
        raise ValueError("Saved decision count must be between zero and eight")
    # Reuse exactly the same routing guards as an uninterrupted graph.
    if state["dialogue"] and state["dialogue"][-1]["role"] == "orchestrator":
        return route_decision(state)
    return return_turn(state)


def build_graph(nodes, start_at="orchestrator"):
    builder = StateGraph(AgentState)
    for name, function in nodes.items():
        builder.add_node(name, function)
    # Resuming a saved decision begins at its selected specialist, without
    # spending a new decision or resetting the eight-decision budget.
    builder.add_edge(START, start_at)
    builder.add_conditional_edges("orchestrator", route_decision, {
        "scientist": "scientist", "general": "general", "executor": "executor",
        "orchestrator": "orchestrator", END: END})
    for name in ("scientist", "general", "executor"):
        builder.add_conditional_edges(name, return_turn, {"orchestrator": "orchestrator", END: END})
    return builder.compile()
