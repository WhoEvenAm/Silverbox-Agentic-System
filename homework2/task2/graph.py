"""Model-selected four-role routing and the original eight-decision budget."""

from langgraph.graph import END, START, StateGraph
from agents import latest_plan, verified_results
from state import AgentState


def return_turn(state):
    return END if state["decision_count"] >= 8 else "orchestrator"


def route_decision(state):
    choice = state["decision"]["next_role"]
    if choice == "executor" and latest_plan(state["dialogue"]) is None:
        return return_turn(state)
    if choice == "finish" and len(verified_results(state["dialogue"])) < 2:
        return return_turn(state)
    return choice


def finish_node(state):
    results = verified_results(state["dialogue"])
    if len(results) < 2:
        raise ValueError("Finish requires at least two measured experiments")
    best = max(results, key=lambda result: result["final"]["enhancement"])
    final = best["final"]
    return {"dialogue": [{"role": "finish", "text":
        f"Best tool-returned enhancement: {final['enhancement']:.6f} "
        f"(validation-shot SEM {final['enhancement_sem']:.6f}). "
        f"Camera images: {best['camera_images']}."}]}


def build_graph(nodes, checkpointer=None):
    builder = StateGraph(AgentState)
    for name, function in nodes.items():
        builder.add_node(name, function)
    builder.add_node("finish", finish_node)
    builder.add_edge(START, "orchestrator")
    builder.add_conditional_edges("orchestrator", route_decision,
        {name: name for name in ("scientist", "general", "executor", "finish", "orchestrator", END)})
    for name in ("scientist", "general", "executor"):
        builder.add_conditional_edges(name, return_turn, {"orchestrator": "orchestrator", END: END})
    builder.add_edge("finish", END)
    return builder.compile(checkpointer=checkpointer)
