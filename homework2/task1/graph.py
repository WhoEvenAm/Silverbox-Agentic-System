"""Follow model-selected routes while enforcing approval and decision limits."""

import json

from langgraph.graph import END, START, StateGraph
from agents import latest_plan, verified_result
from human import action_is_approved, human_node
from state import AgentState


def route_decision(state):
    choice = state["decision"]["next_role"]
    if choice in {"executor", "finish"}:
        # Model prose and an earlier approval cannot authorize a new action.
        if not action_is_approved(state, choice):
            return return_turn(state)
        if choice == "finish" and verified_result(state["dialogue"]) is None:
            return return_turn(state)
        if choice == "executor" and latest_plan(state["dialogue"]) is None:
            return return_turn(state)
    return choice


def return_turn(state):
    # Let decision eight's selected specialist run, then stop without decision nine.
    return END if state["decision_count"] >= 8 else "orchestrator"


def route_human(state):
    # Acceptance routes to the exact action. Refusal and clarification return
    # to the orchestrator, which still sees the person's full reply.
    if state.get("approved_action"):
        return route_decision(state)
    return return_turn(state)


def finish_node(state):
    # Finishing is a control step, not a fifth model agent. Consume permission
    # and report only the independently verified measurement.
    if not action_is_approved(state, "finish"):
        raise ValueError("Finishing requires approval of this exact action")
    evidence = verified_result(state["dialogue"])
    if evidence is None:
        raise ValueError("Finishing requires a verified executor result")
    score = evidence["verification"]["metrics"]["test_rmse"]
    lines = [f"Approved final test free-run RMSE: {score} V (1024 test samples)."]
    metrics = evidence["verification"]["metrics"]
    # Report available checked metrics and settings, never model-invented values.
    for key, label in (("train_rmse", "Train free-run RMSE"),
                       ("zero_test_rmse", "Zero-prediction test baseline RMSE")):
        if key in metrics:
            lines.append(f"{label}: {metrics[key]} V.")
    config = evidence["verification"].get("best_config")
    if config is not None:
        lines.append("Selected settings: " + json.dumps(config, sort_keys=True))
    return {"approved_action": None, "dialogue": [{"role": "finish", "text": "\n".join(lines)}]}


def resume_node(state):
    """Import legacy JSON progress; native checkpoints resume their own tasks."""
    if not 0 <= state["decision_count"] <= 8:
        raise ValueError("Saved decision count must be between zero and eight")
    if state.get("approved_action"):
        return route_decision(state)
    if state["dialogue"] and state["dialogue"][-1]["role"] == "orchestrator":
        return route_decision(state)
    return return_turn(state)


def build_graph(nodes, start_at="orchestrator", checkpointer=None):
    builder = StateGraph(AgentState)
    for name, function in nodes.items():
        if name == "executor":
            def guarded_executor(state, execute=function):
                if not action_is_approved(state, "executor"):
                    raise ValueError("Execution requires approval of this exact action")
                # Clear permission even if the specialist returns a tool failure.
                return {**execute(state), "approved_action": None}
            builder.add_node(name, guarded_executor)
        else:
            builder.add_node(name, function)
    builder.add_node("human", human_node)
    builder.add_node("finish", finish_node)
    # Resuming a saved decision begins at its selected specialist, without
    # spending a new decision or resetting the eight-decision budget.
    builder.add_edge(START, start_at)
    destinations = {name: name for name in
                    ("scientist", "general", "executor", "human", "finish", "orchestrator")}
    destinations[END] = END
    builder.add_conditional_edges("orchestrator", route_decision, destinations)
    builder.add_conditional_edges("human", route_human, destinations)
    for name in ("scientist", "general", "executor"):
        builder.add_conditional_edges(name, return_turn, {"orchestrator": "orchestrator", END: END})
    builder.add_edge("finish", END)
    return builder.compile(checkpointer=checkpointer)
