"""Four model agents with full dialogue and executor-owned optical-bench tools."""

import json
import math
from pathlib import Path
import time
import traceback

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import HumanMessage, SystemMessage

from coding_tool import make_coding_tool
from focusing_tool import make_focusing_tool
from schemas import Decision, FocusPlan, FocusSettings
import prompts


def verified_results(dialogue):
    results = []
    for turn in dialogue:
        result = turn.get("tool_result", {})
        score = result.get("final", {}).get("enhancement")
        if (turn["role"] == "executor" and result.get("tool") == "focus_beam"
                and result.get("verified") is True and isinstance(score, (int, float))
                and math.isfinite(score) and score > 0):
            results.append(result)
    return results


def latest_plan(dialogue):
    for turn in reversed(dialogue):
        if turn["role"] == "scientist":
            try:
                return FocusPlan.model_validate_json(turn["text"])
            except ValueError:
                pass
    return None


def messages_for(prompt, state):
    dialogue = "\n\n".join(f"[{turn['role']}]\n{turn['text']}" for turn in state["dialogue"])
    return [SystemMessage(content=prompt), HumanMessage(content=dialogue)]


def make_nodes(model, executor_model, directory, event):
    structured = {"orchestrator": model.with_structured_output(Decision, method="json_schema"),
                  "scientist": model.with_structured_output(FocusPlan, method="json_schema"),
                  "executor": executor_model.with_structured_output(FocusSettings, method="json_schema")}
    # Keep the existing coding tool and add the optical tool to this role's registry.
    tools = {"write_and_run_python": make_coding_tool(directory, directory),
             "focus_beam": make_focusing_tool(directory, event)}

    def invoke(role, selected, messages):
        event("model_started", {"role": role})
        started = time.perf_counter()
        record = {}
        try:
            response = selected.invoke(messages)
            record = response.model_dump() if hasattr(response, "model_dump") else {"text": response.content}
            event("model_completed", {"role": role, "response": record})
            return response
        except Exception as error:
            record = {"error": str(error)}
            event("model_error", {"role": role, "error": str(error)[:1200]})
            raise
        finally:
            with (Path(directory) / "model_calls.jsonl").open("a") as file:
                file.write(json.dumps({"role": role, "duration_seconds": time.perf_counter() - started,
                    "messages": [{"role": item.type, "text": item.content} for item in messages],
                    "response": record}) + "\n")

    def validated_call(role, messages):
        # A malformed answer is corrected by the model, not repaired into a Python-chosen route.
        for attempt in range(3):
            try:
                return invoke(role, structured[role], messages)
            except OutputParserException as error:
                if attempt == 2:
                    raise
                messages.append(HumanMessage(content="Correct the JSON/schema error and return only valid JSON: " + str(error)[:2000]))

    def orchestrator(state):
        count = state["decision_count"] + 1
        if count > 8:
            raise ValueError("Cannot make more than eight decisions")
        messages = messages_for(prompts.ORCHESTRATOR + f"\nDecision {count} of 8.", state)
        results = verified_results(state["dialogue"])
        plan = latest_plan(state["dialogue"])
        # Model prose can wrongly claim that planning already happened. Repeat
        # authoritative runtime facts after the transcript so it can correct that.
        messages.append(HumanMessage(content="Authoritative runtime status: " + json.dumps({
            "validated_scientist_plan_exists": plan is not None,
            "completed_focus_experiments": len(results),
            "executor_requires_validated_scientist_plan": True,
            "finish_requires_at_least_two_tool_results": True})
            + (" No scientist plan exists yet. The user's request is a goal, not a FocusPlan. "
               "Ask scientist to choose actual settings before execution." if plan is None else "")))
        if results:
            messages.append(HumanMessage(content="Runtime verified experiment results: " + json.dumps([
                {"settings": result["settings"], "enhancement": result["final"]["enhancement"]}
                for result in results])))
        decision = validated_call("orchestrator", messages)
        text = decision.model_dump_json(indent=2)
        if decision.next_role == "executor" and plan is None:
            text += "\nRuntime check: executor blocked. No validated scientist FocusPlan exists. Request actual settings from scientist."
        if decision.next_role == "finish" and len(results) < 2:
            text += "\nRuntime check: finish blocked. At least two actual bench results are required."
        event("decision", {"number": count, **decision.model_dump()})
        return {"decision_count": count, "decision": decision.model_dump(),
                "dialogue": [{"role": "orchestrator", "text": text}]}

    def scientist(state):
        plan = validated_call("scientist", messages_for(prompts.SCIENTIST, state))
        return {"dialogue": [{"role": "scientist", "text": plan.model_dump_json(indent=2)}]}

    def general(state):
        messages = messages_for(prompts.GENERAL, state)
        # Label initial/final statistics explicitly: a long transcript can make
        # the model attach the initial uncertainty to the final measurement.
        messages.append(HumanMessage(content="Authoritative measured statistics: " + json.dumps([
            {"settings": result["settings"], "initial": result["initial"], "final": result["final"]}
            for result in verified_results(state["dialogue"])])))
        reply = invoke("general", model, messages)
        return {"dialogue": [{"role": "general", "text": str(reply.content)}]}

    def executor(state):
        plan = latest_plan(state["dialogue"])
        if plan is None:
            return {"dialogue": [{"role": "executor", "text": "No valid focus plan exists."}]}
        try:
            messages = messages_for(prompts.EXECUTOR, state)
            messages.append(HumanMessage(content="Latest validated scientist plan: " + plan.model_dump_json()))
            settings = validated_call("executor", messages)
            completed = verified_results(state["dialogue"])
            if any(result["settings"] == settings.model_dump() for result in completed):
                raise ValueError("These settings were already completed; choose an untried experiment")
            if settings.model_dump() not in [item.model_dump() for item in plan.experiments]:
                raise ValueError("Choose an experiment from the latest scientist plan")
            event("tool_call", {"role": "executor", "tool": "focus_beam", "arguments": settings.model_dump()})
            result = tools["focus_beam"].invoke(settings.model_dump())
            return {"dialogue": [{"role": "executor", "text": json.dumps(result, indent=2), "tool_result": result}]}
        except Exception:
            text = "Focus tool call failed; no measured result accepted.\n" + traceback.format_exc()
            event("executor_error", {"error": text[-1600:]})
            return {"dialogue": [{"role": "executor", "text": text}]}

    executor.tools = tools
    return {"orchestrator": orchestrator, "scientist": scientist, "general": general, "executor": executor}
