"""Four node functions: read shared dialogue and append a role-tagged message."""

import json
import math
from pathlib import Path
from time import perf_counter
import traceback
from langchain_core.messages import HumanMessage, SystemMessage

import prompts
from coding_tool import make_coding_tool
from schemas import CodeSubmission, Decision, ScientificPlan
from verification import verify_execution
from persistence import write_json


def latest_plan(dialogue):
    # Search backward for the most recent valid scientist message.
    for turn in reversed(dialogue):
        if turn["role"] == "scientist":
            try:
                return ScientificPlan.model_validate_json(turn["text"])
            except ValueError:
                continue
    return None


def verified_result(dialogue):
    """Model prose cannot manufacture the attached Python execution evidence."""
    for turn in reversed(dialogue):
        evidence = turn.get("execution", {})
        check = evidence.get("verification", {})
        value = check.get("metrics", {}).get("test_rmse")
        if (turn["role"] == "executor" and evidence.get("exit_code") == 0
                and not evidence.get("timed_out") and check.get("verified") is True
                and isinstance(value, (int, float)) and math.isfinite(value) and value >= 0):
            return evidence
    return None


def messages_for(system_prompt, state):
    # Workflow speaker names remain explicit in the transcript sent to every role.
    transcript = "\n\n".join(f"[{turn['role']}]\n{turn['text']}" for turn in state["dialogue"])
    return [SystemMessage(content=system_prompt), HumanMessage(content=transcript)]


class ModelCalls:
    """Keep real prompts, model responses, and elapsed time for the report."""

    def __init__(self, directory, role_models):
        self.path = Path(directory) / "model_calls.jsonl"
        self.role_models = role_models

    def invoke(self, role, model, messages):
        print(f"[{role}] Calling local model...", flush=True)
        started = perf_counter()
        record = {"error": "Model call interrupted before completion"}
        try:
            response = model.invoke(messages)
            record = response.model_dump() if hasattr(response, "model_dump") else response
        except Exception as error:
            record = {"error": str(error)}
            raise
        finally:
            elapsed = perf_counter() - started
            with self.path.open("a") as file:
                file.write(json.dumps({"role": role, "model": self.role_models[role], "duration_seconds": elapsed,
                    "messages": [{"role": message.type, "text": message.content} for message in messages],
                    "response": record}, default=str) + "\n")
        print(f"[{role}] Model responded in {elapsed:.1f}s", flush=True)
        return response


def make_nodes(model, run_directory, support_directory, executor_model=None):
    names = {role: getattr(model, "model", "test-double")
             for role in ("orchestrator", "scientist", "general")}
    names["executor"] = getattr(executor_model or model, "model", "test-double")
    calls = ModelCalls(run_directory, names)
    router_model = model.with_structured_output(Decision, method="json_schema")
    scientist_model = model.with_structured_output(ScientificPlan, method="json_schema")
    coding_tool = make_coding_tool(run_directory, support_directory)
    # Only this node has the coding tool. Structured arguments avoid unreliable
    # native tool-call parsing in smaller local models. The code still comes
    # entirely from the executor model, rather than a prewritten fitting script.
    executor_model = (executor_model or model).with_structured_output(CodeSubmission, method="json_schema")

    def orchestrator(state):
        # The language model chooses the next role; Python only enforces limits.
        count = state["decision_count"] + 1
        if count > 8:
            raise RuntimeError("More than eight orchestrator decisions attempted")
        system = prompts.ORCHESTRATOR + f"\nThis is decision {count} of 8."
        messages = messages_for(system, state)
        evidence = verified_result(state["dialogue"])
        if evidence is not None:
            # Repeat checked facts after the long transcript so the model can
            # recognize completed work. The model still chooses the next role.
            messages.append(HumanMessage(content="Runtime verification status: the complete study succeeded. "
                "All four planned fits, lowest-test-RMSE model selection, the final continuous "
                "4096-sample free run, train/test/zero-baseline scores, saved checkpoints, "
                "predictions, and comparison plot are complete and independently verified. "
                "The script ALREADY printed TEST_RMSE_V = "
                + str(evidence["verification"]["metrics"]["test_rmse"])
                + ". No further fitting, selection, or final evaluation is pending. "
                "Do not request another execution merely to print or confirm this score. "
                "Choose whether an outstanding explanation is needed or the task is finished."))
        decision = calls.invoke("orchestrator", router_model, messages)
        text = decision.model_dump_json(indent=2)
        if decision.next_role == "finish" and verified_result(state["dialogue"]) is None:
            text += "\nRuntime check: finish rejected; obtain a verified executor result."
        if decision.next_role == "executor" and latest_plan(state["dialogue"]) is None:
            text += "\nRuntime check: execution rejected; obtain a valid scientist plan."
        print(f"[orchestrator] Decision {count}: {decision.next_role}", flush=True)
        return {"decision_count": count, "decision": decision.model_dump(),
                "dialogue": [{"role": "orchestrator", "text": text}]}

    def scientist(state):
        # Validate the plan's fields and preserve its settings during code repairs.
        try:
            system = prompts.SCIENTIST + "\nResponse JSON schema:\n" + json.dumps(ScientificPlan.model_json_schema())
            previous = latest_plan(state["dialogue"])
            if previous is not None:
                system += "\nKeep this existing scientific plan unchanged: " + previous.model_dump_json()
            plan = calls.invoke("scientist", scientist_model, messages_for(system, state))
            if previous is not None and plan.model_dump() != previous.model_dump():
                raise ValueError("The original scientific plan must remain unchanged")
            text = plan.model_dump_json(indent=2)
        except ValueError as error:
            text = "Plan validation failed; request a corrected plan.\n" + str(error)[:1400]
        return {"dialogue": [{"role": "scientist", "text": text}]}

    def general(state):
        # This role explains concepts and receives no code execution tool.
        reply = calls.invoke("general", model, messages_for(prompts.GENERAL, state))
        return {"dialogue": [{"role": "general", "text": str(reply.content)}]}

    def executor(state):
        # Generate tool arguments, run the script, then independently check results.
        plan = latest_plan(state["dialogue"])
        if plan is None:
            return {"dialogue": [{"role": "executor", "text": "No valid scientist plan exists."}]}
        try:
            system = prompts.EXECUTOR + "\nResponse JSON schema:\n" + json.dumps(CodeSubmission.model_json_schema())
            messages = messages_for(system, state)
            # Keep the exact helper/artifact interface salient after the dialogue,
            # including on repairs whose routing instruction names one error.
            choices = {"architectures": plan.architectures,
                       "settings": [setting.model_dump() for setting in plan.settings]}
            messages.append(HumanMessage(content=prompts.EXECUTION_CONTRACT
                + "\nValidated scientist choices to use exactly:\n" + json.dumps(choices)))
            reply = calls.invoke("executor", executor_model, messages)
            print("[executor] Writing and running generated Python...", flush=True)
            result = coding_tool.invoke(reply.model_dump())
            print(f"[executor] Script exit code: {result['exit_code']}", flush=True)
            try:
                result["verification"] = verify_execution(result, plan, support_directory)
            except Exception:
                result["verification"] = {"verified": False, "error": traceback.format_exc()}
            write_json(Path(result["working_directory"]) / "execution.json", result)
            # Retain the generated code in shared dialogue so later executor
            # turns can repair the actual previous implementation.
            text = ("Generated Python (" + reply.filename + "):\n```python\n" + reply.code
                    + "\n```\nActual coding-tool result:\n" + json.dumps(result, indent=2))
            return {"dialogue": [{"role": "executor", "text": text, "execution": result}]}
        except Exception:
            return {"dialogue": [{"role": "executor", "text":
                "Coding tool failed; no accepted measurement.\n" + traceback.format_exc()}]}

    return {"orchestrator": orchestrator, "scientist": scientist,
            "general": general, "executor": executor}
