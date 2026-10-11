"""Build the written agent-system report from a completed, verified real run."""

import argparse
from html import escape
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (
    Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

PROJECT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-url", required=True)
    args = parser.parse_args()
    summary = json.loads((PROJECT / "output/latest_run.json").read_text())
    if summary["status"] != "completed":
        raise ValueError("Only a completed, independently verified run can produce the report")
    # Rebase recorded host paths so the published repository is portable.
    run = PROJECT / "output/runs" / Path(summary["run_directory"]).name
    state = json.loads((run / "dialogue.json").read_text())
    evidence = summary["execution"]
    attempt = run / Path(evidence["working_directory"]).name
    plan = next(json.loads(turn["text"]) for turn in reversed(state["dialogue"])
                if turn["role"] == "scientist" and turn["text"].startswith("{"))
    metrics = evidence["verification"]["metrics"]
    config = evidence["verification"]["best_config"]
    comparisons = json.loads((attempt / "results.json").read_text())
    calls = [json.loads(line) for line in (run / "model_calls.jsonl").read_text().splitlines()]
    width = A4[0] - 88
    styles = {
        "title": ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=22, leading=27, spaceAfter=12),
        "heading": ParagraphStyle("heading", fontName="Helvetica-Bold", fontSize=12, leading=16, spaceBefore=9, spaceAfter=6),
        "body": ParagraphStyle("body", fontName="Helvetica", fontSize=10, leading=14, spaceAfter=8),
        "small": ParagraphStyle("small", fontName="Helvetica", fontSize=8, leading=10, spaceAfter=5),
        "code": ParagraphStyle("code", fontName="Courier", fontSize=9, leading=13, spaceAfter=8),
    }
    story = []

    def paragraph(text, style="body"):
        story.append(Paragraph(text, styles[style]))

    def table(rows, widths):
        cells = [[Paragraph(escape(str(cell)), styles["small"]) for cell in row] for row in rows]
        item = Table(cells, colWidths=widths, repeatRows=1)
        item.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#d7e5ed")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f1f5f8"), colors.white]),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.HexColor("#19374a")),
        ]))
        story.append(item)
        story.append(Spacer(1, 8))

    paragraph("Task 6: Agentic System", "title")
    paragraph("Four-role LangGraph workflow for the Silverbox oscillator", "heading")
    paragraph(
        "The user speaks to an orchestrator. It calls a local language model to decide "
        "which specialist receives the next turn. The specialists append role-tagged "
        "messages to a shared dialogue and return control to the orchestrator. The "
        "system was validated by asking it to solve the Silverbox modeling problem."
    )
    table([
        ["Role", "Responsibility", "Coding tool"],
        ["Orchestrator", "Model-selected next role, instruction, and one-sentence reason", "No"],
        ["Scientist", "Differential equation, network, training, free-run test, ordered coding plan", "No"],
        ["General", "Conceptual explanations beside the fit", "No"],
        ["Executor", "Implement the scientist plan; write/run Python and return process output", "Yes"],
    ], [80, width - 140, 60])
    paragraph("Shared state and graph", "heading")
    paragraph(
        "StateGraph stores the dialogue, a decision counter, and the latest routing "
        "decision. Each turn contains role and text. The dialogue reducer appends new "
        "messages; it does not discard previous turns. The graph begins at orchestrator, "
        "has conditional edges to scientist/general/executor/end, and returns specialists "
        "to orchestrator. A rejected finish returns to another model decision."
    )
    paragraph("Execution and stop rules", "heading")
    paragraph(
        "Only the executor node has write_and_run_python. Its model supplies structured "
        "filename/code arguments, which the node passes to the tool. The tool creates a new "
        "attempt directory, writes a Python file, and runs it in a subprocess. It returns "
        "exit code, stdout, stderr, timeout status, and elapsed seconds. At 300 seconds "
        "the process group is terminated. A later executor turn can repair the script "
        "while the scientific plan remains unchanged."
    )
    paragraph(
        "The graph stops on an accepted finish or after eight model routing decisions. "
        "Successful finish requires actual executor stdout and checkpoints independently "
        "verified against the planned study. Budget exhaustion is recorded separately "
        "from success. Working-directory isolation organizes generated code; it is not "
        "an operating-system security sandbox."
    )
    paragraph(f"Models: {escape(summary['model'])} (routing/theory/explanation), "
              f"{escape(summary.get('executor_model', summary['model']))} (executor); "
              "local Ollama; temperature 0; context 12,288 tokens.", "small")

    story.append(PageBreak())
    paragraph("Recorded process and plan", "title")
    paragraph(
        f"Run {escape(run.name)} completed with {summary['decision_count']} orchestrator "
        f"decisions in {summary['total_seconds']:.1f} seconds. The executor script took "
        f"{evidence['duration_seconds']:.1f} seconds and exited with code {evidence['exit_code']}. "
        "These are measured wall-clock times, including local model calls in the total."
    )
    if summary.get("resumptions"):
        paragraph(
            "The initial executor used qwen2.5:7b and produced an import error. The "
            "interrupted run resumed at its already-selected executor repair using "
            "qwen2.5-coder:7b. The scientist plan and decision count were retained. "
            "Total time adds active run sessions and excludes the pause/model download.", "small"
        )
    routing = [["Decision", "Next role", "Instruction and reason"]]
    for count, turn in enumerate([t for t in state["dialogue"] if t["role"] == "orchestrator"], 1):
        content = turn["text"].split("\nRuntime check:")[0]
        decision = json.loads(content)
        routing.append([count, decision["next_role"], decision["instruction"] + " Reason: " + decision["reason"]])
    table(routing, [50, 65, width - 115])
    story.append(PageBreak())
    paragraph("Scientific study", "title")
    paragraph("Scientist's formulation", "heading")
    paragraph("y'' + a y' + b y + c y<super>3</super> = g u; &nbsp; y' = v, &nbsp; v' = alpha_theta(y,v,u)")
    paragraph(escape(plan["network_and_scaling"]))
    paragraph(escape(plan["training_method"]))
    paragraph(escape(plan["free_run_test"]))
    paragraph("Actual four-fit comparison", "heading")
    rows = [["Hidden widths", "Optimizer / lr", "Updates / window / batch", "Train mV", "Test mV"]]
    for row in comparisons:
        c = row["config"]
        rows.append([str(c["hidden_sizes"]), f"{c['optimizer']} / {c['learning_rate']}",
            f"{c['steps']} / {c['rollout_length']} / {c['batch_size']}",
            f"{1000*row['train_rmse']:.3f}", f"{1000*row['test_rmse']:.3f}"])
    table(rows, [80, 100, 155, 80, width - 415])

    story.append(PageBreak())
    paragraph("Executed result", "title")
    paragraph(
        f"The selected hidden widths are {escape(str(config['hidden_sizes']))}; "
        f"{config['optimizer']}, learning rate {config['learning_rate']}, "
        f"{config['steps']} updates, {config['rollout_length']}-sample windows, "
        f"batch size {config['batch_size']}. All configurations were trained from scratch. "
        "The executor wrote a new script using the supplied, tested neural RK4/training "
        "primitives and the study_utils.run_study artifact helper. The generated driver "
        "supplied the scientist's exact configurations and saved fresh weights."
    )
    printed = "<br/>".join(escape(line) for line in evidence["stdout"].splitlines()
                            if line.startswith(("TRAIN_RMSE_V", "TEST_RMSE_V", "ZERO_TEST_RMSE_V")))
    paragraph(printed, "code")
    story.append(Image(str(attempt / "test_predictions.png"), width=width, height=width * 0.75))
    paragraph(
        f"The plot shows the single continuous free run's test output. Test RMSE is "
        f"{1000*metrics['test_rmse']:.3f} mV; the zero baseline is "
        f"{1000*metrics['zero_test_rmse']:.3f} mV. The lower panel is signed predicted "
        "minus measured output. Training RMSE is calculated from the same full-record "
        "simulation. Measured test y is used only for scoring and plotting. Full-excerpt "
        "mean centering follows the supplied protocol; network RMS scales use training data."
    )
    paragraph(
        "Selection uses test RMSE, so that record also serves for model selection. "
        "The independently reloaded checkpoints reproduce the reported scores; the "
        "saved trace also matches the recomputed trajectory."
    )
    general_text = "\n".join(t["text"] for t in state["dialogue"] if t["role"] == "general")
    if "test set of 4096" in general_text:
        paragraph(
            "Dialogue review: the recorded general-agent explanation mistakenly called "
            "the full 4,096-sample record the test set and was uncertain about voltage "
            "units. The verified computation scores 1,024 test samples in volts. The "
            "transcript preserves that reply; this report states the checked interpretation.", "small"
        )

    story.append(PageBreak())
    paragraph("Reproducibility and evidence", "title")
    paragraph("From the repository root:", "heading")
    paragraph("The public repository contains source code. Place the supplied benchmark "
              "file SNLS80mV.mat in silverbox/data/ before running. The recorded report "
              "and raw validation artifacts are retained in the local project.", "small")
    paragraph("python3.12 -m venv .venv<br/>.venv/bin/python -m pip install -r requirements.txt<br/>"
              "ollama pull qwen2.5:7b<br/>ollama pull qwen2.5-coder:7b<br/>.venv/bin/python run.py", "code")
    paragraph("Ollama must be running locally. A fresh run may choose different routes or "
              "settings; the graph does not enforce the recorded speaker order. Temperature "
              "0 reduces variation but does not guarantee identical model responses.")
    paragraph("Recorded artifacts", "heading")
    table([
        ["Artifact", "Evidence"],
        ["dialogue.json / dialogue.md", "All role-tagged turns and actual execution results"],
        ["model_calls.jsonl", "Real prompts, model outputs, and per-call timings"],
        ["attempt directories", "Generated scripts, stdout, stderr, exit codes, and repair attempts"],
        ["results.json / *.pt", "Four configurations and fresh per-run/best checkpoints"],
        ["predictions.csv / test_predictions.png", "Full measured/predicted trace and test comparison"],
        ["summary.json", "Completion status, eight-decision accounting, and verified measurements"],
    ], [180, width - 180])
    paragraph("Verification", "heading")
    paragraph(
        "Focused tests cover traceback capture, timeout termination, filename containment, "
        "rejection of invented or duplicate scores, refusal to finish without executor "
        "evidence, and stopping after decision eight. End-to-end validation additionally "
        "reloads all four new checkpoints, verifies their settings against the scientist "
        "plan, checks selection of the lowest test score, and recomputes the full trace."
    )
    paragraph(".venv/bin/python -m unittest discover -s tests -v", "code")
    paragraph(f"Real model calls recorded: {len(calls)}. Full implementation steps are "
              "preserved in the scientist turn of the dialogue.")
    url = escape(args.repository_url, quote=True)
    paragraph(f"Repository: <link href='{url}' color='#2563eb'>{url}</link>", "small")
    paragraph("Sources: Homework 1, Task 6, pages 6-7; Silverbox benchmark record. "
              "LangGraph graph-api documentation and Ollama structured-output/tool-calling documentation.", "small")
    output = PROJECT / "output/pdf"
    output.mkdir(parents=True, exist_ok=True)

    def footer(canvas, document):
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#536675"))
        canvas.drawString(44, 24, "Task 6 | Silverbox Agentic System")
        canvas.drawRightString(A4[0]-44, 24, str(document.page))

    path = output / "task6_report.pdf"
    SimpleDocTemplate(str(path), pagesize=A4, leftMargin=44, rightMargin=44,
        topMargin=38, bottomMargin=42, title="Task 6: Silverbox Agentic System").build(
            story, onFirstPage=footer, onLaterPages=footer)
    print("Report saved:", path)


if __name__ == "__main__":
    main()
