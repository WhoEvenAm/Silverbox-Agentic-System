"""Build the Homework 2 Task 1 section from two actual interactive runs."""

import argparse
from html import escape
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from agents import verified_result
from persistence import load_run
from schemas import Decision

PROJECT = Path(__file__).resolve().parent
ACCEPT = {"yes", "y", "approve", "accept"}


def human_events(state):
    """Pair each actual reply with the model question that caused its interrupt."""
    events = []
    turns = state["dialogue"]
    for index, turn in enumerate(turns):
        if turn["role"] != "orchestrator":
            continue
        decision = Decision.model_validate_json(turn["text"].split("\nRuntime check:")[0])
        if decision.next_role != "human":
            continue
        for reply_index in range(index + 1, len(turns)):
            if turns[reply_index]["role"] == "orchestrator":
                break  # A later question is a different event.
            if turns[reply_index]["role"] == "human":
                events.append({"request": decision.human_request.model_dump(),
                               "reply": turns[reply_index]["text"], "index": reply_index})
                break
    return events


def collect_evidence(clarification, approval):
    """Reject missing demonstrations rather than substituting sample answers."""
    clarified = next((event for event in human_events(clarification)
                      if event["request"]["kind"] == "clarification"), None)
    if clarified is None:
        raise ValueError("The clarification run needs an actual human reply")
    following = clarification["dialogue"][clarified["index"] + 1:]
    if not following or following[0]["role"] != "orchestrator":
        raise ValueError("The clarified reply must be followed by a model routing decision")
    next_decision = Decision.model_validate_json(following[0]["text"].split("\nRuntime check:")[0])
    explanation = next((turn for turn in following if turn["role"] in
                        {"general", "scientist", "executor"}), None)

    events = [event for event in human_events(approval) if event["request"]["kind"] == "approval"]
    for refused in events:
        if refused["reply"].strip().casefold() in ACCEPT:
            continue
        role = refused["request"]["proposed_action"]["next_role"]
        accepted = next((event for event in events if event["index"] > refused["index"]
                         and event["reply"].strip().casefold() in ACCEPT
                         and event["request"]["proposed_action"]["next_role"] == role), None)
        if accepted is not None:
            break
    else:
        raise ValueError("The hard-decision run needs a refusal followed by acceptance")
    turns = approval["dialogue"]
    notice = turns[refused["index"] + 1]
    if notice["role"] != "system" or "step did not run" not in notice["text"]:
        raise ValueError("Missing runtime evidence that the refused step did not run")
    # The refused action must not appear before the next orchestrator decision.
    for turn in turns[refused["index"] + 1:]:
        if turn["role"] == "orchestrator":
            break
        if turn["role"] == role:
            raise ValueError("The refused action appears to have executed")
    executed = next((turn for turn in turns[accepted["index"] + 1:] if turn["role"] == role), None)
    if executed is None:
        raise ValueError("The accepted action has no following execution/finish message")
    evidence = verified_result(turns)
    if evidence is None:
        raise ValueError("The study needs independently verified execution evidence")
    return clarified, next_decision, explanation, refused, notice, accepted, executed, evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clarification-run", required=True)
    parser.add_argument("--approval-run", required=True)
    parser.add_argument("--repository-url", required=True)
    parser.add_argument("--test-count", required=True, type=int)
    args = parser.parse_args()
    # Read local evidence only; the source repository does not include these runs.
    root = PROJECT / "output/runs"
    clarification, _ = load_run(root / args.clarification_run)
    approval, summary = load_run(root / args.approval_run)
    clarified, routed, explained, refused, notice, accepted, executed, evidence = collect_evidence(
        clarification, approval)
    out = PROJECT / "output/pdf"
    out.mkdir(parents=True, exist_ok=True)
    width = A4[0] - 88
    styles = {
        "title": ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=21, leading=26, spaceAfter=15),
        "heading": ParagraphStyle("heading", fontName="Helvetica-Bold", fontSize=12, leading=16, spaceBefore=12, spaceAfter=7),
        "body": ParagraphStyle("body", fontName="Helvetica", fontSize=10, leading=14, spaceAfter=8),
        "small": ParagraphStyle("small", fontName="Helvetica", fontSize=8, leading=11, spaceAfter=7),
    }
    story, markdown = [], []

    def text(value, style="body"):
        # Escape all model/human text before it enters ReportLab's markup parser.
        # Render simple inline math delimiters as text; retain the original in Markdown.
        displayed = value.replace("\\(", "").replace("\\)", "")
        story.append(Paragraph(escape(displayed).replace("\n", "<br/>"), styles[style]))
        markdown.extend([value, ""])

    def heading(value):
        text(value, "heading")

    text("Homework 2 - Task 1", "title")
    heading("Human-in-the-loop Silverbox system")
    text("The Homework 1 system retains four model agents and the complete shared dialogue. "
         "The orchestrator reads that dialogue and makes a model call to choose scientist, "
         "general, executor, human, or finish. It makes at most eight validated routing "
         "decisions; invalid structured replies are retried at most twice and grant no permission.")
    rows = [["Role", "Responsibility"], ["Orchestrator", "Choose the next destination, instruction, question, and reason"],
            ["Scientist", "Specify the equations, architectures, fitting settings, and scientific plan"],
            ["General", "Explain concepts and what the error measures"],
            ["Executor", "Generate and run Python; attach independently checked execution evidence"]]
    cells = [[Paragraph(escape(cell), styles["small"]) for cell in row] for row in rows]
    table = Table(cells, colWidths=[90, width - 90])
    table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dce9f0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f3f6f8"), colors.white]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7)]))
    story.extend([table, Spacer(1, 10)])
    markdown.extend(["| Role | Responsibility |", "|---|---|", *[f"| {a} | {b} |" for a, b in rows[1:]], ""])
    heading("Interrupt, persistence, and approval")
    text("The human control node calls LangGraph interrupt() with a model-selected question. "
         "The graph is compiled with SqliteSaver and called with the run folder name as "
         "thread_id. Command(resume=reply) resumes that same thread. The actual reply is "
         "appended with role human; an EOF leaves the question paused for a later restart.")
    text("Clarification grants no execution permission. An approval describes one executor "
         "or finish action. Explicit yes/y/approve/accept authorizes that action once; other "
         "replies skip it and return control to the orchestrator. Runtime guards check the "
         "exact destination and instruction. Execution approvals are consumed on disk before "
         "the tool is invoked, preventing replay after a crash. Finish also requires verified evidence.")
    heading("Validation and source")
    text(f"All {args.test_count} automated tests passed, including real subprocess cleanup, "
         "refusal without execution, single-use acceptance, action matching, SQLite reopen, "
         "and runner restart without duplicate replies. The demonstrations below use actual "
         "Ollama calls and the person's submitted replies, not test doubles.")
    text(f"Models: {summary['model']} for routing, science, and explanation; "
         f"{summary.get('executor_model', summary['model'])} for generated code. "
         "Live validation exposed inconsistent model JSON; schema constraints and "
         "bounded corrective retries were added while preserving the saved threads.", "small")
    text("Source: " + args.repository_url, "small")
    story.append(Paragraph(f'<link href="{escape(args.repository_url, quote=True)}">Open Task 1 source on GitHub</link>', styles["small"]))

    story.append(PageBreak())
    text("Run (a) - ambiguous request", "title")
    text("Run/thread: " + args.clarification_run, "small")
    heading("Initial request")
    text(clarification["dialogue"][0]["text"].strip())
    heading("Question surfaced by the graph")
    text(clarified["request"]["question"])
    heading("Actual human reply")
    text(clarified["reply"])
    heading("Next dialogue message: orchestrator")
    text(f"Destination: {routed.next_role}\nInstruction: {routed.instruction}\nReason: {routed.reason}")
    if explained is not None:
        heading("Following specialist message: " + explained["role"])
        text(explained["text"])
    text("The routing decision after the reply follows the selected reading: test free-run RMSE. "
         "Any illustrative number in the explanation is an example, not a measurement from this run.", "small")
    text("Free-run evaluation carries the simulated state through all 4096 samples without "
         "resetting at the train/test boundary. Known test inputs drive the simulation; "
         "measured test outputs only score the final 1024 samples. This run remains saved "
         "at a later question after the required clarification and following messages.", "small")

    story.append(PageBreak())
    text("Run (b) - refuse, then accept", "title")
    text("Run/thread: " + args.approval_run, "small")
    text("Request: plan, implement, and run the Silverbox modeling study. The scientist "
         "selected two architectures and two settings for four fresh fits. The person "
         "approved each execution that occurred; the hard decision below was refused first.")
    heading("First question and proposed action")
    text(refused["request"]["question"])
    text(refused["request"]["proposed_action"]["description"], "small")
    text("Actual reply: " + refused["reply"])
    heading("Runtime message proving the refused step did not run")
    text(notice["text"])
    text("The model subsequently sent the explanation request to the scientist, whose "
         "structured response repeated the existing plan, and then asked for finish "
         "approval again. No finish message or additional coding-tool attempt occurred "
         "during the refusal; the local refusal_audit.json records that check.", "small")
    heading("Second question and proposed action")
    text(accepted["request"]["question"])
    text(accepted["request"]["proposed_action"]["description"], "small")
    text("Actual reply: " + accepted["reply"])
    heading("Message after acceptance")
    if executed["role"] == "finish":
        text(executed["text"])
    else:
        text("The executor invoked the coding tool and attached actual process output and verified results.")
    metrics = evidence["verification"]["metrics"]
    text(f"Verified test free-run RMSE: {metrics['test_rmse']:.8f} V. "
         f"Script exit code: {evidence['exit_code']}. "
         f"Orchestrator decisions used: {approval['decision_count']} of 8. "
         f"Saved run status: {summary['status']}.")
    text("Raw dialogue, model-call logs, SQLite checkpoints, and measurement artifacts remain local. "
         "This PDF is the Task 1 section; the assignment's final submission combines it with Task 2.", "small")

    def footer(canvas, document):
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#536475"))
        canvas.drawString(44, 24, "Homework 2 | Task 1 | Actual interactive evidence")
        canvas.drawRightString(A4[0] - 44, 24, str(document.page))

    pdf = out / "homework2_task1_report.pdf"
    SimpleDocTemplate(str(pdf), pagesize=A4, leftMargin=44, rightMargin=44,
                      topMargin=42, bottomMargin=42).build(story, onFirstPage=footer, onLaterPages=footer)
    (PROJECT / "output/task1_report.md").write_text("\n".join(markdown))
    print(pdf)


if __name__ == "__main__":
    main()
