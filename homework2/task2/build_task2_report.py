"""Create Task 2 and combined PDFs using actual saved agent/tool evidence."""

import argparse
from html import escape
import json
from pathlib import Path
import shutil
import subprocess

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from agents import verified_results

PROJECT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--repository-url", required=True)
    parser.add_argument("--recording", type=Path, help="Actual screen recording, if already available")
    args = parser.parse_args()
    if not args.run_name or not all(c.isalnum() or c in "_-" for c in args.run_name):
        parser.error("Use a simple run-folder basename")
    root = PROJECT / "output/runs" / args.run_name
    state = json.loads((root / "dialogue.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    events = json.loads((root / "events.json").read_text())
    results = verified_results(state["dialogue"])
    if summary["status"] != "completed" or len(results) < 2:
        raise ValueError("The report requires a completed real run with at least two tool results")
    # Model claims are insufficient: require matching executor calls in the log.
    calls = [event["data"] for event in events if event["kind"] == "tool_call"]
    for result in results:
        if not any(call.get("role") == "executor" and call.get("arguments") == result["settings"] for call in calls):
            raise ValueError("Missing actual executor tool-call evidence")
        for path in result["camera_images"].values():
            if not Path(path).is_file():
                raise ValueError("Missing actual camera image: " + path)
    if args.recording and not args.recording.is_file():
        raise ValueError("The supplied recording does not exist")

    out = PROJECT / "output/pdf"
    out.mkdir(parents=True, exist_ok=True)
    width = A4[0] - 88
    styles = {
        "title": ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=21, leading=26, spaceAfter=12),
        "heading": ParagraphStyle("heading", fontName="Helvetica-Bold", fontSize=12, leading=16, spaceBefore=10, spaceAfter=6),
        "body": ParagraphStyle("body", fontName="Helvetica", fontSize=9.5, leading=13, spaceAfter=7),
        "small": ParagraphStyle("small", fontName="Helvetica", fontSize=8, leading=10.5, spaceAfter=6),
    }
    story, markdown = [], []

    def text(value, style="body"):
        # Treat model strings as text rather than executable ReportLab markup.
        story.append(Paragraph(escape(str(value)).replace("\n", "<br/>"), styles[style]))
        markdown.extend([str(value), ""])

    def heading(value):
        text(value, "heading")

    def table(rows, widths):
        cells = [[Paragraph(escape(str(cell)), styles["small"]) for cell in row] for row in rows]
        item = Table(cells, colWidths=widths, repeatRows=1)
        item.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dce9f0")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f3f6f8"), colors.white]),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(item)
        markdown.extend(["| " + " | ".join(map(str, row)) + " |" for row in rows] + [""])

    text("Homework 2 - Task 2", "title")
    heading("Autonomous focusing on a simulated optical bench")
    text("The user gives a focusing goal. Four agents retain the full shared dialogue: "
         "the orchestrator selects each destination, the scientist chooses experiment "
         "settings, the general agent explains results, and only the executor has the "
         "focus_beam tool. A validated scientist plan is required before execution. "
         "Finish requires at least two actual tool results; prose-only scores cannot finish the graph.")
    text("The simulated optical system is a 32 x 32 binary-amplitude DMD, a fixed random "
         "complex transmission matrix, and a 32 x 32 camera: I = |T E_in|^2. "
         "A fixed Gaussian beam illuminates the DMD; macropixels gate amplitude without "
         "changing phase. Every exposure has fresh Poisson noise (400 photons per "
         "normalized intensity unit) plus Gaussian read noise (standard deviation 0.005). "
         "Both experiments use the same medium, transmission seed 17, and beam waist fraction 0.38.")
    text("Enhancement = target mean intensity / background mean intensity. The square "
         "target is compared with a nearby annulus split into angular regions, combined "
         "by pixel count. Starting from a random binary pattern, the tool randomly flips "
         "macropixels and remeasures both old and proposed patterns. It accepts greater "
         "noisy enhancement. The transmission matrix is used only for forward simulation. "
         "Final results are means of fresh validation exposures; SEM describes shot noise, "
         "not variation between optimization runs or different scattering media.")
    heading("All agent-selected experiment settings")
    labels = {
        "macropixel_size": "Macropixel side (DMD pixels)", "focus_x": "Target center x",
        "focus_y": "Target center y", "focus_size": "Target square side (camera pixels)",
        "background_inner_radius": "Background inner radius", "background_outer_radius": "Background outer radius",
        "background_regions": "Number of background regions", "iterations": "Flip iterations",
        "flip_count": "Macropixels flipped per proposal", "frames_per_measurement": "Frames per search measurement",
        "validation_shots": "Fresh shots before / after", "seed": "Pattern / proposal seed",
    }
    table([["Setting"] + [f"Experiment {i}" for i in range(1, len(results) + 1)]] +
          [[label] + [result["settings"][key] for result in results] for key, label in labels.items()],
          [230] + [(width - 230) / len(results)] * len(results))
    text("These values are actual structured model outputs. No person supplied scientific "
         "settings during the run. Geometry/schema constraints bound the simulator's valid inputs.", "small")

    story.append(PageBreak())
    text("Actual measurements and camera images", "title")
    table([["Experiment", "Initial enhancement", "Final enhancement +/- SEM", "Camera shots", "Accepted proposals"]] +
          [[str(i), f"{result['initial']['enhancement']:.4f}",
            f"{result['final']['enhancement']:.4f} +/- {result['final']['enhancement_sem']:.4f}",
            str(result["camera_shots"]), str(result["accepted_flips"])]
           for i, result in enumerate(results, 1)], [55, 100, 160, 83, width - 398])
    for i, result in enumerate(results, 1):
        # Preserve the saved shared color scale; do not brighten the final image independently.
        heading(f"Experiment {i}: before and after feedback search")
        image = Image(result["camera_images"]["comparison"])
        image.drawHeight = width * image.imageHeight / image.imageWidth
        image.drawWidth = width
        story.append(image)
        s = result["settings"]
        text(f"Target: {s['focus_size']} x {s['focus_size']} pixels centered at ({s['focus_x']}, {s['focus_y']}); "
             "cyan marks the target; left is the initial pattern, right is the final pattern. Both panels share an intensity scale. "
             f"Final active DMD fraction: {result['final_active_fraction']:.3f}.", "small")
        markdown.extend(["![](../runs/" + args.run_name + "/" + Path(result["working_directory"]).name + "/before_after.png)", ""])
        # Two large camera comparisons fit one page; additional experiments use another page.
        if i % 2 == 0 and i < len(results):
            story.append(PageBreak())
    best = max(results, key=lambda result: result["final"]["enhancement"])
    text(f"The best returned enhancement was {best['final']['enhancement']:.6f}. "
         "This measures target contrast, not absolute optical power. Binary gating can "
         "reduce total throughput; residual speckle remains. The study varies settings "
         "together, so it does not isolate a causal macropixel-size effect.", "small")

    story.append(PageBreak())
    text("Agent evidence and reference comparison", "title")
    heading("Actual autonomous execution")
    text("Initial request:\n" + state["dialogue"][0]["text"], "small")
    decisions = [event["data"] for event in events if event["kind"] == "decision"]
    table([["Decision", "Selected role", "Model's reason"]] +
          [[str(item["number"]), item["next_role"], item["reason"]] for item in decisions],
          [50, 70, width - 120])
    text(f"Saved status: {summary['status']}; decisions: {state['decision_count']} of 8; "
         f"models: {summary['model']} and {summary['executor_model']}; run/thread: {args.run_name}. "
         "Local evidence includes model_calls.jsonl, dialogue.json, events.json, SQLite "
         "checkpoints, result.json, camera arrays, masks, and search history.", "small")
    text("All eight automated tests passed. A preceding real run exhausted its budget "
         "because the model repeatedly claimed a scientist plan already existed. "
         "Runtime plan/completion facts and blocked-action feedback were added; the "
         "successful run above retains model-selected routing and settings.", "small")
    text("In this run the general agent attached experiment 1's initial uncertainty "
         "to its final score and over-attributed differences to macropixel size. "
         "The report table reads the actual final tool statistics and does not make "
         "that causal claim. Explicitly labeled statistics and a clearer explanation "
         "prompt were subsequently added for future runs.", "small")
    heading("Comparison with the lecture and PhotonicsAI Lab")
    text("The supplied Lecture 2 deck (pages 18-19) shows Gaussian illumination, "
         "amplitude DMD control, a transmission matrix, and camera intensity, matching "
         "this simulation's components. Its demo page has no recorded final camera "
         "result or measured enhancement. The PhotonicsAI focusing guide's Figure 1 "
         "(page 2) schematically contrasts initial speckle with a focused output. "
         "Our camera measurements show increased target-to-background contrast with "
         "remaining speckle, qualitatively consistent with that goal. The guide uses "
         "a phase-only SLM, while this homework uses amplitude-only DMD gating; their "
         "controls and achievable enhancements are not directly interchangeable. "
         "The guide's measurement table is blank. An exact numerical comparison "
         "with the classroom demonstration is therefore unavailable.")
    reference_url = "https://photonicsai.com/wp-content/uploads/2026/08/Lab_work_2_focusing.pdf"
    story.append(Paragraph(f'<link href="{reference_url}">PhotonicsAI Lab: Lab work 2 - Focusing</link>', styles["small"]))
    heading("Screen recording and source")
    if args.recording:
        text("Actual recording supplied locally: " + str(args.recording.resolve()) +
             ". Attach this video or publish it as required by the assignment; its "
             "existence alone does not verify that every required scene is visible.", "small")
    else:
        text("SCREEN RECORDING PENDING. This PDF is a draft until a real video shows "
             "the agent deciding, the executor calling focus_beam, and the experiment. "
             "The user-operated start_recording_demo.py launcher supports recording "
             "a new actual run; saved plots or a rendered animation are not a screen recording.", "small")
    text("Task 2 source: " + args.repository_url, "small")
    story.append(Paragraph(f'<link href="{escape(args.repository_url, quote=True)}">Open Task 2 source on GitHub</link>', styles["small"]))

    def footer(canvas, document):
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#536475"))
        canvas.drawString(44, 24, "Homework 2 | Task 2 | Actual simulated-bench evidence")
        canvas.drawRightString(A4[0] - 44, 24, str(document.page))

    task2_pdf = out / "homework2_task2_report.pdf"
    SimpleDocTemplate(str(task2_pdf), pagesize=A4, leftMargin=44, rightMargin=44,
                      topMargin=40, bottomMargin=40).build(story, onFirstPage=footer, onLaterPages=footer)
    (PROJECT / "output/task2_report.md").write_text("\n".join(markdown))
    task1_pdf = PROJECT.parent / "task1/output/pdf/homework2_task1_report.pdf"
    merger = shutil.which("pdfunite")
    if not merger:
        raise RuntimeError("Install Poppler (brew install poppler) for pdfunite")
    combined_dir = PROJECT.parent / "output/pdf"
    combined_dir.mkdir(parents=True, exist_ok=True)
    combined = combined_dir / "homework2_report.pdf"
    subprocess.run([merger, str(task1_pdf), str(task2_pdf), str(combined)], check=True)
    print(task2_pdf)
    print(combined)


if __name__ == "__main__":
    main()
