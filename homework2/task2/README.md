# Homework 2 Task 2: autonomous beam focusing

The four-role LangGraph agent chooses experiment settings and asks its executor
to call `focus_beam`. The tool simulates a binary-amplitude DMD, a fixed random
complex transmission matrix, and a noisy camera. No optical hardware is required.

## Run on this Mac

The Python libraries are already installed in this folder's `.venv`.
Start Ollama if it is not running, then use:

```sh
cd /Users/yzuss/Desktop/homework2/task2
.venv/bin/python run.py
```

The initial request contains the goal only. The scientist chooses all experiment
settings; the executor chooses an untried experiment from that plan and invokes
the tool. The orchestrator decides which role runs next, with at most eight
decisions. Finish requires at least two actual tool results. Runtime checks
report blocked actions to the model rather than selecting replacement routes.

Default local models are `qwen2.5:7b` and `qwen2.5-coder:7b`, served by Ollama at
`http://localhost:11434`. Model calls can take several minutes on a laptop.
Press Ctrl+C to save an interrupted run. Resume an unfinished graph with:

```sh
.venv/bin/python run.py --resume-run <run-folder-name>
```

Completed runs and exhausted decision budgets cannot resume. Each run has its
own SQLite checkpoints, full dialogue, model-call log, event stream, and results
under `output/runs/`. Failed runs remain visible as failed evidence.

## Record the real demonstration

```sh
.venv/bin/python start_recording_demo.py
```

This user-operated launcher opens a local browser view of actual agent events.
Use **Shift+Command+5** to start recording the view. Press Enter in the terminal
when recording is ready; this submits the initial focusing request. The agent
then chooses its settings autonomously. Record a decision, the executor's
`focus_beam` call with arguments, camera progress, and the final result table.
Stop recording before closing the viewer. A screen recording is a separate
artifact; the launcher does not manufacture or record a video itself.

## Install on another machine

Use Python 3.12 and install Ollama, then:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
ollama pull qwen2.5:7b
ollama pull qwen2.5-coder:7b
```

## What the measurement means

The simulator uses `I = abs(T @ E_in)**2`. A fixed Gaussian illumination
envelope and fixed input phase are gated by binary DMD macropixels. Every camera
exposure draws fresh Poisson photon noise and Gaussian read noise. A square
target region is compared with a nearby annulus split into angular background
regions. Enhancement is target mean intensity divided by pixel-weighted mean
background intensity.

The search randomly flips model-selected numbers of macropixels. It remeasures
both the retained and proposed patterns, accepting the proposal when its noisy
enhancement is greater. It never reads the transmission matrix to choose flips.
Fresh validation exposures measure the final pattern. The returned value is
their mean enhancement, accompanied by standard deviation and standard error
of the mean (SEM), rather than the largest optimization score.

`verified` checks consistency of the returned validation scores and mean. It
does not imply experimental calibration or a validated model of a real lab.
Changing the target region changes the objective, so compare such experiments
with that qualification. Repeat runs can differ because camera noise is present;
explicit seeds make a given simulation run reproducible.

## Source map and checks

| File | Purpose |
|---|---|
| `bench.py` | Optical propagation, regions, noise, feedback search |
| `schemas.py` | Validated model-selected settings, plans, and routing |
| `focusing_tool.py` | Executor tool and actual measurement artifacts |
| `agents.py`, `prompts.py` | Four model roles and full shared dialogue |
| `graph.py` | Routing guards and eight-decision budget |
| `run.py`, `persistence.py` | CLI, event log, SQLite checkpoints |
| `live_view.py` | Read-only loopback observer for the experiment |
| `start_recording_demo.py` | User-operated recording workflow |
| `coding_tool.py` | Original executor coding tool retained in its registry |
| `build_task2_report.py` | Reports from completed actual runs; combines both tasks |

```sh
.venv/bin/python -m pip check
.venv/bin/python -m unittest discover -s tests -v
```

Lecture-reference provenance and the limits of the available comparison are in
`references/REFERENCE_NOTES.md`. Local references and run artifacts are excluded
from source publication.

## Build the homework PDF locally

Once a real run has completed, use its folder name:

```sh
.venv/bin/python build_task2_report.py --run-name <completed-run-name> --repository-url <task2-github-url>
```

This requires the local Task 1 PDF at
`../task1/output/pdf/homework2_task1_report.pdf` and Poppler's `pdfunite` command
(already installed on this Mac). It writes a Task 2 section and a combined PDF
at `../output/pdf/homework2_report.pdf`. Until an actual recording is available,
the report marks the recording as pending. Supply its local path with
`--recording /path/to/video.mov`; review its contents and attach/publish it
separately as the assignment requires.
