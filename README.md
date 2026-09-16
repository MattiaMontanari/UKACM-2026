# QuadAgent v3 — a minimal deep agent for 2D quad meshing (Gmsh + GLM)

Academic-lecture baseline: the **simplest possible deep agent** — a single
ReAct loop, no subagents, no frameworks — that meshes PCB-like boards with
**quad-dominant 2D meshes for explicit FEA** using Gmsh.

The tedious task it attacks: Gmsh cannot *automatically* produce structured
quads around holes and slots (the O-grid problem — see gmsh work item
[#1236](https://gitlab.onelab.info/gmsh/gmsh/-/work_items/1236)). Engineers
hand-build transfinite annuli per hole. This agent automates the
script → mesh → measure → improve loop. **This baseline targets "mostly
quads" via a playbook**; solving the O-grid rings is the next milestone.

## Architecture (one file each, ~700 lines total)

```
src/quadagent/
├── llm.py       GLM via the z.ai coding endpoint (OpenAI-compatible)
├── agent.py     the deep-agent loop: plan -> act (tools) -> observe -> ...
├── tools.py     write_todos | ls | read_file | write_file
│                execute_python | mesh_quality | read_reference
├── sandbox.py   subprocess jail: cwd-pinned, timeout + killpg, rlimits,
│                macOS Seatbelt (no network, no writes outside workspace)
├── quality.py   trusted host-side metrics: quad fraction, scaled Jacobian,
│                aspect ratio, angles, h_min -> dt_crit ~ h_min/c
├── prompts.py   system prompt + task template
└── cli.py       quadagent run | boards | selftest
references/      playbook.md (levels 0-2 + quality gates), gmsh_quads.md
boards/          PCB-like board JSONs (holes / cutouts / slots / header)
```

Deep-agent lineage (mirrors the `deepagents` library, minus subagents):

| deepagents concept   | here                                            |
|----------------------|-------------------------------------------------|
| planning middleware  | `write_todos` tool + `_todos.md`                |
| virtual filesystem   | `ls`/`read_file`/`write_file` jailed to session |
| backend/tools        | `execute_python` (sandboxed Gmsh), `mesh_quality` |
| skills               | `read_reference` (playbook + Gmsh cheat sheet)  |

## Setup

```bash
uv sync                       # creates .venv with gmsh, numpy, matplotlib, openai
cp .env.example .env          # then put your z.ai coding-plan key in it
uv run quadagent selftest     # offline check: sandbox -> gmsh -> vtk/png -> metrics
```

`.env` keys: `ZAI_API_KEY` (required for real runs), `ZAI_BASE_URL`
(default `https://api.z.ai/api/coding/paas/v4`), `QUADAGENT_MODEL`
(default `glm-5.3`). Endpoint per <https://docs.z.ai/devpack/quick-start>.

## Run

```bash
uv run quadagent boards                          # list the bundled boards
uv run quadagent run boards/train-holes-00.json  # baseline agent run
uv run quadagent run boards/train-slots-00.json --target-edge 1.5 --max-steps 40
```

Each run creates `workspace/<board>-<timestamp>/` containing `board.json`,
every sandboxed script (`_sandbox/run_*.py`), the deliverables
(`mesh.vtk`, `mesh.png`, `mesh.py`), `quality.json`, `_todos.md`, and a
replayable `transcript.jsonl`.

## What the agent is asked to do

1. `write_todos` a 3-6 item plan.
2. `read_reference` the playbook (Level 0: Frontal-Delaunay + Blossom
   recombination; Level 1: targeted size fields; Level 2: hand-built
   O-grids = stretch goal).
3. Write `mesh.py` (Gmsh OCC domain: rectangle minus holes/cutouts/slots/
   header), run it via `execute_python`, produce `mesh.msh` + `mesh.vtk`
   + `mesh.png` (triangles drawn red).
4. `mesh_quality` after each attempt; gates: quad fraction >= 0.90,
   scaled Jacobian >= 0.40, AR <= 5, min angle >= 35 deg.
5. Report and stop.

## Quality gates (explicit FEA)

| metric              | target | acceptable |
|---------------------|--------|------------|
| quad fraction       | >= 0.98| >= 0.90    |
| scaled Jacobian min | >= 0.60| >= 0.40    |
| max aspect ratio    | <= 3   | <= 5       |
| min interior angle  | >= 45° | >= 35°     |
| elements            | <= 10k | <= 20k     |

`dt_crit ~ h_min / c` (c = 3000 m/s FR-4 default) is reported as the
explicit time-step proxy.

## Sandbox threat model (lecture-grade, honest)

`execute_python` runs the agent's code in a subprocess with: workspace as
cwd, wall-clock timeout (process-group kill), POSIX rlimits, an env
allowlist (no secrets, no repo-path leaks), and — on macOS — a Seatbelt
profile denying network and writes outside the workspace. That stops
accidents and casual exfiltration; it is **not** a security boundary. For
untrusted settings use a container (see v2's Dockerfile) — swap
`sandbox.py`'s command construction.

## Recipe isolation (anti-cheating)

The certified whole-board recipes live in `recipes/` — **outside** the
installed package — and the sandbox runs a dedicated interpreter
(`.venv-sandbox`, gmsh/numpy/matplotlib only) that cannot `import
quadagent` at all. `verify_isolation()` fails closed if the interpreter
can reach the package. The agent only receives *primitives* (single-hole
O-grid quadrants, hole-in-box block pattern) via the playbook; whole-
domain composition is the agent's own work. `read_reference` serves an
explicit allowlist (playbook + gmsh cheat sheet); maintainer-only
algorithms live in `references/maintainer_level23.md`.

Setup after cloning:

```bash
uv sync
uv venv .venv-sandbox && uv pip install --python .venv-sandbox/bin/python gmsh numpy matplotlib
uv run pytest -q   # includes the isolation tests
```

Escape hatch (documented as cheating): `QUADAGENT_ALLOW_RECIPE_LEAK=1`.

## Tests

```bash
uv run pytest -q
```

Covers sandbox isolation/timeout, quality metrics on analytic meshes
(perfect quads), the path jail, and an offline end-to-end agent loop with
a scripted model (no network).

## Roadmap (next lecture steps)

1. Baseline (this repo): mostly-quads playbook agent. ✅
2. O-grid agent: transfinite annuli per hole, clearance-aware ring sizing
   (see `references/playbook.md` Level 2 and v2's oracle for the recipe).
3. Subagents: a mesher + a critic (v2 used the full `deepagents` stack
   with budgets and an immutable scorer).
