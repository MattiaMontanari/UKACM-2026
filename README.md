# QuadAgent — Part 1 baseline

QuadAgent is a small classroom example of a LangChain Deep Agent that writes
and runs a Gmsh Python program for one of four finite 2D board domains.

Part 1 deliberately stops at a Level-0 mesh. The agent can repair Python or
Gmsh execution failures, but it cannot revise a successful mesh in response to
quality measurements. `mesh_quality` is called once as the terminal tool.

## What the example teaches

[`baseline.py`](baseline.py) keeps the Deep Agents construction visible:

- `ChatOpenAI` connects GLM to the z.ai OpenAI-compatible endpoint;
- `LocalShellBackend` gives the agent a persistent workspace and Python;
- filesystem middleware exposes only `read_file`, `write_file`, and `execute`;
- todo and call-limit middleware bound the ReAct loop;
- one application tool measures the final mesh and ends the run.

There are no subagents, skills, memory, or custom agent loop in Part 1.

## Setup

Install [uv](https://docs.astral.sh/uv/), then run:

```bash
uv sync --locked
cp .env.example .env
```

Add your z.ai coding-plan key to `.env`. The configuration is:

```dotenv
ZAI_API_KEY=...
ZAI_BASE_URL=https://api.z.ai/api/coding/paas/v4
QUADAGENT_MODEL=glm-5.3
```

The project pins Python 3.12.8 and Deep Agents 0.7.14 for reproducible
lecture runs.

## Run the baseline

Choose one representative domain:

```bash
uv run python baseline.py holes
uv run python baseline.py rects
uv run python baseline.py slots
uv run python baseline.py dense
```

Each run creates `workspace/<board>-<timestamp>/` containing:

- `board.json` and the agent-written `mesh.py`;
- `mesh.msh`, `mesh.vtk`, and `mesh.png`;
- `quality.json`, `todos.json`, `transcript.json`, and `run.json`.

The fixed recipe is a uniform 2 mm mesh using Gmsh Frontal-Delaunay and
Blossom recombination. Quality gates are reported but are not Part 1 pass
criteria.

## Verification

Deterministic checks do not call an LLM:

```bash
uv run pytest -q
uv run ruff check .
```

The live gate always calls the configured GLM model. Use one board while
developing and all four before accepting the Part 1 milestone:

```bash
uv run python scripts/live_glm_test.py holes
uv run python scripts/live_glm_test.py holes rects slots dense
```

The validator checks real token usage, tool order and budgets, required
artifacts, positive scaled Jacobian, nonempty quad content, and mesh area.
Failed workspaces are preserved for inspection.

## Security boundary

`LocalShellBackend` executes unrestricted shell commands on the host. Its
filesystem paths are rooted in the session workspace and its child environment
does not inherit the API key, but shell commands can still access anything the
current user can access. Run this example only on a trusted teaching machine
with trusted inputs. It is not a sandbox or a production security boundary.

## Later lecture material

The previous experimental recipes, datasets, and reports are parked under
`lecture/part2-assets/`. They are inactive in Part 1. Part 2 will only be
started when explicitly requested.
