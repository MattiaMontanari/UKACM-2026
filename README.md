# QuadAgent

A minimal Deep Agents exercise: an agent reads a JSON board description,
writes and runs a Gmsh Python program, and reports basic mesh quality. Each run
keeps its generated mesh, preview, and score under `workspace/`.

## Run

Install [uv](https://docs.astral.sh/uv/), then configure the model:

```bash
uv sync --locked
cp .env.example .env
```

Set `ZAI_API_KEY` in `.env`, then choose a board:

```bash
uv run python baseline.py holes
uv run python baseline.py rects
uv run python baseline.py slots
uv run python baseline.py dense
```

To inspect the agent trace in LangSmith, set its API key and change
`LANGSMITH_TRACING` to `true`.

## Check

```bash
uv run pytest -q
uv run ruff check .
```
