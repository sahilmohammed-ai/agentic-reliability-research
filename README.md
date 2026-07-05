# agentic-reliability-research

Research project on a learned reliability layer (turn-level agentic verifier + RL coordinator)
for an LLM multi-agent system. See `.info/CLAUDE.MD` for the full architecture and research plan.

Current stage: rollout collection with a fixed coordinator, comparing frozen thinker/worker models
(see `reports/`). No verifier or learned coordinator yet.

## Setup

### 1. Python version

The venv must be **Python 3.12 or lower**. ALFWorld's textworld backend breaks on Python 3.13
(PEP 667 changes `locals()` semantics that textworld's grammar engine relies on).

```bash
uv venv --python 3.12 .venv
```

### 2. Install dependencies

```bash
uv pip install --python .venv/bin/python -r requirements.txt
```

### 3. Download ALFWorld task data

```bash
.venv/bin/alfworld-download
```

This downloads to `~/.cache/alfworld` by default. Note the path, you need it as an environment
variable every time you run the pipeline.

### 4. Set up API keys

Copy `.env.example` to `.env` and fill in:

```bash
cp .env.example .env
```

- `ANTHROPIC_API_KEY` — required for Claude-based agents (Haiku, Sonnet, Opus)
- `LANGSMITH_*` — optional, only needed if tracing is active

### 5. (Optional) Ollama for local open-source models

If you want to run open-source models locally (e.g. Qwen2.5-3B) instead of the Anthropic API:

```bash
# install ollama: https://ollama.com/download
ollama pull qwen2.5:3b-instruct
```

Make sure the Ollama server is running (`ollama serve`, or it starts automatically on most
installs) before collecting rollouts with an Ollama model.

## Running the pipeline

### Sanity check the environment

```bash
ALFWORLD_DATA=~/.cache/alfworld .venv/bin/python scripts/alfworld_smoke_test.py
```

Runs one episode with a trivial policy (always picks the first admissible command). Confirms
ALFWorld is installed correctly and returns a real win/loss signal. Does not call any LLM.

### Collect rollouts

```bash
ALFWORLD_DATA=~/.cache/alfworld .venv/bin/python -m rollout.collect \
  --n 50 \
  --out data/rollouts/build_N \
  --model <model>
```

- `--model` accepts either an Anthropic model name (e.g. `claude-haiku-4-5-20251001`,
  `claude-opus-4-8`) or an Ollama model tag (e.g. `qwen2.5:3b-instruct`). Anything with a colon
  is routed to Ollama, everything else goes to the Anthropic API. Defaults to Haiku 4.5 if omitted.
- Output is one JSON trajectory file per episode in `data/rollouts/build_N/`.
- Each performance test (a different model, same fixed coordinator) should get its own
  `build_N` directory and a matching `reports/build_N.md` write-up. See existing reports for
  the format: overview, environment, agent setup, metrics. No tables.

### Inspect a trajectory

```bash
cat data/rollouts/build_N/train_0000.json | python -m json.tool | less
```

Each `worker` turn has `obs_before`, `action`, `obs_after`, `env_reward` (0 except on the final
step of a won episode), and `done`. The `won` field at the top of the file is the episode-level
outcome used later for TD/GAE labeling.

## Repo layout

```
agents/       frozen role agents (thinker, worker) + shared llm.py backend router
envs/         gym-style ALFWorld wrapper
rollout/      schemas, episode runner (fixed coordinator), collection CLI
scripts/      standalone environment sanity check
configs/      ALFWorld config
data/         collected rollouts, one build_N subdirectory per performance test (gitignored)
reports/      one markdown report per build, written after each rollout collection run
.info/        CLAUDE.MD, the full project knowledge base and research plan
```
