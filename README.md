# agentic-reliability-research

Research project on a learned, turn-level verifier for an LLM agentic system, evaluated on
TextWorldExpress. The verifier is trained purely from real environment outcomes (no hand-written
rules) and evaluated as an online failure detector and as a real-time decision-quality signal
(Best-of-N action selection). See `reports/` for the full build-by-build experiment log.

Two independently-trained verifiers are compared throughout:

- **`verifier_mc`** — trained via per-step Monte-Carlo return regression on real environment
  reward.
- **`verifier_dpo`** — trained via a Bradley-Terry/DPO-style preference objective on real
  per-episode won/lost outcomes only, with per-turn scores read off as a byproduct.

An earlier phase of this project also trained a learned PPO coordinator on top of the verifier's
signal. That line was tested across six independent training runs, diagnosed precisely (turn-level
verifier reward, however constructed, could not train a policy to reliably beat a strong
single-agent-loop baseline on this environment), and removed from the current codebase; see
`reports/04_coordinator_replan.md` through `reports/07_backtrack.md` for the fixed-coordinator
baselines that motivated it. The current methodology and all reported results center on the
verifier itself.

## Setup

### 1. Python version

The venv must be **Python 3.12 or lower** (a legacy `textworld` dependency breaks on 3.13's
`PEP 667` changes to `locals()` semantics).

```bash
uv venv --python 3.12 .venv
```

### 2. Install dependencies

```bash
uv pip install --python .venv/bin/python -r requirements.txt
```

### 3. Set up API keys

Copy `.env.example` to `.env` and fill in what you need:

```bash
cp .env.example .env
```

- `ANTHROPIC_API_KEY` — needed for Claude-based agents (used in early baseline builds).
- `OPENAI_API_KEY` — needed for GPT-based baseline comparisons.
- `LANGSMITH_*` — optional, only needed if LangSmith tracing is enabled.
- Local, open-source models (the default for all current verifier/Best-of-N work,
  `hf:Qwen/Qwen2.5-3B-Instruct`) need no API key — they run locally via `transformers`.

### 4. GPU access for verifier training

Both verifiers were trained on a Lightning AI Studio GPU (full fine-tune, `bitsandbytes` 8-bit
AdamW). Training and evaluation can be run without a GPU, but expect it to be very slow — most
scripts in this repo were run and validated on CUDA.

## Repository layout

| directory | contents |
|---|---|
| `envs/` | `TextWorldExpressEnvWrapper` — the only environment used for every reported result. |
| `agents/` | `thinker.py` (plans once per episode), `worker.py` (acts each turn). |
| `rollout/` | episode runner (`runner.py`), rollout collection (`collect.py`), and labeling scripts (`label_twx.py` — the Monte-Carlo labeler used for `verifier_mc`'s reported checkpoint). |
| `verifier/` | `verifier_mc`'s model, training, and inference code, plus the frozen/untrained LLM judge baseline (`frozen_llm.py`). |
| `verifier_dpo/` | `verifier_dpo`'s model, training, and inference code. |
| `scripts/` | evaluation entry points — AUC scoring, online failure detection, Best-of-N, difficulty calibration, dataset consolidation. |
| `reports/` | the numbered build-by-build experiment log (01 through 12), written contemporaneously as each experiment ran. |
| `checkpoints/`, `data/` | trained model weights and collected/labeled episode data (gitignored — see below). |

Some early-project files (the single-agent baseline agent, and a few labeling/collection scripts
representing closed, superseded experiments) are kept on disk for local replication but are not
tracked in this repository; see `.gitignore` for the full list and the reasoning next to each.

## Data and checkpoints

`data/` and `checkpoints/` are gitignored — they contain large, locally-generated artifacts
(collected rollouts, labeled training sets, and multi-GB model checkpoints) that don't belong in
version control. To regenerate them:

1. **Collect rollouts**: `python -m rollout.collect --n <N> --out data/rollouts/<name> --model hf:Qwen/Qwen2.5-3B-Instruct`
2. **Label with real per-step reward**: `python -m rollout.label_twx --in data/rollouts/<name>/<game> --out data/labeled/<name>/<game>` (once per game), then `python -m scripts.consolidate_labeled --in data/labeled/<name> --out data/labeled/<name>_combined --games coin,simonsays,peckingorder,mapreader` to flatten per-game directories into one training set.
3. **Train a verifier**: `python -m verifier.train --data data/labeled/<name>_combined --out checkpoints/<name> --full-finetune --batch-size 16 --epochs 3 --lr 1e-5 --unbounded-q-value` (see `verifier_dpo/train.py` for the preference-trained variant's equivalent).
4. **Evaluate**: `scripts/score_trained_verifier.py` (AUC), `scripts/failure_detection_eval.py` (online failure detection), `scripts/best_of_n_eval.py` (real-time decision quality).

## Results

The full, numbered experiment log lives in `reports/` (baselines, heuristic coordination,
verifier development, and difficulty calibration), written up as each experiment ran. Cross-cutting
comparisons across both verifiers — per-game and pooled AUC, online failure detection
(precision/recall/F1/lead-time), and Best-of-N win-rate lift, all computed on the same held-out
`eval_ood` episodes — are the current headline results; regenerate them with the scripts in
`scripts/` listed above.
