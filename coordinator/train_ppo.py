"""
PPO training loop for the coordinator policy (coordinator/model.py), reward supplied by the
already-trained, already-validated verifier_v5 checkpoint acting as a FROZEN critic -- no new
critic is trained here (see .info/CLAUDE.md's coordinator-design decision: verifier_v5's q_value
IS the value estimate, and its own `advantage` field, q(t) - q(t-1), is already a baseline-
subtracted one-step advantage, so it's used directly as the PPO advantage rather than computing a
separate GAE over a second, untrained value function).

online PPO by construction: each iteration collects fresh episodes with the CURRENT policy
(rollout.runner.run_learned_coordinated_episode, greedy=False so log_probs are captured), then
updates on that batch, then discards it and collects again next iteration -- deliberately not
reusing the fixed build-12 dataset, since that data has no coordinator-action labels at all (it
was collected zero-coordination). see reports/13_ppo_vs_dpo_decision.md for why PPO was chosen
over DPO/GRPO for this project (agent_prm's DPO precedent needs branching multi-candidate rollouts
we don't have; PPO consumes the verifier's scalar value directly with no restructuring).

usage (real training run, needs live LLM calls -- Lightning AI, not local):
    python -m coordinator.train_ppo --iterations 10 --episodes-per-iter 20 \\
        --verifier-checkpoint checkpoints/verifier_v5 --out checkpoints/coordinator_v1 \\
        --worker-model hf:Qwen/Qwen2.5-3B-Instruct
"""

import argparse
import os
import random
from collections import Counter

import torch
from torch.optim import AdamW
from transformers import AutoTokenizer

from dotenv import load_dotenv
load_dotenv()

from coordinator.infer import CoordinatorPolicy, build_coordinator_input_text
from coordinator.model import ACTIONS, BASE_MODEL, CoordinatorModel
from envs.textworldexpress_env import TextWorldExpressEnvWrapper, TRAINING_GAMES
from rollout.runner import run_learned_coordinated_episode
from verifier.infer import Verifier

CLIP_EPS = 0.2          # PPO clipped-objective epsilon
PPO_EPOCHS = 4           # optimization passes over each collected batch
LEARNING_RATE = 1e-5
BATCH_SIZE = 8           # coordinator-decision minibatch size for the policy update
MAX_LENGTH = 384

# ENTROPY_COEF raised 0.01 -> 0.08 (coordinator_v1 postmortem, 2026-07-26): the first real
# training run collapsed to "retry" on 83-92% of ALL decisions (won AND lost episodes alike,
# confirmed via coordinator/evaluate.py's action-distribution logging), driving a real win-rate
# DROP vs the zero-coordination baseline (worst on the easy games peckingorder -33pts, simonsays
# -23pts -- exactly where "retry" masks a correct last action and forces a disruptive re-attempt
# on turns that didn't need one).
#
# coordinator_v2 postmortem (2026-07-26, same day): raising entropy to 0.08 and adding a
# KL-to-reference penalty (removed below, see that note) did NOT fix this -- retry-share went
# 99% (iteration 1) -> oscillated flat at 80-85% for the rest of training, no trend. Root cause,
# confirmed via an independent review reading train()'s actual call order: iteration 1's rollouts
# are collected from the UNTRAINED policy head (collect_batch runs before ppo_update each
# iteration -- see train()), so 99% retry was never a training pathology, it was
# coordinator/model.py's policy_head having a non-uniform initial bias BEFORE any gradient step
# (nn.Linear's default init on top of a pretrained backbone's non-random hidden state does not
# produce a uniform 3-way split -- fixed in coordinator/model.py via zero-init). Once that's
# fixed, ENTROPY_COEF=0.08 is kept as the regularizer that keeps genuine reward-driven learning
# from re-collapsing prematurely -- it was never the wrong knob, just insufficient to overcome a
# biased init it never caused and couldn't see.
ENTROPY_COEF = 0.08


def collect_batch(
    policy: CoordinatorPolicy, verifier: Verifier, games: tuple[str, ...],
    n_episodes: int, worker_model: str, split: str = "train",
) -> list[dict]:
    """run n_episodes with the current policy, return one dict per coordinator DECISION (not per
    episode) -- flattened across all episodes in the batch, each carrying everything the PPO
    update needs: the state text, the action taken, its log_prob under the policy that generated
    it (old_log_prob, frozen for the ratio computation), and its reward (the verifier's advantage
    for the resulting turn)."""
    examples = []
    for i in range(n_episodes):
        game = random.choice(games)
        env = TextWorldExpressEnvWrapper(split=split, games=(game,))
        traj = run_learned_coordinated_episode(
            env, coordinator=policy, verifier=verifier, task_id=f"ppo_{i}",
            model=worker_model, greedy=False,
        )
        env.close()

        history_so_far: list[str] = []
        for turn in traj.turns:
            meta = turn.metadata
            if meta.get("coordinator_log_prob") is None:
                history_so_far.append(turn.action)
                continue  # turn 1 always defaults to "continue" with no sampled log_prob, skip
            # history_so_far reflects exactly what the coordinator saw live: the action history
            # BEFORE this turn's action (appended after the coordinator's decision was made)
            text = build_coordinator_input_text(
                traj.task_goal, traj.plan, meta["coordinator_state_obs"],
                history_so_far, meta["coordinator_prior_q_value"], meta["coordinator_prior_advantage"],
            )
            examples.append({
                "text": text,
                "action": meta["coordinator_action"],
                "old_log_prob": meta["coordinator_log_prob"],
                "reward": meta["advantage"],  # verifier's baseline-subtracted advantage == PPO advantage
            })
            history_so_far.append(turn.action)
        print(f"  episode {i+1}/{n_episodes} [{game}] won={traj.won} steps={traj.total_steps}", flush=True)

    return examples


def ppo_update(
    model: CoordinatorModel, tokenizer, examples: list[dict], optimizer, device: torch.device,
) -> tuple[float, float]:
    """clipped-ratio PPO policy update over one collected batch. no value-function loss term --
    the advantage is precomputed from the frozen verifier (see module docstring), so this is a
    pure policy-gradient update, not the usual actor+critic joint loss.

    NO KL-to-reference penalty (removed 2026-07-26, coordinator_v2 postmortem): it anchored the
    policy toward a FROZEN snapshot of its own initial weights -- which coordinator_v2 confirmed
    were already biased toward "retry" before any training (nn.Linear's default init on a
    pretrained backbone does not produce a uniform 3-way split, see coordinator/model.py's
    zero-init fix). Penalizing divergence from an already-biased reference actively fights escaping
    that bias. With the policy head now zero-initialized (truly uniform at iteration 1), entropy
    alone is the right regularizer -- it has no direction of its own, so it can't anchor the policy
    toward a bad prior the way KL-to-reference did.

    returns (avg_loss, avg_entropy) so the caller can log both per iteration and catch a
    re-collapse early instead of only finding out at the next full evaluation."""
    action_to_idx = {a: i for i, a in enumerate(ACTIONS)}
    total_loss = 0.0
    total_entropy = 0.0
    num_updates = 0

    # normalize advantages ONCE, over the whole iteration's batch, not per-minibatch. builds
    # 10-12 confirmed the verifier's advantage scale differs a lot across games (coin/mapreader vs
    # simonsays/peckingorder); per-minibatch normalization with random.shuffle would make a
    # decision's normalized advantage depend on which OTHER games happened to land in the same
    # size-BATCH_SIZE minibatch, pure noise from the shuffle rather than a stable target.
    raw_advantages = torch.tensor([ex["reward"] for ex in examples], dtype=torch.float32)
    if raw_advantages.std() > 1e-6:
        norm_advantages = (raw_advantages - raw_advantages.mean()) / (raw_advantages.std() + 1e-8)
    else:
        norm_advantages = raw_advantages
    for ex, adv in zip(examples, norm_advantages.tolist()):
        ex["_norm_advantage"] = adv

    for epoch in range(PPO_EPOCHS):
        random.shuffle(examples)
        for start in range(0, len(examples), BATCH_SIZE):
            batch = examples[start:start + BATCH_SIZE]
            if not batch:
                continue

            texts = [ex["text"] for ex in batch]
            action_idx = torch.tensor([action_to_idx[ex["action"]] for ex in batch], device=device)
            old_log_probs = torch.tensor([ex["old_log_prob"] for ex in batch], device=device)
            advantages = torch.tensor([ex["_norm_advantage"] for ex in batch], device=device, dtype=torch.float32)

            encoded = tokenizer(
                texts, truncation=True, max_length=MAX_LENGTH, padding=True, return_tensors="pt",
            ).to(device)

            logits = model(encoded["input_ids"], encoded["attention_mask"])
            dist = torch.distributions.Categorical(logits=logits)
            new_log_probs = dist.log_prob(action_idx)

            ratio = torch.exp(new_log_probs - old_log_probs)
            unclipped = ratio * advantages
            clipped = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * advantages
            policy_loss = -torch.min(unclipped, clipped).mean()

            # entropy bonus keeps the policy from collapsing to a single action too early
            entropy_bonus = dist.entropy().mean()
            loss = policy_loss - ENTROPY_COEF * entropy_bonus

            optimizer.zero_grad()
            loss.backward()
            # gradient clipping: cheap insurance against a full-finetune (1.5B params) diverging
            # on a small (~20-episode), potentially high-variance batch -- no safeguard against
            # this existed before, a real risk on a paid Lightning AI run.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            total_entropy += entropy_bonus.item()
            num_updates += 1

    n = max(num_updates, 1)
    return total_loss / n, total_entropy / n


def train(
    verifier_checkpoint: str,
    out_dir: str,
    iterations: int = 10,
    episodes_per_iter: int = 20,
    worker_model: str = "hf:Qwen/Qwen2.5-3B-Instruct",
    learning_rate: float = LEARNING_RATE,
    resume_from: str | None = None,
) -> None:
    device = torch.device("cuda") if torch.cuda.is_available() else (
        torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    )
    print(f"using device: {device}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    # forced explicitly, not left to the tokenizer's default: the model reads the LAST non-pad
    # token via attention_mask.sum(dim=1)-1 (coordinator/model.py forward()), which is only
    # correct under right-padding. Qwen2.5-1.5B-Instruct's default happens to already be "right"
    # but relying on that silently is fragile -- a future checkpoint/tokenizer-config change would
    # break every batched log_prob without erroring.
    tokenizer.padding_side = "right"
    model = CoordinatorModel(BASE_MODEL, freeze_backbone=False).to(device)
    if resume_from:
        model.load_state_dict(torch.load(os.path.join(resume_from, "coordinator.pt"), map_location="cpu"))
        print(f"resumed policy weights from {resume_from}", flush=True)
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=learning_rate)

    verifier = Verifier(verifier_checkpoint, bound_q_value=False)  # v5 was trained --unbounded-q-value
    os.makedirs(out_dir, exist_ok=True)

    for it in range(1, iterations + 1):
        print(f"\n=== PPO iteration {it}/{iterations}: collecting {episodes_per_iter} episodes ===", flush=True)
        model.eval()
        # wrap the training model itself as the sampling policy so this iteration's rollouts
        # reflect the latest update -- on-policy by construction, not the frozen-checkpoint
        # CoordinatorPolicy used at eval time
        policy = _LivePolicyAdapter(model, tokenizer, device)
        examples = collect_batch(policy, verifier, TRAINING_GAMES, episodes_per_iter, worker_model)
        print(f"collected {len(examples)} coordinator decisions", flush=True)

        if not examples:
            print("no decisions collected this iteration (all episodes hit the turn-1-only edge "
                  "case), skipping update", flush=True)
            continue

        # action-distribution logging BEFORE the update: catches a collapse in the data the
        # policy is ABOUT to be reinforced on, one iteration earlier than waiting for the next
        # full coordinator.evaluate.py run (which is what surfaced coordinator_v1's ~85-90%
        # "retry" collapse only after all 10 iterations had already completed).
        action_counts = Counter(ex["action"] for ex in examples)
        total_actions = sum(action_counts.values())
        dist_str = ", ".join(
            f"{a}={action_counts.get(a, 0)}/{total_actions} ({action_counts.get(a, 0)/total_actions*100:.0f}%)"
            for a in ACTIONS
        )
        print(f"  action distribution this iteration: {dist_str}", flush=True)

        model.train()
        avg_loss, avg_entropy = ppo_update(model, tokenizer, examples, optimizer, device)
        mean_reward = sum(ex["reward"] for ex in examples) / len(examples)
        print(
            f"iteration {it}: avg_loss={avg_loss:.4f} mean_reward={mean_reward:.4f} "
            f"avg_entropy={avg_entropy:.4f}", flush=True,
        )

        torch.save(model.state_dict(), os.path.join(out_dir, "coordinator_last.pt"))
        torch.save(model.state_dict(), os.path.join(out_dir, "coordinator.pt"))

    tokenizer.save_pretrained(out_dir)
    print(f"\nsaved coordinator checkpoint to {out_dir}/", flush=True)


class _LivePolicyAdapter:
    """wraps the IN-TRAINING model with CoordinatorPolicy's sample()/act_greedy() interface, so
    run_learned_coordinated_episode can call it identically whether the policy is a saved
    checkpoint or the live model being updated this iteration."""

    def __init__(self, model: CoordinatorModel, tokenizer, device: torch.device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    @torch.no_grad()
    def sample(self, task_goal, plan, obs_before, action_history, q_value, advantage):
        text = build_coordinator_input_text(task_goal, plan, obs_before, action_history, q_value, advantage)
        encoded = self.tokenizer(text, truncation=True, max_length=MAX_LENGTH, return_tensors="pt").to(self.device)
        was_training = self.model.training
        self.model.eval()
        logits = self.model(encoded["input_ids"], encoded["attention_mask"])[0]
        self.model.train(was_training)
        dist = torch.distributions.Categorical(logits=logits)
        idx = dist.sample()
        return ACTIONS[idx.item()], dist.log_prob(idx).item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verifier-checkpoint", type=str, default="checkpoints/verifier_v5")
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--episodes-per-iter", type=int, default=20)
    parser.add_argument("--worker-model", type=str, default="hf:Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--resume-from", type=str, default=None)
    args = parser.parse_args()
    train(
        args.verifier_checkpoint, args.out, args.iterations, args.episodes_per_iter,
        args.worker_model, args.lr, args.resume_from,
    )
