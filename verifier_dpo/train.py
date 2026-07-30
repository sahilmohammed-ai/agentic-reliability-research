"""
train the episode-pair preference scorer (verifier_dpo/model.py's PreferenceScorer).

loss: bradley-terry / DPO-style preference loss on WHOLE-EPISODE scores --
    loss = -log(sigmoid(score(won_episode) - score(lost_episode)))
this only needs each episode's real won/lost outcome (never sparse -- every collected episode has
exactly one), unlike per-step q_value/advantage regression, which inherits TextWorldExpress's
near-zero non-terminal reward and has been confirmed (across MC return-to-go, TD/GAE, and
agent_prm's own shaped-target formula) to carry no same-state action-discrimination signal no
matter how it's computed.

what this DOES test (the actual purpose of this local sanity check, per plan): whether training
on this preference objective produces REAL per-step variance when scores are pooled per-turn
afterward (see infer.py), rather than collapsing flat the way label_td.py's TD/GAE relabeling did
("a wash", build 11). if per-turn scores come out flat/near-constant within an episode, that's a
real negative result and this line should stop here, before any Lightning AI GPU time is spent.

what this does NOT test and is not expected to show: same-state, two-different-actions
discrimination. this objective never sees two actions at one state side by side -- it only sees
whole trajectories against each other. a flat local sanity-check pass would validate "the verifier
can separate winning-flavored turns from losing-flavored turns," which is real signal for episode-
level failure detection, not a claim that it resolves the coordinator's exact same-state reward
problem (see verifier_dpo/model.py's module docstring and .info/CLAUDE.md for the full context).
"""

import argparse
import json
import os

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from verifier_dpo.dataset import EpisodePairDataset, collate_fn, load_episodes_by_game, make_pairs
from verifier_dpo.model import BASE_MODEL, PreferenceScorer

BATCH_SIZE = 2
LEARNING_RATE = 1e-3   # frozen-backbone-only default, matches verifier/train.py's local-test rate
NUM_EPOCHS = 3
LOG_EVERY = 20


def preference_loss(won_scores: torch.Tensor, lost_scores: torch.Tensor) -> torch.Tensor:
    """bradley-terry pairwise loss: -log(sigmoid(won - lost)). pushes won episode's summed score
    above the paired lost episode's, no margin needed (sigmoid already saturates gently)."""
    return -torch.nn.functional.logsigmoid(won_scores - lost_scores).mean()


def train(
    labeled_dir: str, out_dir: str, freeze_backbone: bool, pairs_per_game: int | None, num_epochs: int,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    by_game = load_episodes_by_game(labeled_dir)
    for game, buckets in by_game.items():
        print(f"  {game}: {len(buckets['won'])} won, {len(buckets['lost'])} lost")
    pairs = make_pairs(by_game, pairs_per_game=pairs_per_game)
    print(f"total pairs: {len(pairs)}")

    dataset = EpisodePairDataset(pairs, tokenizer)
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
    )

    model = PreferenceScorer(freeze_backbone=freeze_backbone).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable, lr=LEARNING_RATE)

    model.train()
    step = 0
    for epoch in range(num_epochs):
        epoch_loss, epoch_correct, epoch_n = 0.0, 0, 0
        for batch in loader:
            won_scores = model.episode_score(
                batch["won_input_ids"].to(device), batch["won_attention_mask"].to(device),
            )
            lost_scores = model.episode_score(
                batch["lost_input_ids"].to(device), batch["lost_attention_mask"].to(device),
            )
            loss = preference_loss(won_scores, lost_scores)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(batch["games"])
            epoch_correct += (won_scores > lost_scores).sum().item()
            epoch_n += len(batch["games"])
            step += 1
            if step % LOG_EVERY == 0:
                print(f"  epoch {epoch} step {step}: loss={loss.item():.4f}")

        acc = epoch_correct / epoch_n
        print(f"epoch {epoch}: avg_loss={epoch_loss/epoch_n:.4f} pairwise_acc={acc:.4f}")

    ckpt_path = os.path.join(out_dir, "model.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"saved checkpoint -> {ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--labeled-dir", type=str, default="data/labeled/v5_train_combined")
    parser.add_argument("--out", type=str, default="verifier_dpo/checkpoints/sanity_check")
    parser.add_argument("--freeze-backbone", action="store_true", default=True)
    parser.add_argument("--no-freeze-backbone", dest="freeze_backbone", action="store_false")
    parser.add_argument("--pairs-per-game", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    args = parser.parse_args()
    train(args.labeled_dir, args.out, args.freeze_backbone, args.pairs_per_game, args.epochs)
