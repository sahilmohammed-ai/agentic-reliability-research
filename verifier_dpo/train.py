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
import os

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from verifier_dpo.dataset import (
    EpisodePairDataset, collate_fn, load_episodes_by_game, make_pairs, split_by_game,
)
from verifier_dpo.model import BASE_MODEL, PreferenceScorer

# 8-bit adamw reduces optimizer state memory on cuda (needed for full fine-tune), same pattern as
# verifier/train.py -- 1.5B backbone here is smaller than that file's 3B, but full backprop
# through all backbone layers is still real memory pressure a frozen-head run never pays.
try:
    import bitsandbytes as bnb
    HAS_BITSANDBYTES = True
except ImportError:
    HAS_BITSANDBYTES = False

BATCH_SIZE = 2               # frozen-head default (use --batch-size for full fine-tune)
# lowered from 1e-3 after review (2026-07-29): the first GPU run showed most individual step
# losses saturated at exactly 0.0000 with occasional spikes to 100+ (a few high-confidence-gap
# pairs dominating each batch's gradient) -- switching episode_score() from sum to mean pooling
# (see model.py) fixed the length-dependence but not this instability by itself. a lower lr plus
# grad clipping (below) reduces how much any single outlier pair's gradient can move the head.
LEARNING_RATE = 2e-4          # frozen-head default (use --lr ~1e-5 for full fine-tune, matches
                               # verifier/train.py's full-finetune rate -- fine-tuning a whole
                               # pretrained model needs a much smaller step than training a fresh
                               # linear head from scratch)
MAX_GRAD_NORM = 1.0    # added for the same reason -- caps any one batch's gradient magnitude
NUM_EPOCHS = 3
LOG_EVERY = 20
VAL_FRACTION = 0.2


def preference_loss(won_scores: torch.Tensor, lost_scores: torch.Tensor) -> torch.Tensor:
    """bradley-terry pairwise loss: -log(sigmoid(won - lost)). pushes won episode's mean per-token
    score above the paired lost episode's, no margin needed (sigmoid already saturates gently)."""
    return -torch.nn.functional.logsigmoid(won_scores - lost_scores).mean()


def _make_loader(
    by_game: dict, pairs_per_game, tokenizer, shuffle: bool, batch_size: int = BATCH_SIZE,
) -> DataLoader:
    pairs = make_pairs(by_game, pairs_per_game=pairs_per_game)
    dataset = EpisodePairDataset(pairs, tokenizer)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
    ), len(pairs)


@torch.no_grad()
def _evaluate(model, loader, device) -> tuple[float, float]:
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for batch in loader:
        won_scores = model.episode_score(
            batch["won_input_ids"].to(device), batch["won_attention_mask"].to(device),
        )
        lost_scores = model.episode_score(
            batch["lost_input_ids"].to(device), batch["lost_attention_mask"].to(device),
        )
        loss = preference_loss(won_scores, lost_scores)
        total_loss += loss.item() * len(batch["games"])
        correct += (won_scores > lost_scores).sum().item()
        n += len(batch["games"])
    model.train()
    return total_loss / n, correct / n


@torch.no_grad()
def _length_confound_check(model, loader, device) -> dict:
    """answers the open question from the 2026-07-29 sanity check: does episode-level
    pairwise_acc survive controlling for episode LENGTH (token count), or is it substantially
    riding on the same length shortcut the per-turn readout showed (score_vs_turn_length_correlation
    = -0.62 on that run)? two things computed on the same val pairs used for val_pairwise_acc:

    1. correlation between episode_score and episode token-length, pooling won+lost together --
       if strongly negative (mirroring the per-turn finding), the episode-level score is also
       substantially a length proxy.
    2. "length-only" pairwise accuracy: for each pair, predict whichever episode is SHORTER as
       the winner (using length alone, zero model signal) and compare against the true won label.
       if this length-only heuristic alone gets close to val_pairwise_acc, the trained model isn't
       adding much beyond what raw length already tells you."""
    model.eval()
    scores, lengths, is_won = [], [], []
    length_only_correct, n_pairs = 0, 0

    for batch in loader:
        won_scores = model.episode_score(
            batch["won_input_ids"].to(device), batch["won_attention_mask"].to(device),
        )
        lost_scores = model.episode_score(
            batch["lost_input_ids"].to(device), batch["lost_attention_mask"].to(device),
        )
        won_lens = batch["won_attention_mask"].sum(dim=1).tolist()
        lost_lens = batch["lost_attention_mask"].sum(dim=1).tolist()

        scores.extend(won_scores.tolist() + lost_scores.tolist())
        lengths.extend(won_lens + lost_lens)
        is_won.extend([1] * len(won_lens) + [0] * len(lost_lens))

        for wl, ll in zip(won_lens, lost_lens):
            # length-only heuristic: predict the SHORTER episode is the winner (matches the
            # direction of the per-turn finding, where shorter turns scored higher)
            length_only_correct += int(wl < ll)
            n_pairs += 1

    model.train()

    s_t = torch.tensor(scores)
    l_t = torch.tensor([float(x) for x in lengths])
    score_length_corr = 0.0
    if s_t.std() > 0 and l_t.std() > 0:
        score_length_corr = torch.corrcoef(torch.stack([s_t, l_t]))[0, 1].item()

    return {
        "episode_score_vs_length_correlation": round(score_length_corr, 4),
        "length_only_pairwise_acc": round(length_only_correct / n_pairs, 4) if n_pairs else 0.0,
        "n_pairs": n_pairs,
    }


def train(
    labeled_dir: str, out_dir: str, freeze_backbone: bool, pairs_per_game: int | None, num_epochs: int,
    learning_rate: float, batch_size: int,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    by_game = load_episodes_by_game(labeled_dir)
    for game, buckets in by_game.items():
        print(f"  {game}: {len(buckets['won'])} won, {len(buckets['lost'])} lost")

    # episode-grouped train/val split (added after review, 2026-07-29): the first run had no
    # held-out split at all, so pairwise_acc was pure in-sample fit -- with peckingorder's 63 lost
    # episodes reused ~4x across 237 pairs, high train accuracy alone could reflect memorizing a
    # handful of episodes, not real generalization. splitting by episode (not by pair) guarantees
    # no episode's text appears in both a training pair and a validation pair.
    train_by_game, val_by_game = split_by_game(by_game, val_fraction=VAL_FRACTION)

    train_loader, n_train_pairs = _make_loader(
        train_by_game, pairs_per_game, tokenizer, shuffle=True, batch_size=batch_size,
    )
    val_loader, n_val_pairs = _make_loader(
        val_by_game, pairs_per_game, tokenizer, shuffle=False, batch_size=batch_size,
    )
    print(f"train pairs: {n_train_pairs}, val pairs: {n_val_pairs}")

    model = PreferenceScorer(freeze_backbone=freeze_backbone).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable parameters: {sum(p.numel() for p in trainable):,}")

    # 8-bit optimizer on cuda for full fine-tune, same pattern as verifier/train.py -- plain
    # AdamW's fp32 momentum+variance buffers (2x params x 4 bytes) is real memory pressure once
    # gradients flow through all of a 1.5B backbone's layers, not just one linear head.
    use_8bit = (not freeze_backbone) and device == "cuda" and HAS_BITSANDBYTES
    if use_8bit:
        optimizer = bnb.optim.AdamW8bit(trainable, lr=learning_rate)
        print("using 8-bit AdamW (bitsandbytes)")
    else:
        if (not freeze_backbone) and device == "cuda" and not HAS_BITSANDBYTES:
            print("WARNING: full-finetune on cuda without bitsandbytes, may OOM")
        optimizer = AdamW(trainable, lr=learning_rate)

    model.train()
    step = 0
    for epoch in range(num_epochs):
        epoch_loss, epoch_correct, epoch_n = 0.0, 0, 0
        for batch in train_loader:
            won_scores = model.episode_score(
                batch["won_input_ids"].to(device), batch["won_attention_mask"].to(device),
            )
            lost_scores = model.episode_score(
                batch["lost_input_ids"].to(device), batch["lost_attention_mask"].to(device),
            )
            loss = preference_loss(won_scores, lost_scores)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, MAX_GRAD_NORM)
            optimizer.step()

            epoch_loss += loss.item() * len(batch["games"])
            epoch_correct += (won_scores > lost_scores).sum().item()
            epoch_n += len(batch["games"])
            step += 1
            if step % LOG_EVERY == 0:
                print(f"  epoch {epoch} step {step}: loss={loss.item():.4f}")

        train_acc = epoch_correct / epoch_n
        val_loss, val_acc = _evaluate(model, val_loader, device)
        print(
            f"epoch {epoch}: train_avg_loss={epoch_loss/epoch_n:.4f} train_pairwise_acc={train_acc:.4f} "
            f"val_avg_loss={val_loss:.4f} val_pairwise_acc={val_acc:.4f}"
        )

    ckpt_path = os.path.join(out_dir, "model.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"saved checkpoint -> {ckpt_path}")

    # answers the open question from the prior sanity check (per-turn score_vs_turn_length_
    # correlation was -0.62 -- does episode-level val_pairwise_acc survive controlling for length,
    # or is it substantially the same shortcut at a coarser grain?), on the SAME held-out val pairs
    # already used for val_pairwise_acc above -- no extra data collection needed.
    confound = _length_confound_check(model, val_loader, device)
    print("length-confound check (val set):")
    print(f"  episode_score_vs_length_correlation: {confound['episode_score_vs_length_correlation']}")
    print(f"  length_only_pairwise_acc (predict shorter=won, zero model signal): "
          f"{confound['length_only_pairwise_acc']} (n={confound['n_pairs']})")
    print(f"  for comparison, trained model's val_pairwise_acc: {val_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--labeled-dir", type=str, default="data/labeled/v5_train_combined")
    parser.add_argument("--out", type=str, default="verifier_dpo/checkpoints/sanity_check")
    parser.add_argument("--freeze-backbone", action="store_true", default=True)
    parser.add_argument("--no-freeze-backbone", dest="freeze_backbone", action="store_false")
    parser.add_argument("--pairs-per-game", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--lr", type=float, default=None,
                         help="default: 2e-4 if frozen-backbone, 1e-5 if --no-freeze-backbone "
                              "(full fine-tune needs a much smaller step)")
    parser.add_argument("--batch-size", type=int, default=None,
                         help="default: 2 if frozen-backbone, 1 if --no-freeze-backbone "
                              "(full backprop through the backbone is far more memory per example)")
    args = parser.parse_args()

    lr = args.lr if args.lr is not None else (LEARNING_RATE if args.freeze_backbone else 1e-5)
    bs = args.batch_size if args.batch_size is not None else (BATCH_SIZE if args.freeze_backbone else 1)
    train(args.labeled_dir, args.out, args.freeze_backbone, args.pairs_per_game, args.epochs, lr, bs)
