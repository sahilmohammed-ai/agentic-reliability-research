"""
train verifier: qwen backbone + value head for q-value and advantage prediction.
frozen mode: trains head only (local testing). unfrozen: full fine-tune (needs gpu).
loss: L_Q + beta * L_A (q-value weighted higher, advantage secondary).
"""

import argparse
import functools
import os
import random
from collections import defaultdict

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer

from verifier.dataset import VerifierDataset, collate_fn, load_examples
from verifier.model import VerifierModel, BASE_MODEL

# 8-bit adamw reduces optimizer state memory on cuda (needed for full fine-tune)
try:
    import bitsandbytes as bnb
    HAS_BITSANDBYTES = True
except ImportError:
    HAS_BITSANDBYTES = False

BETA = 0.5              # advantage loss weight (q-value is primary)
BATCH_SIZE = 2          # frozen mode default (use --batch-size for full fine-tune)
LEARNING_RATE = 1e-3    # frozen mode default (use --lr ~1e-5 for full fine-tune)
NUM_EPOCHS = 3
VAL_FRACTION = 0.1      # held-out validation data
LOG_EVERY = 200         # progress logging frequency


def stratified_group_split(
    examples: list[dict], val_fraction: float = VAL_FRACTION, seed: int = 42
) -> tuple[list[int], list[int]]:
    """split example indices into (train_indices, val_indices), grouped by episode_id and
    stratified by task_goal, so no episode's turns are ever split across train/val (the bug this
    replaces: turn-level random_split let consecutive, near-identical turns from the same episode
    leak across the split, so val loss was measuring memorization, not generalization -- the same
    failure mode scikit-learn's GroupKFold/GroupShuffleSplit exist to prevent).

    stratified by task_goal (not just grouped by episode) so each task template's episodes are
    proportionally divided rather than left to chance: with ~372 templates across 500 episodes,
    many templates have only 1-2 episodes, and a template with a single episode is assigned
    entirely to train (a template can't be "proportionally split" with one member, and pulling it
    into val would only add an untestable, high-variance singleton rather than a stable validation
    signal -- full template-level holdout was considered and rejected for the same reason at a
    larger scale: too few independent groups for a validation loss trustworthy enough to drive
    epoch-to-epoch early stopping).

    known residual limitation, not fixed here: alfworld reuses a small pool of underlying room
    layouts across different task_goals, so two episodes with different task_goal strings can still
    share a very similar observation-text distribution. episode/task_goal grouping fixes the
    within-episode leak (the confirmed, measured problem) but does not guarantee zero train/val
    similarity from shared room layouts, since scene/floorplan id is not currently captured
    anywhere in collected trajectory data (would need a rollout/envs/alfworld_env.py change to fix
    for future collections)."""
    by_task: dict[str, list[str]] = defaultdict(list)  # task_goal -> [episode_id, ...] (deduped)
    episode_task: dict[str, str] = {}
    for ex in examples:
        eid, task = ex["episode_id"], ex["task_goal"]
        episode_task[eid] = task
    for eid, task in episode_task.items():
        by_task[task].append(eid)

    rng = random.Random(seed)
    val_episodes: set[str] = set()
    for task, episodes in by_task.items():
        if len(episodes) < 2:
            continue  # singleton-template episodes go entirely to train, see docstring
        episodes = sorted(episodes)
        rng.shuffle(episodes)
        n_val = max(1, round(len(episodes) * val_fraction))
        val_episodes.update(episodes[:n_val])

    train_indices = [i for i, ex in enumerate(examples) if ex["episode_id"] not in val_episodes]
    val_indices = [i for i, ex in enumerate(examples) if ex["episode_id"] in val_episodes]
    return train_indices, val_indices


def get_device() -> torch.device:
    """prefer cuda (lightning ai), then apple silicon mps, then cpu."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_epoch(model, loader, optimizer, device, train: bool, log_prefix: str = "") -> tuple[float, float]:
    """one pass over data. returns (avg q_loss, avg advantage_loss)."""
    model.train(mode=train)
    total_q_loss = 0.0
    total_a_loss = 0.0
    num_batches = 0
    total_batches = len(loader)

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        q_target = batch["q_value"].to(device)
        a_target = batch["advantage"].to(device)

        with torch.set_grad_enabled(train):
            predictions = model(input_ids, attention_mask)
            q_pred = predictions[:, 0]
            a_pred = predictions[:, 1]

            # dual loss: L_Q + beta * L_A
            q_loss = torch.nn.functional.mse_loss(q_pred.float(), q_target.float())
            a_loss = torch.nn.functional.mse_loss(a_pred.float(), a_target.float())
            loss = q_loss + BETA * a_loss

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_q_loss += q_loss.item()
        total_a_loss += a_loss.item()
        num_batches += 1

        # periodic progress logging
        if log_prefix and num_batches % LOG_EVERY == 0:
            print(
                f"{log_prefix} batch {num_batches}/{total_batches} | "
                f"running q_loss={total_q_loss / num_batches:.4f} adv_loss={total_a_loss / num_batches:.4f}",
                flush=True,
            )

    return total_q_loss / num_batches, total_a_loss / num_batches


def train(
    data_dir: str,
    out_dir: str,
    num_epochs: int = NUM_EPOCHS,
    freeze_backbone: bool = True,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    max_length: int = 256,
    bound_q_value: bool = True,
) -> None:
    device = get_device()
    print(f"using device: {device}, freeze_backbone={freeze_backbone}", flush=True)

    # load examples and create dataset
    examples = load_examples(data_dir)
    print(f"loaded {len(examples)} labeled worker turns from {data_dir}", flush=True)
    if len(examples) == 0:
        raise SystemExit(f"no labeled examples found in {data_dir}, run rollout.label first")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    dataset = VerifierDataset(examples, tokenizer, max_length=max_length)

    # train/val split: stratified-group by (episode_id, task_goal), not a flat turn-level random
    # split, so no episode's turns are split across train/val. see stratified_group_split()'s
    # docstring for why (confirmed group-leakage bug in the previous random_split(dataset, ...)
    # approach) and what it still doesn't cover (residual alfworld scene-reuse risk).
    train_indices, val_indices = stratified_group_split(examples, VAL_FRACTION)
    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)

    collate = functools.partial(collate_fn, pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=collate)

    print(f"train examples: {len(train_set)}, val examples: {len(val_set)}", flush=True)

    # model and trainable parameters
    model = VerifierModel(BASE_MODEL, freeze_backbone=freeze_backbone, bound_q_value=bound_q_value).to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable parameters: {sum(p.numel() for p in trainable_params):,}", flush=True)

    # optimizer: 8-bit on cuda (full fine-tune), plain adamw elsewhere
    use_8bit = (not freeze_backbone) and device.type == "cuda" and HAS_BITSANDBYTES
    if use_8bit:
        optimizer = bnb.optim.AdamW8bit(trainable_params, lr=learning_rate)
        print("using 8-bit AdamW (bitsandbytes)", flush=True)
    else:
        if (not freeze_backbone) and device.type == "cuda" and not HAS_BITSANDBYTES:
            print("WARNING: full-finetune on cuda without bitsandbytes, may OOM", flush=True)
        optimizer = AdamW(trainable_params, lr=learning_rate)

    os.makedirs(out_dir, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        train_q_loss, train_a_loss = run_epoch(
            model, train_loader, optimizer, device, train=True, log_prefix=f"epoch {epoch} [train]"
        )
        val_q_loss, val_a_loss = run_epoch(
            model, val_loader, optimizer, device, train=False, log_prefix=f"epoch {epoch} [val]"
        )
        val_loss = val_q_loss + BETA * val_a_loss
        print(
            f"epoch {epoch}/{num_epochs} | "
            f"train: q_loss={train_q_loss:.4f} adv_loss={train_a_loss:.4f} | "
            f"val: q_loss={val_q_loss:.4f} adv_loss={val_a_loss:.4f} (combined={val_loss:.4f})",
            flush=True,
        )

        # save latest (recover from interruption) and best (early stopping)
        torch.save(model.state_dict(), os.path.join(out_dir, "verifier_last.pt"))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(out_dir, "verifier.pt"))
            print(f"  -> new best val loss ({val_loss:.4f}), saved as verifier.pt", flush=True)

    tokenizer.save_pretrained(out_dir)
    print(f"saved verifier checkpoint to {out_dir}/", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="directory of labeled trajectory jsons")
    parser.add_argument("--out", type=str, required=True, help="directory to save checkpoint")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--full-finetune", action="store_true", help="train full backbone (needs gpu)")
    parser.add_argument(
        "--unbounded-q-value", action="store_true",
        help="disable the sigmoid on q_value (use for TextWorldExpress / verifier v3+ data, whose "
             "labels can be genuinely negative -- see verifier/model.py's bound_q_value docstring). "
             "omit for ALFWorld-era data, where q_value is a [0,1] win-probability target.",
    )
    args = parser.parse_args()
    train(
        args.data,
        args.out,
        num_epochs=args.epochs,
        freeze_backbone=not args.full_finetune,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_length=args.max_length,
        bound_q_value=not args.unbounded_q_value,
    )
