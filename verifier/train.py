"""
train the verifier: a qwen2.5-3b backbone plus a 2-output value head, trained to predict
q_value and advantage for every worker turn in a labeled rollout dataset.

two modes, controlled by --freeze-backbone:
  - frozen (default off): only the value head trains, backbone stays in inference mode.
    light enough to run locally on apple silicon (mps), useful for a quick correctness
    check of the training loop, but the verifier's quality is capped by how well qwen's
    off-the-shelf representations already separate "on track" from "stuck" states.
  - unfrozen (full fine-tune): the whole backbone trains alongside the head, same as
    agentprm's own setup. needs real gpu memory, run this on lightning ai. this is the
    mode that produces the actual verifier checkpoint used downstream.

loss follows agentprm's dual-objective form: L = L_Q + beta * L_A, both plain MSE.
q_value is the harder, more important signal (it drives the detector), advantage is
the secondary term, hence beta < 1 by default.

usage:
    # quick local check, frozen backbone, small batch (mps-safe)
    python -m verifier.train --data data/labeled/build_5 --out checkpoints/verifier_v1_local

    # full fine-tune on a cuda gpu (lightning ai)
    python -m verifier.train --data data/labeled/build_5 --out checkpoints/verifier_v1 \\
        --full-finetune --batch-size 16 --epochs 5
"""

import argparse
import functools
import os

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer

from verifier.dataset import VerifierDataset, collate_fn, load_examples
from verifier.model import VerifierModel, BASE_MODEL

# 8-bit adamw is only available (and only needed) on cuda. full-finetune on cuda puts
# adamw's fp32 momentum+variance state on all 3b backbone params, ~25gb by itself, more
# than a single gpu (e.g. 22gb) can hold alongside the model weights and activations.
# 8-bit optimizer state cuts that to ~6gb. mps has no bitsandbytes support, and frozen-mode
# runs there only ever optimize the tiny value head anyway, so plain adamw is fine there.
try:
    import bitsandbytes as bnb
    HAS_BITSANDBYTES = True
except ImportError:
    HAS_BITSANDBYTES = False

BETA = 0.5              # weight on the advantage loss term, q_value loss is the primary signal
# defaults assume a frozen backbone on mps. full fine-tuning on a cuda gpu should override
# these via cli flags (bigger batch, lower lr since the whole backbone now trains).
BATCH_SIZE = 2
LEARNING_RATE = 1e-3    # fine for a frozen linear head; use ~1e-5 when fine-tuning the backbone
NUM_EPOCHS = 3
VAL_FRACTION = 0.1      # held-out slice of the labeled data, purely for tracking overfitting
LOG_EVERY = 200         # print running loss every N batches, so a stall is visible immediately


def get_device() -> torch.device:
    """prefer cuda (lightning ai), then apple silicon mps, then cpu."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_epoch(model, loader, optimizer, device, train: bool, log_prefix: str = "") -> tuple[float, float]:
    """one pass over the data. returns (avg q_loss, avg advantage_loss). optimizer is
    None-safe: pass train=False and no gradient step happens, used for the val pass."""
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

            # dual loss: L_Q + beta * L_A, both mse, matches agentprm's objective
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

        # print progress periodically, otherwise a stall or slow pass is invisible until
        # the whole epoch finishes, which can be a long wait on this machine's mps backend
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
) -> None:
    device = get_device()
    print(f"using device: {device}, freeze_backbone={freeze_backbone}", flush=True)

    # load and flatten every worker turn across the labeled trajectories
    examples = load_examples(data_dir)
    print(f"loaded {len(examples)} labeled worker turns from {data_dir}", flush=True)
    if len(examples) == 0:
        raise SystemExit(f"no labeled examples found in {data_dir}, run rollout.label first")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    # 256 tokens covers task+plan+observation+action comfortably (most alfworld
    # observations are short) while keeping per-batch activation memory down
    dataset = VerifierDataset(examples, tokenizer, max_length=max_length)

    # simple random split for a validation slice, just to watch for overfitting
    val_size = int(len(dataset) * VAL_FRACTION)
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    collate = functools.partial(collate_fn, pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=collate)

    print(f"train examples: {len(train_set)}, val examples: {len(val_set)}", flush=True)

    model = VerifierModel(BASE_MODEL, freeze_backbone=freeze_backbone).to(device)
    # when frozen, only the head has requires_grad=True; when fine-tuning, this covers
    # the whole model, so filter to trainable params either way rather than branching
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable parameters: {sum(p.numel() for p in trainable_params):,}", flush=True)

    # full-finetune on cuda: use 8-bit adamw so optimizer state doesn't itself oom the gpu.
    # everything else (frozen-head mode, or any non-cuda device): plain fp32 adamw is fine,
    # since the head alone is a few thousand parameters.
    use_8bit = (not freeze_backbone) and device.type == "cuda" and HAS_BITSANDBYTES
    if use_8bit:
        optimizer = bnb.optim.AdamW8bit(trainable_params, lr=learning_rate)
        print("using 8-bit AdamW (bitsandbytes) to fit optimizer state on the GPU", flush=True)
    else:
        if (not freeze_backbone) and device.type == "cuda" and not HAS_BITSANDBYTES:
            print(
                "WARNING: full-finetune on cuda without bitsandbytes installed, "
                "optimizer state may not fit in gpu memory. pip install bitsandbytes.",
                flush=True,
            )
        optimizer = AdamW(trainable_params, lr=learning_rate)

    for epoch in range(1, num_epochs + 1):
        train_q_loss, train_a_loss = run_epoch(
            model, train_loader, optimizer, device, train=True, log_prefix=f"epoch {epoch} [train]"
        )
        val_q_loss, val_a_loss = run_epoch(
            model, val_loader, optimizer, device, train=False, log_prefix=f"epoch {epoch} [val]"
        )
        print(
            f"epoch {epoch}/{num_epochs} | "
            f"train: q_loss={train_q_loss:.4f} adv_loss={train_a_loss:.4f} | "
            f"val: q_loss={val_q_loss:.4f} adv_loss={val_a_loss:.4f}",
            flush=True,
        )

        # checkpoint after every epoch, not just at the end, so a stall or kill later
        # doesn't lose progress already made
        os.makedirs(out_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(out_dir, "verifier.pt"))

    tokenizer.save_pretrained(out_dir)
    print(f"saved verifier checkpoint to {out_dir}/", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="directory of labeled trajectory jsons")
    parser.add_argument("--out", type=str, required=True, help="directory to save the trained checkpoint")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument(
        "--full-finetune", action="store_true",
        help="train the whole backbone, not just the head. needs a real gpu, not mps.",
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
    )
