"""
standalone length-confound check against an ALREADY-TRAINED checkpoint -- no retraining needed.

rebuilds the identical held-out val split train.py used (same split_by_game() call, same seed),
loads the saved model.pt, and runs train.py's _length_confound_check() against it. use this when
you already have a checkpoint from a prior training run and just want the confound numbers without
paying for another full training pass.

usage:
    python -m verifier_dpo.check_length_confound \
      --checkpoint verifier_dpo/checkpoints/sanity_check/model.pt \
      --pairs-per-game 40
(pairs-per-game must match whatever the checkpoint was originally trained with, so the val split's
pair count lines up with what val_pairwise_acc was computed on during training.)
"""

import argparse

import torch
from transformers import AutoTokenizer

from verifier_dpo.dataset import load_episodes_by_game, split_by_game
from verifier_dpo.model import BASE_MODEL, PreferenceScorer
from verifier_dpo.train import VAL_FRACTION, _length_confound_check, _evaluate, _make_loader


def main(checkpoint: str, labeled_dir: str, pairs_per_game: int | None) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    by_game = load_episodes_by_game(labeled_dir)
    _, val_by_game = split_by_game(by_game, val_fraction=VAL_FRACTION)
    val_loader, n_val_pairs = _make_loader(val_by_game, pairs_per_game, tokenizer, shuffle=False)
    print(f"val pairs: {n_val_pairs}")

    model = PreferenceScorer(freeze_backbone=True).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    val_loss, val_acc = _evaluate(model, val_loader, device)
    print(f"val_avg_loss={val_loss:.4f} val_pairwise_acc={val_acc:.4f}")

    confound = _length_confound_check(model, val_loader, device)
    print("length-confound check (val set):")
    print(f"  episode_score_vs_length_correlation: {confound['episode_score_vs_length_correlation']}")
    print(f"  length_only_pairwise_acc (predict shorter=won, zero model signal): "
          f"{confound['length_only_pairwise_acc']} (n={confound['n_pairs']})")
    print(f"  for comparison, trained model's val_pairwise_acc: {val_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="verifier_dpo/checkpoints/sanity_check/model.pt")
    parser.add_argument("--labeled-dir", type=str, default="data/labeled/v5_train_combined")
    parser.add_argument("--pairs-per-game", type=int, default=40)
    args = parser.parse_args()
    main(args.checkpoint, args.labeled_dir, args.pairs_per_game)
