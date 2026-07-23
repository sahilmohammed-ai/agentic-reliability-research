"""
consolidate per-game labeled trajectory directories into one flat directory with game-prefixed
filenames (so files from different games don't collide), ready for verifier/train.py's --data.

usage:
    python -m scripts.consolidate_labeled --in data/labeled/v4_train_td \\
        --out data/labeled/v4_train_td_combined --games coin,simonsays,peckingorder,mapreader
"""
import argparse, glob, os, shutil

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_dir", required=True, help="parent dir containing per-game subdirs")
    p.add_argument("--out", dest="out_dir", required=True)
    p.add_argument("--games", default="coin,simonsays,peckingorder,mapreader",
                   help="comma-separated game subdirectory names")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    total = 0
    for game in args.games.split(","):
        files = sorted(glob.glob(os.path.join(args.in_dir, game, "*.json")))
        for f in files:
            shutil.copy(f, os.path.join(args.out_dir, f"{game}_{os.path.basename(f)}"))
            total += 1
        print(f"{game}: {len(files)} files")
    print(f"total: {total} -> {args.out_dir}/")
