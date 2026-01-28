#!/usr/bin/env python3
# select_best_flow.py

from pathlib import Path
import argparse
import shutil
import torch


def get_best_val(run_dir: Path) -> float:
    meta_p = run_dir / "meta.pt"
    losses_p = run_dir / "losses.pt"

    if meta_p.exists():
        meta = torch.load(meta_p, map_location="cpu", weights_only=False)
        bv = meta.get("best_val_loss", None)
        if bv is not None:
            return float(bv)

    if losses_p.exists():
        losses = torch.load(losses_p, map_location="cpu", weights_only=False)
        v = losses.get("val_losses", None)
        if v is None or len(v) == 0:
            raise ValueError(f"{losses_p} has no val_losses")
        return float(torch.min(v).item())

    raise ValueError(f"Missing meta.pt and losses.pt in {run_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", type=str, required=True,
                    help="e.g. outputs/protein_dataset (contains alpha*/lr*_wd*/*)")
    ap.add_argument("--outdir", type=str, required=True,
                    help="where to copy best {model.pt, meta.pt, losses.pt}")
    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Your structure: dataset_dir/alpha*/lr*_wd*/
    run_dirs = sorted([p for p in dataset_dir.glob("alpha*/lr*_wd*") if p.is_dir()])

    if not run_dirs:
        raise SystemExit(f"No run dirs found under {dataset_dir}/alpha*/lr*_wd*")

    best = None  # (best_val, run_dir)

    for rd in run_dirs:
        try:
            val = get_best_val(rd)
        except Exception as e:
            print(f"SKIP {rd}: {e}")
            continue

        print(f"{val: .6e}  {rd}")
        if best is None or val < best[0]:
            best = (val, rd)

    if best is None:
        raise SystemExit("No valid runs found (all skipped).")

    best_val, best_dir = best
    print(f"\nBEST: {best_val:.6e}  {best_dir}")

    for fname in ["model.pt", "meta.pt", "losses.pt"]:
        src = best_dir / fname
        if src.exists():
            shutil.copy2(src, outdir / fname)

    # also save a tiny pointer file so you know where it came from
    (outdir / "best_path.txt").write_text(str(best_dir) + "\n")
    (outdir / "best_val.txt").write_text(f"{best_val:.12e}\n")

    print(f"Copied to: {outdir}")


if __name__ == "__main__":
    main()
