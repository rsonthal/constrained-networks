#!/usr/bin/env python3
# train_flowmatch.py
#
# Trains one FlowVelocityNet run from a .pt dataset file and writes:
#   - model.pt   (state_dict + architecture + best stats + hparams + T_used)
#   - losses.pt  (train_losses, val_losses as float tensors)
#   - meta.pt    (small metadata: dataset path, shapes, hparams, best stats, T_used)
#
# No JSON is written.

import argparse
import os
import numpy as np
import torch

from flow_matching import FlowVelocityNet
from train_flow_matching import train_flow_matching_model


def flatten_if_needed(X: torch.Tensor) -> torch.Tensor:
    # Handles cs_dataset: N x M x d -> (N*M) x d
    return X.reshape(-1, X.shape[-1]) if X.ndim == 3 else X


def atomic_torch_save(obj, path: str) -> None:
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def parse_T(T_arg: str, train_init: np.ndarray, velocity_scale: float, alpha: float):
    """
    If T_arg == "auto": T = alpha * median(||x||) / velocity_scale.
    Else: parse float.
    Returns (T_used, r_median).
    """
    if isinstance(T_arg, str) and T_arg.lower() == "auto":
        r = np.linalg.norm(train_init, axis=1)
        r_median = float(np.median(r))
        T_used = float(alpha * r_median / velocity_scale)
        return T_used, r_median

    # fixed numeric T
    T_used = float(T_arg)
    return T_used, None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, required=True, help="Path to .pt file, e.g. ../Data/so3_dataset.pt")
    p.add_argument("--outdir", type=str, required=True)

    # model
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--num-layers", type=int, default=8)

    # optimizer
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)

    # training
    p.add_argument("--num-epochs", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--early-stop", type=int, default=0)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", type=str, default="cuda")

    # lr scheduler
    p.add_argument("--scheduler-patience", type=int, default=100)
    p.add_argument("--scheduler-factor", type=float, default=0.5)

    # flow params (must match train_flow_matching_model signature)
    # T can be a float (as string) or "auto"
    p.add_argument("--T", type=str, default="2.0")
    p.add_argument("--alpha", type=float, default=0.5, help="Used only when --T auto")
    p.add_argument("--num-timesteps", type=int, default=30)
    p.add_argument("--velocity-scale", type=float, default=0.5)
    p.add_argument("--velocity-cov-scale", type=float, default=1.0)

    args = p.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    loaded = torch.load(args.dataset, map_location="cpu", weights_only=False)

    X_train = torch.tensor(loaded["X_train"], dtype=torch.float32)
    X_val = torch.tensor(loaded["X_val"], dtype=torch.float32)

    X_train = flatten_if_needed(X_train)
    X_val = flatten_if_needed(X_val)

    train_init = X_train.numpy()
    val_init = X_val.numpy()

    # Choose T
    T_used, r_median = parse_T(args.T, train_init, args.velocity_scale, args.alpha)

    input_dim = train_init.shape[1]
    device = torch.device(args.device)

    model = FlowVelocityNet(input_dim=input_dim, hidden_dim=args.hidden_dim, num_layers=args.num_layers)

    model, logs = train_flow_matching_model(
        model,
        train_init=train_init,
        val_init=val_init,
        device=device,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        early_stop=args.early_stop,
        seed=args.seed,
        verbose=True,
        T=T_used,
        num_timesteps=args.num_timesteps,
        velocity_scale=args.velocity_scale,
        velocity_cov_scale=args.velocity_cov_scale,
        scheduler_patience=args.scheduler_patience,
        scheduler_factor=args.scheduler_factor,
    )

    # Loss curves (small + robust)
    train_losses = torch.tensor(logs.get("train_losses", []), dtype=torch.float32)
    val_losses = torch.tensor(logs.get("val_losses", []), dtype=torch.float32)

    atomic_torch_save(
        {"train_losses": train_losses, "val_losses": val_losses},
        os.path.join(args.outdir, "losses.pt"),
    )

    # Store exact run metadata for selection / reproducibility
    hparams = vars(args).copy()
    hparams["T_used"] = T_used
    hparams["r_median"] = r_median

    meta = {
        "dataset": args.dataset,
        "train_shape": tuple(X_train.shape),
        "val_shape": tuple(X_val.shape),
        "hparams": hparams,
        "best_val_loss": logs.get("best_val_loss", None),
        "best_epoch": logs.get("best_epoch", None),
        "epochs_ran": int(train_losses.numel()),
    }
    atomic_torch_save(meta, os.path.join(args.outdir, "meta.pt"))

    # Model checkpoint (include enough info to reconstruct)
    ckpt = {
        "state_dict": model.state_dict(),
        "input_dim": input_dim,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "hparams": hparams,
        "best_val_loss": logs.get("best_val_loss", None),
        "best_epoch": logs.get("best_epoch", None),
    }
    atomic_torch_save(ckpt, os.path.join(args.outdir, "model.pt"))

    print(f"T_used = {T_used:.6g}" + (f"  (r_median={r_median:.6g}, alpha={args.alpha})" if r_median is not None else ""))
    print(f"Saved: {os.path.join(args.outdir, 'model.pt')}")
    print(f"Saved: {os.path.join(args.outdir, 'losses.pt')}")
    print(f"Saved: {os.path.join(args.outdir, 'meta.pt')}")


if __name__ == "__main__":
    main()
