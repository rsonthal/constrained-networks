#!/usr/bin/env python3
"""
Train RegularFeedForward / ProjectedFeedForward / ExponentialFeedForward on a dataset
and save the *best-val* checkpoint.

Place in: <repo>/src/Models/

IMPORTANT (per your note):
- The constructor arg named `d_hid` is misnamed; it is actually the OUTPUT dimension.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from projected import ProjectedTransformer
from exponential import ExponentialTransformer
from regular import RegularTransformer


# -----------------------------
# Dataset configuration
# -----------------------------

DATASET_TO_FILE = {
    "sphere": "sphere_dataset.pt",
    "disk": "disk_dataset.pt",
    "so3": "so3_dataset.pt",
    "cs": "cs_dataset.pt",
    "protein": "protein_dataset.pt",
}

# Fixed input dims (as you specified)
INPUT_DIMS = {
    "sphere": 3,
    "disk": 2,
    "so3": 9,
    "cs": 9,        # lives on SO(3)
    "protein": 16,  # lives on SE(3)
}

# Output dims:
# - regular/projected: out_dim == in_dim
# - exponential: manifold tangent dims (as you specified)
EXP_OUTPUT_DIMS = {
    "sphere": 3,
    "so3": 3,
    "cs": 3,
    "protein": 6,
}


# -----------------------------
# Utils
# -----------------------------

def _as_float_tensor(x: Any) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.detach().clone().to(dtype=torch.float32)
    return torch.tensor(x, dtype=torch.float32)

def infer_split_tensors(loaded: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    X_train = _as_float_tensor(loaded["X_train"])
    Y_train = _as_float_tensor(loaded["Y_train"])
    X_val   = _as_float_tensor(loaded["X_val"])
    Y_val   = _as_float_tensor(loaded["Y_val"])
    X_test  = _as_float_tensor(loaded["X_test"])
    Y_test  = _as_float_tensor(loaded["Y_test"])

    for name, t in [("X_train", X_train), ("Y_train", Y_train), ("X_val", X_val), ("Y_val", Y_val), ("X_test", X_test), ("Y_test", Y_test)]:
        if not torch.isfinite(t).all():
            raise ValueError(f"Found NaN/Inf in {name}")

    return X_train, Y_train, X_val, Y_val, X_test, Y_test

def get_projection_fn(dataset: str) -> Callable:
    """
    projections available: sphere, so3, se3, disk
    Note: cs uses so3; protein uses se3.
    """
    import projections
    if dataset == "sphere":
        return projections.sphere
    if dataset in ("so3", "cs"):
        return projections.so3
    if dataset == "protein":
        return projections.se3
    if dataset == "disk":
        return projections.disk
    raise ValueError(f"Unknown dataset for projection: {dataset}")

def get_exponential_fn(dataset: str) -> Callable:
    """
    exponentials available: sphere, so3, se3
    Note: cs uses so3; protein uses se3.
    """
    import exponentials
    if dataset == "sphere":
        return exponentials.sphere
    if dataset in ("so3", "cs"):
        return exponentials.so3
    if dataset == "protein":
        return exponentials.se3
    raise ValueError(f"Dataset {dataset} is not configured for exponential models.")

def get_manifold_fn(dataset: str) -> str:
    """
    Returns manifold type for probabilistic models.
    Note: cs uses so3; protein uses se3.
    """
    if dataset == "sphere":
        return "sphere"
    if dataset in ("so3", "cs"):
        return "so3"
    if dataset == "protein":
        return "se3"
    if dataset == "disk":
        return "disk"
    raise ValueError(f"Unknown dataset for probabilistic model: {dataset}")

def get_flow_matching_projection_fn(dataset: str, outputsflow_dir: Optional[str] = None) -> Callable:
    """
    Loads the best flow matching model from outputsflow/{dataset}/BEST/ and returns
    a projection function that uses it.
    
    Args:
        dataset: Dataset name (sphere, so3, protein, etc.)
        outputsflow_dir: Directory containing the flow matching models (default: "outputsflow")
        
    Returns:
        A callable projection function that takes a tensor and returns projected tensor
    """
    from flow_matching import FlowVelocityNet, flow_matching_projection
    
    if outputsflow_dir is None:
        outputsflow_dir = "outputsflow"
    
    # Path to the BEST folder for this dataset
    best_path = Path(outputsflow_dir) / f"{dataset}_dataset" / "BEST"
    model_path = best_path / "model.pt"
    meta_path = best_path / "meta.pt"
    
    if not model_path.exists():
        raise FileNotFoundError(
            f"Flow matching model not found at {model_path}. "
            f"Expected path: {outputsflow_dir}/{dataset}_dataset/BEST/model.pt"
        )
    
    # Load the model checkpoint
    checkpoint = torch.load(str(model_path), map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    
    # Try to get input_dim from meta.pt if available, otherwise infer from dataset
    input_dim = INPUT_DIMS[dataset]
    if meta_path.exists():
        try:
            meta = torch.load(str(meta_path), map_location="cpu", weights_only=False)
            hparams = meta.get("hparams", {})
            # Try to get input_dim from hparams or infer from model structure
            if "input_dim" in hparams:
                input_dim = hparams["input_dim"]
            elif "train_shape" in meta:
                # train_shape is (N, input_dim, ...) or (N, input_dim)
                train_shape = meta["train_shape"]
                if isinstance(train_shape, (list, tuple)) and len(train_shape) >= 2:
                    input_dim = train_shape[1]
        except Exception as e:
            print(f"Warning: Could not load meta.pt from {meta_path}: {e}")
            print(f"Using default input_dim={input_dim} for dataset {dataset}")
    
    # Create FlowVelocityNet with standard architecture
    # Default parameters match typical flow matching training
    flow_model = FlowVelocityNet(
        input_dim=input_dim,
        hidden_dim=256,
        num_layers=8,
        dropout=0.1
    )
    
    # Load the state dict
    flow_model.load_state_dict(state_dict)
    flow_model.eval()
    
    # Create a closure that captures the flow_model
    # The flow_model will be moved to the correct device when the projection is first called
    def proj_fn(x: torch.Tensor) -> torch.Tensor:
        """Projection function using the loaded flow matching model."""
        if dataset == "cs":
            S, B, F = x.shape
            x_flat = x.reshape(B * S, F)  # [batch*seq, features]
            
            # Move flow_model to the same device as input x
            flow_model_device = flow_model.to(x.device)
            x_proj_flat = flow_matching_projection(x_flat, flow_model_device, T=2.0, num_steps=40, differentiable=True)
            
            x_proj = x_proj_flat.reshape(S, B, F)  # [seq, batch, features]
            return x_proj
        else:
            flow_model_device = flow_model.to(x.device)
            return flow_matching_projection(x, flow_model_device, T=2.0, num_steps=40, differentiable=True)
    
    print(f"Loaded flow matching model from {model_path} (input_dim={input_dim})")
    return proj_fn


# -----------------------------
# Metadata
# -----------------------------

@dataclass
class HParams:
    model_type: str
    dataset: str
    depth: int
    d_out: int
    dropout: float
    residual: bool
    dt: float
    lr: float
    weight_decay: float
    batch_size: int
    num_epochs: int
    eval_every: int
    seed: int
    device: str
    scheduler_patience_epochs: int
    scheduler_factor: float
    grad_clip: float
    data_dir: str
    outdir: str
    use_internal: bool


# -----------------------------
# Model build
# -----------------------------

def build_model(
    model_type: str,
    dataset: str,
    *,
    depth: int,
    d_out: int,          # <-- passed into your misnamed `d_hid`
    dropout: float,
    residual: bool,
    dt: float,
    use_internal: bool,
    outputsflow_dir: Optional[str] = None,
    nhead: int = 3,     
    d_hid: int = 2048,  
) -> nn.Module:
    """
    Matches your actual constructors, with the correction:
      - arg named `d_hid` is actually OUTPUT DIM.

    Therefore we call:
      RegularFeedForward(d_model=in_dim, d_hid=out_dim, nlayers=depth, ...)
      ProjectedFeedForward(d_model=in_dim, d_hid=out_dim, nlayers=depth, ...)
      ExponentialFeedForward(d_model=in_dim, d_hid=out_dim, nlayers=depth, ...)
    """
    from regular import RegularFeedForward
    from projected import ProjectedFeedForward
    from exponential import ExponentialFeedForward

    model_type = model_type.lower()
    dataset = dataset.lower()

    d_in = INPUT_DIMS[dataset]

    if dataset == "cs":
        if model_type == "regular":
            return RegularTransformer(
                input_dim=d_in,
                nhead=nhead,
                d_hid=d_hid,
                nlayers=depth,
                dropout=dropout,
                dt=dt,
            )
    
        if model_type == "projected":
            proj_fn = get_projection_fn(dataset)
            return ProjectedTransformer(
                input_dim=d_in,
                nhead=nhead,
                d_hid=d_hid,
                nlayers=depth,
                dropout=dropout,
                proj_func=proj_fn,
                use_internal_projection=use_internal,
                dt=dt,
            )
    
        if model_type == "exponential":
            exp_fn = get_exponential_fn(dataset)
            return ExponentialTransformer(
                input_dim=d_in,
                nhead=nhead,
                d_hid=d_hid,
                nlayers=depth,
                dropout=dropout,
                internal_exp_func=exp_fn if use_internal else None,
                end_exp_func=exp_fn,
                use_internal_exp=use_internal,
                m=d_out,  # tangent space dimension
                dt=dt,
            )
    
        if model_type == "probabilistic":
            # Probabilistic model: RegularTransformer + Linear, but we need a wrapper
            # to extract the last timestep for sequential data
            return nn.Sequential(RegularTransformer(
                input_dim=d_in,
                nhead=nhead,
                d_hid=d_hid,
                nlayers=depth,
                dropout=dropout,
                dt=dt), nn.Linear(d_in,100))

        if model_type == "flow_matching":
            # Flow matching model: ProjectedTransformer with flow matching projection
            proj_fn = get_flow_matching_projection_fn(dataset, outputsflow_dir=outputsflow_dir)
            return ProjectedTransformer(
                input_dim=d_in,
                nhead=nhead,
                d_hid=d_hid,
                nlayers=depth,
                dropout=dropout,
                proj_func=proj_fn,
                use_internal_projection=use_internal,
                dt=dt,
            )
        

    if model_type == "regular":
        return RegularFeedForward(
            d_model=d_in,
            d_hid=d_out,       # misnamed: output dim
            nlayers=depth,
            dropout=dropout,
            residual=residual,
            dt=dt,
        )

    if model_type == "projected":
        proj_fn = get_projection_fn(dataset)
        return ProjectedFeedForward(
            d_model=d_in,
            d_hid=d_out,       # misnamed: output dim
            nlayers=depth,
            dropout=dropout,
            proj_func=proj_fn,
            use_internal_projection=use_internal,
            residual=residual,
            dt=dt,
        )

    if model_type == "exponential":
        if dataset == "disk":
            raise ValueError("ExponentialFeedForward not intended for disk.")
        exp_fn = get_exponential_fn(dataset)
        print(exp_fn, d_in, d_out, depth, dropout, use_internal)
        return ExponentialFeedForward(
            d_model=d_in,
            d_hid=d_out,       # misnamed: output dim (tangent dim)
            nlayers=depth,
            dropout=dropout,
            exp_func=exp_fn,
            use_internal_exponential=use_internal,
            dt=dt,
        )

    if model_type == "probabilistic":
        # Probabilistic model: nn.Sequential(RegularFeedForward(...), nn.Linear(...))
        # Note: d_out here is the hidden dimension for RegularFeedForward
        # The final Linear layer maps to num_anchors (set separately)
        return nn.Sequential(
            RegularFeedForward(
                d_model=d_in,
                d_hid=d_out,   # hidden/output dim for RegularFeedForward
                nlayers=depth,
                dropout=dropout,
                residual=residual,
                dt=dt,
            ),
            # Note: num_anchors will be set in main() after anchors are created
            # This is a placeholder - will be replaced
            nn.Linear(d_out, 100)  # default, will be updated
        )

    if model_type == "flow_matching":
        # Flow matching model: ProjectedFeedForward with flow matching projection
        proj_fn = get_flow_matching_projection_fn(dataset, outputsflow_dir)
        return ProjectedFeedForward(
            d_model=d_in,
            d_hid=d_out,       # misnamed: output dim
            nlayers=depth,
            dropout=dropout,
            proj_func=proj_fn,
            use_internal_projection=use_internal,
            residual=residual,
            dt=dt,
        )

    raise ValueError(f"Unknown model_type: {model_type}")


# -----------------------------
# Train (no early stopping; ReduceLROnPlateau)
# -----------------------------

@torch.no_grad()
def eval_mse(model: nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    total = 0.0
    n = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        loss = F.mse_loss(pred, y, reduction="sum")
        total += float(loss.item())
        n += int(y.numel())
    return total / max(n, 1)


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    lr: float,
    weight_decay: float,
    num_epochs: int,
    eval_every: int,
    device: str,
    grad_clip: Optional[float],
    scheduler_patience_epochs: int,
    scheduler_factor: float,
    verbose: bool,
    loss_fn: Optional[Callable] = None,
    eval_fn: Optional[Callable] = None,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Generic training function that works for both regular and probabilistic models.
    
    Args:
        loss_fn: Optional loss function. If None, uses F.mse_loss(pred, y).
        eval_fn: Optional evaluation function. If None, uses eval_mse.
    """
    # Default to MSE loss and eval for regular models
    if loss_fn is None:
        def loss_fn(pred, target):
            return F.mse_loss(pred, target)
    
    if eval_fn is None:
        eval_fn = eval_mse
    
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Scheduler steps on evaluation events. Convert patience from epochs -> eval steps.
    patience_evals = max(1, scheduler_patience_epochs // max(1, eval_every))

    # NOTE: Some PyTorch versions do NOT accept `verbose=` here.
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=scheduler_factor, patience=patience_evals
    )

    best_val = float("inf")
    best_epoch = -1
    best_state = None

    # Use generic loss names that work for both
    logs: Dict[str, Any] = {
        "train_loss": [],
        "val_loss": [],
        "lr": [],
        "eval_epochs": [],
        "best_val": None,
        "best_epoch": None,
        "patience_evals": patience_evals,
    }

    for epoch in range(1, num_epochs + 1):
        model.train()
        for batch in train_loader:
            x, target = batch
            x = x.to(device)
            target = target.to(device)

            opt.zero_grad(set_to_none=True)
            output = model(x)
            loss = loss_fn(output, target)
            loss.backward()

            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            opt.step()

        if epoch % eval_every != 0 and epoch != 1 and epoch != num_epochs:
            continue

        train_loss = eval_fn(model, train_loader, device)
        val_loss = eval_fn(model, val_loader, device)
        sched.step(val_loss)

        logs["train_loss"].append(train_loss)
        logs["val_loss"].append(val_loss)
        logs["lr"].append(float(opt.param_groups[0]["lr"]))
        logs["eval_epochs"].append(epoch)

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if verbose:
            print(
                f"Epoch {epoch:6d} | train_loss={train_loss:.6e} | "
                f"val_loss={val_loss:.6e} | lr={opt.param_groups[0]['lr']:.3e}"
            )

    logs["best_val"] = best_val
    logs["best_epoch"] = best_epoch

    if best_state is None:
        raise RuntimeError("No best checkpoint captured (unexpected).")

    model.load_state_dict(best_state)
    return model, logs

def normalize_se3_translation(X16: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    """
    X16: [N,16] row-major flatten of 4x4.
    Normalizes translation column (rows 0..2, col 3) by tau.
    """
    G = X16.view(-1, 4, 4).clone()
    G[:, :3, 3] = G[:, :3, 3] / tau
    return G.view(-1, 16)

# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, required=True, choices=["regular", "projected", "exponential", "probabilistic", "flow_matching"])
    parser.add_argument("--dataset", type=str, required=True, choices=list(DATASET_TO_FILE.keys()))
    parser.add_argument("--depth", type=int, required=True)
    parser.add_argument("--nhead", type=int, default=3, help="Number of attention heads (only used for CS dataset transformers; must divide input_dim; default=3 for CS with dim=9)")
    parser.add_argument("--d_hid", type=int, default=2048, help="Feedforward dimension in transformer (only used for CS dataset transformers; ignored for feedforward models)")

    # model hparams
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--residual", action="store_true", help="Enable residual connections (default True in your classes).")
    parser.add_argument("--no_residual", action="store_true", help="Disable residual connections.")
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--use_internal", action="store_true", help="Use internal projection/exponential if supported.")

    # training hparams (defaults per your spec)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--num_epochs", type=int, default=100_000)

    # Evaluate/scheduler scaling
    parser.add_argument("--eval_every", type=int, default=100, help="Compute full train/val MSE + step scheduler every N epochs.")
    parser.add_argument("--scheduler_patience_epochs", type=int, default=10_000, help="Patience in *epochs* (converted to eval steps).")
    parser.add_argument("--scheduler_factor", type=float, default=0.5)

    parser.add_argument("--grad_clip", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--data_dir", type=str, default=str(Path(__file__).resolve().parents[1] / "Data"))
    parser.add_argument("--outdir", type=str, default="outputs")
    parser.add_argument("--outputsflow_dir", type=str, default="outputsflow", help="Directory containing flow matching BEST models")
    parser.add_argument("--train_limit", type=int, default=0, help="If >0, cap train set to this many examples (debug).")
    parser.add_argument("--no_verbose", action="store_true")

    parser.add_argument("--no_protein_translation_norm", action="store_true", help="Disable protein SE(3) translation rescaling by train-set std.")
    parser.add_argument("--num_anchors", type=int, default=100, help="Number of anchors for probabilistic models.")

    args = parser.parse_args()
    verbose = not args.no_verbose

    residual = True
    if args.no_residual:
        residual = False
    elif args.residual:
        residual = True

    # Ensure local imports work even if launched from elsewhere.
    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    # Load dataset
    data_path = Path(args.data_dir) / DATASET_TO_FILE[args.dataset]
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {data_path}")

    print(data_path)

    loaded = torch.load(str(data_path), map_location="cpu", weights_only=False)
    X_train, Y_train, X_val, Y_val, X_test, Y_test = infer_split_tensors(loaded)

    print(X_train.shape)

    if args.train_limit and args.train_limit > 0:
        X_train = X_train[: args.train_limit]
        Y_train = Y_train[: args.train_limit]

    # Sanity checks against your stated dims
    in_dim = INPUT_DIMS[args.dataset]
    if args.dataset == "cs":
        if int(X_train.shape[2]) != in_dim:
            raise ValueError(f"Expected X feature dim {in_dim} for dataset={args.dataset}, got {tuple(X_train.shape)}")
    else:
        if int(X_train.shape[1]) != in_dim:
            raise ValueError(f"Expected X dim {in_dim} for dataset={args.dataset}, got {tuple(X_train.shape)}")

    if args.model_type in ("regular", "projected", "flow_matching", "probabilistic"):
        out_dim = in_dim
    else:
        if args.dataset == "disk":
            raise ValueError("ExponentialFeedForward not intended for disk.")
        out_dim = EXP_OUTPUT_DIMS[args.dataset]

    protein_tau = None
    if args.dataset == "protein" and not args.no_protein_translation_norm:
        # Compute tau from train translations only
        t = X_train.view(-1, 4, 4)[:, :3, 3]
        tau = t.std().clamp_min(1e-8)
        protein_tau = float(tau.item())
    
        X_train = normalize_se3_translation(X_train, tau)
        Y_train = normalize_se3_translation(Y_train, tau)
        X_val   = normalize_se3_translation(X_val,   tau)
        Y_val   = normalize_se3_translation(Y_val,   tau)
        X_test  = normalize_se3_translation(X_test,  tau)
        Y_test  = normalize_se3_translation(Y_test,  tau)
    
        print(f"[protein] translation tau = {protein_tau:.6g}")

    # Build model (note: d_out is passed into the misnamed `d_hid`)
    model = build_model(
        model_type=args.model_type,
        dataset=args.dataset,
        depth=args.depth,
        d_out=out_dim,
        dropout=args.dropout,
        residual=residual,
        dt=args.dt,
        use_internal=args.use_internal,
        outputsflow_dir=args.outputsflow_dir,
        nhead=args.nhead,
        d_hid=args.d_hid,
    )

    # For probabilistic models, create anchors and update final Linear layer
    anchors = None
    anchors_tensor = None
    if args.model_type == "probabilistic":
        from probabilistic import create_anchors, create_label_dataset
        
        # Create anchors from training data (after normalization if applicable)
        Y_train_np = Y_train.numpy()
        manifold = get_manifold_fn(args.dataset)
        anchors = create_anchors(args.num_anchors, training_data=Y_train_np, manifold=manifold, dataset=args.dataset)
        anchors_tensor = torch.tensor(anchors, dtype=torch.float32)
        
        # Update the final Linear layer to output num_anchors
        # Replace the placeholder Linear layer
        if isinstance(model, nn.Sequential):
            # Remove the last layer (placeholder Linear)
            model_list = list(model.children())
            model_list[-1] = nn.Linear(out_dim, args.num_anchors)
            model = nn.Sequential(*model_list)
        
        print(f"Created {args.num_anchors} anchors with shape {anchors.shape}")

    # Dataloaders
    if args.model_type == "probabilistic":
        # Create label datasets using Voronoi partitioning
        X_train_np = X_train.numpy()
        Y_train_np = Y_train.numpy()
        train_loader, _ = create_label_dataset(
            X_train_np, Y_train_np, anchors, 
            batch_size=args.batch_size, metric='euclidean', dataset=args.dataset
        )
        
        # For validation, train_probabilistic_model expects (x, y) pairs where y is the actual target
        # It uses predict() internally to convert logits to predictions and compares with y
        val_ds = TensorDataset(X_val, Y_val)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)
    else:
        train_ds = TensorDataset(X_train, Y_train)
        val_ds   = TensorDataset(X_val, Y_val)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
        val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, drop_last=False)

    # Train
    if args.model_type == "probabilistic":
        from probabilistic import train_probabilistic_model
        # train_probabilistic_model uses different parameter names and structure
        # Map our parameters to its signature
        model, logs = train_probabilistic_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            anchors=anchors_tensor,
            lr=args.lr,
            num_epochs=args.num_epochs,
            device=device,
            weight_decay=args.weight_decay,
            grad_clip=args.grad_clip if args.grad_clip > 0 else 1.0,
            scheduler_patience=args.scheduler_patience_epochs,
            scheduler_factor=args.scheduler_factor,
            early_stop=0,  # Disable early stopping to match train() behavior
            verbose=verbose,
        )
        # Convert logs format to match expected format
        # train_probabilistic_model returns: train_losses, val_losses, best_val_loss, best_epoch, lrs
        # Adapt to match the format expected by the rest of the code
        original_logs = logs
        logs = {
            "train_loss": original_logs.get("train_losses", []),
            "val_loss": original_logs.get("val_losses", []),
            "lr": original_logs.get("lrs", []),
            "eval_epochs": list(range(1, len(original_logs.get("train_losses", [])) + 1)),
            "best_val": original_logs.get("best_val_loss"),
            "best_epoch": original_logs.get("best_epoch"),
            "patience_evals": None,
        }
    else:
        model, logs = train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            lr=args.lr,
            weight_decay=args.weight_decay,
            num_epochs=args.num_epochs,
            eval_every=args.eval_every,
            device=device,
            grad_clip=args.grad_clip if args.grad_clip > 0 else None,
            scheduler_patience_epochs=args.scheduler_patience_epochs,
            scheduler_factor=args.scheduler_factor,
            verbose=verbose,
        )

    # Save best-val model (already loaded)
    if args.model_type == "probabilistic":
        run_dir = (
            Path(args.outdir)
            / args.dataset
            / args.model_type
            / f"depth{args.depth}"
            / f"out{out_dim}"
            / f"anchors{args.num_anchors}"
            / f"lr{args.lr:g}_wd{args.weight_decay:g}"
            / f"seed{args.seed}"
        )
    else:
        run_dir = (
            Path(args.outdir)
            / args.dataset
            / args.model_type
            / f"depth{args.depth}"
            / f"out{out_dim}"
            / f"lr{args.lr:g}_wd{args.weight_decay:g}"
            / f"seed{args.seed}"
        )
    run_dir.mkdir(parents=True, exist_ok=True)

    torch.save({"state_dict": model.state_dict()}, str(run_dir / "model.pt"))

    hps = HParams(
        model_type=args.model_type,
        dataset=str(data_path),
        depth=args.depth,
        d_out=out_dim,
        dropout=args.dropout,
        residual=residual,
        dt=args.dt,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        eval_every=args.eval_every,
        seed=args.seed,
        device=device,
        scheduler_patience_epochs=args.scheduler_patience_epochs,
        scheduler_factor=args.scheduler_factor,
        grad_clip=args.grad_clip,
        data_dir=str(args.data_dir),
        outdir=str(run_dir),
        use_internal=bool(args.use_internal),
    )

    meta = {
        "dataset": str(data_path),
        "train_shape": tuple(X_train.shape),
        "val_shape": tuple(X_val.shape),
        "test_shape": tuple(X_test.shape),
        "hparams": asdict(hps),
        "logs": logs,
    }
    
    # Add probabilistic-specific metadata
    if args.model_type == "probabilistic":
        meta["num_anchors"] = args.num_anchors
        if anchors_tensor is not None:
            meta["anchors_shape"] = tuple(anchors_tensor.shape)
            meta["anchors"] = anchors_tensor.cpu().numpy()

    torch.save(meta, str(run_dir / "meta.pt"))

    with open(run_dir / "meta.json", "w") as f:
        json.dump(
            {
        "dataset": meta["dataset"],
        "train_shape": meta["train_shape"],
        "val_shape": meta["val_shape"],
        "test_shape": meta["test_shape"],
        "hparams": meta["hparams"],
        "best_epoch": logs.get("best_epoch"),
        "best_val": logs.get("best_val"),
        "patience_evals": logs.get("patience_evals"),
            },
            f,
            indent=2,
        )

    best_val = logs.get("best_val", logs.get("val_loss", 0.0))
    print(f"\nSaved best checkpoint (epoch={logs['best_epoch']}, best_val={best_val:.6e}) to: {run_dir}")

if __name__ == "__main__":
    main()
