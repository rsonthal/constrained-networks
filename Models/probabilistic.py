import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation

def create_label_dataset(X_train, Y_train, Y_particles, batch_size=32, metric='euclidean', dataset=None):
    """
    Generates the classification labels for the Probabilistic Transformer.
    
    Args:
        X_train (np.array): Input features (T, input_dim) or (T, seq_len, input_dim) for sequential
        Y_train (np.array): Target outputs (T, output_dim) or (T, seq_len, output_dim) for sequential
        Y_particles (np.array): Anchor points (N, output_dim) - "The Representatives"
        batch_size (int): Batch size for the DataLoader
        metric (str): Distance metric ('euclidean', 'cosine', etc.)
                      If 'euclidean', it corresponds to ||y - Y|| in the paper.
        dataset (str, optional): Dataset name (e.g., 'cs' for sequential data).

    Returns:
        dataloader (DataLoader): PyTorch DataLoader yielding (X_batch, L_indices_batch)
        L_indices (torch.Tensor): The raw indices of the closest anchors (T,)
    """
    
    if dataset == "cs":
        Y_train = Y_train.reshape(-1, Y_train.shape[-1])  # [T*seq_len, output_dim]
        X_train = X_train.reshape(-1, X_train.shape[-1])  # [T*seq_len, input_dim]
    
    print(f"--- Generating Labels (Voronoi Partitioning) ---")
    print(f"    Training Data (T): {Y_train.shape[0]}")
    print(f"    Anchors/Particles (N): {Y_particles.shape[0]}")

    # --------------------------------------------------------------------------
    #    Vectorized Distance Calculation
    #    Instead of nested loops, we use cdist to compute the (T x N) matrix
    #    Distance[t, n] = dist(Y_train[t], Y_particles[n])
    # --------------------------------------------------------------------------
    # Note: If using specific Riemannian metrics, you would replace 'euclidean' 
    # with a custom vectorized function.
    Distances = cdist(Y_train, Y_particles, metric=metric)

    # --------------------------------------------------------------------------
    #    Find Nearest Anchor (Argmin)
    #    This implements the indicator function from Algorithm 1:
    #    n* = argmin_m ||y_t - Y_m||
    # --------------------------------------------------------------------------
    closest_particle_indices = np.argmin(Distances, axis=1)

    # --------------------------------------------------------------------------
    #    Create Tensors
    # --------------------------------------------------------------------------
    # X: Inputs (Standard float tensor)
    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    
    # L: Labels (Long tensor for CrossEntropy, or convert to Float one-hot for MSE)
    L_tensor = torch.tensor(closest_particle_indices, dtype=torch.long)
    
    # Create the dataset and loader
    dataset = TensorDataset(X_tensor, L_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print(f"    Done. Created DataLoader with {len(dataloader)} batches.")
    
    return dataloader, L_tensor

def probabilistic_transformer_loss(logits, target_indices):
    """
    Computes the MSE loss using built-in torch functions.
    Matches Algorithm 1: Sum of squares over particles, Mean over batch.
    """
    # 1. Logits -> Probabilities
    probs = F.softmax(logits, dim=1)
    
    # 2. Indices -> One-Hot Floats
    num_particles = logits.shape[1]
    L_one_hot = F.one_hot(target_indices, num_classes=num_particles).float()
    
    # 3. Compute MSE
    #    We use reduction='sum' to get the total squared error across all Batch x Particles
    #    Then divide by batch_size to get the average loss per example.
    #    (Standard reduction='mean' would divide by Batch * Num_Particles, which is too small)
    total_sse = F.mse_loss(probs, L_one_hot, reduction='sum')
    
    return total_sse / logits.shape[0]

def predict(network, X_test, anchors, method='expectation'):
    """
    Performs inference using the Probabilistic Transformer.
    
    Args:
        network (nn.Module): Trained network outputting logits.
        X_test (torch.Tensor): Input data (Batch, Input_Dim) or (Batch, Seq_Len, Input_Dim) for sequential.
        anchors (torch.Tensor): The Y_n particles (N, Output_Dim).
        method (str): 'expectation' (Weighted Sum) or 'argmax' (Nearest Anchor).
    
    Returns:
        y_pred (torch.Tensor): Predictions (Batch, Output_Dim).
    """
    network.eval()
    with torch.no_grad():
        is_transformer = (isinstance(network, nn.Sequential) and 
                         len(network) > 0 and 
                         hasattr(network[0], 'transformer_encoder'))
        
        if X_test.dim() == 2 and is_transformer:
            X_test = X_test.unsqueeze(1)
        # 1. Forward Pass -> Logits
        logits = network(X_test)
        
        # 2. Logits -> Probabilities (Weights)
        probs = F.softmax(logits, dim=-1) # Shape: (Batch, N)
        
        # --- Method A: Weighted Average (Expectation) ---
        # y = p1*Y1 + p2*Y2 + ...
        # Matrix Mult: (Batch, N) x (N, Out_Dim) -> (Batch, Out_Dim)
        if method == 'expectation':
            y_pred = torch.matmul(probs, anchors)
            
        # --- Method B: Argmax (Snap to Grid) ---
        # Pick the index with highest probability, return that anchor
        elif method == 'argmax':
            best_indices = torch.argmax(probs, dim=1) # Shape: (Batch,)
            y_pred = anchors[best_indices]            # Shape: (Batch, Out_Dim)
            
        # --- Method C: Fréchet Mean (Concept) ---
        # (Requires iterative optimization or geomstats library)
        elif method == 'frechet':
            raise NotImplementedError(
                "Fréchet Mean requires a manifold optimization solver (e.g., geomstats)."
            )
            
    return y_pred

def sample_sphere(num_samples, dim=3):
    """
    Samples uniformly from the unit sphere S^(dim-1) embedded in R^dim.
    (e.g., dim=3 is the standard 2-sphere).
    
    Method: Sample from Standard Normal, then normalize.
    
    Args:
        num_samples (int): Number of anchors.
        dim (int): Dimension of the embedding space (3 for standard sphere).
        
    Returns:
        anchors (np.array): Shape (num_samples, dim)
    """
    # 1. Sample from N(0, 1)
    vecs = np.random.randn(num_samples, dim)
    
    # 2. Normalize to unit length
    # Avoid division by zero just in case
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    anchors = vecs / (norms + 1e-8)
    
    return anchors

def sample_so3(num_samples):
    """
    Samples uniformly from SO(3) (3x3 Rotation Matrices).
    
    Method: Uses Scipy's implementation of the subgroup algorithm 
            (generating random quaternions uniformly on S^3).
    
    Returns:
        anchors (np.array): Shape (num_samples, 3, 3)
    """
    # Scipy handles the correct uniform measure (Haar measure)
    rotations = Rotation.random(num_samples)
    anchors = rotations.as_matrix()
    
    return anchors

def sample_se3(num_samples, translation_bounds=(-1.0, 1.0)):
    """
    Samples from SE(3) (Rigid Body Transformations).
    
    Note: SE(3) is non-compact (infinite translation), so you MUST 
    specify bounds for the translation component.
    
    Args:
        num_samples (int): Number of anchors.
        translation_bounds (tuple): (min, max) for the XYZ translation box.
        
    Returns:
        anchors (np.array): Shape (num_samples, 4, 4) homogeneous matrices.
    """
    # 1. Sample Rotations (Upper left 3x3)
    R = sample_so3(num_samples) # (N, 3, 3)
    
    # 2. Sample Translations (Right column 3x1)
    #    Uniform sampling within the bounding box
    t_min, t_max = translation_bounds
    t = np.random.uniform(t_min, t_max, size=(num_samples, 3, 1))
    
    # 3. Construct 4x4 Matrices
    #    Start with Identity matrices
    anchors = np.eye(4).reshape(1, 4, 4).repeat(num_samples, axis=0)
    
    #    Inject Rotation and Translation
    anchors[:, :3, :3] = R
    anchors[:, :3, 3:] = t
    
    return anchors

def create_anchors(num_anchors, training_data=None, manifold='sphere', se3_bounds=(-1.0, 1.0), dataset=None):
    """
    Creates anchor points (particles) for the Probabilistic Transformer.
    
    Args:
        num_anchors (int): Number of anchors (N).
        training_data (np.array, optional): If provided, samples from this data (Subset Strategy).
        manifold (str): 'sphere', 'so3', or 'se3'. Used if training_data is None.
        se3_bounds (tuple): Bounds for translation sampling if manifold is 'se3'.
        dataset (str, optional): Dataset name (e.g., 'cs' for sequential data).

    Returns:
        anchors (np.array): Flattened float32 array of shape (N, Output_Dim).
                            (e.g., N x 3, N x 9, or N x 16)
    """
    
    # Strategy A: Subset of Real Training Data (Recommended)
    if training_data is not None:
        print(f"--- Creating Anchors: Sampling {num_anchors} from Training Data ---")
        if dataset == "cs":
            training_data = training_data.reshape(-1, training_data.shape[-1])  # [N*seq_len, features]
        total_samples = training_data.shape[0]
        if num_anchors > total_samples:
             raise ValueError(f"Requested {num_anchors} anchors, but only {total_samples} available.")
        
        indices = np.random.choice(total_samples, size=num_anchors, replace=False)
        anchors = training_data[indices]
        
    # Strategy B: Synthetic Manifold Sampling
    else:
        print(f"--- Creating Anchors: Synthetic Sampling on {manifold} ---")
        if manifold == 'sphere':
            # Default to 3D sphere (S^2)
            anchors = sample_sphere(num_anchors, dim=3)
            
        elif manifold == 'so3':
            raw_anchors = sample_so3(num_anchors)
            anchors = raw_anchors.reshape(-1, 9)
            
        elif manifold == 'se3':
            raw_anchors = sample_se3(num_anchors, translation_bounds=se3_bounds)
            anchors = raw_anchors.reshape(-1, 16)
            
        else:
            raise ValueError(f"Unknown manifold: {manifold}")

    return anchors.astype(np.float32)

import torch.nn as nn
import torch.nn.functional as F

# Custom training function for probabilistic model
def train_probabilistic_model(
    model: nn.Module,
    train_loader,
    val_loader,
    anchors: torch.Tensor,
    lr: float,
    num_epochs: int,
    device,
    weight_decay: float = 1e-4,
    grad_clip: float = 1.0,
    scheduler_patience: int = 300,
    scheduler_factor: float = 0.8,
    early_stop: int = 30,
    verbose: bool = True,
):
    import copy
    model = model.to(device)
    anchors = anchors.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", patience=scheduler_patience, factor=scheduler_factor
    )

    train_losses, val_losses, lrs = [], [], []
    best_val, best_epoch = float("inf"), -1
    best_state, patience = None, 0

    for epoch in range(num_epochs):
        # ---- train ----
        model.train()
        tr = 0.0
        n_tr = 0
        is_transformer = (isinstance(model, nn.Sequential) and 
                         len(model) > 0 and 
                         hasattr(model[0], 'transformer_encoder'))
        
        for batch in train_loader:
            x, labels = batch
            x, labels = x.to(device), labels.to(device)
            
            if x.dim() == 2 and is_transformer:
                x = x.unsqueeze(1)

            opt.zero_grad(set_to_none=True)
            logits = model(x)
            if logits.dim() == 3:
                logits = logits[:, -1, :]  
            loss = probabilistic_transformer_loss(logits, labels)
            loss.backward()

            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            opt.step()

            bs = x.shape[0]
            tr += loss.item() * bs
            n_tr += bs

        tr /= n_tr
        train_losses.append(tr)

        # ---- val ----
        model.eval()
        va = 0.0
        n_va = 0
        is_transformer = (isinstance(model, nn.Sequential) and 
                         len(model) > 0 and 
                         hasattr(model[0], 'transformer_encoder'))
        
        with torch.no_grad():
            for batch in val_loader:
                x, y = batch
                x, y = x.to(device), y.to(device)
                if x.dim() == 2 and is_transformer:
                    x = x.unsqueeze(1)
                if y.dim() == 3:
                    y = y[:, -1, :]
                
                loss = F.mse_loss(predict(model, x, anchors.to(device)), y)

                bs = x.shape[0]
                va += loss.item() * bs
                n_va += bs

        va /= n_va
        val_losses.append(va)

        # ---- scheduler ----
        sched.step(va)
        lrs.append(opt.param_groups[0]["lr"])

        # ---- best model ----
        if va < best_val:
            best_val, best_epoch = va, epoch
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1

        if verbose and ((epoch == 0) or ((epoch + 1) % 20 == 0)):
            print(
                f"epoch {epoch+1:4d} | train {tr:.3e} | val {va:.3e} | lr {opt.param_groups[0]['lr']:.1e}"
            )

        if early_stop and patience >= early_stop:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    logs = dict(
        train_losses=train_losses,
        val_losses=val_losses,
        lrs=lrs,
        best_val_loss=best_val,
        best_epoch=best_epoch,
    )
    return model, logs
