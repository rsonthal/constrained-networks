"""
Main training function for all model types.

This module provides a unified training interface that routes to the appropriate
training function based on model type.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Callable
import numpy as np

from flow_matching import FlowVelocityNet
from train_constrained import train_constrained_model
from train_flow_matching import train_flow_matching_model


def train_model(
    dataset: Dict[str, np.ndarray],
    model: nn.Module,
    device: torch.device,
    projection: Optional[Callable] = None,
    flow_matching_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Train a model using the provided dataset, model, and configurations.
    
    Arguments
    ---------
    dataset : dict
        Dictionary containing training data with keys:
        - 'train_init': np.ndarray, shape [N_train, input_dim]
        - 'train_final': np.ndarray, shape [N_train, output_dim]
    model : nn.Module
        Model to train (RegularTransformer, ProjectedTransformer, ProbabilisticTransformer, etc.)
    device : torch.device
        Device to train on (CPU or CUDA)
    projection : callable or None, default=None
        Projection function to apply (for projected/exponential models)
    flow_matching_config : dict or None, default=None
        Flow matching config (required only for flow matching models)
        
    Returns
    -------
    results : dict
        Training results with model, losses, etc.
    """
    # Route to appropriate training function
    if isinstance(model, FlowVelocityNet):
        return train_flow_matching_model(dataset, device)
    else:
        return train_constrained_model(dataset, model, device, projection)
