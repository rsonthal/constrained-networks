"""
Projected Transformer - A generalized transformer model with geometric projections.
This model includes projection hooks that can be customized for different geometries.
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple, Callable

class ProjectedTransformer(nn.Module):
    """
    Projected Transformer model that works with any input dimension.
    
    This transformer includes projection hooks that can be customized for different
    geometries (sphere, SO(3), SE(3), etc.). By default, projections are identity
    functions (no-op) and can be overridden or passed as callable functions.
    
    Parameters
    ----------
    input_size : int
        Dimension of input features. Can be any positive integer.
        Examples:
            - Sphere: 3 (3D points)
            - SO(3): 9 (flattened 3x3 rotation matrices)
            - KITTI: 16 (flattened 4x4 SE(3) matrices) or any other dimension
            - Protein: variable (depends on representation)
    nhead : int, default=8
        Number of attention heads.
    d_hid : int, default=2048
        Dimension of the feedforward network.
    nlayers : int, default=6
        Number of transformer encoder layers.
    dropout : float, default=0.5
        Dropout probability.
    dt : float, default=1.0
        Initial value for learnable residual connection scaling factor.
    use_internal_projection : bool, default=True
        If True, applies projection between transformer layers.
        If False, skips internal projections (only end projection is applied).
    proj_func : Callable or None, default=None
        Custom projection function.
        If None, uses the default internal_projection method (identity by default).
    """
    def __init__(
        self, 
        input_dim: int, 
        nhead: int = 8, 
        d_hid: int = 2048,
        nlayers: int = 6, 
        dropout: float = 0.5, 
        dt: float = 1.0,
        use_internal_projection: bool = True,
        proj_func: Optional[Callable] = None,
    ):
        super().__init__()
        self.model_type = 'ProjectedTransformer'
        self.input_dim = input_dim
        self.nlayers = nlayers
        self.use_internal_projection = use_internal_projection
        
        # Store custom projection functions
        self.internal_proj_func = proj_func
        self.end_proj_func = proj_func
       
        # Transformer encoder layers
        self.transformer_encoder = nn.ModuleList([
            nn.TransformerEncoderLayer(input_dim, nhead, d_hid, dropout, batch_first=False)
            for _ in range(nlayers)
        ])
       
        # Input and output layers (generalized to any input_size)
        self.linear = nn.Linear(input_dim, input_dim)  # Output layer matches input dimension
       
        # Learnable residual connection scaling
        self.dt = nn.Parameter(torch.full((nlayers,), float(dt)))

    def forward(self, x: torch.Tensor, return_intermediate: bool = False) -> torch.Tensor:
        """
        Forward pass through the transformer with projections.
        
        Arguments
        ---------
        x : torch.Tensor
            Input tensor of shape [batch_size, seq_len, input_size]
        return_intermediate : bool, default=False
            If True, returns intermediate outputs after each layer.
            
        Returns
        -------
        output : torch.Tensor
            Output tensor of shape [batch_size, seq_len, input_size]
            If return_intermediate=True, returns (intermediates, output) tuple
        """
        # Add positional encoding (transformer expects [seq_len, batch_size, d_model])
        x = x.permute(1, 0, 2)  # [seq_len, batch_size, d_model]

        intermediates = []
        # Process through transformer layers with optional internal projections
        for i in range(self.nlayers):
            x_in = x
            x = self.transformer_encoder[i](x)  # [seq_len, batch_size, d_model]
            x = torch.nn.functional.relu(x)
            
            # Apply internal projection if enabled
            if self.use_internal_projection:
                if self.internal_proj_func is not None:
                    x = self.internal_proj_func(x)
            
            # Residual connection with learnable scaling
            x = x_in + self.dt[i] * x
            
            if return_intermediate:
                intermediates.append(x.clone())
        
        # Convert to [batch_size, seq_len, d_model] for output layer
        x = x.permute(1, 0, 2)
        
        # Final output layer: [batch_size, seq_len, d_model] -> [batch_size, seq_len, input_size]
        output = self.linear(x)
        
        # ALWAYS apply end projection (cannot be disabled)
        if self.end_proj_func is not None:
            output = self.end_proj_func(output)
        
        if return_intermediate:
            return intermediates, output
        return output

class FFBlock(nn.Module):
    """One FF block: d_model -> d_hid -> d_model."""
    def __init__(self, d_model: int, d_hid: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_hid)
        self.act = torch.nn.ReLU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_hid, d_hid)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        return x

class ProjectedFeedForward(nn.Module):
    """
    Stacked FF network with nlayers blocks.
    Output shape matches input shape: [..., d_model]

    By default each block is wrapped in a residual: x <- x + block(x).
    """
    def __init__(
        self,
        d_model: int,
        d_hid: int,
        nlayers: int,
        dropout: float = 0.1,
        proj_func: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        use_internal_projection: bool = True,
        residual: bool = True,
        dt: float = 1.0,   # scale each residual update
    ):
        super().__init__()
        if nlayers < 1:
            raise ValueError("nlayers must be >= 1")
        self.blocks = nn.ModuleList([FFBlock(d_model, d_hid, dropout) for _ in range(nlayers)])
        self.use_internal_projection = use_internal_projection
        self.internal_proj_func = proj_func
        self.final_proj_func = proj_func        
        self.residual = residual
        self.dt = nn.Parameter(torch.full((nlayers,), float(dt)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, blk in enumerate(self.blocks):
            y = blk(x)
            if self.residual:
                x = x + self.dt[i] * y
            else:
                x = y
                
            # Internal projection BETWEEN blocks (i.e., after block i, before block i+1)
            if i < len(self.blocks) - 1 and self.use_internal_projection and self.internal_proj_func is not None:
                x = self.internal_proj_func(x)
                
        # ALWAYS apply final projection (identity if final_proj_func is None)
        if self.final_proj_func is not None:
            x = self.final_proj_func(x)
        return x
